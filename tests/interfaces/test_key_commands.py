from pathlib import Path

import keyring
import pytest
from keyring.errors import KeyringLocked
from typer.testing import CliRunner

from synthia.gateway.providers import PROVIDERS_FILE, remote_key, select_remote
from synthia.interfaces.cli import app
from synthia.kernel.config import load_settings
from synthia.kernel.keys import KEY_SERVICE
from tests.keyrings import LockedKeyring, MemoryKeyring

runner = CliRunner()
KEY = "sk-typed-0123456789abcdef"  # pragma: allowlist secret
ENV_KEY = "sk-env-0123456789abcdef"  # pragma: allowlist secret
OPENAI = """
[providers.openai]
base_url = "https://api.openai.com/v1"
model = "a-model"
context_window = 8192
"""


@pytest.fixture(autouse=True)
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Run from an empty directory with one extra provider, openai."""
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    (home / PROVIDERS_FILE).write_text(OPENAI, encoding="utf-8")
    monkeypatch.setenv("SYNTHIA_HOME", str(home))
    return home


def test_set_stores_the_typed_key_without_echoing_it(
    memory_keyring: MemoryKeyring,
) -> None:
    result = runner.invoke(app, ["keys", "set", "openai"], input=f"  {KEY}  \n")

    assert result.exit_code == 0, result.output
    assert memory_keyring.saved == {(KEY_SERVICE, "SYNTHIA_OPENAI_API_KEY"): KEY}
    assert KEY not in result.output
    assert "stored openai's key in the system key store" in result.output


def test_a_stored_key_reaches_its_provider_and_wins_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYNTHIA_OPENAI_API_KEY", ENV_KEY)
    monkeypatch.setenv("SYNTHIA_REMOTE", "openai")
    runner.invoke(app, ["keys", "set", "openai"], input=f"{KEY}\n")

    settings = load_settings(env_file=None)
    key = remote_key(settings, select_remote(settings))

    assert key is not None
    assert key.get_secret_value() == KEY


def test_a_stored_openrouter_key_wins_over_its_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYNTHIA_OPENROUTER_API_KEY", ENV_KEY)
    runner.invoke(app, ["keys", "set", "openrouter"], input=f"{KEY}\n")

    settings = load_settings(env_file=None)
    key = remote_key(settings, select_remote(settings))

    assert key is not None
    assert key.get_secret_value() == KEY


def test_list_says_where_each_key_comes_from_and_never_the_key(
    monkeypatch: pytest.MonkeyPatch, memory_keyring: MemoryKeyring
) -> None:
    runner.invoke(app, ["keys", "set", "openai"], input=f"{KEY}\n")
    monkeypatch.setenv("SYNTHIA_OPENAI_API_KEY", ENV_KEY)
    monkeypatch.setenv("SYNTHIA_OPENROUTER_API_KEY", ENV_KEY)
    runner.invoke(app, ["keys", "remove", "openrouter"])
    memory_keyring.saved[KEY_SERVICE, "SYNTHIA_GONE_API_KEY"] = KEY

    result = runner.invoke(app, ["keys", "list"])

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        "openrouter: .env or environment",
        "openai: key store, SYNTHIA_OPENAI_API_KEY in .env or environment unused",
    ]
    assert KEY not in result.output
    assert ENV_KEY not in result.output


def test_list_names_a_stored_key_no_provider_uses(home: Path) -> None:
    runner.invoke(app, ["keys", "set", "openai"], input=f"{KEY}\n")
    (home / PROVIDERS_FILE).unlink()

    result = runner.invoke(app, ["keys", "list"])

    assert result.stdout.splitlines() == [
        "openrouter: no key",
        "SYNTHIA_OPENAI_API_KEY: key store, no provider uses it",
    ]


def test_remove_deletes_the_key_and_says_when_env_still_has_one(
    monkeypatch: pytest.MonkeyPatch, memory_keyring: MemoryKeyring
) -> None:
    runner.invoke(app, ["keys", "set", "openai"], input=f"{KEY}\n")
    monkeypatch.setenv("SYNTHIA_OPENAI_API_KEY", ENV_KEY)

    removed = runner.invoke(app, ["keys", "remove", "openai"])
    again = runner.invoke(app, ["keys", "remove", "openai"])

    assert memory_keyring.saved == {}
    assert removed.stdout.strip() == (
        "removed openai's key from the system key store; "
        "SYNTHIA_OPENAI_API_KEY in .env or environment is still used; "
        "a running daemon keeps the keys it started with until synthia stop"
    )
    assert again.stdout.strip() == (
        "the system key store held no key for openai; "
        "SYNTHIA_OPENAI_API_KEY in .env or environment is still used"
    )


def test_remove_says_nothing_of_env_when_env_has_no_key() -> None:
    runner.invoke(app, ["keys", "set", "openai"], input=f"{KEY}\n")

    result = runner.invoke(app, ["keys", "remove", "openai"])

    assert result.stdout.strip() == (
        "removed openai's key from the system key store; "
        "a running daemon keeps the keys it started with until synthia stop"
    )


@pytest.mark.parametrize("raised", [KeyringLocked, RuntimeError])
def test_a_locked_store_gives_a_way_out_and_never_a_traceback(
    raised: type[Exception],
) -> None:
    runner.invoke(app, ["keys", "set", "openai"], input=f"{KEY}\n")
    keyring.set_keyring(LockedKeyring(leaked=KEY, error=raised))

    for command in (["keys", "list"], ["keys", "remove", "openai"], ["providers"]):
        result = runner.invoke(app, command)

        assert result.exit_code == 2, (command, result.output)
        assert f"({raised.__name__})" in result.stderr
        assert "delete keys.json in SYNTHIA_HOME" in result.stderr
        assert "Traceback" not in result.output
        assert KEY not in result.output
    assert runner.invoke(app, ["stop"]).stdout.strip() == "no daemon running"


def test_set_refuses_an_unknown_provider_before_asking(
    memory_keyring: MemoryKeyring,
) -> None:
    result = runner.invoke(app, ["keys", "set", "nobody"], input=f"{KEY}\n")

    assert result.exit_code == 2
    assert "no provider is named 'nobody'" in result.stderr
    assert "openai, openrouter" in result.stderr
    assert memory_keyring.saved == {}


def test_set_refuses_a_value_that_cannot_be_a_key(
    memory_keyring: MemoryKeyring,
) -> None:
    result = runner.invoke(app, ["keys", "set", "openai"], input="two words\n")

    assert result.exit_code == 2
    assert "space" in result.stderr
    assert "two words" not in result.output
    assert memory_keyring.saved == {}


def test_remove_refuses_a_name_no_provider_can_have() -> None:
    result = runner.invoke(app, ["keys", "remove", "a b"])

    assert result.exit_code == 2
    assert "is not a SYNTHIA_<NAME>_API_KEY variable" in result.stderr
