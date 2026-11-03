import json
import logging
import tempfile
from pathlib import Path

import keyring
import pytest
from keyring.errors import KeyringLocked
from pydantic import SecretStr

from synthia.kernel.config import load_settings
from synthia.kernel.errors import ConfigError
from synthia.kernel.keys import (
    KEY_SERVICE,
    KEYS_INDEX_FILE,
    MAX_KEY_CHARS,
    MAX_KEYS_INDEX_BYTES,
    KeyStore,
    KeyStoreError,
    SystemCredentials,
    check_key,
)
from synthia.kernel.logs import logging_to
from tests.keyrings import LockedKeyring, MemoryKeyring

OPENAI = "SYNTHIA_OPENAI_API_KEY"
KEY = "sk-stored-0123456789abcdef"  # pragma: allowlist secret
ENV_KEY = "sk-from-env-0123456789abcdef"  # pragma: allowlist secret


class Refusing:
    """Credentials that fail the test if anything asks them."""

    def get(self, variable: str) -> str | None:
        raise AssertionError(variable)

    def put(self, variable: str, value: str) -> None:
        raise AssertionError((variable, len(value)))

    def delete(self, variable: str) -> bool:
        raise AssertionError(variable)


def store(home: Path) -> KeyStore:
    return KeyStore(home, SystemCredentials())


def test_set_key_is_loaded_back_and_only_its_name_is_on_disk(
    tmp_path: Path, memory_keyring: MemoryKeyring
) -> None:
    store(tmp_path).set(OPENAI, SecretStr(KEY))

    loaded = store(tmp_path).load()

    assert {n: k.get_secret_value() for n, k in loaded.items()} == {OPENAI: KEY}
    assert memory_keyring.saved == {(KEY_SERVICE, OPENAI): KEY}
    index = (tmp_path / KEYS_INDEX_FILE).read_bytes()
    assert KEY.encode() not in index
    assert json.loads(index) == {"version": 1, "variables": [OPENAI]}
    assert b"\r\n" not in index


def test_set_twice_replaces_the_key_and_lists_it_once(tmp_path: Path) -> None:
    store(tmp_path).set(OPENAI, SecretStr("first"))
    store(tmp_path).set(OPENAI, SecretStr(KEY))

    assert store(tmp_path).names() == (OPENAI,)
    assert store(tmp_path).load()[OPENAI].get_secret_value() == KEY


def test_remove_deletes_the_key_and_its_name(tmp_path: Path) -> None:
    store(tmp_path).set(OPENAI, SecretStr(KEY))

    assert store(tmp_path).remove(OPENAI) is True
    assert store(tmp_path).names() == ()
    assert store(tmp_path).remove(OPENAI) is False


def test_a_listed_name_the_store_lost_is_skipped(
    tmp_path: Path, memory_keyring: MemoryKeyring
) -> None:
    store(tmp_path).set(OPENAI, SecretStr(KEY))
    memory_keyring.saved.clear()

    assert store(tmp_path).load() == {}
    assert store(tmp_path).remove(OPENAI) is False
    assert store(tmp_path).names() == ()


def test_no_index_never_opens_the_store(tmp_path: Path) -> None:
    assert KeyStore(tmp_path, Refusing()).load() == {}


def test_several_keys_are_kept_apart(tmp_path: Path) -> None:
    one = store(tmp_path)
    one.set(OPENAI, SecretStr(KEY))
    one.set("SYNTHIA_DEEPSEEK_API_KEY", SecretStr(ENV_KEY))

    loaded = one.load()

    assert {n: k.get_secret_value() for n, k in loaded.items()} == {
        OPENAI: KEY,
        "SYNTHIA_DEEPSEEK_API_KEY": ENV_KEY,
    }
    assert one.names() == ("SYNTHIA_DEEPSEEK_API_KEY", OPENAI)


def test_an_unreadable_index_is_reported_by_path(tmp_path: Path) -> None:
    (tmp_path / KEYS_INDEX_FILE).mkdir()

    with pytest.raises(KeyStoreError, match=r"cannot read .*keys\.json"):
        KeyStore(tmp_path, Refusing()).names()


def test_an_unwritable_index_is_reported_by_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_: object, **__: object) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", refuse)

    with pytest.raises(KeyStoreError, match=r"cannot write .*keys\.json: Permission"):
        store(tmp_path).set(OPENAI, SecretStr(KEY))


@pytest.mark.parametrize(
    "content",
    [
        b"not json",
        b'{"version": 2, "variables": []}',
        b'{"version": 1, "variables": ["PATH"]}',
        b'{"version": 1, "variables": ["SYNTHIA_X_API_KEY"], "keys": {}}',
        b'{"version": 1, "variables": "SYNTHIA_X_API_KEY"}',
    ],
)
def test_a_damaged_index_is_refused_by_name(tmp_path: Path, content: bytes) -> None:
    (tmp_path / KEYS_INDEX_FILE).write_bytes(content)

    with pytest.raises(KeyStoreError, match=r"keys\.json is damaged; delete it"):
        KeyStore(tmp_path, Refusing()).load()


def test_an_oversized_index_is_refused_before_it_is_parsed(tmp_path: Path) -> None:
    (tmp_path / KEYS_INDEX_FILE).write_bytes(b" " * (MAX_KEYS_INDEX_BYTES + 1))

    with pytest.raises(KeyStoreError, match="larger than"):
        KeyStore(tmp_path, Refusing()).load()


@pytest.mark.parametrize("variable", ["PATH", "SYNTHIA_REMOTE", "synthia_x_api_key"])
def test_only_key_variables_can_be_stored(tmp_path: Path, variable: str) -> None:
    with pytest.raises(KeyStoreError, match="is not a SYNTHIA_<NAME>_API_KEY"):
        KeyStore(tmp_path, Refusing()).set(variable, SecretStr(KEY))


@pytest.mark.parametrize(
    ("typed", "problem"),
    [
        ("", "empty"),
        ("x" * (MAX_KEY_CHARS + 1), "longer than"),
        ("sk-two words", "space"),
        ("sk-line\nbreak", "control"),
        ("sk-café", "outside ASCII"),
        ("sk-\u200bzero-width", "outside ASCII"),
    ],
)
def test_a_typed_value_that_cannot_be_a_key_is_refused_without_echo(
    typed: str, problem: str
) -> None:
    with pytest.raises(KeyStoreError, match=problem) as caught:
        check_key(typed)

    assert typed == "" or typed not in str(caught.value)


def test_a_key_of_the_longest_length_is_accepted() -> None:
    assert check_key("k" * MAX_KEY_CHARS).get_secret_value() == "k" * MAX_KEY_CHARS


@pytest.mark.parametrize("raised", [KeyringLocked, RuntimeError, OSError])
def test_a_failing_store_names_only_the_error_type(
    tmp_path: Path, raised: type[Exception]
) -> None:
    store(tmp_path).set(OPENAI, SecretStr(KEY))
    keyring.set_keyring(LockedKeyring(leaked=KEY, error=raised))

    with pytest.raises(KeyStoreError, match=rf"\({raised.__name__}\)") as caught:
        store(tmp_path).load()
    with pytest.raises(KeyStoreError) as on_write:
        store(tmp_path).set(OPENAI, SecretStr(KEY))
    with pytest.raises(KeyStoreError) as on_delete:
        store(tmp_path).remove(OPENAI)

    for error in (caught, on_write, on_delete):
        assert KEY not in str(error.value)
        assert error.value.__cause__ is None
        assert error.value.__context__ is None


def test_a_locked_store_is_not_opened_when_stored_keys_are_not_wanted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SYNTHIA_HOME", str(tmp_path))
    store(tmp_path).set(OPENAI, SecretStr(KEY))
    keyring.set_keyring(LockedKeyring(leaked=KEY))

    assert load_settings(env_file=None, stored_keys=False).api_keys() == {}


def test_settings_prefer_the_stored_key_over_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SYNTHIA_HOME", str(tmp_path))
    monkeypatch.setenv(OPENAI, ENV_KEY)
    monkeypatch.setenv("SYNTHIA_DEEPSEEK_API_KEY", ENV_KEY)
    store(tmp_path).set(OPENAI, SecretStr(KEY))

    keys = load_settings(env_file=None).api_keys()

    assert keys[OPENAI].get_secret_value() == KEY
    assert keys["SYNTHIA_DEEPSEEK_API_KEY"].get_secret_value() == ENV_KEY


def test_settings_fail_loudly_when_stored_keys_cannot_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SYNTHIA_HOME", str(tmp_path))
    store(tmp_path).set(OPENAI, SecretStr(KEY))
    keyring.set_keyring(LockedKeyring(leaked=KEY))

    with pytest.raises(ConfigError, match="could not read the system key store"):
        load_settings(env_file=None)


def test_a_stored_key_is_masked_in_logs_and_absent_from_repr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SYNTHIA_HOME", str(tmp_path))
    store(tmp_path).set(OPENAI, SecretStr(KEY))
    settings = load_settings(env_file=None)
    log = tmp_path / "synthia.log"
    refused = f"provider refused {KEY}"

    with logging_to(settings, log):
        try:
            raise RuntimeError(refused)  # noqa: TRY301
        except RuntimeError:
            logging.getLogger("synthia.test").exception("sent %s", KEY)

    written = log.read_text(encoding="utf-8")
    assert KEY not in written
    assert "sent " in written
    assert "provider refused" in written
    assert KEY not in repr(settings)
    assert KEY not in str(settings.model_dump())
