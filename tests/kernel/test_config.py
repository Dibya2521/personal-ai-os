import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from synthia.kernel.config import (
    DEFAULT_DISK_BUDGET_GB,
    DEFAULT_REMOTE_DAILY_CAP,
    ENV_PREFIX,
    LogFormat,
    LogLevel,
    Settings,
    default_home,
    load_settings,
)
from synthia.kernel.errors import ConfigError, SynthiaError

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_VARIABLE = re.compile(r"^([A-Z][A-Z0-9_]*)=", re.MULTILINE)


def test_defaults_apply_when_nothing_is_set() -> None:
    settings = load_settings(env_file=None)

    assert settings.home == default_home()
    assert settings.log_level is LogLevel.INFO
    assert settings.log_format is LogFormat.CONSOLE
    assert settings.disk_budget_gb == DEFAULT_DISK_BUDGET_GB
    assert settings.openrouter_api_key is None
    assert settings.remote_daily_cap == DEFAULT_REMOTE_DAILY_CAP == 50


def test_environment_variables_are_read_with_the_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SYNTHIA_HOME", str(tmp_path))
    monkeypatch.setenv("SYNTHIA_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("SYNTHIA_LOG_FORMAT", "json")
    monkeypatch.setenv("SYNTHIA_DISK_BUDGET_GB", "2.5")
    monkeypatch.setenv("SYNTHIA_REMOTE_DAILY_CAP", "1000")

    settings = load_settings(env_file=None)

    assert settings.home == tmp_path
    assert settings.log_level is LogLevel.DEBUG
    assert settings.log_format is LogFormat.JSON
    assert settings.disk_budget_gb == 2.5
    assert settings.remote_daily_cap == 1000


def test_env_file_is_read_and_a_real_variable_beats_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("SYNTHIA_LOG_LEVEL=WARNING\nSYNTHIA_DISK_BUDGET_GB=3\n")
    monkeypatch.setenv("SYNTHIA_DISK_BUDGET_GB", "4")

    settings = load_settings(env_file=env_file)

    assert settings.log_level is LogLevel.WARNING
    assert settings.disk_budget_gb == 4


def test_missing_env_file_is_not_an_error(tmp_path: Path) -> None:
    assert load_settings(env_file=tmp_path / "absent.env").log_level is LogLevel.INFO


def test_an_empty_value_means_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNTHIA_LOG_LEVEL", "")
    monkeypatch.setenv("SYNTHIA_DISK_BUDGET_GB", "")

    settings = load_settings(env_file=None)

    assert settings.log_level is LogLevel.INFO
    assert settings.disk_budget_gb == DEFAULT_DISK_BUDGET_GB


def test_a_secret_never_appears_in_repr_or_str(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planted = "sk-or-v1-not-a-real-key-0123456789"
    monkeypatch.setenv("SYNTHIA_OPENROUTER_API_KEY", planted)

    settings = load_settings(env_file=None)

    assert planted not in repr(settings)
    assert planted not in str(settings)
    assert settings.openrouter_api_key is not None
    assert settings.openrouter_api_key.get_secret_value() == planted


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("SYNTHIA_LOG_LEVEL", "LOUD-and-secret"),
        ("SYNTHIA_LOG_FORMAT", "xml-and-secret"),
        ("SYNTHIA_DISK_BUDGET_GB", "-7"),
        ("SYNTHIA_DISK_BUDGET_GB", "lots"),
        ("SYNTHIA_REMOTE_DAILY_CAP", "-3"),
    ],
)
def test_invalid_values_name_the_variable_but_not_the_value(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)

    with pytest.raises(ConfigError) as caught:
        load_settings(env_file=None)

    assert variable in str(caught.value)
    assert value not in str(caught.value)
    assert isinstance(caught.value, SynthiaError)
    assert caught.value.__cause__ is None


def test_settings_are_immutable() -> None:
    settings = load_settings(env_file=None)

    with pytest.raises(ValidationError):
        settings.log_level = LogLevel.DEBUG  # pyright: ignore[reportAttributeAccessIssue]


def test_env_example_documents_exactly_the_settings() -> None:
    example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    documented = set(EXAMPLE_VARIABLE.findall(example))
    declared = {f"{ENV_PREFIX}{name.upper()}" for name in Settings.model_fields}

    assert documented == declared


def test_env_example_holds_no_values() -> None:
    example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")

    assert re.findall(r"^[A-Z0-9_]+=(.+)$", example, re.MULTILINE) == []
