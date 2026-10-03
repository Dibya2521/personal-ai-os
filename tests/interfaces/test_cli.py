import io
import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from synthia import __version__
from synthia.interfaces import doctor
from synthia.interfaces.cli import app, main

runner = CliRunner()


@pytest.fixture(autouse=True)
def no_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run from an empty directory, so no developer .env is ever read."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SYNTHIA_HOME", str(tmp_path / "home"))


def test_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == f"synthia {__version__}"


def test_half_an_emoji_is_printed_escaped_instead_of_ending_the_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    written = io.BytesIO()
    console = io.TextIOWrapper(written, encoding="utf-8", errors="strict")
    monkeypatch.setattr(sys, "stdout", console)

    main()
    sys.stdout.write("a\ud83db")
    console.flush()

    assert written.getvalue() == b"a\\ud83db"


def test_an_output_that_cannot_be_reconfigured_is_left_as_it_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = io.StringIO()
    monkeypatch.setattr(sys, "stdout", text)

    main()
    sys.stdout.write("a\ud83db")

    assert text.getvalue() == "a\ud83db"


def test_no_arguments_prints_help() -> None:
    result = runner.invoke(app, [])

    assert "doctor" in result.stdout


def test_doctor_json_lists_every_check_and_exits_with_the_worst() -> None:
    result = runner.invoke(app, ["doctor", "--json"])

    report = json.loads(result.stdout)
    assert {c["name"] for c in report} >= {"python", "memory", "disk", "openrouter"}
    worst = max(doctor.Status[c["status"].upper()] for c in report)
    assert result.exit_code == worst


def test_doctor_table_is_readable_text() -> None:
    result = runner.invoke(app, ["doctor"])

    assert "python" in result.stdout
    assert "openrouter" in result.stdout


def test_doctor_is_ok_with_a_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNTHIA_OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("SYNTHIA_DISK_BUDGET_GB", "0.001")

    result = runner.invoke(app, ["doctor", "--json"])

    statuses = {c["name"]: c["status"] for c in json.loads(result.stdout)}
    assert statuses["openrouter"] == "ok"
    assert "sk-test" not in result.stdout


def test_a_variable_that_is_not_a_setting_is_named_on_stderr_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text("SYNTHIA_REMOTE_RPM=5\n")
    monkeypatch.setenv("SYNTHIA_OPENROUTER_MODEL", "vendor/model")

    result = runner.invoke(app, ["doctor", "--json"])

    assert result.stderr == (
        "warning: not a setting, ignored: SYNTHIA_OPENROUTER_MODEL, "
        "SYNTHIA_REMOTE_RPM (.env.example lists every setting)\n"
    )
    assert {c["name"] for c in json.loads(result.stdout)} >= {"python", "openrouter"}
    assert "SYNTHIA_REMOTE_RPM" not in result.stdout
    assert "SYNTHIA_OPENROUTER_MODEL" not in result.stdout


def test_with_only_settings_set_there_is_no_warning() -> None:
    result = runner.invoke(app, ["doctor", "--json"])

    assert "warning" not in result.stderr


def test_invalid_configuration_fails_without_echoing_the_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYNTHIA_LOG_LEVEL", "sk-or-v1-pasted-in-the-wrong-place")

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == doctor.Status.FAIL
    assert "SYNTHIA_LOG_LEVEL" in result.stderr
    assert "sk-or-v1" not in result.stderr + result.stdout
