import sys
from pathlib import Path

import pytest

from synthia.tools.run import environment_without_own, run_once


def test_the_environment_drops_only_synthias_own_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYNTHIA_HOME", "x")
    monkeypatch.setenv("synthia_lower", "x")
    monkeypatch.setenv("NOT_SYNTHIA_HOME", "kept")

    environment = environment_without_own({"ADDED": "1"})

    assert "SYNTHIA_HOME" not in environment
    assert "synthia_lower" not in environment
    assert "SYNTHIA_LOWER" not in environment
    assert (environment["NOT_SYNTHIA_HOME"], environment["ADDED"]) == ("kept", "1")


async def test_errors_sent_to_a_file_stay_out_of_the_result(tmp_path: Path) -> None:
    code = b"import sys\nprint('answer')\nprint('progress', file=sys.stderr)\n"
    with (tmp_path / "errors.log").open("wb") as errors:
        run = await run_once(
            [sys.executable, "-"],
            cwd=tmp_path,
            env=environment_without_own({}),
            stdin=code,
            timeout_s=60,
            max_output_bytes=1000,
            errors=errors,
        )

    assert (run.output, run.exit_code) == ("answer\n", 0)
    assert (tmp_path / "errors.log").read_text().split() == ["progress"]
