"""Run Python code the model writes, in a separate process with limits.

This is not an operating-system sandbox. The code runs as the person running
SYNTHIA and may read or change any file they can, or reach the network. What
is limited is time (the whole process tree is killed at the deadline), output
size, the starting directory (a new empty one, deleted afterwards) and the
environment (none of the person's variables, so no keys). That is why every
call asks first.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Annotated, Final

from pydantic import Field

from synthia.agent.tools import Effect, FunctionTool, Reach
from synthia.tools.run import Run, describe_run, run_once

# Below the agent's 60 s limit per call, so the tool stops the code itself and
# reports what it printed, instead of being cancelled with nothing to show.
PYTHON_TIMEOUT_S: Final = 30.0
MAX_CODE_CHARS: Final = 20_000
MAX_OUTPUT_BYTES: Final = 20_000
# -I ignores PYTHON* variables, the user site and the current directory on
# sys.path; -X utf8 makes output UTF-8 whatever the console's code page.
PYTHON_FLAGS: Final = ("-I", "-X", "utf8", "-")


def child_environment(work: Path) -> dict[str, str]:
    """Return the only variables the code sees: temp files go into ``work``."""
    environment = {"TEMP": str(work), "TMP": str(work), "TMPDIR": str(work)}
    if sys.platform == "win32":
        # Without it Python on Windows cannot get random numbers and will not start.
        environment["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
    return environment


async def run_code(
    code: str,
    *,
    timeout_s: float = PYTHON_TIMEOUT_S,
    max_output_bytes: int = MAX_OUTPUT_BYTES,
    executable: str = sys.executable,
) -> Run:
    """Run ``code`` in a new Python process in a new empty directory."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        work = Path(directory)
        return await run_once(
            [executable, *PYTHON_FLAGS],
            cwd=work,
            env=child_environment(work),
            stdin=code.encode(),
            timeout_s=timeout_s,
            max_output_bytes=max_output_bytes,
        )


def python_tool(
    *,
    timeout_s: float = PYTHON_TIMEOUT_S,
    max_output_bytes: int = MAX_OUTPUT_BYTES,
    executable: str = sys.executable,
) -> FunctionTool:
    """Return the tool that runs Python code; every call asks the person first."""

    async def run_python(
        code: Annotated[
            str,
            Field(
                min_length=1,
                max_length=MAX_CODE_CHARS,
                description="a whole Python 3 program; print what you need back",
            ),
        ],
    ) -> str:
        """Run a Python 3 program in a new process and return what it printed.

        It starts in a new empty directory and is stopped after a time limit.
        """
        run = await run_code(
            code,
            timeout_s=timeout_s,
            max_output_bytes=max_output_bytes,
            executable=executable,
        )
        return describe_run(run, timeout_s, max_output_bytes)

    return FunctionTool.of(run_python, reach=Reach.LOCAL, effect=Effect.CHANGE)
