"""Run Python code the model writes, in a separate process with limits.

This is not an operating-system sandbox. The code runs as the person running
SYNTHIA and may read or change any file they can, or reach the network. What
is limited is time (the whole process tree is killed at the deadline), output
size, the starting directory (a new empty one, deleted afterwards) and the
environment (none of the person's variables, so no keys). That is why every
call asks first.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final

from pydantic import Field

from synthia.agent.tools import Effect, FunctionTool, Reach, ToolError
from synthia.tools.process_tree import ProcessTree

# Below the agent's 60 s limit per call, so the tool stops the code itself and
# reports what it printed, instead of being cancelled with nothing to show.
PYTHON_TIMEOUT_S: Final = 30.0
MAX_CODE_CHARS: Final = 20_000
MAX_OUTPUT_BYTES: Final = 20_000
# -I ignores PYTHON* variables, the user site and the current directory on
# sys.path; -X utf8 makes output UTF-8 whatever the console's code page.
PYTHON_FLAGS: Final = ("-I", "-X", "utf8", "-")


@dataclass(frozen=True, slots=True)
class Run:
    """What one run printed and how it ended."""

    output: str
    exit_code: int | None
    timed_out: bool = False
    output_cut: bool = False


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
        tree = await ProcessTree.start(
            [executable, *PYTHON_FLAGS],
            stdin=code.encode(),
            cwd=work,
            env=child_environment(work),
            limit=max_output_bytes,
        )
        try:
            timed_out = not await _exit_or_overflow(tree, timeout_s)
        finally:
            # Also ends whatever the code left running, which would hold the
            # output open.
            await tree.end()
    head = _text(tree.output[:max_output_bytes])
    if timed_out:
        return Run(head, None, timed_out=True)
    if tree.overflowed.is_set():
        return Run(head, None, output_cut=True)
    return Run(_text(tree.output), tree.exit_code)


async def _exit_or_overflow(tree: ProcessTree, timeout_s: float) -> bool:
    """Wait until the process exits or prints too much; False if time ran out."""
    waits = [asyncio.ensure_future(e.wait()) for e in (tree.exited, tree.overflowed)]
    try:
        _, pending = await asyncio.wait(
            waits, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        for wait in waits:
            wait.cancel()
    return len(pending) < len(waits)


def _text(output: bytes | bytearray) -> str:
    # Windows text mode prints "\r\n"; the model gets the same text on every system.
    return bytes(output).decode("utf-8", "replace").replace("\r\n", "\n")


def describe_run(run: Run, timeout_s: float, max_output_bytes: int) -> str:
    """Return the result text for the model.

    Raises:
        ToolError: If the code failed, ran out of time or printed too much;
            the message carries what it printed.
    """
    printed = run.output or "(nothing printed)"
    if run.timed_out:
        message = f"stopped after {timeout_s:g} s; printed so far:\n{printed}"
        raise ToolError(message)
    if run.output_cut:
        message = (
            f"stopped: the output passed {max_output_bytes:,} bytes; "
            f"the first {max_output_bytes:,}:\n{printed}"
        )
        raise ToolError(message)
    if run.exit_code:
        message = f"exit code {run.exit_code}:\n{printed}"
        raise ToolError(message)
    return printed


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
