"""Run a program once in a process tree, with a time limit and an output limit.

Shared by the tools that run a program per call (Python, outside agents): one
place decides how a run is bounded and how its ending is told to the model.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from dataclasses import dataclass
from typing import IO, TYPE_CHECKING, Final

from synthia.agent.tools import ToolError
from synthia.tools.process_tree import ProcessTree

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

_OWN_PREFIX: Final = "SYNTHIA_"


@dataclass(frozen=True, slots=True)
class Run:
    """What one run printed and how it ended."""

    output: str
    exit_code: int | None
    timed_out: bool = False
    output_cut: bool = False


def environment_without_own(extra: Mapping[str, str]) -> dict[str, str]:
    """Return the environment without any ``SYNTHIA_*`` variable, plus ``extra``.

    SYNTHIA's own settings, its key among them, never reach a program it starts.
    """
    inherited = {
        name: value
        for name, value in os.environ.items()
        if not name.upper().startswith(_OWN_PREFIX)
    }
    return inherited | dict(extra)


async def run_once(  # noqa: PLR0913
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    stdin: bytes,
    timeout_s: float,
    max_output_bytes: int,
    errors: int | IO[bytes] = subprocess.STDOUT,
) -> Run:
    """Run ``argv`` with ``stdin`` as its whole input until it exits or a limit is hit.

    Whatever it left running is ended with it, since that would hold the
    output open.
    """
    output = bytearray()
    overflowed = asyncio.Event()

    def take(data: bytes) -> None:
        output.extend(data)
        if len(output) > max_output_bytes:
            overflowed.set()

    tree = await ProcessTree.start(
        argv, cwd=cwd, env=env, on_output=take, errors=errors
    )
    try:
        tree.send(stdin)
        tree.close_input()
        timed_out = not await _first(tree.exited, overflowed, timeout_s)
    finally:
        await tree.end()
    head = _text(output[:max_output_bytes])
    if timed_out:
        return Run(head, None, timed_out=True)
    if overflowed.is_set():
        return Run(head, None, output_cut=True)
    return Run(_text(output), tree.exit_code)


async def _first(
    exited: asyncio.Event, overflowed: asyncio.Event, timeout_s: float
) -> bool:
    """Wait until the process exits or prints too much; False if time ran out."""
    waits = [asyncio.ensure_future(e.wait()) for e in (exited, overflowed)]
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
        ToolError: If the program failed, ran out of time or printed too much;
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
