"""SYNTHIA's own tools: those that run on this machine, and those that reach outside."""

from __future__ import annotations

from typing import TYPE_CHECKING

from synthia.agent.tools import Toolbox
from synthia.tools.agents import agent_tools
from synthia.tools.basic import calculator_tool, clock_tool
from synthia.tools.files import AllowedRoots, file_tools
from synthia.tools.python import python_tool
from synthia.tools.web import fetch_tool

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    import httpx

    from synthia.agent.tools import FunctionTool


def local_tools(file_roots: Iterable[Path] = ()) -> Toolbox:
    """Return the clock, calculator, files inside ``file_roots``, and Python."""
    return Toolbox(
        (
            clock_tool(),
            calculator_tool(),
            *file_tools(AllowedRoots.of(file_roots)),
            python_tool(),
        )
    )


def outside_tools(
    home: Path, transport: httpx.AsyncBaseTransport | None = None
) -> list[FunctionTool]:
    """Return web fetch and every outside agent on PATH; each call asks first."""
    return [fetch_tool(transport), *agent_tools(home)]
