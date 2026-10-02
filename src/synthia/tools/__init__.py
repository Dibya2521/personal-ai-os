"""SYNTHIA's own tools, each running on this machine."""

from __future__ import annotations

from typing import TYPE_CHECKING

from synthia.agent.tools import Toolbox
from synthia.tools.basic import calculator_tool, clock_tool
from synthia.tools.files import AllowedRoots, file_tools
from synthia.tools.python import python_tool

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


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
