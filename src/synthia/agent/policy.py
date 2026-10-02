"""Which tools may run, decided in code the model cannot reach.

A model only asks for a call. Whether it runs is decided here, from what the
tool declares about itself, so nothing a model reads (a web page, a file, a
tool's output) can grant a permission.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from synthia.agent.tools import Effect, Reach

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from synthia.agent.tools import Tool


class Decision(StrEnum):
    """What happens to a call."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


type Approver = Callable[["Tool", str], Awaitable[bool]]
"""Asked with the tool and the call's arguments; True runs the call."""


async def nobody_approves(_tool: Tool, _arguments: str) -> bool:
    """Answer no: without someone to ask, a call that needs approval never runs."""
    return False


@dataclass(frozen=True, slots=True)
class Policy:
    """Run what only looks at this machine; ask for anything else.

    A tool that changes something, or reaches a service outside, is asked
    about on every call. A name in ``denied`` never runs.
    """

    denied: frozenset[str] = frozenset()

    def decide(self, tool: Tool) -> Decision:
        """Return what happens to a call of ``tool``."""
        if tool.spec.name in self.denied:
            return Decision.DENY
        if tool.reach is Reach.LOCAL and tool.effect is Effect.READ:
            return Decision.ALLOW
        return Decision.ASK
