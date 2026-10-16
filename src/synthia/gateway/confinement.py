"""A chat that has read private text stays on this machine until it starts again.

The rule is enforced in the router, because the router is the one place every
request of a turn passes. The chat builds only the first request of a turn;
the agent then copies that request's ``use_remote`` into every later step
(``agent/loop.py``), and the planner builds its plan, step and answer
requests from the same original (``agent/plan.py``). A turn that reads
private text halfway through would let those later requests leave if the
rule lived where the first request is built.

So each chat keeps one :class:`Confinement`, a latch, and puts it in
:data:`CONFINEMENT` for the turn it runs. Whatever reads private text during
the turn closes the latch, and the router reads it for every request, sending
each one local while it is closed. Only starting the chat again opens it.

The latch is not a field of ``ChatRequest``: the response cache keys a
request by serialising every one of its fields (``gateway/cache.py``) and
refuses any value it cannot serialise, a latch object among them.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Final


class Confinement:
    """One chat's latch: open until something private is read, then closed."""

    def __init__(self) -> None:
        """Start open, so the chat's own remote switch decides."""
        self.reason: str | None = None

    @property
    def confined(self) -> bool:
        """Return whether this chat must stay on this machine."""
        return self.reason is not None

    def confine(self, reason: str) -> None:
        """Keep the chat on this machine from now on, for ``reason``.

        A chat already confined keeps its first reason.
        """
        if self.reason is None:
            self.reason = reason

    def release(self) -> None:
        """Open the latch again, for a chat that starts again."""
        self.reason = None


CONFINEMENT: Final[ContextVar[Confinement | None]] = ContextVar(
    "confinement", default=None
)
"""The latch of the chat whose turn is running; None outside a chat's turn."""


def confined() -> bool:
    """Return whether the chat whose turn is running must stay on this machine."""
    latch = CONFINEMENT.get()
    return latch is not None and latch.confined
