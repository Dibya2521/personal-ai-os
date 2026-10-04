"""The commands about one conversation, answered as text, with no terminal.

A command (switch persona, allow the remote, show the budget) changes the
session or reads the gateway and says what happened in a :class:`Reply`. The
terminal prints a reply; the daemon sends it to its client. Either way the
words are the same, because they are made here once.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from synthia.agent.policy import Decision
from synthia.persona.model import PersonaError

if TYPE_CHECKING:
    from synthia.gateway.assemble import Gateway
    from synthia.gateway.types import Reasoning
    from synthia.server.session import ChatSession

RULE_SHOWN: Final = {
    Decision.ALLOW: "",
    Decision.ASK: " | asks first",
    Decision.DENY: " | never runs",
}
NO_REMOTE: Final = (
    "no remote model is configured: set SYNTHIA_OPENROUTER_API_KEY in .env"
)


@dataclass(frozen=True, slots=True)
class Reply:
    """What a command says back: lines of news, and lines of what went wrong."""

    notes: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


def note(*lines: str) -> Reply:
    """Return a reply of ``lines`` of news."""
    return Reply(notes=lines)


def error(*lines: str) -> Reply:
    """Return a reply of ``lines`` saying what went wrong."""
    return Reply(errors=lines)


class Conversation:
    """One conversation and the commands about it."""

    def __init__(self, session: ChatSession, gateway: Gateway) -> None:
        self.session = session
        self.gateway = gateway
        self.learning: asyncio.Task[int | None] | None = None

    def think(self, level: Reasoning | None) -> Reply:
        """Set how much each answer may think, or with None say what it is."""
        if level is not None:
            self.session.reasoning = level
        return note(f"thinking: {self.session.reasoning.value}")

    def remote(self, *, on: bool | None) -> Reply:
        """Allow or stop turns going to the remote model, or with None say which."""
        if on and self.gateway.router.remote is None:
            return error(NO_REMOTE)
        if on is not None:
            self.session.use_remote = on
        if self.session.use_remote and self.learning is None:
            # Not before remote is on, so the key never leaves unasked; beside
            # the turns, so no answer waits for it.
            self.learning = asyncio.get_running_loop().create_task(
                self.gateway.learn_daily_cap()
            )
        if self.session.use_remote:
            return note("remote: on, turns may leave this machine")
        return note("remote: off, every turn stays on this machine")

    def persona(self, key: str) -> Reply:
        """Continue as the persona ``key``; an empty key lists them."""
        if not key:
            return note(f"personas: {', '.join(self.session.persona_names())}")
        try:
            self.session.switch(key)
        except PersonaError as problem:
            return error(str(problem))
        return self._now()

    def adjust(self, values: dict[str, float]) -> Reply:
        """Move some trait sliders of the current persona."""
        try:
            self.session.adjust(values)
        except PersonaError as problem:
            return error(str(problem))
        return self._now()

    async def budget(self) -> Reply:
        """Say how much of today's remote budget is used."""
        health = self.gateway.health
        if health is None:
            return note("no remote model is configured, so there is no budget")
        status = await health.ledger.status(health.provider)
        return note(
            f"{status.used} of {status.cap} remote requests used today (UTC); "
            f"{status.remaining} left, {health.reserve} kept in reserve"
        )

    def model(self) -> Reply:
        """Say what the model behind the router can do, and where the last turn went."""
        info = self.gateway.router.info
        images = "yes" if info.vision else "no"
        tools = "yes" if info.tools else "no"
        first = f"context {info.context_window} tokens, images {images}, tools {tools}"
        last = self.session.routes.decision
        if last is None:
            return note(first, "no turn yet")
        return note(first, f"last turn: {last.route} to {last.model} ({last.reason})")

    def tools(self) -> Reply:
        """List the tools, where each works, what it may change and whether it asks."""
        tools = list(self.session.tools)
        if not tools:
            return note("no tools")
        return note(
            *(
                f"{tool.spec.name} | {tool.reach.value}, {tool.effect.value}"
                f"{RULE_SHOWN[self.session.policy.decide(tool)]} | "
                f"{tool.spec.description}"
                for tool in tools
            )
        )

    def reset(self) -> Reply:
        """Forget the conversation so far."""
        self.session.reset()
        return note("conversation forgotten")

    async def close(self) -> None:
        """Stop asking for the daily cap, if that is still under way."""
        if self.learning is not None:
            self.learning.cancel()
            await asyncio.wait([self.learning])

    def _now(self) -> Reply:
        persona = self.session.persona
        sliders = ", ".join(
            f"{k}={v:g}" for k, v in persona.traits.model_dump().items()
        )
        return note(f"now {persona.name}: {sliders}")
