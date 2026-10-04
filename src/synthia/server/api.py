"""One conversation served over JSON-RPC 2.0.

The client calls methods; while a ``turn`` runs, the daemon tells it what is
happening with notifications, and asks it before each tool call that needs a
yes. Every method's params are checked against a model, so a client with the
wrong shape gets ``-32602`` naming the field, not a failure deep inside.

Methods, client to daemon:

- ``hello`` {} returns the daemon's ``version``, the conversation's ``persona``
  how many ``sessions`` the daemon holds, and the ``warnings`` it had at start
  (an MCP server that did not start, a broken ``mcp.toml``).
- ``turn`` {``text``, ``plan``, ``images``: paths} returns the turn's report,
  after notifications ``chunk``, ``tool`` and ``plan``. An answer that failed
  is error ``-32001`` with the reason, an image that cannot be sent ``-32002``.
- ``think`` {``level``}, ``remote`` {``on``}, ``private`` {``on``}, ``persona``
  {``key``}, ``adjust`` {``values``}, ``budget``, ``model``, ``tools``,
  ``reset``, ``forget`` return a reply: ``notes`` and ``errors``, lines to show.
- ``status`` {} returns how many ``sessions`` the daemon holds and the
  ``local`` model's state: ``ready``, ``starting``, ``stopped`` or ``none``.
- ``stop`` {} stops the daemon.

Request, daemon to client: ``approve`` {``tool``, ``arguments``} returns true to
run the call. A client that is gone, or answers anything but true, is a no.
"""

from __future__ import annotations

import logging
from contextlib import aclosing
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from synthia import __version__
from synthia.agent.loop import ToolFinished
from synthia.agent.plan import PlanAnswerBegun, Planned, PlanStepBegun
from synthia.gateway.errors import GatewayError
from synthia.gateway.types import Reasoning
from synthia.kernel.jsonrpc import ConnectionClosedError, Peer, RpcError
from synthia.server.host import CURRENT_ROUTES
from synthia.server.session import ImageError, TurnReport

if TYPE_CHECKING:
    from collections.abc import Callable

    from synthia.agent.tools import Tool
    from synthia.gateway.types import ChatChunk
    from synthia.kernel.jsonrpc import Method, Params
    from synthia.server.conversation import Reply
    from synthia.server.host import Host
    from synthia.server.session import PlanMark

INVALID_PARAMS: Final = -32602
ANSWER_FAILED: Final = -32001
IMAGE_REFUSED: Final = -32002
APPROVE: Final = "approve"

logger = logging.getLogger(__name__)


class _Params(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TurnParams(_Params):
    """A turn: what was said, whether to plan it, and image files to show."""

    text: str = Field(min_length=1)
    plan: bool = False
    images: tuple[Path, ...] = ()


class ThinkParams(_Params):
    """The thinking level to set, or None to say what it is."""

    level: Reasoning | None = None


class SwitchParams(_Params):
    """A switch to turn on or off, or None to say how it is."""

    on: bool | None = None


class PersonaParams(_Params):
    """The persona to continue as; empty lists them."""

    key: str = ""


class AdjustParams(_Params):
    """Trait sliders to move."""

    values: dict[str, float] = Field(min_length=1)


class EmptyParams(_Params):
    """A method that takes nothing."""


def checked[P: _Params](kind: type[P], params: Params) -> P:
    """Return ``params`` as ``kind``.

    Raises:
        RpcError: ``-32602`` with the problem, if they do not fit.
    """
    try:
        return kind.model_validate(params)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'params'}: {e['msg']}"
            for e in error.errors(include_url=False)
        )
        raise RpcError(INVALID_PARAMS, problems) from None


def reply_of(reply: Reply) -> dict[str, object]:
    """Return ``reply`` as a method's result."""
    return {"notes": list(reply.notes), "errors": list(reply.errors)}


def chunk_of(chunk: ChatChunk) -> dict[str, object]:
    """Return what a client shows of a streamed chunk."""
    progress = chunk.progress
    return {
        "text": chunk.text,
        "reasoning": chunk.reasoning,
        "tool_calls": bool(chunk.tool_calls),
        "progress": None if progress is None else [progress.processed, progress.total],
    }


def tool_of(finished: ToolFinished) -> dict[str, object]:
    """Return a finished tool call as a client shows it."""
    return {
        "name": finished.call.name,
        "arguments": finished.call.arguments,
        "result": finished.result,
        "ok": finished.ok,
        "seconds": finished.seconds,
        "flags": list(finished.flags),
    }


def mark_of(mark: PlanMark) -> dict[str, object]:
    """Return where a planned turn has got to."""
    match mark:
        case Planned(steps=steps, revised=revised):
            return {"kind": "planned", "steps": list(steps), "revised": revised}
        case PlanStepBegun(number=number, text=text):
            return {"kind": "step", "number": number, "text": text}
        case _:
            return {"kind": "answer"}


def report_of(report: TurnReport) -> dict[str, object]:
    """Return a turn's report as its result."""
    fields = asdict(report)
    fields["reasoning"] = None if report.reasoning is None else report.reasoning.value
    return fields


class Served:
    """One client's conversation, served through ``peer``.

    ``on_stop`` is called when the client asks the daemon to stop.
    """

    def __init__(
        self,
        host: Host,
        send: Callable[[bytes], None],
        on_stop: Callable[[], None],
        sessions: Callable[[], int] = lambda: 1,
    ) -> None:
        self.peer = Peer(send, methods=self._methods())
        self.conversation = host.conversation(self._approve)
        self._on_stop = on_stop
        self._sessions = sessions
        self._warnings = host.warnings
        self._local_state = host.local_state

    async def close(self) -> None:
        """Stop serving: running turns are cancelled, the conversation closed."""
        self.peer.close("the conversation ended")
        await self.conversation.close()

    def _methods(self) -> dict[str, Method]:
        return {
            "hello": self._hello,
            "turn": self._turn,
            "think": self._think,
            "remote": self._remote,
            "persona": self._persona,
            "adjust": self._adjust,
            "budget": self._budget,
            "model": self._model,
            "tools": self._tools,
            "reset": self._reset,
            "private": self._private,
            "forget": self._forget,
            "status": self._status,
            "stop": self._stop,
        }

    async def _hello(self, params: Params) -> object:
        checked(EmptyParams, params)
        return {
            "version": __version__,
            "persona": self.conversation.persona_name,
            "sessions": self._sessions(),
            "warnings": list(self._warnings),
        }

    async def _think(self, params: Params) -> object:
        level = checked(ThinkParams, params).level
        return reply_of(await self.conversation.think(level))

    async def _remote(self, params: Params) -> object:
        on = checked(SwitchParams, params).on
        return reply_of(await self.conversation.remote(on=on))

    async def _private(self, params: Params) -> object:
        on = checked(SwitchParams, params).on
        return reply_of(await self.conversation.private(on=on))

    async def _forget(self, params: Params) -> object:
        checked(EmptyParams, params)
        return reply_of(await self.conversation.forget())

    async def _persona(self, params: Params) -> object:
        key = checked(PersonaParams, params).key
        return reply_of(await self.conversation.persona(key))

    async def _adjust(self, params: Params) -> object:
        values = checked(AdjustParams, params).values
        return reply_of(await self.conversation.adjust(values))

    async def _budget(self, params: Params) -> object:
        checked(EmptyParams, params)
        return reply_of(await self.conversation.budget())

    async def _model(self, params: Params) -> object:
        checked(EmptyParams, params)
        return reply_of(await self.conversation.model())

    async def _tools(self, params: Params) -> object:
        checked(EmptyParams, params)
        return reply_of(await self.conversation.tools())

    async def _reset(self, params: Params) -> object:
        checked(EmptyParams, params)
        return reply_of(await self.conversation.reset())

    async def _status(self, params: Params) -> object:
        checked(EmptyParams, params)
        return {"sessions": self._sessions(), "local": self._local_state()}

    async def _stop(self, params: Params) -> object:
        checked(EmptyParams, params)
        self._on_stop()
        return {}

    async def _turn(self, params: Params) -> object:
        asked = checked(TurnParams, params)
        # This handler runs in its own task, so the setting is this turn's alone.
        CURRENT_ROUTES.set(self.conversation.session.routes)
        try:
            turn = self.conversation.turn(
                asked.text, images=asked.images, plan=asked.plan
            )
        except ImageError as error:
            raise RpcError(IMAGE_REFUSED, str(error)) from None
        report: TurnReport | None = None
        try:
            async with aclosing(turn) as steps:
                async for step in steps:
                    if isinstance(step, TurnReport):
                        report = step
                    else:
                        self._tell(step)
        except GatewayError as error:
            raise RpcError(ANSWER_FAILED, f"no answer: {error}") from None
        return None if report is None else report_of(report)

    def _tell(self, step: ChatChunk | ToolFinished | PlanMark) -> None:
        if isinstance(step, ToolFinished):
            self.peer.notify("tool", tool_of(step))
        elif isinstance(step, Planned | PlanStepBegun | PlanAnswerBegun):
            self.peer.notify("plan", mark_of(step))
        else:
            self.peer.notify("chunk", chunk_of(step))

    async def _approve(self, tool: Tool, arguments: str) -> bool:
        try:
            answer = await self.peer.request(
                APPROVE, {"tool": tool.spec.name, "arguments": arguments}
            )
        except (ConnectionClosedError, RpcError):
            return False
        return answer is True
