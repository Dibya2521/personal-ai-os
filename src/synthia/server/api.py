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
  ``reset``, ``forget``, ``facts``, ``forget_fact`` {``fact``: its number},
  ``forget_about`` {``words``}, ``resume`` return a reply: ``notes`` and
  ``errors``, lines to show. ``resume`` takes up the newest remembered
  conversation, before the first turn.
- ``status`` {} returns how many ``sessions`` the daemon holds and the
  ``local`` model's state: ``ready``, ``starting``, ``stopped`` or ``none``.
- ``consolidate`` {} merges the facts held that the local model calls one,
  once no turn is being learned from, and returns ``asked``, ``merged``,
  ``seconds``, ``scheduled``, ``stopped`` and ``error`` (null, or why the run
  failed). A daemon that learns no facts (no local model, or no embedding
  model) answers error ``-32003``.
- ``knowledge_add`` {``root``: absolute, ``private``, ``excludes``} adds a folder as a
  source, or changes the flags of one added before (``private`` and
  ``excludes`` null keep them), and returns it: ``name``, ``root``, ``kind``,
  ``private``, ``excludes``. ``knowledge_remove`` {``root``} removes one and
  everything read from it, returning ``removed``; ``knowledge_sources`` {}
  returns ``sources``, in the order added.
- ``knowledge_index`` {} reads every source's folder now, without waiting
  for turns, and returns the pass's counts (``added``, ``changed``,
  ``unchanged``, ``removed``, ``unreadable``, ``held_back``, ``passed_over``,
  ``chunks``, ``cut``, ``embedded``), ``unavailable`` sources, ``seconds``
  and ``error`` null; a pass that failed returns ``error`` alone.
- ``knowledge_search`` {``query``, ``limit``: 1 to 50, default 6,
  ``variant``} returns ``passages`` best first, each with its ``citation``
  (``source/path:start-end``), ``title``, ``header``, ``body``, ``score``
  and ``private``, and ``withheld``. Private sources are searched: the
  client is the person's own. A daemon with no embedding model has no
  index and answers each of these with error ``-32004``; a folder knowledge
  refuses is error ``-32005`` with the reason.
- ``stop`` {} stops the daemon.

Request, daemon to client: ``approve`` {``tool``, ``arguments``} returns true to
run the call. A client that is gone, or answers anything but true, is a no.
"""

from __future__ import annotations

import logging
from contextlib import aclosing
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Annotated, Final

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError

from synthia import __version__
from synthia.agent.loop import ToolFinished
from synthia.agent.plan import PlanAnswerBegun, Planned, PlanStepBegun
from synthia.gateway.errors import GatewayError
from synthia.gateway.types import Reasoning
from synthia.kernel.jsonrpc import ConnectionClosedError, Peer, RpcError
from synthia.knowledge.retrieve import Variant, VariantError
from synthia.knowledge.service import SEARCH_LIMIT, KnowledgeError
from synthia.server.host import CURRENT_ROUTES
from synthia.server.session import ImageError, TurnReport

if TYPE_CHECKING:
    from collections.abc import Callable

    from synthia.agent.tools import Tool
    from synthia.gateway.types import ChatChunk
    from synthia.kernel.jsonrpc import Method, Params
    from synthia.knowledge.model import Source
    from synthia.knowledge.retrieve import Passage
    from synthia.server.conversation import Reply
    from synthia.server.host import Host
    from synthia.server.session import PlanMark
    from synthia.server.upkeep import Indexed

INVALID_PARAMS: Final = -32602
ANSWER_FAILED: Final = -32001
IMAGE_REFUSED: Final = -32002
NO_LEARNING: Final = -32003
NOT_LEARNING: Final = (
    "this daemon learns no facts: it needs a local model and an embedding model"
)
NO_KNOWLEDGE: Final = -32004
NOT_INDEXING: Final = (
    "no knowledge index: it needs an embedding model; "
    "install one with `synthia models install`"
)
KNOWLEDGE_REFUSED: Final = -32005
MAX_SEARCH_LIMIT: Final = 50
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


class FactParams(_Params):
    """The number of a fact, as ``facts`` lists it."""

    fact: int = Field(ge=1)


class WordsParams(_Params):
    """Words that whatever is forgotten holds, all of them."""

    words: str = Field(min_length=1)


class EmptyParams(_Params):
    """A method that takes nothing."""


def _absolute(path: Path) -> Path:
    """Return ``path`` if absolute; a relative one would start at the daemon's folder.

    Raises:
        ValueError: If ``path`` is relative.
    """
    if not path.is_absolute():
        message = "must be an absolute path"
        raise ValueError(message)
    return path


type AbsolutePath = Annotated[Path, AfterValidator(_absolute)]


class SourceParams(_Params):
    """A folder to add as a source; a flag left None keeps what the source has."""

    root: AbsolutePath
    private: bool | None = None
    excludes: tuple[str, ...] | None = None


class RootParams(_Params):
    """The folder of a source."""

    root: AbsolutePath


class SearchParams(_Params):
    """A question, how many passages to return, and the ranking by name."""

    query: str = Field(min_length=1)
    limit: int = Field(default=SEARCH_LIMIT, ge=1, le=MAX_SEARCH_LIMIT)
    variant: str | None = None


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


def source_of(source: Source) -> dict[str, object]:
    """Return a knowledge source as a client shows it."""
    return {
        "name": source.name,
        "root": str(source.root),
        "kind": source.kind.value,
        "private": source.private,
        "excludes": [exclude.as_posix() for exclude in source.excludes],
    }


def indexed_of(done: Indexed) -> dict[str, object]:
    """Return an asked pass of the knowledge index as its result."""
    fields: dict[str, object] = {} if done.report is None else asdict(done.report)
    fields["error"] = done.error
    return fields


def passage_of(passage: Passage) -> dict[str, object]:
    """Return a passage found in knowledge as a client shows it."""
    return {
        "citation": passage.citation(),
        "title": passage.document.title,
        "header": passage.chunk.header,
        "body": passage.chunk.body,
        "score": passage.score,
        "private": passage.private,
    }


def _needed[T](held: T | None) -> T:
    """Return ``held``, the daemon's knowledge or its re-indexing.

    Raises:
        RpcError: ``-32004`` if the daemon has none.
    """
    if held is None:
        raise RpcError(NO_KNOWLEDGE, NOT_INDEXING)
    return held


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
        self._learning = host.learning
        self._knowledge = host.knowledge
        self._reindexing = host.reindexing
        self._activity = host.activity

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
            "facts": self._facts,
            "forget_fact": self._forget_fact,
            "forget_about": self._forget_about,
            "resume": self._resume,
            "status": self._status,
            "consolidate": self._consolidate,
            "knowledge_add": self._knowledge_add,
            "knowledge_remove": self._knowledge_remove,
            "knowledge_sources": self._knowledge_sources,
            "knowledge_index": self._knowledge_index,
            "knowledge_search": self._knowledge_search,
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

    async def _facts(self, params: Params) -> object:
        checked(EmptyParams, params)
        return reply_of(await self.conversation.facts())

    async def _forget_fact(self, params: Params) -> object:
        fact = checked(FactParams, params).fact
        return reply_of(await self.conversation.forget_fact(fact))

    async def _forget_about(self, params: Params) -> object:
        words = checked(WordsParams, params).words
        return reply_of(await self.conversation.forget_about(words))

    async def _resume(self, params: Params) -> object:
        checked(EmptyParams, params)
        return reply_of(await self.conversation.resume())

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

    async def _consolidate(self, params: Params) -> object:
        checked(EmptyParams, params)
        if self._learning is None:
            raise RpcError(NO_LEARNING, NOT_LEARNING)
        return asdict(await self._learning.consolidate())

    async def _knowledge_add(self, params: Params) -> object:
        asked = checked(SourceParams, params)
        knowledge = _needed(self._knowledge)
        excludes = (
            None
            if asked.excludes is None
            else [PurePosixPath(exclude) for exclude in asked.excludes]
        )
        try:
            source = await knowledge.add(
                asked.root, private=asked.private, excludes=excludes
            )
        except KnowledgeError as error:
            raise RpcError(KNOWLEDGE_REFUSED, str(error)) from None
        return source_of(source)

    async def _knowledge_remove(self, params: Params) -> object:
        root = checked(RootParams, params).root
        return {"removed": await _needed(self._knowledge).remove(root)}

    async def _knowledge_sources(self, params: Params) -> object:
        checked(EmptyParams, params)
        sources = await _needed(self._knowledge).sources()
        return {"sources": [source_of(source) for source in sources]}

    async def _knowledge_index(self, params: Params) -> object:
        checked(EmptyParams, params)
        return indexed_of(await _needed(self._reindexing).index(scheduled=False))

    async def _knowledge_search(self, params: Params) -> object:
        asked = checked(SearchParams, params)
        try:
            variant = None if asked.variant is None else Variant.parse(asked.variant)
        except VariantError as error:
            raise RpcError(INVALID_PARAMS, f"variant: {error}") from None
        found = await _needed(self._knowledge).search(
            asked.query, limit=asked.limit, variant=variant, private=True
        )
        return {
            "passages": [passage_of(passage) for passage in found.passages],
            "withheld": found.withheld,
        }

    async def _stop(self, params: Params) -> object:
        checked(EmptyParams, params)
        self._on_stop()
        return {}

    async def _turn(self, params: Params) -> object:
        asked = checked(TurnParams, params)
        with self._activity.turn():
            return await self._answer(asked)

    async def _answer(self, asked: TurnParams) -> object:
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
