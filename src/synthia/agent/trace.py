"""A trace of what the agent did: every step of every turn, one JSON line each.

A chat writes one file per session under ``<home>/traces``, named by the time
it started, so ``synthia trace`` can show afterwards which tools ran, with what,
what came back, and where each model step was sent. The daemon writes one
too, for the services it supervises: when each started, ended and was
restarted. Texts are cut to
:data:`MAX_TRACED_CHARS` with their full length kept: a trace shows what
happened, it is not a copy of every file a tool read. The key never appears,
because no step carries it.
"""

from __future__ import annotations

import logging
import secrets
from contextlib import aclosing
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
)

from synthia.agent.loop import Finished, ModelChunk, ModelTurn, ToolStarted
from synthia.agent.plan import PlanAnswerBegun, Planned, PlanStepBegun
from synthia.gateway.errors import GatewayError
from synthia.kernel.supervisor import ServiceExited, ServiceStarted, ServiceStopped

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from synthia.agent.plan import PlanEvent
    from synthia.gateway.router import RouteDecided
    from synthia.gateway.types import Reasoning
    from synthia.kernel.supervisor import ServiceEvent

logger = logging.getLogger(__name__)

TRACES: Final = Path("traces")
SUFFIX: Final = ".jsonl"
# read_file returns up to 20,000 characters; a tenth shows what a result held
# without the trace growing by every file read.
MAX_TRACED_CHARS: Final = 2000
_STAMP: Final = "%Y%m%dT%H%M%SZ"
_TOKEN_BYTES: Final = 3


def _mend(text: str) -> str:
    # A model's JSON may escape half an emoji ("\ud83d"); that string cannot
    # be written as UTF-8, so it is kept as the visible escape instead.
    return text.encode("utf-8", "backslashreplace").decode("utf-8")


Mended = Annotated[str, AfterValidator(_mend)]


class Clip(BaseModel):
    """The start of a text, and how long the whole text was."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: Mended
    chars: int

    @classmethod
    def of(cls, text: str) -> Clip:
        """Keep the first :data:`MAX_TRACED_CHARS` characters of ``text``."""
        return cls(text=text[:MAX_TRACED_CHARS], chars=len(text))


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    turn: int
    at: datetime


class TurnBegan(_Record):
    """A turn was asked."""

    kind: Literal["turn"] = "turn"
    question: Clip
    persona: Mended
    reasoning: Mended | None
    use_remote: bool


class ModelAnswered(_Record):
    """One model step ended.

    ``route`` is where the router sent it and ``thinking`` the level it was
    sent, ``auto`` already decided; both come from the router, so a step
    sent straight to a model has route ``direct`` and no level.
    """

    kind: Literal["model"] = "model"
    route: Mended
    thinking: Mended | None
    model: Mended | None
    text: Clip
    reasoning_chars: int
    finish: Mended
    prompt_tokens: int | None
    completion_tokens: int | None
    seconds: float


class ToolBegan(_Record):
    """A tool call began."""

    kind: Literal["tool_started"] = "tool_started"
    call_id: Mended
    name: Mended
    arguments: Clip


class ToolEnded(_Record):
    """A tool call ended; ``flags`` name what in the result reads as an instruction."""

    kind: Literal["tool_finished"] = "tool_finished"
    call_id: Mended
    name: Mended
    result: Clip
    ok: bool
    seconds: float
    flags: tuple[Mended, ...]


class PlanMade(_Record):
    """A plan was made, or made again after a step did not finish."""

    kind: Literal["plan"] = "plan"
    steps: tuple[Mended, ...]
    revised: bool


class PlanStepStarted(_Record):
    """Step ``number`` (from 1) of the plan began."""

    kind: Literal["plan_step"] = "plan_step"
    number: int
    text: Mended


class PlanAnswerStarted(_Record):
    """Every step was done; the answer is written from their results."""

    kind: Literal["plan_answer"] = "plan_answer"


class TurnEnded(_Record):
    """The turn was answered, or stopped at a limit."""

    kind: Literal["finished"] = "finished"
    outcome: Mended
    text: Clip


class TurnFailed(_Record):
    """The turn ended without an answer."""

    kind: Literal["failed"] = "failed"
    reason: Mended


class ServiceChanged(_Record):
    """A supervised service started, ended, or stopped for good; never in a turn.

    ``restart_in_s`` is set when an ended service will be started again.
    """

    kind: Literal["service"] = "service"
    service: Mended
    change: Literal["started", "exited", "stopped"]
    attempt: int | None = None
    error: Clip | None = None
    restart_in_s: float | None = None


type Record = Annotated[
    TurnBegan
    | ModelAnswered
    | ToolBegan
    | ToolEnded
    | PlanMade
    | PlanStepStarted
    | PlanAnswerStarted
    | TurnEnded
    | TurnFailed
    | ServiceChanged,
    Field(discriminator="kind"),
]
_RECORD: Final[TypeAdapter[Record]] = TypeAdapter(Record)


def utc_now() -> datetime:
    """Return the time now, in UTC."""
    return datetime.now(UTC)


def record_of(
    step: PlanEvent, *, turn: int, at: datetime, route: RouteDecided | None = None
) -> Record | None:
    """Return the record of ``step``; None for a streamed piece, which is not traced.

    ``route`` is the router's decision for the step's model answer; without
    one the answer came straight from a model.
    """
    match step:
        case ModelChunk():
            return None
        case ModelTurn(response=response, seconds=seconds):
            usage = response.usage
            sent = route.reasoning if route else None
            return ModelAnswered(
                turn=turn,
                at=at,
                route=route.route.value if route else "direct",
                thinking=sent.value if sent else None,
                model=response.model,
                text=Clip.of(response.text),
                reasoning_chars=len(response.reasoning),
                finish=response.finish_reason.value,
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                seconds=seconds,
            )
        case ToolStarted(call=call):
            return ToolBegan(
                turn=turn,
                at=at,
                call_id=call.id,
                name=call.name,
                arguments=Clip.of(call.arguments),
            )
        case Finished(text=text, outcome=outcome):
            return TurnEnded(
                turn=turn, at=at, outcome=outcome.value, text=Clip.of(text)
            )
        case Planned() | PlanStepBegun() | PlanAnswerBegun():
            return _plan_record(step, turn=turn, at=at)
        case _:
            return ToolEnded(
                turn=turn,
                at=at,
                call_id=step.call.id,
                name=step.call.name,
                result=Clip.of(step.result),
                ok=step.ok,
                seconds=step.seconds,
                flags=step.flags,
            )


def _plan_record(
    mark: Planned | PlanStepBegun | PlanAnswerBegun, *, turn: int, at: datetime
) -> Record:
    match mark:
        case Planned(steps=steps, revised=revised):
            return PlanMade(turn=turn, at=at, steps=steps, revised=revised)
        case PlanStepBegun(number=number, text=text):
            return PlanStepStarted(turn=turn, at=at, number=number, text=text)
        case _:
            return PlanAnswerStarted(turn=turn, at=at)


def service_record(event: ServiceEvent, *, at: datetime) -> ServiceChanged:
    """Return the record of a supervisor's ``event``."""
    match event:
        case ServiceStarted(service=name, attempt=attempt):
            return ServiceChanged(
                turn=0, at=at, service=name, change="started", attempt=attempt
            )
        case ServiceExited(service=name, error=error, restart_in_s=delay):
            return ServiceChanged(
                turn=0,
                at=at,
                service=name,
                change="exited",
                error=None if error is None else Clip.of(error),
                restart_in_s=delay,
            )
        case ServiceStopped(service=name):
            return ServiceChanged(turn=0, at=at, service=name, change="stopped")


class Trace:
    """Append one session's records to its file, created at the first record.

    A record that cannot be written is logged and the trace stops, so a full
    disk never ends a conversation.
    """

    def __init__(self, path: Path, now: Callable[[], datetime] = utc_now) -> None:
        self.path = path
        self._now = now
        self._turn = 0
        self._broken = False

    @classmethod
    def start(cls, folder: Path, now: Callable[[], datetime] = utc_now) -> Trace:
        """Return the trace of a session starting now, in ``folder``.

        The name is the start time and a random part, so names sort by time
        and two sessions started in the same second do not share a file.
        """
        name = f"{now().strftime(_STAMP)}-{secrets.token_hex(_TOKEN_BYTES)}"
        return cls(folder / f"{name}{SUFFIX}", now)

    def begin(
        self,
        question: str,
        persona: str,
        reasoning: Reasoning | None,
        *,
        use_remote: bool,
    ) -> None:
        """Record that a new turn asked ``question``."""
        self._turn += 1
        self._write(
            TurnBegan(
                turn=self._turn,
                at=self._now(),
                question=Clip.of(question),
                persona=persona,
                reasoning=None if reasoning is None else reasoning.value,
                use_remote=use_remote,
            )
        )

    async def watch[E: PlanEvent](
        self,
        steps: AsyncGenerator[E],
        route: Callable[[], RouteDecided | None] = lambda: None,
    ) -> AsyncGenerator[E]:
        """Pass ``steps`` on unchanged, recording each into the current turn.

        ``route`` returns the router's latest decision, read as each model
        answer ends. A turn that ends without :class:`Finished` (an error, a
        Ctrl+C, a consumer that stopped reading) is recorded as failed.

        Raises:
            GatewayError: What ``steps`` raised.
        """
        finished = False
        reason = "stopped"
        try:
            async with aclosing(steps) as running:
                async for step in running:
                    record = record_of(
                        step, turn=self._turn, at=self._now(), route=route()
                    )
                    if record is not None:
                        self._write(record)
                    finished = isinstance(step, Finished)
                    yield step
        except GatewayError as error:
            reason = f"no answer: {error}"
            raise
        finally:
            if not finished:
                self._write(TurnFailed(turn=self._turn, at=self._now(), reason=reason))

    def service(self, event: ServiceEvent) -> None:
        """Record a change in a supervised service."""
        self._write(service_record(event, at=self._now()))

    def _write(self, record: Record) -> None:
        if self._broken:
            return
        line = record.model_dump_json() + "\n"
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="") as file:
                file.write(line)
        except OSError:
            self._broken = True
            logger.exception("trace not written; tracing stops for this session")


def read_trace(path: Path) -> tuple[list[Record], int]:
    """Return the records in ``path`` and how many lines could not be read.

    A session stopped hard can leave its last line cut short; that line is
    counted, not fatal.

    Raises:
        OSError: If the file cannot be read.
    """
    records: list[Record] = []
    unreadable = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(_RECORD.validate_json(line))
        except ValidationError:
            unreadable += 1
    return records, unreadable


def sessions(folder: Path) -> list[Path]:
    """Return every trace in ``folder``, oldest first."""
    if not folder.is_dir():
        return []
    return sorted(folder.glob(f"*{SUFFIX}"))
