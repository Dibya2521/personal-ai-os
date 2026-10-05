"""One conversation: its history, its persona, and a report on each turn.

The session knows nothing about terminals, so a web UI or a voice loop can
drive the same object later. Only a turn that finished enters the history and
the memory: an answer that failed or was interrupted is dropped whole, question
and all, so the model is never shown half of an exchange as if it had happened.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from contextlib import aclosing
from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING, Final

from synthia.agent.loop import (
    Agent,
    Finished,
    Limits,
    ModelChunk,
    ModelTurn,
    ToolFinished,
)
from synthia.agent.plan import PlanAnswerBegun, Planned, Planner, PlanStepBegun
from synthia.agent.policy import Policy, nobody_approves
from synthia.agent.tools import Toolbox
from synthia.agent.trace import utc_now
from synthia.gateway.errors import GatewayError, IncompleteResponseError
from synthia.gateway.router import RouteDecided
from synthia.gateway.types import (
    ChatChunk,
    ChatRequest,
    ChatResponse,
    ImagePart,
    Message,
    Reasoning,
)
from synthia.memory.store import ToolUse, Turn
from synthia.memory.window import ContextWindow, message_chars, tools_chars

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable
    from pathlib import Path

    from synthia.agent.plan import PlanEvent
    from synthia.agent.policy import Approver
    from synthia.agent.trace import Trace
    from synthia.gateway.protocol import ChatModel
    from synthia.kernel.bus import Event
    from synthia.memory.remembering import Remembering
    from synthia.memory.summary import Summarizer
    from synthia.persona.library import PersonaLibrary
    from synthia.persona.model import Persona

logger = logging.getLogger(__name__)

MEDIA_TYPES: Final = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
# Inline images grow a third as base64; a wrong path (a video renamed .png)
# should fail here, not as a request of hundreds of megabytes.
MAX_IMAGE_BYTES: Final = 20 * 1024 * 1024
# Each turn's depth is decided from what it asks, until the user sets a level.
DEFAULT_REASONING: Final = Reasoning.AUTO
# No deadline: the answer streams in view and Ctrl+C stops it when the person
# chooses; a long local answer at 2 threads can pass the agent's 5 minutes.
CHAT_LIMITS: Final = Limits(deadline_s=None)

type PlanMark = Planned | PlanStepBegun | PlanAnswerBegun
"""Where a planned turn is: the plan made, a step begun, the answer begun."""
type TurnItem = ChatChunk | ToolFinished | PlanMark | TurnReport
"""What a turn yields: streamed answer, tool calls, plan marks, then its report."""


class ImageError(ValueError):
    """An image cannot be sent: missing, too large, or not a known type."""


def load_image(path: Path) -> ImagePart:
    """Read the image at ``path``.

    Raises:
        ImageError: If it is missing, not a PNG, JPEG, WebP or GIF, or too large.
    """
    media_type = MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None:
        message = f"{path.name} is not a PNG, JPEG, WebP or GIF image"
        raise ImageError(message)
    try:
        size = path.stat().st_size
    except OSError as error:
        message = f"cannot read {path}: {error.strerror}"
        raise ImageError(message) from error
    if size > MAX_IMAGE_BYTES:
        message = f"{path.name} is {size / 2**20:.1f} MB; the limit is 20 MB"
        raise ImageError(message)
    return ImagePart(path.read_bytes(), media_type)


class LastRoute:
    """Remember the router's latest decision; pass it as the gateway's publisher."""

    def __init__(self) -> None:
        self.decision: RouteDecided | None = None

    async def __call__(self, event: Event) -> None:
        """Keep ``event`` if it is a routing decision."""
        if isinstance(event, RouteDecided):
            self.decision = event


@dataclass(frozen=True, slots=True)
class TurnReport:
    """What one finished turn cost and where it went.

    ``reasoning`` is the thinking level the model was sent, with ``auto``
    already decided when a router answered; ``None`` when none was sent.
    """

    route: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    seconds: float
    reasoning: Reasoning | None = None


@dataclass(slots=True)
class _Taken:
    """What a turn's steps leave behind, sorted from what the person is shown."""

    answers: list[ChatResponse] = field(default_factory=list[ChatResponse])
    uses: list[ToolUse] = field(default_factory=list[ToolUse])
    finished: Finished | None = None

    def take(self, step: PlanEvent) -> TurnItem | None:
        """Keep what ``step`` adds to the turn; return it if it is to be shown."""
        match step:
            case ModelChunk(chunk=chunk):
                return chunk
            case ToolFinished(call=call, result=result, ok=ok):
                self.uses.append(ToolUse(call.name, call.arguments, result, ok))
                return step
            case Planned() | PlanStepBegun() | PlanAnswerBegun():
                return step
            case ModelTurn(response=response):
                self.answers.append(response)
            case Finished():
                self.finished = step
            case _:
                pass
        return None


class ChatSession:
    """A conversation with SYNTHIA through one model, usually the router."""

    def __init__(  # noqa: PLR0913
        self,
        model: ChatModel,
        library: PersonaLibrary,
        persona: str,
        routes: LastRoute | None = None,
        clock: Callable[[], float] = time.perf_counter,
        *,
        tools: Toolbox | None = None,
        approver: Approver = nobody_approves,
        trace: Trace | None = None,
        memory: Remembering | None = None,
        summarizer: Summarizer | None = None,
    ) -> None:
        """Start an empty conversation as ``persona``.

        ``routes`` is the recorder the router publishes to, so each turn's
        report can say where the turn went. ``tools`` are what the model may
        call; a call that needs approval is put to ``approver``. ``trace``
        records every step of every turn, including turns that fail;
        ``memory`` keeps each finished turn. Turns that no longer fit in the
        model's window leave it, and ``summarizer`` folds them into a summary
        the model keeps seeing; without one they are only dropped.

        Raises:
            PersonaError: If there is no such persona.
        """
        self._model = model
        self._library = library
        self._clock = clock
        self._approver = approver
        self._trace = trace
        self._summarizer = summarizer
        self.memory = memory
        self.window = ContextWindow(model.info.context_window)
        self.summary = ""
        self._leaving: list[Message] = []
        self._summarizing: asyncio.Task[None] | None = None
        # Where each finished turn's messages begin in the history.
        self._starts: list[int] = []
        self.tools = tools or Toolbox()
        self.policy = Policy()
        self.persona: Persona = library.get(persona)
        self.history: list[Message] = []
        self.routes = routes or LastRoute()
        self.reasoning = DEFAULT_REASONING
        self.use_remote = False

    def persona_names(self) -> tuple[str, ...]:
        """Return the keys of every persona the session can switch to."""
        return self._library.names()

    def switch(self, key: str) -> None:
        """Continue as the persona ``key``, keeping the conversation.

        Raises:
            PersonaError: If there is no such persona.
        """
        self.persona = self._library.get(key)

    def adjust(self, values: dict[str, float]) -> None:
        """Move trait sliders of the current persona from the next turn on.

        Raises:
            PersonaError: If a trait is unknown or a value is outside 0 to 1.
        """
        self.persona = self.persona.adjusted(**values)

    def reset(self) -> None:
        """Forget the conversation; what was remembered stays remembered."""
        self.history.clear()
        self._starts.clear()
        self._leaving.clear()
        self.summary = ""
        if self.memory is not None:
            self.memory.begin_again()

    async def close(self) -> None:
        """Stop writing the summary, if that is under way."""
        if self._summarizing is not None:
            self._summarizing.cancel()
            await asyncio.wait([self._summarizing])

    def drop_last(self) -> bool:
        """Take the last finished turn out of the history; False if there is none."""
        if not self._starts:
            return False
        del self.history[self._starts.pop() :]
        return True

    def request(self, text: str, *images: ImagePart) -> ChatRequest:
        """Return the request a turn saying ``text`` would send."""
        return ChatRequest(
            (self._system(), *self.history, Message.user(text, *images)),
            reasoning=self.reasoning,
            use_remote=self.use_remote,
        )

    def make_room(self, text: str, *images: ImagePart) -> int:
        """Move the oldest turns out of the window until a turn saying ``text`` fits.

        Return how many turns left. They wait to be summarised after the turn.
        """
        bounds = sorted({0, *self._starts}) if self.history else []
        edges = [*bounds, len(self.history)]
        turns = [self.history[a:b] for a, b in pairwise(edges)]
        fixed = (self._system(), Message.user(text, *images))
        leaving = self.window.overflow(fixed, turns, tools_chars(self.tools.specs()))
        if leaving:
            cut = edges[leaving]
            self._leaving.extend(self.history[:cut])
            del self.history[:cut]
            self._starts = [b - cut for b in bounds[leaving:]]
        return leaving

    def _system(self) -> Message:
        prompt = self.persona.system_prompt()
        if self.summary:
            prompt += f"\n\nEarlier in this conversation, in short: {self.summary}"
        return Message.system(prompt)

    async def turn(
        self, text: str, *images: ImagePart, plan: bool = False
    ) -> AsyncGenerator[TurnItem]:
        """Yield the answer as it streams and each tool call as it ends, then a report.

        The turn runs as an agent over :attr:`tools`; with none, it is one
        request, as it always was. With ``plan`` it runs as a plan of steps,
        and the plan, each step's start and the answer's start are yielded too.
        Earlier conversations close to ``text`` go before it in the request,
        never into the history: after the turn the history holds ``text``
        alone, so old reminders do not fill the window, and a server that
        reuses the start of the last prompt has only that question to read again.

        Raises:
            GatewayError: If the answer failed; the history is unchanged.
        """
        asked = await self._asked(text)
        self.make_room(asked, *images)
        request = self.request(asked, *images)
        started = self._clock()
        asked_at = utc_now()
        before = len(self.history)
        taken = _Taken()
        try:
            async with aclosing(self._run(request, text, plan=plan)) as steps:
                async for step in steps:
                    if (shown := taken.take(step)) is not None:
                        yield shown
        except IncompleteResponseError:
            return
        finished = taken.finished
        if finished is None:  # pragma: no cover - the agent always finishes
            return
        # The persona's system message is rebuilt every turn, so it is not kept.
        self.history = list(finished.messages[1:])
        if asked != text:
            self.history[before] = Message.user(text, *images)
        self._starts.append(before)
        report = self._report(request, taken.answers, self._clock() - started)
        if not plan:
            self._learn(request, taken.answers)
        self._summarize_leaving()
        await self._remember(
            Turn(
                text,
                finished.text,
                asked_at,
                self.persona.name,
                report.route,
                report.model,
                tuple(taken.uses),
            )
        )
        yield report

    async def _asked(self, text: str) -> str:
        if self.memory is None:
            return text
        return await self.memory.reminder(text) + text

    async def _remember(self, turn: Turn) -> None:
        if self.memory is not None:
            await self.memory.remember(turn)

    def _learn(self, request: ChatRequest, answers: list[ChatResponse]) -> None:
        usage = answers[0].usage if answers else None
        if usage is not None:
            chars = sum(message_chars(m) for m in request.messages)
            chars += tools_chars(self.tools.specs())
            self.window.learn(chars, usage.prompt_tokens)

    def _summarize_leaving(self) -> None:
        if self._summarizer is None:
            self._leaving.clear()
            return
        if not self._leaving or (
            self._summarizing is not None and not self._summarizing.done()
        ):
            return
        leaving, self._leaving = self._leaving, []
        # A context of its own, so the summary's routing decision is never
        # taken for the next turn's.
        self._summarizing = asyncio.get_running_loop().create_task(
            self._write_summary(self._summarizer, leaving),
            context=contextvars.Context(),
        )

    async def _write_summary(
        self, summarizer: Summarizer, leaving: list[Message]
    ) -> None:
        try:
            self.summary = await summarizer(self.summary, leaving)
        except GatewayError:
            logger.warning("the turns that left the window were not summarised")

    def _run(
        self, request: ChatRequest, text: str, *, plan: bool
    ) -> AsyncGenerator[PlanEvent]:
        agent = Agent(
            self._model,
            self.tools,
            CHAT_LIMITS,
            policy=self.policy,
            approver=self._approver,
        )
        run = Planner(agent).run(request) if plan else agent.run(request)
        if self._trace is None:
            return run
        self._trace.begin(
            text, self.persona.name, self.reasoning, use_remote=self.use_remote
        )
        return self._trace.watch(run, lambda: self.routes.decision)

    def _report(
        self, request: ChatRequest, answers: list[ChatResponse], seconds: float
    ) -> TurnReport:
        route = self.routes.decision
        usages = [a.usage for a in answers]
        counted = all(u is not None for u in usages)
        last = answers[-1].model if answers else None
        return TurnReport(
            route=route.route if route else "direct",
            model=last or (route.model if route else self._model.info.id),
            prompt_tokens=sum(u.prompt_tokens for u in usages if u)
            if counted
            else None,
            completion_tokens=(
                sum(u.completion_tokens for u in usages if u) if counted else None
            ),
            seconds=seconds,
            reasoning=route.reasoning if route else request.reasoning,
        )
