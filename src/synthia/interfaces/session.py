"""One conversation: its history, its persona, and a report on each turn.

The session knows nothing about terminals, so a web UI or a voice loop can
drive the same object later. Only a turn that finished enters the history: an
answer that failed or was interrupted is dropped whole, question and all, so
the model is never shown half of an exchange as if it had happened.
"""

from __future__ import annotations

import time
from contextlib import aclosing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from synthia.agent.loop import (
    Agent,
    Finished,
    Limits,
    ModelChunk,
    ModelTurn,
    ToolFinished,
)
from synthia.agent.policy import Policy, nobody_approves
from synthia.agent.tools import Toolbox
from synthia.gateway.errors import IncompleteResponseError
from synthia.gateway.router import RouteDecided
from synthia.gateway.types import (
    ChatChunk,
    ChatRequest,
    ImagePart,
    Message,
    Reasoning,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable
    from pathlib import Path

    from synthia.agent.policy import Approver
    from synthia.gateway.protocol import ChatModel
    from synthia.gateway.types import ChatResponse
    from synthia.kernel.bus import Event
    from synthia.persona.library import PersonaLibrary
    from synthia.persona.model import Persona

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
    ) -> None:
        """Start an empty conversation as ``persona``.

        ``routes`` is the recorder the router publishes to, so each turn's
        report can say where the turn went. ``tools`` are what the model may
        call; a call that needs approval is put to ``approver``.

        Raises:
            PersonaError: If there is no such persona.
        """
        self._model = model
        self._library = library
        self._clock = clock
        self._approver = approver
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
        """Forget the conversation."""
        self.history.clear()

    def request(self, text: str, *images: ImagePart) -> ChatRequest:
        """Return the request a turn saying ``text`` would send."""
        system = Message.system(self.persona.system_prompt())
        return ChatRequest(
            (system, *self.history, Message.user(text, *images)),
            reasoning=self.reasoning,
            use_remote=self.use_remote,
        )

    async def turn(
        self, text: str, *images: ImagePart
    ) -> AsyncGenerator[ChatChunk | ToolFinished | TurnReport]:
        """Yield the answer as it streams and each tool call as it ends, then a report.

        The turn runs as an agent over :attr:`tools`; with none, it is one
        request, as it always was.

        Raises:
            GatewayError: If the answer failed; the history is unchanged.
        """
        request = self.request(text, *images)
        started = self._clock()
        agent = Agent(
            self._model,
            self.tools,
            CHAT_LIMITS,
            policy=self.policy,
            approver=self._approver,
        )
        answers: list[ChatResponse] = []
        finished: Finished | None = None
        try:
            async with aclosing(agent.run(request)) as steps:
                async for step in steps:
                    if isinstance(step, ModelChunk):
                        yield step.chunk
                    elif isinstance(step, ToolFinished):
                        yield step
                    elif isinstance(step, ModelTurn):
                        answers.append(step.response)
                    elif isinstance(step, Finished):
                        finished = step
        except IncompleteResponseError:
            return
        if finished is None:  # pragma: no cover - the agent always finishes
            return
        # The persona's system message is rebuilt every turn, so it is not kept.
        self.history = list(finished.messages[1:])
        yield self._report(request, answers, self._clock() - started)

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
