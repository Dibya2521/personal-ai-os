"""``synthia chat``: talk to SYNTHIA in the terminal.

Lines are read in the main thread, and each command runs on one long-lived
:class:`asyncio.Runner`. That split is what makes Ctrl+C behave: pressed at
the prompt it ends the chat; pressed while an answer streams, ``Runner.run``
cancels that task, every stream beneath it closes, and the prompt returns.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self

import httpx
from rich.status import Status
from rich.text import Text

from synthia.gateway.assemble import build_gateway
from synthia.gateway.errors import GatewayError
from synthia.interfaces.commands import (
    HELP,
    AdjustPersona,
    Command,
    Exit,
    Help,
    Invalid,
    Reset,
    Say,
    ShowBudget,
    ShowImage,
    ShowModel,
    SwitchPersona,
    Think,
    parse,
)
from synthia.interfaces.session import (
    ChatSession,
    ImageError,
    LastRoute,
    TurnReport,
    load_image,
)
from synthia.persona.library import PersonaLibrary
from synthia.persona.model import PersonaError

if TYPE_CHECKING:
    from collections.abc import Callable

    from rich.console import Console

    from synthia.gateway.assemble import Gateway
    from synthia.gateway.types import ChatChunk, ImagePart
    from synthia.kernel.config import Settings
    from synthia.models.service import LocalService

PROMPT: Final = "you> "
MORE: Final = "...> "
CONTINUES: Final = "\\"
PERSONAS_DIR: Final = Path("personas")
# OpenRouter sends keep-alive comments while a model thinks, so a minute of
# silence mid-stream means the connection is gone, not that the model is slow.
TIMEOUT: Final = httpx.Timeout(60.0, connect=10.0)

type ReadLine = Callable[[str], str]


def read_message(read: ReadLine) -> str:
    """Read one message; a line ending in a backslash continues on the next.

    Raises:
        EOFError: At the end of input.
    """
    lines = [read(PROMPT)]
    while lines[-1].endswith(CONTINUES):
        lines[-1] = lines[-1].removesuffix(CONTINUES)
        lines.append(read(MORE))
    return "\n".join(lines)


def describe(report: TurnReport) -> str:
    """Return the one dim line shown after an answer."""
    if report.prompt_tokens is None or report.completion_tokens is None:
        tokens = "tokens not reported"
    else:
        tokens = f"{report.prompt_tokens} in, {report.completion_tokens} out"
    level = [] if report.reasoning is None else [f"thinking {report.reasoning.value}"]
    parts = [report.route, report.model, *level, tokens, f"{report.seconds:.1f} s"]
    return " | ".join(parts)


class ThinkingTimer:
    """Seconds since thinking began, read again each time the line is drawn."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._started = clock()

    def __rich__(self) -> Text:
        """Return the line as it reads now."""
        return Text(f"thinking {self._clock() - self._started:.0f} s", style="dim")


class ThinkingLine:
    """Show a :class:`ThinkingTimer` from the first thought until the answer starts.

    The line is transient: it leaves nothing on screen once closed, and closing
    it on leaving the ``with`` block also covers a failed or stopped answer.
    """

    def __init__(self, console: Console, clock: Callable[[], float]) -> None:
        self._console = console
        self._clock = clock
        self._status: Status | None = None
        self._answering = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def see(self, chunk: ChatChunk) -> None:
        """Start the line at the first thought; close it once the answer begins."""
        if chunk.text or chunk.tool_calls:
            self._answering = True
            self.close()
        elif chunk.reasoning and self._status is None and not self._answering:
            self._status = Status(
                ThinkingTimer(self._clock), console=self._console, spinner_style="dim"
            )
            self._status.start()

    def close(self) -> None:
        """Remove the line, if it is showing."""
        if self._status is not None:
            self._status.stop()


class ChatApp:
    """Carry out chat commands against a session, printing to a console."""

    def __init__(
        self,
        session: ChatSession,
        gateway: Gateway,
        console: Console,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.session = session
        self.gateway = gateway
        self.console = console
        self._clock = clock

    def greet(self) -> None:
        """Print who is listening and how to get help."""
        self.console.print(
            f"{self.session.persona.name} is listening. /help lists the commands.",
            style="bold",
            markup=False,
        )

    async def handle(self, command: Command) -> bool:
        """Carry out ``command``; return whether the chat goes on."""
        match command:
            case Exit():
                return False
            case Say(text=text):
                if text:
                    await self._answer(text)
            case ShowImage(path=path, text=text):
                await self._image(path, text)
            case SwitchPersona() | AdjustPersona():
                self._persona(command)
            case ShowBudget():
                await self._budget()
            case _:
                self._show(command)
        return True

    def _show(self, command: Think | ShowModel | Reset | Help | Invalid) -> None:
        match command:
            case Think(level=level):
                if level is not None:
                    self.session.reasoning = level
                self._note(f"thinking: {self.session.reasoning.value}")
            case ShowModel():
                self._model()
            case Reset():
                self.session.reset()
                self._note("conversation forgotten")
            case Help():
                self.console.print(HELP, markup=False, highlight=False)
            case _:
                self._error(command.reason)

    def stopped(self) -> None:
        """Say that an answer was stopped with Ctrl+C."""
        self.console.print()
        self._note("stopped; that turn is not kept")

    async def _answer(self, text: str, *images: ImagePart) -> None:
        report: TurnReport | None = None
        printed = False
        try:
            with ThinkingLine(self.console, self._clock) as thinking:
                async for item in self.session.turn(text, *images):
                    if isinstance(item, TurnReport):
                        report = item
                        continue
                    thinking.see(item)
                    self.console.print(item.text, end="", markup=False, highlight=False)
                    printed = printed or bool(item.text)
        except GatewayError as error:
            if printed:
                self.console.print()
            self._error(f"no answer: {error}")
            return
        if printed:
            self.console.print()
        if report is not None:
            self.console.print(describe(report), style="dim", markup=False)
        elif not printed:
            self._note("no answer came back")

    async def _image(self, path: Path, text: str) -> None:
        try:
            image = load_image(path)
        except ImageError as error:
            self._error(str(error))
            return
        await self._answer(text, image)

    def _persona(self, command: SwitchPersona | AdjustPersona) -> None:
        if isinstance(command, SwitchPersona) and not command.key:
            self._note(f"personas: {', '.join(self.session.persona_names())}")
            return
        try:
            if isinstance(command, SwitchPersona):
                self.session.switch(command.key)
            else:
                self.session.adjust(command.values)
        except PersonaError as error:
            self._error(str(error))
            return
        persona = self.session.persona
        sliders = ", ".join(
            f"{k}={v:g}" for k, v in persona.traits.model_dump().items()
        )
        self._note(f"now {persona.name}: {sliders}")

    async def _budget(self) -> None:
        health = self.gateway.health
        status = await health.ledger.status(health.provider)
        self._note(
            f"{status.used} of {status.cap} remote requests used today (UTC); "
            f"{status.remaining} left, {health.reserve} kept in reserve"
        )

    def _model(self) -> None:
        info = self.gateway.router.info
        images = "yes" if info.vision else "no"
        tools = "yes" if info.tools else "no"
        self._note(
            f"context {info.context_window} tokens, images {images}, tools {tools}"
        )
        last = self.session.routes.decision
        if last is None:
            self._note("no turn yet")
        else:
            self._note(f"last turn: {last.route} to {last.model} ({last.reason})")

    def _note(self, text: str) -> None:
        self.console.print(text, style="cyan", markup=False, highlight=False)

    def _error(self, text: str) -> None:
        self.console.print(text, style="red", markup=False, highlight=False)


def converse(runner: asyncio.Runner, app: ChatApp, read: ReadLine) -> None:
    """Read and carry out commands until the user leaves."""
    app.greet()
    while True:
        try:
            line = read_message(read)
        except (EOFError, KeyboardInterrupt):
            app.console.print()
            return
        try:
            if not runner.run(app.handle(parse(line))):
                return
        except KeyboardInterrupt:
            app.stopped()


def run_chat(  # noqa: PLR0913
    settings: Settings,
    persona: str,
    console: Console,
    read: ReadLine = input,
    transport: httpx.AsyncBaseTransport | None = None,
    *,
    local: LocalService | None = None,
) -> None:
    """Hold a chat in the terminal until the user leaves.

    ``local`` starts once the chat can begin and loads while it goes on.

    Raises:
        ConfigError: If no model can be reached.
        PersonaError: If a persona file is invalid or ``persona`` does not exist.
    """
    library = PersonaLibrary(settings.home / PERSONAS_DIR)
    routes = LastRoute()
    with asyncio.Runner() as runner:
        client = httpx.AsyncClient(transport=transport, timeout=TIMEOUT)
        try:
            model = None if local is None else local.model(client)
            gateway = build_gateway(settings, client, routes, model)
            session = ChatSession(gateway.model, library, persona, routes)
            if local is not None:
                local.start()
            converse(runner, ChatApp(session, gateway, console), read)
        finally:
            if local is not None:
                local.stop()
            runner.run(client.aclose())
