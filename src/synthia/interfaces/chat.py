"""``synthia chat``: talk to SYNTHIA in the terminal.

Lines are read in the main thread, and each command runs on one long-lived
:class:`asyncio.Runner`. That split is what makes Ctrl+C behave: pressed at
the prompt it ends the chat; pressed while an answer streams, ``Runner.run``
cancels that task, every stream beneath it closes, and the prompt returns.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self, cast

import httpx
from rich.status import Status
from rich.text import Text

from synthia.agent.loop import ToolFinished
from synthia.agent.plan import PlanAnswerBegun, Planned, PlanStepBegun
from synthia.agent.policy import Decision
from synthia.agent.trace import TRACES, Trace
from synthia.gateway.assemble import build_gateway
from synthia.gateway.errors import GatewayError
from synthia.gateway.providers import OPENROUTER_FREE
from synthia.interfaces.commands import (
    HELP,
    AdjustPersona,
    Command,
    Exit,
    Help,
    Invalid,
    Plan,
    Reset,
    Say,
    ShowBudget,
    ShowImage,
    ShowModel,
    ShowTools,
    SwitchPersona,
    Think,
    UseRemote,
    parse,
)
from synthia.interfaces.session import (
    ChatSession,
    ImageError,
    LastRoute,
    TurnReport,
    load_image,
)
from synthia.kernel.errors import ConfigError
from synthia.mcp.client import MCP_CONFIG, MCP_LOGS, McpServers, load_config
from synthia.persona.library import PersonaLibrary
from synthia.persona.model import PersonaError
from synthia.tools import local_tools, outside_tools

if TYPE_CHECKING:
    from collections.abc import Callable

    from rich.console import Console

    from synthia.agent.policy import Approver
    from synthia.agent.tools import Tool, Toolbox
    from synthia.gateway.assemble import Gateway
    from synthia.gateway.providers import RemoteProvider
    from synthia.gateway.types import ChatChunk, ImagePart
    from synthia.interfaces.session import PlanMark
    from synthia.kernel.config import Settings
    from synthia.models.service import LocalService

PROMPT: Final = "you> "
MORE: Final = "...> "
CONTINUES: Final = "\\"
PERSONAS_DIR: Final = Path("personas")
# OpenRouter sends keep-alive comments while a model thinks, so a minute of
# silence mid-stream means the connection is gone, not that the model is slow.
TIMEOUT: Final = httpx.Timeout(60.0, connect=10.0)
YES: Final = frozenset({"y", "yes"})
MAX_SHOWN: Final = 80
RULE_SHOWN: Final = {
    Decision.ALLOW: "",
    Decision.ASK: " | asks first",
    Decision.DENY: " | never runs",
}

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


def describe_call(finished: ToolFinished) -> str:
    """Return the dim line shown for one tool call."""
    call = finished.call
    outcome = "ok" if finished.ok else shorten(finished.result)
    arguments = shorten(call.arguments or "{}")
    flagged = f" | flagged: {', '.join(finished.flags)}" if finished.flags else ""
    return (
        f"tool {call.name} {arguments} | {outcome}{flagged} | {finished.seconds:.1f} s"
    )


def describe_mark(mark: PlanMark) -> str:
    """Return the dim line shown when a planned turn moves on."""
    match mark:
        case Planned(steps=steps, revised=revised):
            made = "plan again" if revised else "plan"
            listed = " | ".join(f"{n}. {s}" for n, s in enumerate(steps, 1))
            return f"{made}: {visible(listed) or 'no steps left'}"
        case PlanStepBegun(number=number, text=text):
            return f"step {number}: {visible(text)}"
        case _:
            return "answer:"


def approval_question(name: str, arguments: str) -> str:
    """Return the question asked before running ``name`` with ``arguments``.

    Arguments too long for one line are shown whole, each string value on its
    own lines, so nothing is approved unseen. Characters a terminal would act
    on are shown escaped, so they cannot hide or rewrite what is shown.
    """
    shown = visible(arguments or "{}")
    if len(shown) <= MAX_SHOWN:
        return f"run {name} {shown}? [y/N] "
    return f"run {name} with:\n{_whole(arguments)}\n[y/N] "


def visible(text: str) -> str:
    """Return ``text`` with every non-printable character but tab escaped."""
    return "".join(
        c if c.isprintable() or c == "\t" else c.encode("unicode_escape").decode()
        for c in text
    )


def _whole(arguments: str) -> str:
    try:
        values: object = json.loads(arguments)
    except json.JSONDecodeError:
        values = None
    if not isinstance(values, dict):
        return visible(arguments)
    lines: list[str] = []
    for key, value in cast("dict[str, object]", values).items():
        text = value if isinstance(value, str) else json.dumps(value)
        lines.append(f"  {visible(key)}:")
        lines.extend(f"    {visible(line)}" for line in text.split("\n"))
    return "\n".join(lines)


def terminal_approver(read: ReadLine) -> Approver:
    """Return an approver that asks in the terminal; anything but y or yes is no.

    The question is read on the event loop's own thread, so calls made at the
    same time are asked about one after another.
    """

    async def approve(tool: Tool, arguments: str) -> bool:
        question = approval_question(tool.spec.name, arguments)
        try:
            answer = read(question)
        except EOFError:
            return False
        return answer.strip().lower() in YES

    return approve


def shorten(text: str) -> str:
    """Return ``text`` cut to :data:`MAX_SHOWN` characters, ending in ... if cut."""
    return text if len(text) <= MAX_SHOWN else text[: MAX_SHOWN - 3] + "..."


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


class AnswerLines:
    """Print an answer as it streams, with a dim line for each tool call."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self._open = False
        self.said = False

    def text(self, text: str) -> None:
        """Print ``text`` where the answer has got to."""
        self._console.print(text, end="", markup=False, highlight=False)
        self._open = self._open or bool(text)
        self.said = self.said or bool(text)

    def call(self, finished: ToolFinished) -> None:
        """Print one finished tool call on a line of its own."""
        self.end_line()
        self._console.print(describe_call(finished), style="dim", markup=False)

    def mark(self, mark: PlanMark) -> None:
        """Print where a planned turn has got to, on a line of its own."""
        self.end_line()
        self._console.print(
            describe_mark(mark), style="dim", markup=False, highlight=False
        )

    def end_line(self) -> None:
        """End the answer's current line, if text is on it."""
        if self._open:
            self._console.print()
            self._open = False


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
        self.learning: asyncio.Task[int | None] | None = None

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
            case Plan(task=task):
                await self._answer(task, plan=True)
            case ShowImage(path=path, text=text):
                await self._image(path, text)
            case SwitchPersona() | AdjustPersona():
                self._persona(command)
            case ShowBudget():
                await self._budget()
            case _:
                self._show(command)
        return True

    def _show(
        self,
        command: Think | UseRemote | ShowModel | ShowTools | Reset | Help | Invalid,
    ) -> None:
        match command:
            case Think(level=level):
                if level is not None:
                    self.session.reasoning = level
                self._note(f"thinking: {self.session.reasoning.value}")
            case UseRemote():
                self._remote(command)
            case ShowModel():
                self._model()
            case ShowTools():
                self._tools()
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

    async def _answer(self, text: str, *images: ImagePart, plan: bool = False) -> None:
        report: TurnReport | None = None
        shown = AnswerLines(self.console)
        try:
            with ThinkingLine(self.console, self._clock) as thinking:
                async for item in self.session.turn(text, *images, plan=plan):
                    if isinstance(item, TurnReport):
                        report = item
                    elif isinstance(item, ToolFinished):
                        shown.call(item)
                    elif isinstance(item, Planned | PlanStepBegun | PlanAnswerBegun):
                        shown.mark(item)
                    else:
                        thinking.see(item)
                        shown.text(item.text)
        except GatewayError as error:
            shown.end_line()
            self._error(f"no answer: {error}")
            return
        shown.end_line()
        if report is not None:
            self.console.print(describe(report), style="dim", markup=False)
        elif not shown.said:
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

    def _remote(self, command: UseRemote) -> None:
        if command.on and self.gateway.router.remote is None:
            self._error(
                "no remote model is configured: set SYNTHIA_OPENROUTER_API_KEY in .env"
            )
            return
        if command.on is not None:
            self.session.use_remote = command.on
        if self.session.use_remote and self.learning is None:
            # Not before remote is on, so the key never leaves unasked; beside
            # the turns, so no answer waits for it.
            self.learning = asyncio.get_running_loop().create_task(
                self.gateway.learn_daily_cap()
            )
        if self.session.use_remote:
            self._note("remote: on, turns may leave this machine")
        else:
            self._note("remote: off, every turn stays on this machine")

    async def _budget(self) -> None:
        health = self.gateway.health
        if health is None:
            self._note("no remote model is configured, so there is no budget")
            return
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

    def _tools(self) -> None:
        tools = list(self.session.tools)
        if not tools:
            self._note("no tools")
            return
        for tool in tools:
            rule = RULE_SHOWN[self.session.policy.decide(tool)]
            self._note(
                f"{tool.spec.name} | {tool.reach.value}, {tool.effect.value}{rule} | "
                f"{tool.spec.description}"
            )

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
    remote: RemoteProvider = OPENROUTER_FREE,
    use_remote: bool = False,
) -> None:
    """Hold a chat in the terminal until the user leaves.

    ``local`` starts once the chat can begin and loads while it goes on; a
    turn sent before it is ready waits for it. ``use_remote`` starts the chat
    with turns allowed to go to the remote model, as ``/remote on`` does.

    Raises:
        ConfigError: If no model can be reached.
        PersonaError: If a persona file is invalid or ``persona`` does not exist.
    """
    routes = LastRoute()
    with asyncio.Runner() as runner:
        client = httpx.AsyncClient(transport=transport, timeout=TIMEOUT)
        app: ChatApp | None = None
        servers = McpServers([], [])
        try:
            gateway = build_gateway(
                settings,
                client,
                routes,
                None if local is None else local.model(client),
                remote=remote,
            )
            tools, servers = runner.run(chat_tools(settings, console, transport))
            session = ChatSession(
                gateway.model,
                PersonaLibrary(settings.home / PERSONAS_DIR),
                persona,
                routes,
                tools=tools,
                approver=terminal_approver(read),
                trace=Trace.start(settings.home / TRACES),
            )
            app = ChatApp(session, gateway, console)
            if local is not None:
                local.start()
            if use_remote:
                runner.run(app.handle(UseRemote(on=True)))
            converse(runner, app, read)
        finally:
            if local is not None:
                local.stop()
            if app is not None and app.learning is not None:
                app.learning.cancel()
                runner.run(asyncio.wait([app.learning]))
            runner.run(servers.stop())
            runner.run(client.aclose())


async def chat_tools(
    settings: Settings,
    console: Console,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[Toolbox, McpServers]:
    """Return SYNTHIA's own tools, the outside ones, and every MCP server's.

    In that order: what runs on this machine first, then what asks first.
    """
    servers = await start_mcp_servers(settings, console)
    tools = local_tools(settings.file_roots)
    for tool in (*outside_tools(settings.home, transport), *servers.tools()):
        tools.add(tool)
    return tools, servers


async def start_mcp_servers(settings: Settings, console: Console) -> McpServers:
    """Start the servers in ``mcp.toml``; any that fail are named and left out."""
    try:
        configs = load_config(settings.home / MCP_CONFIG)
    except ConfigError as error:
        configs = {}
        _warn(console, f"no MCP servers: {error}")
    servers = await McpServers.start(configs, settings.home / MCP_LOGS)
    for failure in servers.failures:
        _warn(console, f"MCP server not started: {failure}")
    return servers


def _warn(console: Console, text: str) -> None:
    console.print(text, style="red", markup=False, highlight=False)
