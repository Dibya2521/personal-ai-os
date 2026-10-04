"""``synthia chat``: talk to SYNTHIA in the terminal.

Lines are read in the main thread, and each command runs on one long-lived
:class:`asyncio.Runner`. That split is what makes Ctrl+C behave: pressed at
the prompt it ends the chat; pressed while an answer streams, ``Runner.run``
cancels that task, every stream beneath it closes, and the prompt returns.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import TYPE_CHECKING, Final, Self, cast

from rich.status import Status
from rich.text import Text

from synthia.agent.loop import ToolFinished
from synthia.agent.plan import PlanAnswerBegun, Planned, PlanStepBegun
from synthia.gateway.errors import GatewayError
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
from synthia.interfaces.daemon_client import Answer, conversation
from synthia.persona.model import PersonaError
from synthia.server.conversation import Reply, Talk
from synthia.server.session import (
    ImageError,
    TurnReport,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable
    from pathlib import Path

    from rich.console import Console

    from synthia.agent.policy import Approver
    from synthia.agent.tools import Tool
    from synthia.gateway.types import ChatChunk, PromptProgress
    from synthia.server.discovery import DaemonInfo
    from synthia.server.session import PlanMark, TurnItem

PROMPT: Final = "you> "
MORE: Final = "...> "
CONTINUES: Final = "\\"
YES: Final = frozenset({"y", "yes"})
MAX_SHOWN: Final = 80

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


def terminal_answer(read: ReadLine) -> Answer:
    """Return how the terminal answers an approval; anything but y or yes is no.

    The question is read on the event loop's own thread, so calls made at the
    same time are asked about one after another.
    """

    async def answer(tool: str, arguments: str) -> bool:
        try:
            said = read(approval_question(tool, arguments))
        except EOFError:
            return False
        return said.strip().lower() in YES

    return answer


def terminal_approver(read: ReadLine) -> Approver:
    """Return an approver, for a conversation in this process, asking the terminal."""
    answer = terminal_answer(read)

    async def approve(tool: Tool, arguments: str) -> bool:
        return await answer(tool.spec.name, arguments)

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


class ReadingTimer:
    """How much of the message the model has read, and for how long."""

    def __init__(self, clock: Callable[[], float], progress: PromptProgress) -> None:
        self._clock = clock
        self._started = clock()
        self.progress = progress

    def __rich__(self) -> Text:
        """Return the line as it reads now."""
        done, total = self.progress.processed, self.progress.total
        seconds = self._clock() - self._started
        return Text(
            f"reading the message: {done:,} of {total:,} tokens, {seconds:.0f} s",
            style="dim",
        )


class ThinkingLine:
    """Show the model reading the message, then thinking, until the answer starts.

    The line is transient: it leaves nothing on screen once closed, and closing
    it on leaving the ``with`` block also covers a failed or stopped answer.
    """

    def __init__(self, console: Console, clock: Callable[[], float]) -> None:
        self._console = console
        self._clock = clock
        self._status: Status | None = None
        self._reading: ReadingTimer | None = None
        self._thinking = False
        self._answering = False
        self.showing: ReadingTimer | ThinkingTimer | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def see(self, chunk: ChatChunk) -> None:
        """Move the line on with ``chunk``; close it once the answer begins."""
        if chunk.text or chunk.tool_calls:
            self._answering = True
            self.close()
        elif self._answering or self._thinking:
            return
        elif chunk.reasoning:
            self._thinking = True
            self._show(ThinkingTimer(self._clock))
        elif chunk.progress is not None:
            if self._reading is None:
                self._reading = ReadingTimer(self._clock, chunk.progress)
                self._show(self._reading)
            else:
                self._reading.progress = chunk.progress

    def _show(self, line: ReadingTimer | ThinkingTimer) -> None:
        self.showing = line
        if self._status is None:
            self._status = Status(line, console=self._console, spinner_style="dim")
            self._status.start()
        else:
            self._status.update(line)

    def close(self) -> None:
        """Remove the line, if it is showing."""
        self.showing = None
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


class ChatApp[T: Talk]:
    """Carry out chat commands against a conversation, printing to a console."""

    def __init__(
        self,
        conversation: T,
        console: Console,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.conversation = conversation
        self.console = console
        self._clock = clock

    def greet(self) -> None:
        """Print who is listening and how to get help."""
        self.console.print(
            f"{self.conversation.persona_name} is listening. /help lists the commands.",
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
                await self._answer(text, images=(path,))
            case ShowBudget():
                self._say(await self.conversation.budget())
            case Help():
                self.console.print(HELP, markup=False, highlight=False)
            case Invalid(reason=reason):
                self._error(reason)
            case _:
                self._say(await self._reply(command))
        return True

    async def _reply(
        self,
        command: SwitchPersona
        | AdjustPersona
        | Think
        | UseRemote
        | ShowModel
        | ShowTools
        | Reset,
    ) -> Reply:
        conversation = self.conversation
        match command:
            case SwitchPersona(key=key):
                reply = await conversation.persona(key)
            case AdjustPersona(values=values):
                reply = await conversation.adjust(values)
            case Think(level=level):
                reply = await conversation.think(level)
            case UseRemote(on=on):
                reply = await conversation.remote(on=on)
            case ShowModel():
                reply = await conversation.model()
            case ShowTools():
                reply = await conversation.tools()
            case _:
                reply = await conversation.reset()
        return reply

    def _say(self, reply: Reply) -> None:
        for line in reply.notes:
            self._note(line)
        for line in reply.errors:
            self._error(line)

    def stopped(self) -> None:
        """Say that an answer was stopped with Ctrl+C."""
        self.console.print()
        self._note("stopped; that turn is not kept")

    async def _answer(
        self, text: str, *, images: tuple[Path, ...] = (), plan: bool = False
    ) -> None:
        shown = AnswerLines(self.console)
        try:
            with ThinkingLine(self.console, self._clock) as thinking:
                turn = self.conversation.turn(text, images=images, plan=plan)
                report = await _shown(turn, thinking, shown)
        except ImageError as error:
            shown.end_line()
            self._error(str(error))
            return
        except GatewayError as error:
            shown.end_line()
            self._error(f"no answer: {error}")
            return
        shown.end_line()
        if report is not None:
            self.console.print(describe(report), style="dim", markup=False)
        elif not shown.said:
            self._note("no answer came back")

    def _note(self, text: str) -> None:
        self.console.print(text, style="cyan", markup=False, highlight=False)

    def _error(self, text: str) -> None:
        self.console.print(text, style="red", markup=False, highlight=False)


async def _shown(
    turn: AsyncGenerator[TurnItem], thinking: ThinkingLine, shown: AnswerLines
) -> TurnReport | None:
    """Show ``turn`` as it streams; return its report, if it gave one."""
    report: TurnReport | None = None
    async for item in turn:
        if isinstance(item, TurnReport):
            report = item
        elif isinstance(item, ToolFinished):
            shown.call(item)
        elif isinstance(item, Planned | PlanStepBegun | PlanAnswerBegun):
            shown.mark(item)
        else:
            thinking.see(item)
            shown.text(item.text)
    return report


def converse[T: Talk](runner: asyncio.Runner, app: ChatApp[T], read: ReadLine) -> None:
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


def run_chat(
    daemon: DaemonInfo,
    console: Console,
    read: ReadLine = input,
    *,
    persona: str | None = None,
    use_remote: bool = False,
) -> None:
    """Hold a chat in the terminal, held by ``daemon``, until the user leaves.

    ``persona`` switches to that persona first; ``use_remote`` lets turns go to
    the remote model from the start, as ``/remote on`` does.

    Raises:
        PersonaError: If ``persona`` does not exist.
    """
    with asyncio.Runner() as runner:
        stack = contextlib.AsyncExitStack()
        try:
            talk = runner.run(
                stack.enter_async_context(conversation(daemon, terminal_answer(read)))
            )
            for line in talk.warnings:
                console.print(line, style="red", markup=False, highlight=False)
            app = ChatApp(talk, console)
            if persona is not None:
                switched = runner.run(talk.persona(persona))
                if switched.errors:
                    raise PersonaError(switched.errors[0])
            if use_remote:
                runner.run(app.handle(UseRemote(on=True)))
            converse(runner, app, read)
        finally:
            runner.run(stack.aclose())
