"""The chat's side of the daemon: find it or start it, then hold a conversation there.

A conversation in the daemon answers like one in this process (a ``Talk``),
so the terminal shows it the same way: its streamed answer, tool calls and
plan marks are rebuilt from the daemon's notifications. A tool call that
needs a yes is asked here, in the terminal; Ctrl+C cancels the request,
which tells the daemon to cancel the turn.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Final, cast

import httpx
from websockets.asyncio.client import connect

from synthia import __version__
from synthia.agent.loop import ToolFinished
from synthia.agent.plan import PlanAnswerBegun, Planned, PlanStepBegun
from synthia.gateway.errors import GatewayError
from synthia.gateway.types import (
    ChatChunk,
    PromptProgress,
    Reasoning,
    ToolCall,
    ToolCallDelta,
)
from synthia.kernel.errors import SynthiaError
from synthia.kernel.jsonrpc import ConnectionClosedError, Peer, RpcError
from synthia.server.api import IMAGE_REFUSED
from synthia.server.conversation import Reply
from synthia.server.discovery import DAEMON_FILE, DaemonInfo, is_running, read_info
from synthia.server.session import ImageError, TurnReport

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable
    from pathlib import Path

    from websockets.asyncio.client import ClientConnection

    from synthia.kernel.jsonrpc import Params
    from synthia.server.session import TurnItem

# Building the gateway and starting MCP servers (30 s each, at once) comes
# before the daemon writes its file.
START_TIMEOUT_S: Final = 90.0
POLL_S: Final = 0.2
HEALTH_TIMEOUT: Final = httpx.Timeout(2.0)
ANSWER_PREFIX: Final = "no answer: "
SEE_LOG: Final = "see SYNTHIA_HOME/logs/synthia.log"
STOPPED_AT_START: Final = f"the daemon stopped while starting; {SEE_LOG}"
_DONE: Final = object()

type Answer = Callable[[str, str], Awaitable[bool]]
"""Asked with a tool's name and its arguments; True runs the call."""


class DaemonError(SynthiaError):
    """The daemon could not be reached or started."""


def start_daemon() -> subprocess.Popen[bytes]:
    """Start ``synthia serve`` in the background, apart from this terminal.

    On Windows with no console window: the venv's python.exe is a launcher, and
    started without a console its interpreter would open a visible one.
    """
    command = [sys.executable, "-m", "synthia", "serve"]
    if sys.platform == "win32":
        return subprocess.Popen(  # noqa: S603 - our own interpreter and module
            command,
            creationflags=subprocess.CREATE_NO_WINDOW
            | subprocess.CREATE_NEW_PROCESS_GROUP,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
    return subprocess.Popen(  # noqa: S603 - our own interpreter and module
        command,
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )


async def healthy(info: DaemonInfo) -> bool:
    """Return whether the daemon in ``info`` answers ``/health``."""
    try:
        async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as web:
            answer = await web.get(f"{info.base_url}/health")
    except httpx.HTTPError:
        return False
    return answer.is_success


async def running_daemon(home: Path) -> DaemonInfo | None:
    """Return the daemon described in ``home``, if it is running and answers."""
    info = read_info(home / DAEMON_FILE)
    if info is None or not is_running(info) or not await healthy(info):
        return None
    return info


async def find_or_start(
    home: Path,
    *,
    start: Callable[[], subprocess.Popen[bytes]] = start_daemon,
    timeout_s: float = START_TIMEOUT_S,
) -> DaemonInfo:
    """Return the running daemon, starting one if there is none.

    Raises:
        DaemonError: If it stops while starting, or does not answer within
            ``timeout_s``.
    """
    info = await running_daemon(home)
    if info is not None:
        return info
    process = start()
    try:
        async with asyncio.timeout(timeout_s):
            while (info := await running_daemon(home)) is None:
                if process.poll() is not None:
                    raise DaemonError(STOPPED_AT_START)
                await asyncio.sleep(POLL_S)
    except TimeoutError:
        message = f"the daemon did not start within {timeout_s:g} s; {SEE_LOG}"
        raise DaemonError(message) from None
    return info


async def stopped(home: Path, info: DaemonInfo, timeout_s: float) -> None:
    """Wait until the daemon in ``info`` has gone.

    Raises:
        DaemonError: If it is still running after ``timeout_s``.
    """
    try:
        async with asyncio.timeout(timeout_s):
            # Another process is going away: there is no event to wait on.
            while read_info(home / DAEMON_FILE) == info and is_running(info):  # noqa: ASYNC110
                await asyncio.sleep(POLL_S)
    except TimeoutError:
        message = f"the daemon did not stop within {timeout_s:g} s"
        raise DaemonError(message) from None


async def _refuse(_tool: str, _arguments: str) -> bool:
    return False


async def current_daemon(
    home: Path,
    *,
    start: Callable[[], subprocess.Popen[bytes]] = start_daemon,
    timeout_s: float = START_TIMEOUT_S,
) -> DaemonInfo:
    """Return a running daemon of this version, starting or restarting one.

    A daemon of another version (left from before an upgrade) is restarted if
    no other conversation is open in it; otherwise the person is told how.

    Raises:
        DaemonError: If it cannot be started, or another version is busy.
    """
    info = await find_or_start(home, start=start, timeout_s=timeout_s)
    if info.version == __version__:
        return info
    async with conversation(info, _refuse) as talk:
        busy = talk.sessions > 1
        if not busy:
            await talk.stop_daemon()
    if busy:
        message = (
            f"the daemon runs version {info.version} and this is {__version__}, "
            "and other conversations are open in it: run `synthia stop`, "
            "then chat again"
        )
        raise DaemonError(message)
    await stopped(home, info, timeout_s)
    return await find_or_start(home, start=start, timeout_s=timeout_s)


def chunk_from(params: Params) -> ChatChunk:
    """Rebuild a streamed chunk from the daemon's ``chunk`` notification."""
    progress = params.get("progress")
    read = cast("list[int] | None", progress)
    return ChatChunk(
        text=str(params.get("text") or ""),
        reasoning=str(params.get("reasoning") or ""),
        tool_calls=(ToolCallDelta(0),) if params.get("tool_calls") else (),
        progress=None if read is None else PromptProgress(read[0], read[1]),
    )


def tool_from(params: Params) -> ToolFinished:
    """Rebuild a finished tool call from the daemon's ``tool`` notification."""
    call = ToolCall("", str(params["name"]), str(params["arguments"]))
    flags = cast("list[str]", params.get("flags") or [])
    return ToolFinished(
        call,
        str(params["result"]),
        bool(params["ok"]),
        float(cast("float", params["seconds"])),
        tuple(flags),
    )


def mark_from(params: Params) -> Planned | PlanStepBegun | PlanAnswerBegun:
    """Rebuild a plan mark from the daemon's ``plan`` notification."""
    match params.get("kind"):
        case "planned":
            steps = cast("list[str]", params["steps"])
            return Planned(tuple(steps), revised=bool(params["revised"]))
        case "step":
            return PlanStepBegun(
                int(cast("int", params["number"])), str(params["text"])
            )
        case _:
            return PlanAnswerBegun()


def report_from(result: object) -> TurnReport:
    """Rebuild a turn's report from the ``turn`` result."""
    fields = cast("dict[str, object]", result)
    level = fields.get("reasoning")
    return TurnReport(
        route=str(fields["route"]),
        model=str(fields["model"]),
        prompt_tokens=cast("int | None", fields.get("prompt_tokens")),
        completion_tokens=cast("int | None", fields.get("completion_tokens")),
        seconds=float(cast("float", fields["seconds"])),
        reasoning=None if level is None else Reasoning(str(level)),
    )


def reply_from(result: object) -> Reply:
    """Rebuild a command's reply."""
    fields = cast("dict[str, list[str]]", result)
    return Reply(tuple(fields["notes"]), tuple(fields["errors"]))


class RemoteConversation:
    """A conversation the daemon holds, reached over its WebSocket."""

    def __init__(self, socket: ClientConnection, answer: Answer) -> None:
        self._socket = socket
        self._answer = answer
        self._outgoing: asyncio.Queue[bytes] = asyncio.Queue()
        self._items: asyncio.Queue[TurnItem | object] | None = None
        self.peer = Peer(
            self._outgoing.put_nowait,
            methods={"approve": self._approve},
            notices={
                "chunk": lambda p: self._put(chunk_from(p)),
                "tool": lambda p: self._put(tool_from(p)),
                "plan": lambda p: self._put(mark_from(p)),
            },
        )
        self.version = ""
        self.sessions = 0
        self.warnings: tuple[str, ...] = ()
        self.closed = asyncio.Event()
        self._persona = ""

    @property
    def persona_name(self) -> str:
        """Return the name of the persona answering."""
        return self._persona

    async def hello(self) -> None:
        """Learn the daemon's version, the persona and its open conversations."""
        fields = cast("dict[str, object]", await self.peer.request("hello", {}))
        self.version = str(fields["version"])
        self._persona = str(fields["persona"])
        self.sessions = int(cast("int", fields["sessions"]))
        self.warnings = tuple(cast("list[str]", fields.get("warnings") or []))

    async def turn(
        self, text: str, *, images: tuple[Path, ...] = (), plan: bool = False
    ) -> AsyncGenerator[TurnItem]:
        """Yield the turn as the daemon streams it, then its report.

        Raises:
            ImageError: If the daemon could not send an image.
            GatewayError: If the answer failed, or the daemon went away.
        """
        items: asyncio.Queue[TurnItem | object] = asyncio.Queue()
        self._items = items
        params: Params = {
            "text": text,
            "plan": plan,
            "images": [str(p) for p in images],
        }
        request = asyncio.ensure_future(self.peer.request("turn", params))
        request.add_done_callback(lambda _: items.put_nowait(_DONE))
        try:
            while (item := await items.get()) is not _DONE:
                yield cast("TurnItem", item)
            yield report_from(await request)
        except RpcError as error:
            if error.code == IMAGE_REFUSED:
                raise ImageError(str(error)) from None
            raise GatewayError(str(error).removeprefix(ANSWER_PREFIX)) from None
        except ConnectionClosedError as error:
            message = f"the daemon went away: {error}"
            raise GatewayError(message) from None
        finally:
            request.cancel()
            self._items = None

    async def think(self, level: Reasoning | None) -> Reply:
        """Set or show the thinking level."""
        return await self._ask(
            "think", {"level": None if level is None else level.value}
        )

    async def remote(self, *, on: bool | None) -> Reply:
        """Allow, stop or show turns going to the remote model."""
        return await self._ask("remote", {"on": on})

    async def persona(self, key: str) -> Reply:
        """Switch persona, or list them."""
        reply = await self._ask("persona", {"key": key})
        await self.hello()
        return reply

    async def adjust(self, values: dict[str, float]) -> Reply:
        """Move trait sliders."""
        return await self._ask("adjust", {"values": values})

    async def budget(self) -> Reply:
        """Show today's remote budget."""
        return await self._ask("budget", {})

    async def model(self) -> Reply:
        """Show the model and the last route."""
        return await self._ask("model", {})

    async def tools(self) -> Reply:
        """List the tools."""
        return await self._ask("tools", {})

    async def reset(self) -> Reply:
        """Forget the conversation."""
        return await self._ask("reset", {})

    async def stop_daemon(self) -> None:
        """Ask the daemon to stop."""
        await self.peer.request("stop", {})

    async def _ask(self, method: str, params: Params) -> Reply:
        return reply_from(await self.peer.request(method, params))

    def _put(self, item: TurnItem) -> None:
        if self._items is not None:
            self._items.put_nowait(item)

    async def _approve(self, params: Params) -> object:
        return await self._answer(str(params["tool"]), str(params["arguments"]))

    async def run(self) -> None:
        """Carry messages both ways until the connection closes."""

        async def write() -> None:
            while True:
                await self._socket.send((await self._outgoing.get()).decode())

        writer = asyncio.create_task(write())
        try:
            async for message in self._socket:
                self.peer.receive(message)
        finally:
            writer.cancel()
            await asyncio.gather(writer, return_exceptions=True)
            self.peer.close("the daemon closed the conversation")
            self.closed.set()


@asynccontextmanager
async def conversation(
    info: DaemonInfo, answer: Answer
) -> AsyncGenerator[RemoteConversation]:
    """Open a conversation in the daemon described by ``info``."""
    headers = {"Authorization": f"Bearer {info.token}"}
    async with connect(info.socket_url, additional_headers=headers) as socket:
        talk = RemoteConversation(socket, answer)
        carrying = asyncio.create_task(talk.run())
        try:
            await talk.hello()
            yield talk
        finally:
            await socket.close()
            await asyncio.gather(carrying, return_exceptions=True)
