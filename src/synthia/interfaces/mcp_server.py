"""SYNTHIA as an MCP server: another agent searches the person's memory and files.

``synthia mcp serve --host <name>`` speaks MCP (protocol 2025-06-18) over
stdio, one JSON-RPC message per line, to the agent that started it (Claude
Code, Claude Desktop, Codex). It offers two tools, both read-only and both on
this machine: ``knowledge_search`` and ``memory_search``. No tool that reaches
outside or changes anything is offered, so nothing a host asks can run one.

What a host sees is the person's decision, read again at every call so a
change takes effect on the next one:

- ``SYNTHIA_HOME/mcp_hosts.toml``, written by the person: under
  ``[hosts.<name>]``, ``collections``, the knowledge sources by name, and
  ``memory``, whether earlier conversations may be searched (default false).
  A host with no table sees nothing.
- ``SYNTHIA_HOME/consent.json``: private passages, and turns of chats that held
  private text, go only to a host granted with
  ``synthia consent grant --host <name> --private``. That is never asked through
  the host: the result says how many were withheld and how to allow them.

The host's name is the one the person installed it under; it chooses the
rules above, and is not a proof of who is asking (any program of the person's
could pass it, and could read ``SYNTHIA_HOME`` itself).

The searches run in the daemon, the one owner of the index and the memory,
which is started on the first call that needs it. A request line longer than
:data:`MAX_REQUEST_BYTES`, or one that is not a JSON object, is answered with
an error and the server keeps serving. Results are cut to
:data:`~synthia.mcp.client.MAX_RESULT_CHARS` and quoted as untrusted data.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, BinaryIO, Final, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from websockets.exceptions import WebSocketException

from synthia import __version__
from synthia.agent.quoting import quote
from synthia.interfaces.daemon_client import (
    DaemonError,
    conversation,
    current_daemon,
    searched_from,
)
from synthia.interfaces.mcp_hosts import HOSTS_FILE, HostPolicy, policy_of
from synthia.kernel.errors import SynthiaError
from synthia.kernel.jsonrpc import ConnectionClosedError, Peer, RpcError
from synthia.knowledge.tool import MAX_BODY_CHARS
from synthia.mcp.client import MAX_RESULT_CHARS, PROTOCOL_VERSION, SUPPORTED_VERSIONS
from synthia.memory.recall import MAX_FOUND, Day, preview
from synthia.server.api import MAX_QUERY_CHARS

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from synthia.interfaces.daemon_client import RemoteConversation
    from synthia.kernel.consent import Host
    from synthia.kernel.jsonrpc import Method, Params

MAX_REQUEST_BYTES: Final = 1_048_576
# A host may send calls without waiting; more than this many searches at once
# would only queue in the daemon, so the rest wait here.
MAX_RUNNING_CALLS: Final = 4
# Each waiting call holds its request, so past this many a call is refused.
MAX_CALLS_IN_FLIGHT: Final = 16
SERVER_BUSY: Final = -32000
MAX_KNOWLEDGE_LIMIT: Final = 20
KNOWLEDGE_LIMIT: Final = 6
KNOWLEDGE_SEARCH: Final = "knowledge_search"
MEMORY_SEARCH: Final = "memory_search"
PARSE_ERROR: Final = -32700
INVALID_REQUEST: Final = -32600
INVALID_PARAMS: Final = -32602
NOTHING_FOUND: Final = "nothing found"
CUT: Final = "..."

logger = logging.getLogger(__name__)


class ArgumentsError(SynthiaError):
    """A tool was called with arguments that do not fit its schema."""


class Daemon(Protocol):
    """Where the searches run: the daemon, or a stand-in in tests."""

    async def request(self, method: str, params: Params) -> object:
        """Make one request and return its result.

        Raises:
            DaemonError: If the daemon cannot be reached.
            RpcError: If it refuses the request.
        """
        ...


async def _never_approve(_tool: str, _arguments: str, *, confined: bool) -> bool:
    del confined
    return False


class DaemonLink:
    """One conversation with the daemon, opened at the first request.

    A conversation that closed (the daemon stopped) is opened again at the
    next request. Nothing it asks is ever approved.
    """

    def __init__(self, home: Path) -> None:
        self._home = home
        self._stack: AsyncExitStack | None = None
        self._talk: RemoteConversation | None = None
        # Calls that find no daemon at once would each start one.
        self._opening = asyncio.Lock()

    async def request(self, method: str, params: Params) -> object:
        """Make one request of the daemon, starting it if it is not running.

        Raises:
            DaemonError: If the daemon cannot be started or went away.
            RpcError: If it refuses the request.
        """
        talk = await self._open()
        try:
            return await talk.peer.request(method, params)
        except ConnectionClosedError as error:
            await self.close()
            message = f"the daemon went away: {error}"
            raise DaemonError(message) from None

    async def _open(self) -> RemoteConversation:
        async with self._opening:
            return await self._opened()

    async def _opened(self) -> RemoteConversation:
        if self._talk is not None and not self._talk.closed.is_set():
            return self._talk
        await self.close()
        info = await current_daemon(self._home)
        stack = AsyncExitStack()
        try:
            self._talk = await stack.enter_async_context(
                conversation(info, _never_approve)
            )
        except (OSError, WebSocketException, ConnectionClosedError, RpcError) as error:
            await stack.aclose()
            message = f"cannot reach the daemon: {error}"
            raise DaemonError(message) from None
        self._stack = stack
        return self._talk

    async def close(self) -> None:
        """Close the conversation, if one is open."""
        stack, self._stack, self._talk = self._stack, None, None
        if stack is not None:
            await stack.aclose()


class _Arguments(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class KnowledgeArguments(_Arguments):
    """What ``knowledge_search`` takes."""

    query: str = Field(
        min_length=1,
        max_length=MAX_QUERY_CHARS,
        description="what to look for, in any words",
    )
    limit: int = Field(
        default=KNOWLEDGE_LIMIT,
        ge=1,
        le=MAX_KNOWLEDGE_LIMIT,
        description="how many passages to return, best first",
    )


class MemoryArguments(_Arguments):
    """What ``memory_search`` takes."""

    query: str = Field(
        default="",
        max_length=MAX_QUERY_CHARS,
        description="what to look for, in any words; empty for the newest turns",
    )
    since: Day | None = Field(
        default=None, description="the first day to include, YYYY-MM-DD"
    )
    until: Day | None = Field(
        default=None, description="the last day to include, YYYY-MM-DD"
    )


@dataclass(frozen=True, slots=True)
class ExposedTool[A: _Arguments]:
    """A tool the server offers: its name, what it says it does, and its handler."""

    name: str
    description: str
    arguments: type[A]
    handler: Callable[[A, HostPolicy], Awaitable[str]]

    def listed(self) -> dict[str, object]:
        """Return the tool as ``tools/list`` shows it."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.arguments.model_json_schema(),
        }

    async def run(self, given: object, policy: HostPolicy) -> str:
        """Return what the tool found for the arguments ``given``.

        Raises:
            ArgumentsError: Naming each problem, if ``given`` does not fit.
            DaemonError: If the daemon cannot be reached.
            RpcError: If the daemon refuses the search.
        """
        try:
            arguments = self.arguments.model_validate(given)
        except ValidationError as error:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'arguments'}: {e['msg']}"
                for e in error.errors(include_url=False)
            )
            raise ArgumentsError(problems) from None
        return await self.handler(arguments, policy)


def _cut(text: str, most: int) -> str:
    return text if len(text) <= most else text[:most] + CUT


def _withheld(count: int, what: str, host: Host) -> str:
    return (
        f"{count} private {what}{'' if count == 1 else 's'} withheld: the person "
        "can allow private text on their own terminal with "
        f"`synthia consent grant --host {host.name} --private`"
    )


def _not_shared(what: str, setting: str, host: Host) -> str:
    return (
        f"no {what} is shared with {host.name}: the person can set `{setting}` "
        f"under [hosts.{host.name}] in SYNTHIA_HOME/{HOSTS_FILE}"
    )


class Exposure:
    """The MCP methods: ``initialize``, ``tools/list`` and ``tools/call``.

    ``policy`` is read at every call; ``daemon`` runs the searches.
    """

    def __init__(self, policy: Callable[[], HostPolicy], daemon: Daemon) -> None:
        self._policy = policy
        self._daemon = daemon
        self._calls = asyncio.Semaphore(MAX_RUNNING_CALLS)
        self._in_flight = 0
        knowledge = ExposedTool(
            KNOWLEDGE_SEARCH,
            "Search the person's own files (notes, documents and code) that "
            "they share with this agent. Each passage comes under its citation "
            "in square brackets.",
            KnowledgeArguments,
            self._knowledge,
        )
        memory = ExposedTool(
            MEMORY_SEARCH,
            "Search what the person and their assistant SYNTHIA said in earlier "
            "conversations, best match first, each with its day.",
            MemoryArguments,
            self._memory,
        )
        self.tools: dict[
            str, ExposedTool[KnowledgeArguments] | ExposedTool[MemoryArguments]
        ] = {knowledge.name: knowledge, memory.name: memory}

    def methods(self) -> dict[str, Method]:
        """Return the methods a :class:`Peer` serves."""
        return {
            "initialize": self._initialize,
            "tools/list": self._list,
            "tools/call": self._call,
        }

    async def _initialize(self, params: Params) -> object:
        asked = params.get("protocolVersion")
        return {
            "protocolVersion": asked
            if isinstance(asked, str) and asked in SUPPORTED_VERSIONS
            else PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "synthia", "version": __version__},
        }

    async def _list(self, params: Params) -> object:
        del params
        return {"tools": [tool.listed() for tool in self.tools.values()]}

    async def _call(self, params: Params) -> object:
        name = params.get("name")
        tool = self.tools.get(name) if isinstance(name, str) else None
        if tool is None:
            raise RpcError(INVALID_PARAMS, f"unknown tool: {name!r}")
        if self._in_flight >= MAX_CALLS_IN_FLIGHT:
            message = (
                f"{MAX_CALLS_IN_FLIGHT} calls are already waiting or running; "
                "call again once some have answered"
            )
            raise RpcError(SERVER_BUSY, message)
        given = params.get("arguments", {})
        self._in_flight += 1
        try:
            async with self._calls:
                # Read once a slot is free: a decision made while it waited holds.
                text = await tool.run(given, self._policy())
        except ArgumentsError as error:
            raise RpcError(INVALID_PARAMS, str(error)) from None
        except (DaemonError, RpcError) as error:
            return _content(f"{tool.name} failed: {error}", error=True)
        finally:
            self._in_flight -= 1
        return _content(quote(tool.name, _cut(text, MAX_RESULT_CHARS)), error=False)

    async def _knowledge(self, asked: KnowledgeArguments, policy: HostPolicy) -> str:
        if not policy.collections:
            return _not_shared("collection", "collections = [...]", policy.host)
        searched = searched_from(
            await self._daemon.request(
                KNOWLEDGE_SEARCH,
                {
                    "query": asked.query,
                    "limit": asked.limit,
                    "private": policy.private,
                    "sources": sorted(policy.collections),
                },
            )
        )
        parts = [
            f"[{p.citation}] {p.header or p.title}\n{_cut(p.body, MAX_BODY_CHARS)}"
            for p in searched.passages
        ]
        if searched.withheld:
            parts.append(_withheld(searched.withheld, "passage", policy.host))
        return "\n\n".join(parts) or NOTHING_FOUND

    async def _memory(self, asked: MemoryArguments, policy: HostPolicy) -> str:
        if not policy.memory:
            return _not_shared("memory", "memory = true", policy.host)
        result = cast(
            "dict[str, object]",
            await self._daemon.request(
                MEMORY_SEARCH,
                {
                    "query": asked.query,
                    "since": None if asked.since is None else asked.since.isoformat(),
                    "until": None if asked.until is None else asked.until.isoformat(),
                    "limit": MAX_FOUND,
                    "private": policy.private,
                },
            ),
        )
        turns = cast("list[dict[str, object]]", result["turns"])
        parts = [_turn(turn) for turn in turns]
        withheld = int(cast("int", result["withheld"]))
        if withheld:
            parts.append(_withheld(withheld, "turn", policy.host))
        return "\n".join(parts) or NOTHING_FOUND


def _turn(turn: dict[str, object]) -> str:
    at = dt.datetime.fromisoformat(str(turn["at"])).astimezone()
    return "\n".join(
        [
            f"{at:%A %Y-%m-%d %H:%M}",
            f"  person: {preview(str(turn['question']))}",
            f"  SYNTHIA: {preview(str(turn['answer']))}",
        ]
    )


def _content(text: str, *, error: bool) -> dict[str, object]:
    return {"content": [{"type": "text", "text": text}], "isError": error}


async def serve_host(
    home: Path, host: Host, read: BinaryIO, write: Callable[[bytes], None]
) -> None:
    """Serve ``host`` under the rules in ``home`` until it closes ``read``."""
    link = DaemonLink(home)
    try:
        await serve_stdio(Exposure(partial(policy_of, home, host), link), read, write)
    finally:
        await link.close()


def _refusal(code: int, message: str) -> bytes:
    error = {"code": code, "message": message}
    return json.dumps({"jsonrpc": "2.0", "id": None, "error": error}).encode()


async def serve_stdio(
    exposure: Exposure, read: BinaryIO, write: Callable[[bytes], None]
) -> None:
    """Serve ``exposure`` to the host on ``read`` and ``write`` until ``read`` ends."""

    def send(message: bytes) -> None:
        write(message + b"\n")

    peer = Peer(send, methods=exposure.methods())
    try:
        while line := await asyncio.to_thread(read.readline, MAX_REQUEST_BYTES):
            if len(line) == MAX_REQUEST_BYTES and not line.endswith(b"\n"):
                await _skip_rest(read, line)
                send(
                    _refusal(
                        INVALID_REQUEST,
                        f"a request was longer than {MAX_REQUEST_BYTES:,} bytes",
                    )
                )
            elif line.strip():
                _receive(peer, send, line)
    finally:
        peer.close("the host closed the connection")


async def _skip_rest(read: BinaryIO, line: bytes) -> None:
    while line and not line.endswith(b"\n"):
        line = await asyncio.to_thread(read.readline, MAX_REQUEST_BYTES)


def _receive(peer: Peer, send: Callable[[bytes], None], line: bytes) -> None:
    try:
        parsed: object = json.loads(line)
    except (ValueError, RecursionError):
        send(_refusal(PARSE_ERROR, "a request was not JSON, or nested too deeply"))
        return
    if not isinstance(parsed, dict):
        send(_refusal(INVALID_REQUEST, "a request was not a JSON object"))
        return
    peer.receive(line)
