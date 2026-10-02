"""Tools from MCP servers, each run as a local process that speaks over stdio.

Servers are listed in ``<home>/mcp.toml``. Each one is started, asked which
tools it has, and its tools join SYNTHIA's under ``<server>__<tool>``. A
server is OUTSIDE and CHANGE unless the file says otherwise, so every call
asks first: a local process may itself reach the network, and what a server
says about its own tools (read-only hints) is not trusted.

A server that will not start, answers wrongly or dies costs only its own
tools; the rest keep working. None of SYNTHIA's own ``SYNTHIA_*`` variables
reach a server, so its key never does.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, BinaryIO, Final, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from synthia import __version__
from synthia.agent.tools import TOOL_NAME, Effect, Reach, ToolError
from synthia.gateway.types import ToolSpec
from synthia.kernel.errors import ConfigError, SynthiaError
from synthia.mcp.jsonrpc import Connection, ConnectionClosedError, RpcError
from synthia.tools.process_tree import ProcessTree
from synthia.tools.run import environment_without_own

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

MCP_CONFIG: Final = Path("mcp.toml")
MCP_LOGS: Final = Path("logs") / "mcp"
# The version this client speaks first; a server may answer with any of
# SUPPORTED_VERSIONS, which share the tools/list and tools/call shapes used here.
PROTOCOL_VERSION: Final = "2025-06-18"
SUPPORTED_VERSIONS: Final = frozenset({"2024-11-05", "2025-03-26", "2025-06-18"})
START_TIMEOUT_S: Final = 30.0
STOP_TIMEOUT_S: Final = 5.0
MAX_RESULT_CHARS: Final = 20_000
SEPARATOR: Final = "__"
_SERVER_NAME: Final = r"^[A-Za-z0-9-]{1,24}$"

logger = logging.getLogger(__name__)


class McpError(SynthiaError):
    """An MCP server could not be started or used."""


class ServerConfig(BaseModel):
    """One server from ``mcp.toml``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    command: Annotated[list[str], Field(min_length=1)]
    env: dict[str, str] = {}
    cwd: Path | None = None
    reach: Reach = Reach.OUTSIDE
    effect: Effect = Effect.CHANGE


class _ConfigFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    servers: dict[Annotated[str, Field(pattern=_SERVER_NAME)], ServerConfig] = {}


def load_config(path: Path) -> dict[str, ServerConfig]:
    """Return the servers listed in ``path``; none if the file does not exist.

    Raises:
        ConfigError: If the file is not valid TOML or does not fit the shape.
    """
    if not path.is_file():
        return {}
    try:
        return _ConfigFile.model_validate(
            tomllib.loads(path.read_text("utf-8"))
        ).servers
    except (tomllib.TOMLDecodeError, ValidationError, UnicodeDecodeError) as error:
        message = f"{path} is not a valid MCP server list: {error}"
        raise ConfigError(message) from error


def _open_log(log_dir: Path, name: str) -> BinaryIO:
    log_dir.mkdir(parents=True, exist_ok=True)
    return (log_dir / f"{name}.log").open("ab")


@dataclass(slots=True)
class McpTool:
    """A tool one server offers, callable by the agent."""

    spec: ToolSpec
    reach: Reach
    effect: Effect
    server: McpServer
    remote_name: str
    time_limit_s: float | None = None

    async def run(self, arguments: str) -> str:
        """Call the tool on its server with ``arguments`` (a JSON object).

        Raises:
            ToolError: If the arguments are not an object, the server refused
                or failed, or the tool reported an error.
        """
        try:
            values: object = json.loads(arguments or "{}")
        except ValueError as error:
            message = f"the arguments are not JSON: {error}"
            raise ToolError(message) from error
        if not isinstance(values, dict):
            message = "the arguments must be a JSON object"
            raise ToolError(message)
        return await self.server.call(
            self.remote_name, cast("dict[str, object]", values)
        )


@dataclass(slots=True)
class McpServer:
    """One running server and the tools it offered."""

    name: str
    config: ServerConfig
    tree: ProcessTree
    connection: Connection
    log: BinaryIO
    tools: list[McpTool] = field(default_factory=list[McpTool])
    _watch: asyncio.Task[None] = field(init=False)

    def __post_init__(self) -> None:
        self._watch = asyncio.create_task(self._closed_on_exit())

    @classmethod
    async def start(
        cls,
        name: str,
        config: ServerConfig,
        log_dir: Path,
        *,
        start_timeout_s: float = START_TIMEOUT_S,
    ) -> McpServer:
        """Start the server, agree a protocol version and read its tools.

        Raises:
            McpError: If it cannot be found or started, or does not complete
                the handshake within ``start_timeout_s``.
        """
        server = await cls._launch(name, config, log_dir)
        try:
            async with asyncio.timeout(start_timeout_s):
                await server.handshake()
        except TimeoutError as error:
            await server.stop()
            message = f"{name}: no handshake within {start_timeout_s:g} s"
            raise McpError(message) from error
        except (RpcError, ConnectionClosedError, McpError) as error:
            await server.stop()
            message = f"{name}: {error}"
            raise McpError(message) from error
        return server

    @classmethod
    async def _launch(cls, name: str, config: ServerConfig, log_dir: Path) -> McpServer:
        environment = environment_without_own(config.env)
        program = shutil.which(config.command[0], path=environment.get("PATH"))
        if program is None:
            message = f"{name}: {config.command[0]} was not found"
            raise McpError(message)
        log = _open_log(log_dir, name)
        # The connection must exist before the first byte of output can arrive.
        started: list[ProcessTree] = []

        def send(data: bytes) -> None:
            started[0].send(data)

        connection = Connection(send)
        try:
            tree = await ProcessTree.start(
                [program, *config.command[1:]],
                cwd=config.cwd or Path.cwd(),
                env=environment,
                on_output=connection.feed,
                errors=log,
            )
        except OSError as error:
            log.close()
            message = f"{name}: could not start: {error}"
            raise McpError(message) from error
        started.append(tree)
        return cls(name, config, tree, connection, log)

    async def call(self, tool: str, arguments: dict[str, object]) -> str:
        """Call ``tool`` and return its text result, cut to ``MAX_RESULT_CHARS``.

        Raises:
            ToolError: If the server refused or failed, or the tool reported an error.
        """
        try:
            result = await self.connection.request(
                "tools/call", {"name": tool, "arguments": arguments}
            )
        except RpcError as error:
            message = f"{self.name} refused: {error}"
            raise ToolError(message) from error
        except ConnectionClosedError as error:
            message = f"{self.name} is not running: {error}"
            raise ToolError(message) from error
        fields = cast("dict[str, object]", result) if isinstance(result, dict) else {}
        text = _cut(result_text(fields))
        if fields.get("isError") is True:
            raise ToolError(text)
        return text

    async def stop(self) -> None:
        """Close the server's input, give it time to exit, then end it."""
        self.tree.close_input()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self.tree.exited.wait(), STOP_TIMEOUT_S)
        await self.tree.end()
        self.connection.close(f"{self.name} was stopped")
        self._watch.cancel()
        self.log.close()

    async def _closed_on_exit(self) -> None:
        await self.tree.exited.wait()
        self.connection.close(f"{self.name} exited with code {self.tree.exit_code}")

    async def handshake(self) -> None:
        """Agree a protocol version, say the client is ready, and read the tools.

        Raises:
            McpError: If the server answers with a version this client cannot speak.
        """
        answer = await self.connection.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "synthia", "version": __version__},
            },
        )
        fields = cast("dict[str, object]", answer) if isinstance(answer, dict) else {}
        version = fields.get("protocolVersion")
        if version not in SUPPORTED_VERSIONS:
            message = f"protocol version {version!r} is not supported"
            raise McpError(message)
        self.connection.notify("notifications/initialized")
        tools = [self._tool(listed) for listed in await self._listed()]
        # A name listed twice keeps its first tool, as a toolbox takes one per name.
        unique = {tool.spec.name: tool for tool in reversed(tools) if tool is not None}
        self.tools = list(reversed(unique.values()))

    async def _listed(self) -> list[dict[str, object]]:
        """Return every tool the server lists, following its pages."""
        tools: list[dict[str, object]] = []
        cursor: object = None
        while True:
            params: dict[str, object] = {} if cursor is None else {"cursor": cursor}
            page = await self.connection.request("tools/list", params)
            fields = cast("dict[str, object]", page) if isinstance(page, dict) else {}
            listed = fields.get("tools")
            if isinstance(listed, list):
                tools.extend(
                    cast("dict[str, object]", t)
                    for t in cast("list[object]", listed)
                    if isinstance(t, dict)
                )
            cursor = fields.get("nextCursor")
            if not isinstance(cursor, str):
                return tools

    def _tool(self, listed: dict[str, object]) -> McpTool | None:
        """Return the listed tool, or None if no model would accept its name."""
        remote = str(listed.get("name", ""))
        name = f"{self.name}{SEPARATOR}{remote}"
        if not TOOL_NAME.fullmatch(name):
            logger.warning(
                "%s: skipped tool %r, whose name a model cannot take", self.name, remote
            )
            return None
        schema = listed.get("inputSchema")
        description = listed.get("description")
        return McpTool(
            ToolSpec(
                name,
                str(description) if description else f"{remote} from {self.name}",
                cast("dict[str, object]", schema)
                if isinstance(schema, dict)
                else {"type": "object"},
            ),
            self.config.reach,
            self.config.effect,
            self,
            remote,
        )


def result_text(result: Mapping[str, object]) -> str:
    """Return a ``tools/call`` result as text; non-text parts are named, not dropped."""
    content = result.get("content")
    items = cast("list[object]", content) if isinstance(content, list) else []
    parts = [
        _part(cast("dict[str, object]", item))
        for item in items
        if isinstance(item, dict)
    ]
    if not parts and "structuredContent" in result:
        parts.append(json.dumps(result["structuredContent"]))
    return "\n".join(parts) or "(no content)"


def _part(item: dict[str, object]) -> str:
    match item.get("type"):
        case "text":
            return str(item.get("text", ""))
        case "image" | "audio" as kind:
            return f"[{kind} {item.get('mimeType', 'of unknown type')}]"
        case "resource":
            resource = item.get("resource")
            fields = (
                cast("dict[str, object]", resource)
                if isinstance(resource, dict)
                else {}
            )
            return str(fields.get("text") or f"[resource {fields.get('uri', '')}]")
        case "resource_link":
            return f"[link {item.get('uri', '')}]"
        case other:
            return f"[content of type {other!r}]"


def _cut(text: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return f"{text[:MAX_RESULT_CHARS]}\n[cut: {len(text):,} characters in all]"


@dataclass(slots=True)
class McpServers:
    """The servers that started, and why the others did not."""

    servers: list[McpServer]
    failures: list[str]

    @classmethod
    async def start(
        cls,
        configs: Mapping[str, ServerConfig],
        log_dir: Path,
        *,
        start_timeout_s: float = START_TIMEOUT_S,
    ) -> McpServers:
        """Start every server at once; one that fails is reported, not raised."""
        started = await asyncio.gather(
            *(
                McpServer.start(name, config, log_dir, start_timeout_s=start_timeout_s)
                for name, config in configs.items()
            ),
            return_exceptions=True,
        )
        for outcome in started:
            if not isinstance(outcome, McpServer | McpError):
                raise outcome
        return cls(
            [s for s in started if isinstance(s, McpServer)],
            [str(e) for e in started if isinstance(e, McpError)],
        )

    def tools(self) -> Iterable[McpTool]:
        """Return every server's tools, in the order the servers are listed."""
        return [tool for server in self.servers for tool in server.tools]

    async def stop(self) -> None:
        """Stop every server."""
        await asyncio.gather(*(server.stop() for server in self.servers))
