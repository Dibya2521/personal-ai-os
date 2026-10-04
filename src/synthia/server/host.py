"""The parts of SYNTHIA started once and shared by every conversation.

One gateway serves every conversation, so the remote's rate limit, circuit
breaker and daily budget are counted once for all of them. Each conversation
still learns where its own turns were routed: the gateway's routing
decisions go to the record of the conversation whose turn is running, found
through a context variable that each turn sets for itself.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import httpx

from synthia.agent.trace import TRACES, Trace
from synthia.kernel.errors import ConfigError
from synthia.mcp.client import MCP_CONFIG, MCP_LOGS, McpServers, load_config
from synthia.server.conversation import Conversation
from synthia.server.session import ChatSession, LastRoute
from synthia.tools import local_tools, outside_tools

if TYPE_CHECKING:
    from collections.abc import Callable

    from synthia.agent.policy import Approver
    from synthia.agent.tools import Toolbox
    from synthia.gateway.assemble import Gateway
    from synthia.kernel.bus import Event
    from synthia.kernel.config import Settings
    from synthia.persona.library import PersonaLibrary

PERSONAS_DIR: Final = Path("personas")
# OpenRouter sends keep-alive comments while a model thinks, so a minute of
# silence mid-stream means the connection is gone, not that the model is slow.
TIMEOUT: Final = httpx.Timeout(60.0, connect=10.0)
CURRENT_ROUTES: Final[ContextVar[LastRoute | None]] = ContextVar(
    "current_routes", default=None
)


async def publish_route(event: Event) -> None:
    """Hand ``event`` to the routing record of the turn that is running, if any."""
    routes = CURRENT_ROUTES.get()
    if routes is not None:
        await routes(event)


async def start_tools(
    settings: Settings,
    warn: Callable[[str], None],
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[Toolbox, McpServers]:
    """Return SYNTHIA's own tools, the outside ones, and every MCP server's.

    In that order: what runs on this machine first, then what asks first.
    A broken ``mcp.toml`` or a server that will not start is passed to
    ``warn`` and left out.
    """
    try:
        configs = load_config(settings.home / MCP_CONFIG)
    except ConfigError as error:
        configs = {}
        warn(f"no MCP servers: {error}")
    servers = await McpServers.start(configs, settings.home / MCP_LOGS)
    for failure in servers.failures:
        warn(f"MCP server not started: {failure}")
    tools = local_tools(settings.file_roots)
    for tool in (*outside_tools(settings.home, transport), *servers.tools()):
        tools.add(tool)
    return tools, servers


@dataclass(slots=True)
class Host:
    """What every conversation shares, and how a new one is made."""

    gateway: Gateway
    library: PersonaLibrary
    persona: str
    tools: Toolbox
    traces: Path | None = None

    @classmethod
    def traced(
        cls,
        settings: Settings,
        gateway: Gateway,
        library: PersonaLibrary,
        tools: Toolbox,
    ) -> Host:
        """Return a host whose conversations are traced under ``SYNTHIA_HOME``."""
        return cls(gateway, library, settings.persona, tools, settings.home / TRACES)

    def conversation(self, approver: Approver) -> Conversation:
        """Return a new conversation that asks ``approver`` before each call."""
        session = ChatSession(
            self.gateway.model,
            self.library,
            self.persona,
            LastRoute(),
            tools=self.tools,
            approver=approver,
            trace=None if self.traces is None else Trace.start(self.traces),
        )
        return Conversation(session, self.gateway)
