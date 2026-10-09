"""The parts of SYNTHIA started once and shared by every conversation.

One gateway serves every conversation, so the remote's rate limit, circuit
breaker and daily budget are counted once for all of them. Each conversation
still learns where its own turns were routed: the gateway's routing
decisions go to the record of the conversation whose turn is running, found
through a context variable that each turn sets for itself. One memory is
shared too, so a turn one conversation forgets is forgotten for all of them.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final

import httpx

from synthia.agent.trace import Trace
from synthia.kernel.errors import ConfigError
from synthia.mcp.client import MCP_CONFIG, MCP_LOGS, McpServers, load_config
from synthia.memory.embed import EmbedError, TextEmbedder
from synthia.memory.facts import FactKeeper, FactLearning
from synthia.memory.hybrid import Recall
from synthia.memory.remembering import Remembering
from synthia.memory.store import MEMORY_FILE, MemoryStore
from synthia.memory.summary import summarize
from synthia.models.catalogue import DEFAULT_EMBEDDER, EMBEDDERS
from synthia.models.install import BYTES_PER_GB, Installer
from synthia.server.conversation import Conversation
from synthia.server.session import ChatSession, LastRoute
from synthia.tools import local_tools, outside_tools

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        Callable,
        Coroutine,
        Iterable,
        Sequence,
    )

    from synthia.agent.policy import Approver
    from synthia.agent.tools import Tool, Toolbox
    from synthia.gateway.assemble import Gateway
    from synthia.gateway.protocol import ChatModel
    from synthia.kernel.bus import Event
    from synthia.kernel.config import Settings
    from synthia.models.catalogue import Embedder
    from synthia.persona.library import PersonaLibrary

logger = logging.getLogger(__name__)

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
    local: Iterable[Tool] = (),
) -> tuple[Toolbox, McpServers]:
    """Return SYNTHIA's own tools, the outside ones, and every MCP server's.

    In that order: what runs on this machine first (``local`` after the
    built-in ones), then what asks first.
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
    for tool in (*local, *outside_tools(settings.home, transport), *servers.tools()):
        tools.add(tool)
    return tools, servers


def open_embedder(
    settings: Settings,
    warn: Callable[[str], None],
    choices: Sequence[Embedder] = (DEFAULT_EMBEDDER, *EMBEDDERS),
) -> TextEmbedder | None:
    """Return the first installed embedding model of ``choices``, loaded.

    None if none is installed, and memory is then searched by words alone;
    None after a ``warn`` if the one installed cannot be loaded.
    """
    models = Installer(settings.home, int(settings.disk_budget_gb * BYTES_PER_GB))
    spec = next((s for s in choices if models.installed(s)), None)
    if spec is None:
        logger.info("no embedding model installed: memory is searched by words")
        return None
    try:
        return TextEmbedder.load(models.path_of(spec), spec)
    except EmbedError as error:
        warn(f"memory is searched by words only: {error}")
        return None


async def open_memory(
    settings: Settings,
    warn: Callable[[str], None],
    embedder: TextEmbedder | None = None,
) -> Recall | None:
    """Return the memory under ``SYNTHIA_HOME``, or None after a ``warn``.

    The vectors already kept from ``embedder`` are read in.
    """
    try:
        memory = Recall(MemoryStore(settings.home / MEMORY_FILE), embedder)
        await memory.load()
    except (sqlite3.Error, OSError) as error:
        warn(f"nothing will be remembered: {error}")
        return None
    return memory


async def fill_in(memory: Recall) -> None:
    """Embed the turns remembered without a vector; log a failure, never raise."""
    try:
        done = await memory.backfill()
    except Exception:
        logger.exception("earlier turns were not all embedded")
        return
    if done:
        logger.info("embedded %d earlier turns", done)


async def open_learning(
    memory: Recall | None,
    model: ChatModel,
    warn: Callable[[str], None],
    *,
    local: bool,
) -> FactLearning | None:
    """Return the learner of facts about the person, or None when it cannot run.

    It needs memory, an embedding model and a ``local`` model: facts are
    written on this machine and compared by meaning. A memory whose facts
    cannot be read is passed to ``warn``.
    """
    if memory is None or memory.embedder is None or not local:
        return None
    keeper = FactKeeper(memory.store, memory.embedder, model)
    try:
        await keeper.load()
    except (sqlite3.Error, OSError, EmbedError) as error:
        warn(f"no facts will be learned: {error}")
        return None
    return FactLearning(keeper, memory.store)


@asynccontextmanager
async def in_background(
    *work: Coroutine[object, object, None],
) -> AsyncGenerator[None]:
    """Run each of ``work`` while the block runs; cancel what is left at its end."""
    tasks = [asyncio.create_task(each) for each in work]
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks)


def no_local_model() -> str:
    """Return the local model's state when there is none."""
    return "none"


@dataclass(slots=True)
class Host:
    """What every conversation shares, and how a new one is made."""

    gateway: Gateway
    library: PersonaLibrary
    persona: str
    tools: Toolbox
    traces: Path | None = None
    warnings: tuple[str, ...] = ()
    local_state: Callable[[], str] = no_local_model
    memory: Recall | None = None
    learning: FactLearning | None = None

    def background(self) -> list[Coroutine[object, object, None]]:
        """Return the work that runs beside serving: embedding and learning."""
        work: list[Coroutine[object, object, None]] = []
        if self.memory is not None:
            work.append(fill_in(self.memory))
        if self.learning is not None:
            work.append(self.learning.run())
        return work

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
            memory=(
                None if self.memory is None else Remembering(self.memory, self.learning)
            ),
            summarizer=(
                partial(summarize, self.gateway.model)
                if self.gateway.router.has_local
                else None
            ),
        )
        return Conversation(session, self.gateway)
