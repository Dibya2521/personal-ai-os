"""``synthia serve``: run SYNTHIA as a daemon until it is asked to stop.

The daemon binds a free port on 127.0.0.1 and starts listening before it
writes ``daemon.json``, so a client that reads the file can connect at once;
its connection waits in the queue until the server accepts it. Stopping
(asked by a client, Ctrl+C, or the end of the process's session) ends every
conversation, which cancels the turns still running, then stops the MCP
servers and the local model, and removes ``daemon.json``. Each start, end
and restart of the local model's server, each consolidation of the facts
held, and each pass of the knowledge index goes to the daemon's own trace.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from functools import partial
from typing import TYPE_CHECKING, Final, Protocol

import httpx
import uvicorn

from synthia import __version__
from synthia.agent.trace import TRACES, Trace
from synthia.gateway.assemble import build_gateway
from synthia.gateway.providers import OPENROUTER_FREE
from synthia.memory.recall import memory_tool
from synthia.persona.library import PersonaLibrary
from synthia.server.app import create_app
from synthia.server.discovery import (
    DAEMON_FILE,
    DaemonInfo,
    new_token,
    remove_info,
    started_now,
    write_info,
)
from synthia.server.host import (
    PERSONAS_DIR,
    TIMEOUT,
    Host,
    in_background,
    no_local_model,
    open_embedder,
    open_knowledge,
    open_learning,
    open_memory,
    publish_route,
    start_tools,
    traced_index,
)
from synthia.server.upkeep import Activity, Reindexing

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from synthia.gateway.protocol import LocalChatModel
    from synthia.gateway.providers import RemoteProvider
    from synthia.kernel.config import Settings
    from synthia.kernel.supervisor import ServiceEvent
    from synthia.retrieval.embed import TextEmbedder

type EmbedderOpener = Callable[[Settings, Callable[[str], None]], TextEmbedder | None]
"""Opens the embedding model, passing what goes wrong to the second argument."""

LOOPBACK: Final = "127.0.0.1"
BACKLOG: Final = 16


class LocalModels(Protocol):
    """What the daemon needs of the local model's service."""

    def model(self, client: httpx.AsyncClient) -> LocalChatModel:
        """Return the local model, sending its requests on ``client``."""
        ...

    def state(self) -> str:
        """Return whether the model answers, is starting, or has stopped."""
        ...

    def start(self, observe: Callable[[ServiceEvent], None] | None = None) -> None:
        """Start loading the model; ``observe`` hears each start, end and stop."""
        ...

    def stop(self) -> None:
        """Stop the model's server."""
        ...


logger = logging.getLogger(__name__)


def listening_socket() -> socket.socket:
    """Return a socket on a free loopback port, already listening."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((LOOPBACK, 0))
    server.listen(BACKLOG)
    return server


async def serve(host: Host, home: Path, *, stop: asyncio.Event | None = None) -> None:
    """Serve ``host``'s conversations until ``stop`` is set or a client asks to stop.

    Ctrl+C stops it too: the server catches the signal and shuts down.
    """
    sock = listening_socket()
    stopping = stop or asyncio.Event()
    info = DaemonInfo(
        port=int(sock.getsockname()[1]),
        pid=os.getpid(),
        version=__version__,
        started=started_now(),
        token=new_token(),
    )
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(host, info.token, stopping.set),
            ws="websockets-sansio",
            # The chat's loop is idle while it waits at the prompt, so it cannot
            # answer pings; on 127.0.0.1 a client that dies closes its socket.
            ws_ping_interval=None,
            lifespan="off",
            log_config=None,
        )
    )

    async def stop_when_asked() -> None:
        await stopping.wait()
        server.should_exit = True

    watcher = asyncio.create_task(stop_when_asked())
    write_info(home / DAEMON_FILE, info)
    logger.info("serving on %s", info.base_url)
    try:
        await server.serve(sockets=[sock])
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        remove_info(home / DAEMON_FILE, info.pid)
        sock.close()
        logger.info("stopped")


def _warned(warnings: list[str], text: str) -> None:
    warnings.append(text)
    logger.warning("%s", text)


def _started(local: LocalModels | None, trace: Trace) -> Callable[[], str]:
    """Start loading ``local``, traced; return how to ask for its state."""
    if local is None:
        return no_local_model
    local.start(trace.service)
    return local.state


async def run_daemon(  # noqa: PLR0913 - every outside part is injected for tests
    settings: Settings,
    local: LocalModels | None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    remote: RemoteProvider = OPENROUTER_FREE,
    stop: asyncio.Event | None = None,
    embedder: EmbedderOpener = open_embedder,
) -> None:
    """Start what every conversation shares, serve until stopped, then stop it all.

    ``embedder`` is opened once and shared by memory and knowledge.

    Raises:
        ConfigError: If no model can be reached.
        PersonaError: If a persona file is invalid.
    """
    async with httpx.AsyncClient(transport=transport, timeout=TIMEOUT) as client:
        gateway = build_gateway(
            settings,
            client,
            publish_route,
            None if local is None else local.model(client),
            remote=remote,
        )
        warnings: list[str] = []
        warn = partial(_warned, warnings)
        traces = settings.home / TRACES
        trace = Trace.start(traces)
        model = embedder(settings, warn)
        memory = await open_memory(settings, warn, model)
        learning = await open_learning(
            memory, gateway.model, warn, local=gateway.router.has_local, trace=trace
        )
        knowledge = await open_knowledge(settings, warn, model)
        activity = Activity()
        reindexing = (
            None
            if knowledge is None
            else Reindexing(knowledge, activity, report=partial(traced_index, trace))
        )
        searching = (
            ()
            if memory is None
            else (
                memory_tool(
                    memory, can_stay_local=lambda: gateway.router.can_stay_local
                ),
            )
        )
        tools, servers = await start_tools(settings, warn, transport, searching)
        local_state = _started(local, trace)
        try:
            host = Host(
                gateway,
                PersonaLibrary(settings.home / PERSONAS_DIR),
                settings.persona,
                tools,
                traces,
                tuple(warnings),
                local_state,
                memory,
                learning,
                knowledge,
                activity,
                reindexing,
            )
            async with in_background(*host.background()):
                await serve(host, settings.home, stop=stop)
        finally:
            if local is not None:
                local.stop()
            await servers.stop()
