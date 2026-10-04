"""``synthia serve``: run SYNTHIA as a daemon until it is asked to stop.

The daemon binds a free port on 127.0.0.1 and starts listening before it
writes ``daemon.json``, so a client that reads the file can connect at once;
its connection waits in the queue until the server accepts it. Stopping
(asked by a client, Ctrl+C, or the end of the process's session) ends every
conversation, which cancels the turns still running, then stops the MCP
servers and the local model, and removes ``daemon.json``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from typing import TYPE_CHECKING, Final, Protocol

import httpx
import uvicorn

from synthia import __version__
from synthia.gateway.assemble import build_gateway
from synthia.gateway.providers import OPENROUTER_FREE
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
from synthia.server.host import PERSONAS_DIR, TIMEOUT, Host, publish_route, start_tools

if TYPE_CHECKING:
    from pathlib import Path

    from synthia.gateway.protocol import LocalChatModel
    from synthia.gateway.providers import RemoteProvider
    from synthia.kernel.config import Settings

LOOPBACK: Final = "127.0.0.1"
BACKLOG: Final = 16


class LocalModels(Protocol):
    """What the daemon needs of the local model's service."""

    def model(self, client: httpx.AsyncClient) -> LocalChatModel:
        """Return the local model, sending its requests on ``client``."""
        ...

    def start(self) -> None:
        """Start loading the model."""
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


async def run_daemon(
    settings: Settings,
    local: LocalModels | None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    remote: RemoteProvider = OPENROUTER_FREE,
    stop: asyncio.Event | None = None,
) -> None:
    """Start what every conversation shares, serve until stopped, then stop it all.

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
        tools, servers = await start_tools(settings, logger.warning, transport)
        library = PersonaLibrary(settings.home / PERSONAS_DIR)
        if local is not None:
            local.start()
        try:
            host = Host.traced(settings, gateway, library, tools)
            await serve(host, settings.home, stop=stop)
        finally:
            if local is not None:
                local.stop()
            await servers.stop()
