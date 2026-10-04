"""The daemon's HTTP face: ``/health`` for anyone, ``/ws`` for a client with the token.

``/health`` says only that the daemon is up. ``/ws`` is one conversation per
connection, JSON-RPC in text frames. It needs ``Authorization: Bearer
<token>`` from ``daemon.json``, compared in constant time, and refuses any
request that carries an ``Origin`` header: browsers always send one and the
command line never does, so a web page the person visits cannot reach
SYNTHIA through their browser. FastAPI's documentation pages are off, so
nothing else is served.
"""

from __future__ import annotations

import asyncio
import secrets
from typing import TYPE_CHECKING, Final

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status

from synthia.server.api import Served

if TYPE_CHECKING:
    from collections.abc import Callable

    from synthia.server.host import Host

BEARER: Final = "Bearer "


def authorised(header: str | None, token: str) -> bool:
    """Return whether ``header`` carries ``token``."""
    return header is not None and secrets.compare_digest(header, BEARER + token)


def create_app(host: Host, token: str, on_stop: Callable[[], None]) -> FastAPI:
    """Return the daemon's app; ``on_stop`` is called when a client asks it to stop."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    open_conversations: set[Served] = set()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "up"}

    @app.websocket("/ws")
    async def conversation(websocket: WebSocket) -> None:
        headers = websocket.headers
        if "origin" in headers or not authorised(headers.get("authorization"), token):
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        await websocket.accept()
        outgoing: asyncio.Queue[bytes] = asyncio.Queue()
        served = Served(
            host, outgoing.put_nowait, on_stop, lambda: len(open_conversations)
        )
        open_conversations.add(served)
        writer = asyncio.create_task(_write(websocket, outgoing))
        try:
            while True:
                served.peer.receive(await websocket.receive_text())
        except WebSocketDisconnect:
            pass
        finally:
            open_conversations.discard(served)
            await served.close()
            writer.cancel()
            # A client that went away leaves the writer failed on its last send.
            await asyncio.gather(writer, return_exceptions=True)

    return app


async def _write(websocket: WebSocket, outgoing: asyncio.Queue[bytes]) -> None:
    """Send queued messages in order, one text frame each."""
    while True:
        message = await outgoing.get()
        await websocket.send_text(message.decode())
