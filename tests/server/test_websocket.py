import asyncio

import pytest
import uvicorn
from uvicorn.protocols.websockets.websockets_sansio_impl import (
    WebSocketsSansIOProtocol,
)
from uvicorn.server import ServerState
from websockets.exceptions import InvalidState
from websockets.protocol import State

from synthia.server.websocket import ShutdownSafeWebSocket


class Transport:
    """Stands in for a connection's socket: records what is written and closed."""

    def __init__(self) -> None:
        self.written: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> None:
        self.written.append(data)

    def close(self) -> None:
        self.closed = True


async def _app(scope: object, receive: object, send: object) -> None:
    del scope, receive, send


def connected(
    protocol: type[WebSocketsSansIOProtocol], state: State
) -> tuple[WebSocketsSansIOProtocol, Transport]:
    loop = asyncio.new_event_loop()
    try:
        connection = protocol(uvicorn.Config(_app), ServerState(), {}, _loop=loop)
    finally:
        loop.close()
    transport = Transport()
    connection.transport = transport  # type: ignore[assignment]
    connection.handshake_complete = True
    connection.conn.state = state
    return connection, transport


def test_uvicorns_own_shutdown_fails_on_a_connection_the_client_is_closing() -> None:
    connection, _ = connected(WebSocketsSansIOProtocol, State.CLOSING)

    with pytest.raises(InvalidState, match="connection is closing"):
        connection.shutdown()


def test_a_connection_the_client_is_closing_is_closed_without_a_second_close() -> None:
    connection, transport = connected(ShutdownSafeWebSocket, State.CLOSING)

    connection.shutdown()

    assert transport.closed
    assert transport.written == []


def test_an_open_connection_still_gets_the_going_away_close() -> None:
    connection, transport = connected(ShutdownSafeWebSocket, State.OPEN)

    connection.shutdown()

    assert transport.closed
    assert transport.written
    assert connection.conn.state is State.CLOSING
