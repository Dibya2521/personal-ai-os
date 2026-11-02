"""uvicorn's sans-I/O WebSocket protocol, shut down safely while a client is closing.

On shutdown, uvicorn 0.54.0 sends each connection a close frame unless it sent
one itself. A connection whose client started the closing handshake has
already answered that close and left the OPEN state without uvicorn marking
its own close as sent, so the second close raises ``InvalidState`` and aborts
the server's shutdown part way. A real case: a chat closing its connection as
the daemon stopped for a version restart. Such a connection needs only its
socket closed.
"""

from __future__ import annotations

from typing import override

from uvicorn.protocols.websockets.websockets_sansio_impl import (
    WebSocketsSansIOProtocol,
)
from websockets.protocol import State


class ShutdownSafeWebSocket(WebSocketsSansIOProtocol):
    """The sans-I/O protocol, whose shutdown skips a connection already closing."""

    @override
    def shutdown(self) -> None:
        if (
            self.handshake_complete
            and not self.close_sent
            and self.conn.state is not State.OPEN
        ):
            self.stop_keepalive()
            self.transport.close()
            return
        super().shutdown()
