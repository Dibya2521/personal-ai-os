"""MCP's stdio framing: one JSON-RPC message per line, in UTF-8.

A message never holds a raw newline (JSON escapes it), so a line is a
message. A line longer than :data:`MAX_LINE_BYTES` is not a reply; the peer
is closed rather than buffering without bound.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable

    from synthia.kernel.jsonrpc import Peer

# A whole tool result arrives as one line; past this it is not a reply, and
# keeping it would grow without bound.
MAX_LINE_BYTES: Final = 8 * 1024 * 1024


def line_sender(write: Callable[[bytes], None]) -> Callable[[bytes], None]:
    """Return a ``send`` for a :class:`Peer` that ends each message with a newline."""

    def send(message: bytes) -> None:
        write(message + b"\n")

    return send


class Lines:
    """Split what a process writes into lines and hand each one to ``peer``."""

    def __init__(self, peer: Peer) -> None:
        self._peer = peer
        self._buffer = bytearray()

    def feed(self, data: bytes) -> None:
        """Take bytes the process wrote; every complete line is handled now."""
        self._buffer.extend(data)
        while (end := self._buffer.find(b"\n")) >= 0:
            line = bytes(self._buffer[:end])
            del self._buffer[: end + 1]
            if line.strip():
                self._peer.receive(line)
        if len(self._buffer) > MAX_LINE_BYTES:
            self._buffer.clear()
            self._peer.close(f"a message was longer than {MAX_LINE_BYTES:,} bytes")
