"""A response body that never ends, counting how much of it was read."""

from collections.abc import AsyncIterator

import httpx

PIECE = 64 * 1024


class EndlessBody(httpx.AsyncByteStream):
    """Yield ``piece`` forever; ``pieces_read`` says how many were taken."""

    def __init__(self, piece: bytes = b"x" * PIECE) -> None:
        self.piece = piece
        self.pieces_read = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        while True:
            self.pieces_read += 1
            yield self.piece
