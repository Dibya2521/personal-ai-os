"""JSON-RPC 2.0 over a process's input and output, one message per line.

This is MCP's stdio transport: every message is one JSON object on one line,
in UTF-8, with no newline inside it. A line that is not such a message is
logged and skipped, so one bad line cannot stop the conversation.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
from typing import TYPE_CHECKING, Final, cast

from synthia.kernel.errors import SynthiaError

if TYPE_CHECKING:
    from collections.abc import Callable

# A whole tool result arrives as one line; past this it is not a reply, and
# keeping it would grow without bound.
MAX_LINE_BYTES: Final = 8 * 1024 * 1024
METHOD_NOT_FOUND: Final = -32601
CANCELLED: Final = "notifications/cancelled"
_VERSION: Final = "2.0"

logger = logging.getLogger(__name__)


class RpcError(SynthiaError):
    """The other side answered a request with an error."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


class ConnectionClosedError(SynthiaError):
    """The other side went away before it answered."""


class Connection:
    """Requests and notifications to one peer; feed it what the peer writes."""

    def __init__(self, send: Callable[[bytes], None]) -> None:
        self._send = send
        self._ids = itertools.count(1)
        self._pending: dict[int, asyncio.Future[object]] = {}
        self._buffer = bytearray()
        self._closed: str | None = None

    def feed(self, data: bytes) -> None:
        """Take bytes the peer wrote; every complete line is handled now."""
        self._buffer.extend(data)
        while (end := self._buffer.find(b"\n")) >= 0:
            line = bytes(self._buffer[:end])
            del self._buffer[: end + 1]
            self._handle(line)
        if len(self._buffer) > MAX_LINE_BYTES:
            self._buffer.clear()
            self.close(f"a message was longer than {MAX_LINE_BYTES:,} bytes")

    async def request(self, method: str, params: dict[str, object]) -> object:
        """Send a request and return its result.

        Raises:
            RpcError: If the peer answered with an error.
            ConnectionClosedError: If the peer went away first.
        """
        if self._closed is not None:
            raise ConnectionClosedError(self._closed)
        request_id = next(self._ids)
        answer = asyncio.get_running_loop().create_future()
        self._pending[request_id] = answer
        self._write({"id": request_id, "method": method, "params": params})
        try:
            return await answer
        except asyncio.CancelledError:
            self.notify(CANCELLED, {"requestId": request_id, "reason": "cancelled"})
            raise
        finally:
            self._pending.pop(request_id, None)

    def notify(self, method: str, params: dict[str, object] | None = None) -> None:
        """Send a notification, which gets no answer."""
        if self._closed is None:
            message: dict[str, object] = {"method": method}
            if params is not None:
                message["params"] = params
            self._write(message)

    def close(self, reason: str) -> None:
        """Fail every waiting request with ``reason``; later requests fail at once."""
        if self._closed is None:
            self._closed = reason
        for answer in self._pending.values():
            if not answer.done():
                answer.set_exception(ConnectionClosedError(reason))

    def _write(self, message: dict[str, object]) -> None:
        # ensure_ascii escapes every non-ASCII character, so no raw newline or
        # lone surrogate can reach the line.
        self._send(json.dumps({"jsonrpc": _VERSION, **message}).encode() + b"\n")

    def _handle(self, line: bytes) -> None:
        if not line.strip():
            return
        try:
            message: object = json.loads(line)
        except ValueError:
            logger.warning("skipped a line that is not JSON: %r", line[:200])
            return
        if not isinstance(message, dict):
            logger.warning("skipped a message that is not an object: %r", line[:200])
            return
        fields = cast("dict[str, object]", message)
        if "method" in fields:
            self._answer_peer(fields)
        else:
            self._resolve(fields)

    def _answer_peer(self, fields: dict[str, object]) -> None:
        """Answer the peer's own request: ``ping`` is answered, others refused."""
        if "id" not in fields:
            return
        if fields["method"] == "ping":
            self._write({"id": fields["id"], "result": {}})
        else:
            error = {"code": METHOD_NOT_FOUND, "message": "method not supported"}
            self._write({"id": fields["id"], "error": error})

    def _resolve(self, fields: dict[str, object]) -> None:
        request_id = fields.get("id")
        answer = self._pending.get(request_id) if isinstance(request_id, int) else None
        if answer is None or answer.done():
            logger.warning("skipped an answer to no waiting request: id %r", request_id)
            return
        error = fields.get("error")
        if isinstance(error, dict):
            details = cast("dict[str, object]", error)
            code = details.get("code")
            answer.set_exception(
                RpcError(
                    code if isinstance(code, int) else 0,
                    str(details.get("message", "no message")),
                )
            )
        else:
            answer.set_result(fields.get("result"))
