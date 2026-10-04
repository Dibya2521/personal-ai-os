"""JSON-RPC 2.0 between two peers, one whole message at a time.

Either side may ask (a request, answered with a result or an error), tell (a
notification, never answered), and serve the methods it knows. Framing is the
transport's: one line per message over a process's stdio (MCP), one frame per
message over a WebSocket. Every message is written with non-ASCII escaped, so
no raw newline or lone surrogate is ever inside one. A message that is not a
JSON object is logged and skipped, so one bad message cannot stop the
conversation.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
from typing import TYPE_CHECKING, Final, cast

from synthia.kernel.errors import SynthiaError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

METHOD_NOT_FOUND: Final = -32601
INTERNAL_ERROR: Final = -32603
CANCELLED: Final = "notifications/cancelled"
PING: Final = "ping"
_VERSION: Final = "2.0"

logger = logging.getLogger(__name__)

type Params = dict[str, object]
type Method = Callable[[Params], Awaitable[object]]
"""Serves a request: takes its params, returns its result or raises RpcError."""
type Notice = Callable[[Params], None]
"""Hears a notification."""


class RpcError(SynthiaError):
    """A request was answered with an error, or a method refuses one."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


class ConnectionClosedError(SynthiaError):
    """The other side went away before it answered."""


class Peer:
    """One side of a conversation: asks, tells, and serves ``methods``.

    ``send`` writes one whole message; the transport frames it. A request
    the other side cancels (``notifications/cancelled``) stops its method and
    gets no answer, as MCP specifies.
    """

    def __init__(
        self,
        send: Callable[[bytes], None],
        *,
        methods: Mapping[str, Method] | None = None,
        notices: Mapping[str, Notice] | None = None,
    ) -> None:
        self._send = send
        self._methods = dict(methods or {})
        self._notices = dict(notices or {})
        self._ids = itertools.count(1)
        self._pending: dict[int, asyncio.Future[object]] = {}
        self._serving: dict[int | str, asyncio.Task[None]] = {}
        self._closed: str | None = None

    def receive(self, message: bytes | str) -> None:
        """Take one whole message from the other side."""
        try:
            parsed: object = json.loads(message)
        except ValueError:
            logger.warning("skipped a message that is not JSON: %r", message[:200])
            return
        if not isinstance(parsed, dict):
            logger.warning("skipped a message that is not an object: %r", message[:200])
            return
        fields = cast("dict[str, object]", parsed)
        if "method" not in fields:
            self._resolve(fields)
        elif "id" in fields:
            self._serve(fields)
        else:
            self._hear(fields)

    async def request(self, method: str, params: Params) -> object:
        """Send a request and return its result.

        Raises:
            RpcError: If the other side answered with an error.
            ConnectionClosedError: If the other side went away first.
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

    def notify(self, method: str, params: Params | None = None) -> None:
        """Send a notification, which gets no answer."""
        message: dict[str, object] = {"method": method}
        if params is not None:
            message["params"] = params
        self._write(message)

    def close(self, reason: str) -> None:
        """Fail waiting requests and stop methods being served; nothing more is sent."""
        if self._closed is None:
            self._closed = reason
        for answer in self._pending.values():
            if not answer.done():
                answer.set_exception(ConnectionClosedError(reason))
        for task in self._serving.values():
            task.cancel()

    def _write(self, message: dict[str, object]) -> None:
        if self._closed is None:
            self._send(json.dumps({"jsonrpc": _VERSION, **message}).encode())

    def _serve(self, fields: dict[str, object]) -> None:
        request_id, method = fields["id"], fields["method"]
        if not isinstance(request_id, int | str):
            logger.warning("skipped a request whose id is %r", request_id)
            return
        handler = self._methods.get(method) if isinstance(method, str) else None
        if method == PING:
            self._write({"id": request_id, "result": {}})
        elif handler is None:
            error = {"code": METHOD_NOT_FOUND, "message": "method not supported"}
            self._write({"id": request_id, "error": error})
        else:
            task = asyncio.get_running_loop().create_task(
                self._run(request_id, handler, _params(fields))
            )
            self._serving[request_id] = task
            task.add_done_callback(lambda _: self._serving.pop(request_id, None))

    async def _run(
        self, request_id: int | str, handler: Method, params: Params
    ) -> None:
        try:
            result = await handler(params)
        except RpcError as error:
            failure = {"code": error.code, "message": str(error)}
            self._write({"id": request_id, "error": failure})
        except Exception:
            logger.exception("method failed while serving request %r", request_id)
            failure = {"code": INTERNAL_ERROR, "message": "internal error"}
            self._write({"id": request_id, "error": failure})
        else:
            self._write({"id": request_id, "result": result})

    def _hear(self, fields: dict[str, object]) -> None:
        method, params = fields["method"], _params(fields)
        if method == CANCELLED:
            request_id = params.get("requestId")
            task = (
                self._serving.get(request_id)
                if isinstance(request_id, int | str)
                else None
            )
            if task is not None:
                task.cancel()
            return
        handler = self._notices.get(method) if isinstance(method, str) else None
        if handler is not None:
            handler(params)

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


def _params(fields: dict[str, object]) -> Params:
    params = fields.get("params")
    return cast("Params", params) if isinstance(params, dict) else {}
