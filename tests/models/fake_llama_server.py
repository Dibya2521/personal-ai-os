"""A stand-in for llama-server that accepts its flags and speaks its API.

``/health`` answers as llama-server does. ``/v1/chat/completions`` requires
the launch key and streams back ``echo: `` plus the last user text, after
one piece of reasoning when ``chat_template_kwargs.enable_thinking`` is true.
``/v1/chat/completions/control`` answers as llama-server b11130 did in a real
run: ``reasoning_end`` for a streaming completion gives ``success: true``, an
unknown id 200 with ``success: false``, an unknown action 400.
``GET /fake/controls`` lists the control bodies received, for tests.

Behaviour is set through the environment, which the launch passes on:
``FAKE_EXIT_CODE`` exits at once with that code; ``FAKE_LOAD_S`` answers 503
for that long, as while a model loads; ``FAKE_EXIT_AFTER_S`` exits with code 9
that long after its first healthy answer, as a crash would;
``FAKE_REPORT_SESSION`` prints whether it leads its own session (POSIX only);
``FAKE_THINK_UNTIL_ENDED`` keeps a request sent with ``reasoning_control``
thinking until the control ends it, for at most ``THINK_CAP_S``;
``FAKE_FIRST_BYTE_S`` sends nothing for that long before answering a
completion, as the real server is silent while it reads a long prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, cast, override

if TYPE_CHECKING:
    from collections.abc import Callable

    from synthia.models.server import Launch

CRASH_CODE = 9
THOUGHT = "thinking it over"
CONTROL_PATH = "/v1/chat/completions/control"
CONTROLS_PATH = "/fake/controls"
THINK_CAP_S = 10.0
THINK_EVERY_S = 0.02


class Completions:
    """The completions streaming now, so the control route can end their thinking."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._count = 0
        self._ended: dict[str, threading.Event] = {}
        self.controls: list[object] = []

    def open(self) -> tuple[str, threading.Event]:
        with self._lock:
            self._count += 1
            completion_id = f"chatcmpl-fake-{self._count}"
            ended = self._ended[completion_id] = threading.Event()
        return completion_id, ended

    def close(self, completion_id: str) -> None:
        with self._lock:
            self._ended.pop(completion_id, None)

    def end(self, body: object) -> bool:
        with self._lock:
            self.controls.append(body)
            completion_id = cast("dict[str, object]", body).get("id")
            ended = self._ended.get(str(completion_id))
        if ended is None:
            return False
        ended.set()
        return True


def command(launch: Launch) -> list[str]:
    """Return the command that runs this stand-in in place of ``launch``'s server."""
    return [sys.executable, __file__, *launch.command()[1:]]


def _reply(handler: BaseHTTPRequestHandler, status: HTTPStatus, body: object) -> None:
    content = json.dumps(body).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(content)))
    handler.end_headers()
    handler.wfile.write(content)


def _last_text(body: dict[str, object]) -> str:
    messages = cast("list[dict[str, object]]", body["messages"])
    content = messages[-1]["content"]
    if isinstance(content, str):
        return content
    parts = cast("list[dict[str, str]]", content)
    return "".join(p["text"] for p in parts if p["type"] == "text")


def _thinking(body: dict[str, object]) -> bool:
    kwargs = cast("dict[str, object]", body.get("chat_template_kwargs") or {})
    return kwargs.get("enable_thinking") is True


def _send(
    handler: BaseHTTPRequestHandler,
    completion_id: str,
    delta: dict[str, object],
    *,
    last: bool = False,
) -> None:
    choice: dict[str, object] = {"index": 0, "delta": delta}
    chunk: dict[str, object] = {"id": completion_id, "model": "local"}
    if last:
        choice["finish_reason"] = "stop"
        chunk["usage"] = {"prompt_tokens": 3, "completion_tokens": 2}
    chunk["choices"] = [choice]
    handler.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
    handler.wfile.flush()


def _stream(
    handler: BaseHTTPRequestHandler,
    text: str,
    completions: Completions,
    *,
    think: bool,
    until_ended: bool,
) -> None:
    completion_id, ended = completions.open()
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "text/event-stream")
    handler.end_headers()
    try:
        if think:
            _send(handler, completion_id, {"reasoning_content": THOUGHT})
            if until_ended:
                cap = time.monotonic() + THINK_CAP_S
                while not ended.wait(THINK_EVERY_S) and time.monotonic() < cap:
                    _send(handler, completion_id, {"reasoning_content": "."})
        _send(handler, completion_id, {"content": text})
        _send(handler, completion_id, {}, last=True)
        handler.wfile.write(b"data: [DONE]\n\n")
    finally:
        completions.close(completion_id)


def _control(
    handler: BaseHTTPRequestHandler, body: dict[str, object], completions: Completions
) -> None:
    if body.get("action") != "reasoning_end":
        error = {"code": 400, "message": "unknown control action"}
        _reply(handler, HTTPStatus.BAD_REQUEST, {"error": error})
    elif completions.end(body):
        _reply(handler, HTTPStatus.OK, {"success": True})
    else:
        unknown = {"success": False, "message": "no active completion for this id"}
        _reply(handler, HTTPStatus.OK, unknown)


def _handler(
    ready_at: float,
    key: str,
    seen_ready: Callable[[], None],
    *,
    until_ended: bool,
    first_byte_s: float,
) -> type[BaseHTTPRequestHandler]:
    completions = Completions()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == CONTROLS_PATH:
                _reply(self, HTTPStatus.OK, completions.controls)
            elif self.path != "/health":
                _reply(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
            elif time.monotonic() < ready_at:
                loading = {"code": 503, "message": "Loading model"}
                _reply(self, HTTPStatus.SERVICE_UNAVAILABLE, {"error": loading})
            else:
                _reply(self, HTTPStatus.OK, {"status": "ok"})
                seen_ready()

        def do_POST(self) -> None:
            # Read the body before any refusal: closing a socket with unread
            # data resets the connection on Windows, and the client loses the
            # answer it was about to read.
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if self.path not in {"/v1/chat/completions", CONTROL_PATH}:
                _reply(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            if self.headers.get("Authorization") != f"Bearer {key}":
                _reply(self, HTTPStatus.UNAUTHORIZED, {"error": "Invalid API Key"})
                return
            body = cast("dict[str, object]", json.loads(raw))
            if self.path == CONTROL_PATH:
                _control(self, body, completions)
                return
            time.sleep(first_byte_s)
            _stream(
                self,
                f"echo: {_last_text(body)}",
                completions,
                think=_thinking(body),
                until_ended=until_ended and body.get("reasoning_control") is True,
            )

        @override
        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    options, _ = parser.parse_known_args()
    if code := os.environ.get("FAKE_EXIT_CODE"):
        sys.exit(int(code))
    if sys.platform != "win32" and os.environ.get("FAKE_REPORT_SESSION"):
        sys.stdout.write(f"own session: {os.getsid(0) == os.getpid()}\n")
        sys.stdout.flush()
    ready_at = time.monotonic() + float(os.environ.get("FAKE_LOAD_S", "0"))
    after = os.environ.get("FAKE_EXIT_AFTER_S")
    armed = threading.Lock()

    def crash_once_seen_ready() -> None:
        # Timed from the first healthy answer, not from start-up, so a slow
        # machine can never make the crash land before the caller saw it ready.
        if after is not None and armed.acquire(blocking=False):
            threading.Timer(float(after), os._exit, (CRASH_CODE,)).start()

    handler = _handler(
        ready_at,
        os.environ["LLAMA_API_KEY"],
        crash_once_seen_ready,
        until_ended=bool(os.environ.get("FAKE_THINK_UNTIL_ENDED")),
        first_byte_s=float(os.environ.get("FAKE_FIRST_BYTE_S", "0")),
    )
    server = ThreadingHTTPServer((options.host, options.port), handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
