"""A stand-in for llama-server that accepts its flags and speaks its API.

``/health`` answers as llama-server does. ``/v1/chat/completions`` requires
the launch key and streams back ``echo: `` plus the last user text.

Behaviour is set through the environment, which the launch passes on:
``FAKE_EXIT_CODE`` exits at once with that code; ``FAKE_LOAD_S`` answers 503
for that long, as while a model loads; ``FAKE_EXIT_AFTER_S`` exits with code 9
that long after its first healthy answer, as a crash would;
``FAKE_REPORT_SESSION`` prints whether it leads its own session (POSIX only).
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


def _stream(handler: BaseHTTPRequestHandler, text: str) -> None:
    chunks: list[dict[str, object]] = [
        {"model": "local", "choices": [{"index": 0, "delta": {"content": text}}]},
        {
            "model": "local",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        },
    ]
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "text/event-stream")
    handler.end_headers()
    for chunk in chunks:
        handler.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
    handler.wfile.write(b"data: [DONE]\n\n")


def _handler(
    ready_at: float, key: str, seen_ready: Callable[[], None]
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                _reply(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
            elif time.monotonic() < ready_at:
                loading = {"code": 503, "message": "Loading model"}
                _reply(self, HTTPStatus.SERVICE_UNAVAILABLE, {"error": loading})
            else:
                _reply(self, HTTPStatus.OK, {"status": "ok"})
                seen_ready()

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                _reply(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            if self.headers.get("Authorization") != f"Bearer {key}":
                _reply(self, HTTPStatus.UNAUTHORIZED, {"error": "Invalid API Key"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = cast("dict[str, object]", json.loads(self.rfile.read(length)))
            _stream(self, f"echo: {_last_text(body)}")

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

    handler = _handler(ready_at, os.environ["LLAMA_API_KEY"], crash_once_seen_ready)
    server = ThreadingHTTPServer((options.host, options.port), handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
