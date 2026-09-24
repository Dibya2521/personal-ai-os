"""A stand-in for llama-server that accepts its flags and answers ``/health``.

Behaviour is set through the environment, which the launch passes on:
``FAKE_EXIT_CODE`` exits at once with that code; ``FAKE_LOAD_S`` answers 503
for that long, as while a model loads; ``FAKE_EXIT_AFTER_S`` exits with code 9
that long after its first healthy answer, as a crash would.
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
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    options, _ = parser.parse_known_args()
    if code := os.environ.get("FAKE_EXIT_CODE"):
        sys.exit(int(code))
    ready_at = time.monotonic() + float(os.environ.get("FAKE_LOAD_S", "0"))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                _reply(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
            elif time.monotonic() < ready_at:
                loading = {"code": 503, "message": "Loading model"}
                _reply(self, HTTPStatus.SERVICE_UNAVAILABLE, {"error": loading})
            else:
                _reply(self, HTTPStatus.OK, {"status": "ok"})
                crash_once_seen_ready()

        @override
        def log_message(self, format: str, *args: object) -> None:
            return

    after = os.environ.get("FAKE_EXIT_AFTER_S")
    armed = threading.Lock()

    def crash_once_seen_ready() -> None:
        # Timed from the first healthy answer, not from start-up, so a slow
        # machine can never make the crash land before the caller saw it ready.
        if after is not None and armed.acquire(blocking=False):
            threading.Timer(float(after), os._exit, (CRASH_CODE,)).start()

    server = ThreadingHTTPServer((options.host, options.port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
