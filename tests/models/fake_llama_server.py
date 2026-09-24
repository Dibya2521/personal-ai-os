"""A stand-in for llama-server that accepts its flags and answers ``/health``.

Behaviour is set through the environment, which the launch passes on:
``FAKE_EXIT_CODE`` exits at once with that code; ``FAKE_LOAD_S`` answers 503
for that long, as while a model loads; ``FAKE_EXIT_AFTER_S`` exits with code 9
that long after it is ready, as a crash would.
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

        @override
        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer((options.host, options.port), Handler)
    if after := os.environ.get("FAKE_EXIT_AFTER_S"):
        delay = max(ready_at - time.monotonic(), 0) + float(after)
        threading.Timer(delay, os._exit, (CRASH_CODE,)).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
