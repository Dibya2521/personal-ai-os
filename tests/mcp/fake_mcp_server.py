"""A small MCP server over stdio for tests, standard library only.

Tools: ``echo``, ``add``, ``fail`` (an error result), ``die`` (exits without
answering), ``slow`` (never answers), ``cancelled`` (the request ids the
client cancelled), ``ask_back`` (asks the client ``ping`` and ``roots/list``
before answering), ``env`` (SYNTHIA_ variable names it sees and FAKE_EXTRA),
``big`` (30,000 characters), ``mixed`` (an image and a text part).

Flags: ``--version V`` answers ``initialize`` with V; ``--garbage`` writes
lines that are not messages first; ``--exit-at-start`` exits with code 2;
``--silent`` never answers; ``--pages`` lists the tools in two pages;
``--odd`` lists a page whose tools are not a list, then a page holding a
non-object, a tool named with a space, and ``echo`` twice.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

type Message = dict[str, object]

PLAIN_TOOLS = ("fail", "die", "slow", "cancelled", "ask_back", "env", "big", "mixed")
TOOLS: list[Message] = [
    {
        "name": "echo",
        "description": "Say the text back.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "add",
        "description": "Add two numbers.",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        },
        "annotations": {"readOnlyHint": True},
    },
    *({"name": name, "inputSchema": {"type": "object"}} for name in PLAIN_TOOLS),
]
UNKNOWN_TOOL = -32602
NO_SUCH_METHOD = -32601
EXIT_DIED = 3
EXIT_AT_START = 2
EXIT_NO_ANSWER = 4


def send(message: Message) -> None:
    sys.stdout.buffer.write(json.dumps({"jsonrpc": "2.0", **message}).encode() + b"\n")
    sys.stdout.buffer.flush()


def read() -> Message | None:
    line = sys.stdin.buffer.readline()
    return None if not line else cast("Message", json.loads(line))


def text(value: str) -> Message:
    return {"content": [{"type": "text", "text": value}]}


class Server:
    def __init__(self, options: argparse.Namespace) -> None:
        self.options = options
        self.cancelled: list[object] = []
        self.tools: dict[str, Callable[[Message], Message | None]] = {
            "echo": lambda a: text(str(a["text"])),
            "add": lambda a: text(str(cast("float", a["a"]) + cast("float", a["b"]))),
            "fail": lambda _: {**text("it failed"), "isError": True},
            "die": lambda _: os._exit(EXIT_DIED),
            "slow": lambda _: None,
            "cancelled": lambda _: text(json.dumps(self.cancelled)),
            "ask_back": lambda _: self.ask_back(),
            "env": lambda _: text(
                json.dumps([own_variables(), os.environ.get("FAKE_EXTRA")])
            ),
            "big": lambda _: text("x" * 30_000),
            "mixed": lambda _: {
                "content": [
                    {"type": "image", "mimeType": "image/png", "data": "AAAA"},
                    {"type": "text", "text": "a caption"},
                ]
            },
        }

    def ask_back(self) -> Message:
        pinged = ask("ping", "s1")
        roots = cast("Message", ask("roots/list", "s2")["error"])
        return text(f"ping {pinged.get('result')} roots {roots['code']}")

    def page(self, cursor: object) -> Message:
        if self.options.odd:
            if cursor is None:
                return {"tools": "not a list", "nextCursor": "odd-2"}
            second_echo = {"name": "echo", "description": "A second echo."}
            return {
                "tools": ["not a tool", {"name": "has space"}, TOOLS[0], second_echo]
            }
        if not self.options.pages:
            return {"tools": TOOLS}
        if cursor == "page-2":
            return {"tools": TOOLS[2:]}
        return {"tools": TOOLS[:2], "nextCursor": "page-2"}

    def answer(self, message: Message) -> Message | None:
        """Return the reply to a request, or None to send nothing."""
        method = message.get("method")
        params = cast("Message", message.get("params") or {})
        if method == "initialize":
            return {
                "protocolVersion": self.options.version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "1"},
            }
        if method == "tools/list":
            return self.page(params.get("cursor"))
        if method == "tools/call":
            run = self.tools.get(str(params.get("name")))
            if run is None:
                return {"error": {"code": UNKNOWN_TOOL, "message": "unknown tool"}}
            result = run(cast("Message", params.get("arguments") or {}))
            return None if result is None else {"result": result}
        return {"error": {"code": NO_SUCH_METHOD, "message": "no such method"}}

    def serve(self) -> None:
        while (message := read()) is not None:
            if "id" not in message:
                if message.get("method") == "notifications/cancelled":
                    params = cast("Message", message["params"])
                    self.cancelled.append(params["requestId"])
                continue
            if self.options.silent:
                continue
            reply = self.answer(message)
            if reply is not None:
                if "error" not in reply and "result" not in reply:
                    reply = {"result": reply}
                send({"id": message["id"], **reply})


def ask(method: str, request_id: str) -> Message:
    send({"id": request_id, "method": method})
    while (message := read()) is not None:
        if message.get("id") == request_id:
            return message
    raise SystemExit(EXIT_NO_ANSWER)


def own_variables() -> list[str]:
    return sorted(k for k in os.environ if k.upper().startswith("SYNTHIA_"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="2025-06-18")
    parser.add_argument("--garbage", action="store_true")
    parser.add_argument("--exit-at-start", action="store_true")
    parser.add_argument("--silent", action="store_true")
    parser.add_argument("--pages", action="store_true")
    parser.add_argument("--odd", action="store_true")
    options = parser.parse_args()
    if options.exit_at_start:
        raise SystemExit(EXIT_AT_START)
    sys.stderr.write("fake mcp server started\n")
    sys.stderr.flush()
    if options.garbage:
        sys.stdout.buffer.write(b"this is not json\n[1, 2]\n\n")
        sys.stdout.buffer.flush()
    Server(options).serve()


if __name__ == "__main__":
    main()
