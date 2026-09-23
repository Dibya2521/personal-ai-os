"""An incremental parser for Server-Sent Events.

Streaming chat APIs send tokens as a ``text/event-stream``. This follows the
WHATWG HTML specification's "event stream interpretation": UTF-8, lines ending in
CRLF, LF or CR, a blank line to dispatch, ``:`` lines as comments (OpenRouter
sends ``: OPENROUTER PROCESSING`` to keep a connection alive), and the fields
``event``, ``data``, ``id`` and ``retry``.

Bytes arrive in whatever pieces the network delivers, so the parser holds a
partial line and a multi-byte character split across pieces, and produces the
same events however the stream is cut.
"""

from __future__ import annotations

import codecs
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterable

DEFAULT_EVENT = "message"
BOM = "﻿"
NULL = "\x00"
_LINE_END = re.compile(r"\r\n|\r|\n")


@dataclass(frozen=True, slots=True)
class ServerSentEvent:
    """One dispatched event. ``id`` is the last event id seen so far."""

    data: str
    event: str = DEFAULT_EVENT
    id: str | None = None
    retry: int | None = None


class SSEParser:
    """Turn chunks of bytes into events, carrying state between chunks."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._pending: list[str] = []
        self._started = False
        self._skip_lf = False
        self._data: list[str] = []
        self._event = ""
        self._last_id: str | None = None
        self._retry: int | None = None

    def feed(self, chunk: bytes) -> list[ServerSentEvent]:
        """Consume ``chunk`` and return the events it completed."""
        return self._consume(self._decoder.decode(chunk))

    def close(self) -> None:
        """End the stream, discarding whatever is left.

        What can be left is a last line without a line ending, which the
        specification discards so a cut-off event is never dispatched, or the
        bytes of an incomplete character, which cannot hold a line ending. So
        closing never completes an event.
        """
        self._decoder.reset()
        self._pending = []

    def _consume(self, text: str) -> list[ServerSentEvent]:
        if not text:
            return []
        if not self._started:
            self._started = True
            text = text.removeprefix(BOM)
        if self._skip_lf:
            text = text.removeprefix("\n")
        # A CR ends its line at once; an LF right after it, even in the next
        # chunk, belongs to the same line ending.
        self._skip_lf = text.endswith("\r")
        # Only the new text is scanned and a partial line is kept in pieces, so
        # a long line arriving in many chunks costs linear time, not quadratic.
        first, *complete = _LINE_END.split(text)
        if not complete:
            self._pending.append(first)
            return []
        *complete, rest = complete
        lines = ["".join([*self._pending, first]), *complete]
        self._pending = [rest] if rest else []
        events: list[ServerSentEvent] = []
        for line in lines:
            events.extend(self._line(line))
        return events

    def _line(self, line: str) -> list[ServerSentEvent]:
        if not line:
            return self._dispatch()
        if line.startswith(":"):
            return []
        name, _, value = line.partition(":")
        self._field(name, value.removeprefix(" "))
        return []

    def _field(self, name: str, value: str) -> None:
        if name == "data":
            self._data.append(value)
        elif name == "event":
            self._event = value
        elif name == "id" and NULL not in value:
            self._last_id = value
        elif name == "retry" and value.isascii() and value.isdigit():
            self._retry = int(value)

    def _dispatch(self) -> list[ServerSentEvent]:
        data, event = self._data, self._event
        self._data, self._event = [], ""
        if not data:
            return []
        return [
            ServerSentEvent(
                data="\n".join(data),
                event=event or DEFAULT_EVENT,
                id=self._last_id,
                retry=self._retry,
            )
        ]


async def aiter_events(chunks: AsyncIterable[bytes]) -> AsyncGenerator[ServerSentEvent]:
    """Yield the events in a stream of byte chunks as each one completes."""
    parser = SSEParser()
    async for chunk in chunks:
        for event in parser.feed(chunk):
            yield event
    parser.close()
