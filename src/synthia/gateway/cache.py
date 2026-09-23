"""Serve exact repeats of a request from a cache instead of a model.

Evaluations, extraction jobs and development runs send the same request many
times; each remote one costs a request from the daily budget. Caching is opt-in
by composition, not a flag: wrap a model in :class:`CachingModel` where a repeat
answer is what is wanted. Live conversation is not wrapped, since asking the
same thing twice there should not return the same words.

The key is a SHA-256 over everything that shapes the answer: the model id, the
messages (images by the hash of their bytes), the tools, temperature, token
limit and schema. Only a complete answer is stored; a stream that failed, or
that its listener stopped early, is never cached, because a truncated answer
served later would be a hard bug to find.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import sqlite3
import time
from typing import TYPE_CHECKING, Any, cast

from synthia.gateway.errors import IncompleteResponseError
from synthia.gateway.protocol import collect
from synthia.gateway.types import (
    ChatChunk,
    ChatResponse,
    FinishReason,
    ToolCall,
    ToolCallDelta,
    Usage,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from pathlib import Path

    from synthia.gateway.protocol import ChatModel
    from synthia.gateway.types import ChatRequest, ModelInfo

DEFAULT_TTL_S = 7 * 24 * 3600.0
BUSY_TIMEOUT_S = 5.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key TEXT PRIMARY KEY,
    stored_at REAL NOT NULL,
    response TEXT NOT NULL
)
"""


def _hashable(value: object) -> object:
    if isinstance(value, bytes):
        return {"sha256": hashlib.sha256(value).hexdigest()}
    message = f"cannot key a request containing {type(value).__name__}"
    raise TypeError(message)


def request_key(model_id: str, request: ChatRequest) -> str:
    """Return the cache key for sending ``request`` to the model ``model_id``."""
    fields = {"model_id": model_id, "request": dataclasses.asdict(request)}
    canonical = json.dumps(
        fields, sort_keys=True, default=_hashable, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _encode(response: ChatResponse) -> str:
    return json.dumps(dataclasses.asdict(response), sort_keys=True)


def _decode(stored: str) -> ChatResponse:
    data = cast("dict[str, Any]", json.loads(stored))
    usage = cast("dict[str, int] | None", data["usage"])
    return ChatResponse(
        text=data["text"],
        tool_calls=tuple(ToolCall(**c) for c in data["tool_calls"]),
        finish_reason=FinishReason(data["finish_reason"]),
        usage=Usage(**usage) if usage else None,
        model=data["model"],
        reasoning=data["reasoning"],
    )


def _as_chunks(response: ChatResponse) -> tuple[ChatChunk, ...]:
    calls = tuple(
        ToolCallDelta(index, call.id, call.name, call.arguments)
        for index, call in enumerate(response.tool_calls)
    )
    return (
        ChatChunk(
            text=response.text,
            reasoning=response.reasoning,
            tool_calls=calls,
            model=response.model,
        ),
        ChatChunk(finish_reason=response.finish_reason, usage=response.usage),
    )


class ResponseCache:
    """A SQLite store of complete responses, keyed by request, expiring after a TTL."""

    def __init__(
        self,
        path: Path,
        ttl_s: float = DEFAULT_TTL_S,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Open the cache at ``path``, creating it if needed.

        Raises:
            ValueError: If ``ttl_s`` is not positive.
        """
        if ttl_s <= 0:
            message = "ttl_s must be positive"
            raise ValueError(message)
        self._path = path
        self._ttl = ttl_s
        self._clock = clock
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(_SCHEMA)
        finally:
            connection.close()

    async def get(self, key: str) -> ChatResponse | None:
        """Return the stored response for ``key``, or ``None`` if absent or expired."""
        return await asyncio.to_thread(self._get, key)

    async def put(self, key: str, response: ChatResponse) -> None:
        """Store ``response`` under ``key``, replacing any earlier one."""
        await asyncio.to_thread(self._put, key, response)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, timeout=BUSY_TIMEOUT_S, isolation_level=None)

    def _get(self, key: str) -> ChatResponse | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT stored_at, response FROM responses WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            if self._clock() - float(row[0]) >= self._ttl:
                connection.execute("DELETE FROM responses WHERE key = ?", (key,))
                return None
            return _decode(str(row[1]))
        finally:
            connection.close()

    def _put(self, key: str, response: ChatResponse) -> None:
        connection = self._connect()
        try:
            connection.execute(
                "INSERT OR REPLACE INTO responses (key, stored_at, response) "
                "VALUES (?, ?, ?)",
                (key, self._clock(), _encode(response)),
            )
        finally:
            connection.close()


class CachingModel:
    """A :class:`~synthia.gateway.protocol.ChatModel` answering repeats from a cache."""

    def __init__(self, inner: ChatModel, cache: ResponseCache) -> None:
        self._inner = inner
        self._cache = cache

    @property
    def info(self) -> ModelInfo:
        """Return the wrapped model's capabilities."""
        return self._inner.info

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        """Yield a cached answer if there is one, otherwise the model's, storing it.

        Raises:
            GatewayError: Whatever the wrapped model raised; nothing is cached then.
        """
        key = request_key(self._inner.info.id, request)
        cached = await self._cache.get(key)
        if cached is not None:
            for chunk in _as_chunks(cached):
                yield chunk
            return
        seen: list[ChatChunk] = []
        async for chunk in self._inner.stream(request):
            seen.append(chunk)
            yield chunk
        # Reached only when the stream ran to its end: a failure raised above,
        # and a listener that stopped early never resumes this generator.
        try:
            response = await collect(_replay(seen))
        except IncompleteResponseError:
            return  # not whole, so not cached; judging it is the adapter's job
        await self._cache.put(key, response)


async def _replay(chunks: list[ChatChunk]) -> AsyncIterator[ChatChunk]:
    for chunk in chunks:
        yield chunk
