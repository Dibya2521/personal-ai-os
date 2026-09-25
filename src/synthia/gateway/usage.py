"""Account for every finished model call, and check the count against the provider.

Each call that finishes is recorded per UTC day and per model: calls, tokens
and seconds, in the gateway database beside the budget ledger. The model is
the one that actually answered, so a day on ``openrouter/free`` shows which
free models served it. A :class:`UsageRecorded` event goes out for each call.

The budget ledger is this machine's count; OpenRouter keeps its own. Asking
its key endpoint costs no request, so the two can be compared at any time, and
a difference means requests were made that this machine did not meter.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from contextlib import aclosing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Final, cast

import httpx

from synthia.gateway.errors import (
    ConnectionFailedError,
    IncompleteResponseError,
    ProviderError,
)
from synthia.gateway.openai_compat import error_for, scrub
from synthia.gateway.protocol import join
from synthia.kernel.bus import Event

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable
    from pathlib import Path

    from pydantic import SecretStr

    from synthia.gateway.protocol import ChatModel
    from synthia.gateway.types import ChatChunk, ChatRequest, ModelInfo, Usage

BUSY_TIMEOUT_S: Final = 5.0
SPEED_DAYS: Final = 7

_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS usage (
    day TEXT NOT NULL,
    model TEXT NOT NULL,
    calls INTEGER NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    seconds REAL NOT NULL,
    PRIMARY KEY (day, model)
)
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class UsageRecorded(Event):
    """A model call finished; tokens are ``None`` when the provider did not say."""

    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    seconds: float


@dataclass(frozen=True, slots=True)
class ModelTotals:
    """One model's use on one UTC day."""

    model: str
    calls: int
    prompt_tokens: int
    completion_tokens: int
    seconds: float


def _utc_now() -> datetime:
    return datetime.now(UTC)


class UsageLog:
    """Per-day, per-model totals of finished calls, in SQLite."""

    def __init__(self, path: Path, clock: Callable[[], datetime] = _utc_now) -> None:
        """Open the log at ``path``, creating it if needed; ``clock`` is aware."""
        self._path = path
        self._clock = clock
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(_SCHEMA)
        finally:
            connection.close()

    async def record(self, model: str, usage: Usage | None, seconds: float) -> None:
        """Add one finished call to today's totals for ``model``."""
        await asyncio.to_thread(self._record, model, usage, seconds)

    async def today(self) -> tuple[ModelTotals, ...]:
        """Return today's totals, the busiest model first."""
        return await asyncio.to_thread(self._today)

    async def tokens_per_second(
        self, exclude: frozenset[str] = frozenset()
    ) -> float | None:
        """Return completion tokens per second over the last 7 UTC days.

        The seconds are whole calls, the wait for the first token included, so
        this reads a little below the generation speed. Models in ``exclude``
        are left out. ``None`` until a call with tokens has been recorded.
        """
        rows = await asyncio.to_thread(self._since, SPEED_DAYS)
        kept = [
            (tokens, seconds) for model, tokens, seconds in rows if model not in exclude
        ]
        tokens = sum(t for t, _ in kept)
        seconds = sum(s for _, s in kept)
        return tokens / seconds if tokens and seconds > 0 else None

    def _day(self, days_back: int = 0) -> str:
        today = self._clock().astimezone(UTC).date()
        return (today - timedelta(days=days_back)).isoformat()

    def _since(self, days: int) -> list[tuple[str, int, float]]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT model, SUM(completion_tokens), SUM(seconds) FROM usage "
                "WHERE day >= ? GROUP BY model",
                (self._day(days - 1),),
            ).fetchall()
        finally:
            connection.close()
        return [(str(m), int(t), float(s)) for m, t, s in rows]

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, timeout=BUSY_TIMEOUT_S, isolation_level=None)

    def _record(self, model: str, usage: Usage | None, seconds: float) -> None:
        prompt = usage.prompt_tokens if usage else 0
        completion = usage.completion_tokens if usage else 0
        connection = self._connect()
        try:
            connection.execute(
                "INSERT INTO usage VALUES (?, ?, 1, ?, ?, ?) "
                "ON CONFLICT (day, model) DO UPDATE SET calls = calls + 1, "
                "prompt_tokens = prompt_tokens + excluded.prompt_tokens, "
                "completion_tokens = completion_tokens + excluded.completion_tokens, "
                "seconds = seconds + excluded.seconds",
                (self._day(), model, prompt, completion, seconds),
            )
        finally:
            connection.close()

    def _today(self) -> tuple[ModelTotals, ...]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT model, calls, prompt_tokens, completion_tokens, seconds "
                "FROM usage WHERE day = ? ORDER BY calls DESC, model",
                (self._day(),),
            ).fetchall()
        finally:
            connection.close()
        return tuple(
            ModelTotals(str(m), int(c), int(p), int(o), float(s))
            for m, c, p, o, s in rows
        )


class AccountingModel:
    """A :class:`~synthia.gateway.protocol.ChatModel` recording each finished call."""

    def __init__(
        self,
        inner: ChatModel,
        log: UsageLog,
        publish: Callable[[Event], Awaitable[None]],
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._inner = inner
        self._log = log
        self._publish = publish
        self._clock = clock

    @property
    def info(self) -> ModelInfo:
        """Return the wrapped model's capabilities."""
        return self._inner.info

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        """Yield the wrapped model's answer, recording it once it is whole.

        Raises:
            GatewayError: Whatever the wrapped model raised; nothing is recorded.
        """
        started = self._clock()
        seen: list[ChatChunk] = []
        async with aclosing(self._inner.stream(request)) as chunks:
            async for chunk in chunks:
                seen.append(chunk)
                yield chunk
        try:
            response = join(seen)
        except IncompleteResponseError:
            return
        seconds = self._clock() - started
        model = response.model or self._inner.info.id
        usage = response.usage
        await self._log.record(model, usage, seconds)
        await self._publish(
            UsageRecorded(
                model=model,
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                seconds=seconds,
            )
        )


@dataclass(frozen=True, slots=True)
class ProviderCount:
    """The provider's own count of today's free-model requests."""

    used: int
    limit: int
    remaining: int
    paid: bool


async def free_requests_today(
    client: httpx.AsyncClient, api_key: SecretStr, base_url: str
) -> ProviderCount:
    """Ask OpenRouter how many free-model requests it counted today (UTC).

    This reads the key's record and is not itself a model request.

    Raises:
        GatewayError: If the provider refused, or answered in an unknown shape.
    """
    url = base_url.rstrip("/") + "/key"
    try:
        response = await client.get(
            url, headers={"Authorization": f"Bearer {api_key.get_secret_value()}"}
        )
    except httpx.TransportError as error:
        message = f"no response from {url}: {type(error).__name__}"
        raise ConnectionFailedError(message) from error
    if response.is_error:
        error = error_for(response.status_code, response.content, response.headers)
        raise scrub(error, api_key)
    try:
        data = cast("dict[str, Any]", response.json()["data"])
        free = cast("dict[str, Any]", data["free_model_daily_requests"])
        return ProviderCount(
            used=int(free["used"]),
            limit=int(free["limit"]),
            remaining=int(free["remaining"]),
            paid=not bool(data["is_free_tier"]),
        )
    except (ValueError, KeyError, TypeError) as error:
        message = (
            f"the key endpoint answered in an unknown shape: {type(error).__name__}"
        )
        raise ProviderError(message) from error
