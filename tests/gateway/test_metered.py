import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from synthia.gateway.budget import BudgetExhaustedError, BudgetLedger
from synthia.gateway.errors import GatewayError, ProviderError
from synthia.gateway.metered import MeteredModel
from synthia.gateway.protocol import collect
from synthia.gateway.ratelimit import SlidingWindowLimiter
from synthia.gateway.retry import RetryingModel, RetryPolicy
from synthia.gateway.types import (
    ChatChunk,
    ChatRequest,
    FinishReason,
    Message,
    ModelInfo,
)

INFO = ModelInfo("remote", 1000, vision=False, tools=False)
HELLO = ChatRequest((Message.user("hello"),))
PROVIDER = "openrouter"


class Provider:
    """Answer after raising the scripted errors, counting every call."""

    def __init__(self, *errors: GatewayError) -> None:
        self.errors = list(errors)
        self.calls = 0

    @property
    def info(self) -> ModelInfo:
        return INFO

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        del request
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        yield ChatChunk(text="hi")
        yield ChatChunk(finish_reason=FinishReason.STOP)


def ledger(tmp_path: Path, cap: int = 3) -> BudgetLedger:
    now = datetime(2026, 9, 24, 10, tzinfo=UTC)
    return BudgetLedger(tmp_path / "gateway.db", {PROVIDER: cap}, clock=lambda: now)


async def no_sleep(_: float) -> None:
    return None


def metered(
    tmp_path: Path, inner: Provider, cap: int = 3, rpm: int = 20
) -> tuple[MeteredModel, BudgetLedger]:
    books = ledger(tmp_path, cap)
    return MeteredModel(inner, PROVIDER, books, SlidingWindowLimiter(rpm)), books


async def test_each_send_claims_one_request_and_passes_the_answer_through(
    tmp_path: Path,
) -> None:
    inner = Provider()
    model, books = metered(tmp_path, inner)

    first = await collect(model.stream(HELLO))
    await collect(model.stream(HELLO))

    assert first.text == "hi"
    assert model.info is INFO
    assert (await books.status(PROVIDER)).used == 2


async def test_an_exhausted_budget_refuses_before_anything_is_sent(
    tmp_path: Path,
) -> None:
    inner = Provider()
    model, _ = metered(tmp_path, inner, cap=1)
    await collect(model.stream(HELLO))

    with pytest.raises(BudgetExhaustedError):
        await collect(model.stream(HELLO))
    assert inner.calls == 1


async def test_a_send_cancelled_while_rate_limited_costs_no_budget(
    tmp_path: Path,
) -> None:
    books = ledger(tmp_path)
    blocked = asyncio.Event()

    async def wait_forever(_: float) -> None:
        blocked.set()
        await asyncio.Event().wait()

    limiter = SlidingWindowLimiter(1, clock=lambda: 0.0, sleep=wait_forever)
    model = MeteredModel(Provider(), PROVIDER, books, limiter)
    await collect(model.stream(HELLO))

    waiting = asyncio.create_task(collect(model.stream(HELLO)))
    await asyncio.wait_for(blocked.wait(), timeout=1)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting

    assert (await books.status(PROVIDER)).used == 1


async def test_every_retry_attempt_is_claimed_as_the_provider_counts_it(
    tmp_path: Path,
) -> None:
    inner = Provider(ProviderError("busy"), ProviderError("busy"))
    model, books = metered(tmp_path, inner)
    retrying = RetryingModel(model, RetryPolicy(max_attempts=3), sleep=no_sleep)

    answer = await collect(retrying.stream(HELLO))

    assert answer.text == "hi"
    assert inner.calls == 3
    assert (await books.status(PROVIDER)).used == 3
