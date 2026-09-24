"""Closing the outer stream must release everything under it at once.

On barge-in the listener stops reading and closes the stream. If a wrapper let
its inner generator be finalised by the garbage collector instead, the provider
connection would stay open until the event loop got round to it.
"""

import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from pathlib import Path

import httpx
import pytest

from synthia.gateway.budget import BudgetLedger
from synthia.gateway.cache import CachingModel, ResponseCache
from synthia.gateway.circuit import CircuitBreaker, CircuitBreakerModel
from synthia.gateway.metered import MeteredModel
from synthia.gateway.openai_compat import Endpoint, OpenAICompatibleModel
from synthia.gateway.protocol import ChatModel
from synthia.gateway.ratelimit import SlidingWindowLimiter
from synthia.gateway.retry import RetryingModel
from synthia.gateway.types import (
    ChatChunk,
    ChatRequest,
    FinishReason,
    Message,
    ModelInfo,
)

INFO = ModelInfo("tracked", 1000, vision=False, tools=False)
HELLO = ChatRequest((Message.user("hello"),))


class Tracked:
    """Stream three chunks and record whether the generator was closed."""

    def __init__(self) -> None:
        self.closed = False

    @property
    def info(self) -> ModelInfo:
        return INFO

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        del request
        try:
            yield ChatChunk(text="a")
            yield ChatChunk(text="b")
            yield ChatChunk(finish_reason=FinishReason.STOP)
        finally:
            self.closed = True


type Wrap = Callable[[ChatModel, Path], ChatModel]


def retrying(inner: ChatModel, _: Path) -> ChatModel:
    return RetryingModel(inner)


def breaking(inner: ChatModel, _: Path) -> ChatModel:
    return CircuitBreakerModel(inner, CircuitBreaker("tracked"))


def caching(inner: ChatModel, tmp: Path) -> ChatModel:
    return CachingModel(inner, ResponseCache(tmp / "cache.db"))


def metering(inner: ChatModel, tmp: Path) -> ChatModel:
    ledger = BudgetLedger(tmp / "gateway.db", {"remote": 5})
    return MeteredModel(inner, "remote", ledger, SlidingWindowLimiter(5))


def full_stack(inner: ChatModel, tmp: Path) -> ChatModel:
    return RetryingModel(breaking(metering(caching(inner, tmp), tmp), tmp))


@pytest.mark.parametrize("wrap", [retrying, breaking, caching, metering, full_stack])
async def test_closing_the_outer_stream_closes_the_innermost_at_once(
    wrap: Wrap, tmp_path: Path
) -> None:
    inner = Tracked()
    stream = wrap(inner, tmp_path).stream(HELLO)
    first = await anext(stream)
    assert first.text == "a"
    assert not inner.closed
    await stream.aclose()
    assert inner.closed


class Body(httpx.AsyncByteStream):
    def __init__(self, parts: list[bytes]) -> None:
        self.parts = parts
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for part in self.parts:
            yield part

    async def aclose(self) -> None:
        self.closed = True


def event(text: str) -> bytes:
    data = {"choices": [{"index": 0, "delta": {"content": text}}]}
    return f"data: {json.dumps(data)}\n\n".encode()


async def test_closing_the_adapter_stream_closes_the_http_response() -> None:
    body = Body([event("a"), event("b")])
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=body))
    )
    endpoint = Endpoint("https://example.test/v1/", "m", None, {})
    stream = OpenAICompatibleModel(client=client, endpoint=endpoint, info=INFO).stream(
        HELLO
    )
    assert (await anext(stream)).text == "a"
    await stream.aclose()
    assert body.closed
