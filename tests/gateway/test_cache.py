from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path
from typing import cast

import pytest

from synthia.gateway.cache import CachingModel, ResponseCache, request_key
from synthia.gateway.errors import ProviderError
from synthia.gateway.protocol import collect
from synthia.gateway.types import (
    ChatChunk,
    ChatRequest,
    FinishReason,
    ImagePart,
    Message,
    ModelInfo,
    ToolCall,
    ToolCallDelta,
    ToolSpec,
    Usage,
)

INFO = ModelInfo("qwen3.5-4b", 1000, vision=True, tools=True)
ASK = ChatRequest((Message.user("capital of France?"),), temperature=0.0)


class Counting:
    """Stream a scripted answer and count how often it was asked."""

    def __init__(
        self, chunks: tuple[ChatChunk, ...], error: Exception | None = None
    ) -> None:
        self.chunks = chunks
        self.error = error
        self.calls = 0

    @property
    def info(self) -> ModelInfo:
        return INFO

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        del request
        self.calls += 1
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error


PARIS = (
    ChatChunk(text="Par", model="qwen/qwen3.5-4b", reasoning="recall"),
    ChatChunk(text="is"),
    ChatChunk(finish_reason=FinishReason.STOP, usage=Usage(12, 2)),
)


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def cached(
    tmp_path: Path, inner: Counting, ttl: float = 3600.0
) -> tuple[CachingModel, Clock]:
    clock = Clock()
    return CachingModel(inner, ResponseCache(tmp_path / "cache.db", ttl, clock)), clock


async def test_a_repeat_is_answered_from_the_cache_identically(tmp_path: Path) -> None:
    inner = Counting(PARIS)
    model, _ = cached(tmp_path, inner)

    first = await collect(model.stream(ASK))
    second = await collect(model.stream(ASK))

    assert inner.calls == 1
    assert second == first
    assert (second.text, second.reasoning, second.model) == (
        "Paris",
        "recall",
        "qwen/qwen3.5-4b",
    )


async def test_tool_calls_survive_the_cache(tmp_path: Path) -> None:
    chunks = (
        ChatChunk(tool_calls=(ToolCallDelta(0, "c1", "clock", '{"zone":'),)),
        ChatChunk(tool_calls=(ToolCallDelta(0, arguments='"IST"}'),)),
        ChatChunk(finish_reason=FinishReason.TOOL_CALLS),
    )
    model, _ = cached(tmp_path, Counting(chunks))

    await collect(model.stream(ASK))
    replayed = await collect(model.stream(ASK))

    assert replayed.tool_calls == (ToolCall("c1", "clock", '{"zone":"IST"}'),)
    assert replayed.finish_reason is FinishReason.TOOL_CALLS
    assert replayed.usage is None


async def test_a_failed_stream_is_never_cached(tmp_path: Path) -> None:
    inner = Counting(PARIS[:1], error=ProviderError("HTTP 502"))
    model, _ = cached(tmp_path, inner)

    for _ in range(2):
        with pytest.raises(ProviderError):
            await collect(model.stream(ASK))

    assert inner.calls == 2


async def test_a_stream_its_listener_stopped_early_is_never_cached(
    tmp_path: Path,
) -> None:
    inner = Counting(PARIS)
    model, _ = cached(tmp_path, inner)

    stream = cast("AsyncGenerator[ChatChunk]", model.stream(ASK))
    await anext(stream)
    await stream.aclose()  # barge-in: the listener walked away mid-answer
    await collect(model.stream(ASK))

    assert inner.calls == 2


async def test_an_answer_without_a_finish_reason_passes_through_uncached(
    tmp_path: Path,
) -> None:
    inner = Counting((ChatChunk(text="half"),))
    model, _ = cached(tmp_path, inner)

    seen = [chunk.text async for chunk in model.stream(ASK)]
    [chunk.text async for chunk in model.stream(ASK)]

    assert seen == ["half"]
    assert inner.calls == 2


async def test_entries_expire_after_the_ttl(tmp_path: Path) -> None:
    inner = Counting(PARIS)
    model, clock = cached(tmp_path, inner, ttl=60.0)

    await collect(model.stream(ASK))
    clock.now += 59.0
    await collect(model.stream(ASK))
    clock.now += 1.0
    await collect(model.stream(ASK))

    assert inner.calls == 2


async def test_the_cache_survives_a_restart(tmp_path: Path) -> None:
    await collect(
        CachingModel(Counting(PARIS), ResponseCache(tmp_path / "c.db")).stream(ASK)
    )
    inner = Counting(PARIS)

    await collect(CachingModel(inner, ResponseCache(tmp_path / "c.db")).stream(ASK))

    assert inner.calls == 0


def test_everything_that_shapes_the_answer_changes_the_key() -> None:
    image = ImagePart(b"\x89PNG one", "image/png")
    other_image = ImagePart(b"\x89PNG two", "image/png")
    tool = ToolSpec("clock", "time", {"type": "object"})
    variants = [
        request_key("qwen3.5-4b", ASK),
        request_key("other-model", ASK),
        request_key("qwen3.5-4b", ChatRequest(ASK.messages, temperature=0.7)),
        request_key("qwen3.5-4b", ChatRequest(ASK.messages, max_tokens=5)),
        request_key("qwen3.5-4b", ChatRequest(ASK.messages, tools=(tool,))),
        request_key(
            "qwen3.5-4b", ChatRequest(ASK.messages, response_schema={"type": "object"})
        ),
        request_key("qwen3.5-4b", ChatRequest((Message.user("x", image),))),
        request_key("qwen3.5-4b", ChatRequest((Message.user("x", other_image),))),
    ]

    assert len(set(variants)) == len(variants)
    assert request_key("qwen3.5-4b", ASK) == request_key(
        "qwen3.5-4b",
        ChatRequest((Message.user("capital of France?"),), temperature=0.0),
    )


def test_a_request_holding_something_unkeyable_is_refused() -> None:
    odd = ChatRequest(ASK.messages, tools=(ToolSpec("t", "d", {"enum": {1, 2}}),))

    with pytest.raises(TypeError, match="set"):
        request_key("m", odd)


def test_a_nonpositive_ttl_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ttl_s"):
        ResponseCache(tmp_path / "c.db", ttl_s=0)


def test_info_is_the_wrapped_models(tmp_path: Path) -> None:
    model, _ = cached(tmp_path, Counting(PARIS))

    assert model.info is INFO
