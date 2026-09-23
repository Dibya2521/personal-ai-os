import random
from collections.abc import AsyncGenerator, Sequence

import pytest
from hypothesis import given
from hypothesis import strategies as st

from synthia.gateway.errors import (
    AuthError,
    ConnectionFailedError,
    GatewayError,
    ProviderError,
    RateLimitedError,
)
from synthia.gateway.protocol import collect
from synthia.gateway.retry import RetryingModel, RetryPolicy
from synthia.gateway.types import (
    ChatChunk,
    ChatRequest,
    FinishReason,
    Message,
    ModelInfo,
)

INFO = ModelInfo("scripted", 1000, vision=False, tools=False)
HELLO = ChatRequest((Message.user("hello"),))
ANSWER = (ChatChunk(text="Hi"), ChatChunk(finish_reason=FinishReason.STOP))


class Scripted:
    """Fail with each error in turn, then stream the answer."""

    def __init__(
        self, failures: Sequence[GatewayError], fail_after_chunks: int = 0
    ) -> None:
        self.failures = list(failures)
        self.fail_after_chunks = fail_after_chunks
        self.attempts = 0

    @property
    def info(self) -> ModelInfo:
        return INFO

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        del request
        self.attempts += 1
        if self.failures:
            error = self.failures.pop(0)
            for chunk in ANSWER[: self.fail_after_chunks]:
                yield chunk
            raise error
        for chunk in ANSWER:
            yield chunk


class Sleeps:
    def __init__(self) -> None:
        self.waits: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


def retrying(
    inner: Scripted, policy: RetryPolicy | None = None
) -> tuple[RetryingModel, Sleeps]:
    sleeps = Sleeps()
    return RetryingModel(inner, policy, random.Random(7), sleeps), sleeps


async def test_transient_failures_are_retried_until_the_answer_arrives() -> None:
    inner = Scripted([ProviderError("HTTP 503"), ConnectionFailedError("reset")])
    model, sleeps = retrying(inner)

    response = await collect(model.stream(HELLO))

    assert response.text == "Hi"
    assert inner.attempts == 3
    assert len(sleeps.waits) == 2
    assert 0 <= sleeps.waits[0] <= 0.5
    assert 0 <= sleeps.waits[1] <= 1.0


async def test_a_non_retryable_error_is_raised_at_once() -> None:
    inner = Scripted([AuthError("HTTP 401")])
    model, sleeps = retrying(inner)

    with pytest.raises(AuthError):
        await collect(model.stream(HELLO))

    assert (inner.attempts, sleeps.waits) == (1, [])


async def test_the_last_attempt_raises_the_last_error() -> None:
    inner = Scripted(
        [ProviderError("first"), ProviderError("second"), ProviderError("third")]
    )
    model, sleeps = retrying(inner, RetryPolicy(max_attempts=3))

    with pytest.raises(ProviderError, match="third"):
        await collect(model.stream(HELLO))

    assert inner.attempts == 3
    assert len(sleeps.waits) == 2


async def test_nothing_is_retried_once_output_has_started() -> None:
    inner = Scripted([ConnectionFailedError("dropped mid-answer")], fail_after_chunks=1)
    model, sleeps = retrying(inner)
    seen: list[str] = []

    async def listen() -> None:
        async for chunk in model.stream(HELLO):
            seen.append(chunk.text)  # noqa: PERF401 - recording each chunk as it arrives

    with pytest.raises(ConnectionFailedError):
        await listen()

    assert seen == ["Hi"]
    assert (inner.attempts, sleeps.waits) == (1, [])


async def test_retry_after_is_honoured_instead_of_the_drawn_wait() -> None:
    inner = Scripted([RateLimitedError("HTTP 429", retry_after_s=12.0)])
    model, sleeps = retrying(inner)

    await collect(model.stream(HELLO))

    assert sleeps.waits == [12.0]


async def test_a_retry_after_longer_than_we_will_wait_gives_up_at_once() -> None:
    inner = Scripted([RateLimitedError("HTTP 429", retry_after_s=3600.0)])
    model, sleeps = retrying(inner, RetryPolicy(max_wait_s=30.0))

    with pytest.raises(RateLimitedError):
        await collect(model.stream(HELLO))

    assert (inner.attempts, sleeps.waits) == (1, [])


async def test_a_rate_limit_without_retry_after_uses_the_drawn_wait() -> None:
    inner = Scripted([RateLimitedError("HTTP 429")])
    model, sleeps = retrying(inner)

    await collect(model.stream(HELLO))

    assert 0 <= sleeps.waits[0] <= 0.5


def test_info_is_the_wrapped_models() -> None:
    assert RetryingModel(Scripted([])).info is INFO


@given(
    st.integers(min_value=1, max_value=12), st.integers(min_value=0, max_value=2**32)
)
async def test_every_drawn_wait_stays_inside_its_ceiling(
    failures: int, seed: int
) -> None:
    policy = RetryPolicy(max_attempts=failures + 1, base_s=0.5, cap_s=8.0)
    sleeps = Sleeps()
    inner = Scripted([ProviderError("x")] * failures)
    model = RetryingModel(inner, policy, random.Random(seed), sleeps)

    await collect(model.stream(HELLO))

    assert len(sleeps.waits) == failures
    for attempt, wait in enumerate(sleeps.waits, start=1):
        assert 0 <= wait <= min(8.0, 0.5 * 2 ** (attempt - 1))


def test_the_ceiling_doubles_and_stops_at_the_cap() -> None:
    policy = RetryPolicy(base_s=0.5, cap_s=3.0)

    assert [policy.ceiling(n) for n in range(1, 6)] == [0.5, 1.0, 2.0, 3.0, 3.0]


@pytest.mark.parametrize(
    "kwargs",
    [{"max_attempts": 0}, {"base_s": -1.0}, {"cap_s": -0.1}, {"max_wait_s": -5.0}],
)
def test_nonsense_policies_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(**kwargs)  # type: ignore[arg-type]
