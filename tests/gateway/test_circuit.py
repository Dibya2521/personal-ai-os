import asyncio
from collections.abc import AsyncIterator

import pytest

from synthia.gateway.circuit import (
    CircuitBreaker,
    CircuitBreakerModel,
    CircuitOpenError,
    CircuitState,
)
from synthia.gateway.errors import BadRequestError, GatewayError, ProviderError
from synthia.gateway.protocol import collect
from synthia.gateway.types import (
    ChatChunk,
    ChatRequest,
    FinishReason,
    Message,
    ModelInfo,
)

INFO = ModelInfo("flaky", 1000, vision=False, tools=False)
HELLO = ChatRequest((Message.user("hello"),))


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class Provider:
    """Answer, or raise the next scripted error."""

    def __init__(self) -> None:
        self.errors: list[GatewayError] = []
        self.calls = 0
        self.hang = False

    @property
    def info(self) -> ModelInfo:
        return INFO

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        del request
        self.calls += 1
        if self.hang:
            await asyncio.Event().wait()
        if self.errors:
            raise self.errors.pop(0)
        yield ChatChunk(text="ok")
        yield ChatChunk(finish_reason=FinishReason.STOP)


def guarded(
    threshold: int = 3, cooldown: float = 30.0
) -> tuple[CircuitBreakerModel, CircuitBreaker, Provider, Clock]:
    clock, provider = Clock(), Provider()
    breaker = CircuitBreaker("openrouter", threshold, cooldown, clock)
    return CircuitBreakerModel(provider, breaker), breaker, provider, clock


async def fail_times(model: CircuitBreakerModel, provider: Provider, n: int) -> None:
    for _ in range(n):
        provider.errors.append(ProviderError("HTTP 503"))
        with pytest.raises(ProviderError):
            await collect(model.stream(HELLO))


async def test_the_circuit_opens_after_the_threshold_and_then_fails_fast() -> None:
    model, breaker, provider, _ = guarded(threshold=3)
    await fail_times(model, provider, 3)

    with pytest.raises(
        CircuitOpenError, match="openrouter is unavailable; next attempt in 30 s"
    ):
        await collect(model.stream(HELLO))

    assert breaker.state is CircuitState.OPEN
    assert provider.calls == 3


async def test_a_success_resets_the_consecutive_count() -> None:
    model, breaker, provider, _ = guarded(threshold=3)
    await fail_times(model, provider, 2)
    await collect(model.stream(HELLO))
    await fail_times(model, provider, 2)

    assert breaker.state is CircuitState.CLOSED


async def test_after_the_cooldown_one_successful_probe_closes_it() -> None:
    model, breaker, provider, clock = guarded(threshold=1, cooldown=30.0)
    await fail_times(model, provider, 1)

    clock.now += 29.9
    assert breaker.state is CircuitState.OPEN
    clock.now += 0.1
    assert breaker.state is CircuitState.HALF_OPEN

    assert (await collect(model.stream(HELLO))).text == "ok"
    assert breaker.state is CircuitState.CLOSED


async def test_a_failed_probe_opens_it_for_a_fresh_cooldown() -> None:
    model, breaker, provider, clock = guarded(threshold=3, cooldown=30.0)
    await fail_times(model, provider, 3)
    clock.now += 30.0

    await fail_times(model, provider, 1)

    assert breaker.state is CircuitState.OPEN
    clock.now += 29.0
    assert breaker.state is CircuitState.OPEN
    clock.now += 1.0
    assert breaker.state is CircuitState.HALF_OPEN


async def test_only_one_probe_is_out_at_a_time() -> None:
    model, breaker, provider, clock = guarded(threshold=1)
    await fail_times(model, provider, 1)
    clock.now += 30.0
    provider.hang = True

    probe = asyncio.create_task(collect(model.stream(HELLO)))
    await asyncio.sleep(0)
    with pytest.raises(CircuitOpenError):
        await collect(model.stream(HELLO))

    probe.cancel()
    with pytest.raises(asyncio.CancelledError):
        await probe
    # The cancelled probe gave its slot back, so another probe may go.
    provider.hang = False
    assert (await collect(model.stream(HELLO))).text == "ok"
    assert breaker.state is CircuitState.CLOSED


async def test_a_rejected_request_is_not_an_outage() -> None:
    model, breaker, provider, _ = guarded(threshold=1)
    provider.errors.append(BadRequestError("HTTP 400"))

    with pytest.raises(BadRequestError):
        await collect(model.stream(HELLO))

    assert breaker.state is CircuitState.CLOSED


def test_nonsense_settings_are_rejected() -> None:
    with pytest.raises(ValueError, match="failure_threshold"):
        CircuitBreaker("x", failure_threshold=0)
    with pytest.raises(ValueError, match="cooldown"):
        CircuitBreaker("x", cooldown_s=-1)


def test_info_is_the_wrapped_models() -> None:
    model, *_ = guarded()

    assert model.info is INFO


def test_a_late_failure_from_a_call_already_in_flight_restarts_the_cooldown() -> None:
    clock = Clock()
    breaker = CircuitBreaker(
        "openrouter", failure_threshold=1, cooldown_s=30.0, clock=clock
    )
    breaker.before_call()
    breaker.before_call()

    breaker.record_failure()
    clock.now += 10.0
    breaker.record_failure()

    clock.now += 25.0  # 35 s after opening, 25 s after the late failure
    assert breaker.state is CircuitState.OPEN
    clock.now += 5.0
    assert breaker.state is CircuitState.HALF_OPEN


def test_a_closed_circuit_has_nothing_to_wait_for() -> None:
    breaker = CircuitBreaker("x")

    breaker.before_call()
    breaker.record_success()

    assert breaker.state is CircuitState.CLOSED
