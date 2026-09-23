"""A circuit breaker: stop calling a provider that keeps failing.

When a provider is down, every request to it waits out its timeouts and
retries before failing. After ``failure_threshold`` failures in a row the
circuit opens, and calls fail at once with :class:`CircuitOpenError` so the
router can answer from another model immediately. After ``cooldown_s`` one
probe is let through (half-open): if it succeeds the circuit closes, if it
fails it opens again for another cooldown.

Only failures that say something about the provider's health count, which are
the retryable ones: server errors, timeouts, dropped connections, rate limits.
A rejected request or a bad key is not an outage, and waiting will not fix it.
"""

from __future__ import annotations

import logging
import time
from contextlib import aclosing
from enum import StrEnum
from typing import TYPE_CHECKING

from synthia.gateway.errors import GatewayError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from synthia.gateway.protocol import ChatModel
    from synthia.gateway.types import ChatChunk, ChatRequest, ModelInfo

logger = logging.getLogger(__name__)


class CircuitState(StrEnum):
    """Whether calls are let through."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(GatewayError):
    """The provider failed repeatedly and is not being called for now."""

    def __init__(self, name: str, retry_in_s: float) -> None:
        super().__init__(f"{name} is unavailable; next attempt in {retry_in_s:.0f} s")
        self.retry_in_s = retry_in_s


class CircuitBreaker:
    """Track one provider's health and decide whether to call it."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        cooldown_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a closed circuit; ``clock`` is injectable for tests.

        Raises:
            ValueError: If the threshold is below 1 or the cooldown is negative.
        """
        if failure_threshold < 1 or cooldown_s < 0:
            message = "failure_threshold must be at least 1 and cooldown_s >= 0"
            raise ValueError(message)
        self.name = name
        self._threshold = failure_threshold
        self._cooldown = cooldown_s
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._probing = False

    @property
    def state(self) -> CircuitState:
        """Return the state; an open circuit turns half-open after the cooldown."""
        if self._opened_at is None:
            return CircuitState.CLOSED
        if self._clock() - self._opened_at >= self._cooldown:
            return CircuitState.HALF_OPEN
        return CircuitState.OPEN

    def before_call(self) -> None:
        """Let a call through or refuse it.

        In the half-open state exactly one probe is let through at a time.

        Raises:
            CircuitOpenError: If the circuit is open, or a probe is already out.
        """
        opened_at = self._opened_at
        if opened_at is None:
            return
        waited = self._clock() - opened_at
        if waited >= self._cooldown and not self._probing:
            self._probing = True
            return
        raise CircuitOpenError(self.name, max(self._cooldown - waited, 0.0))

    def record_success(self) -> None:
        """Close the circuit: the provider answered."""
        if self._opened_at is not None:
            logger.info("circuit closed", extra={"provider": self.name})
        self._failures = 0
        self._opened_at = None
        self._probing = False

    def record_failure(self) -> None:
        """Count a health failure.

        The circuit opens at the threshold or on a failed probe. A failure that
        arrives while it is already open, from a call that was in flight when it
        opened, restarts the cooldown: it is fresh evidence of the outage.
        """
        self._failures += 1
        if self._probing or self._failures >= self._threshold:
            if self._opened_at is None or self._probing:
                logger.warning(
                    "circuit opened",
                    extra={"provider": self.name, "failures": self._failures},
                )
            self._opened_at = self._clock()
        self._probing = False

    def release(self) -> None:
        """Give back a probe that ended without a verdict, such as a cancellation."""
        self._probing = False


class CircuitBreakerModel:
    """A :class:`~synthia.gateway.protocol.ChatModel` guarded by a circuit breaker."""

    def __init__(self, inner: ChatModel, breaker: CircuitBreaker) -> None:
        self._inner = inner
        self._breaker = breaker

    @property
    def info(self) -> ModelInfo:
        """Return the wrapped model's capabilities."""
        return self._inner.info

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        """Yield the completion if the circuit lets the call through.

        Raises:
            CircuitOpenError: If the provider is not being called for now.
            GatewayError: Whatever the wrapped model raised.
        """
        self._breaker.before_call()
        try:
            async with aclosing(self._inner.stream(request)) as chunks:
                async for chunk in chunks:
                    yield chunk
        except GatewayError as error:
            if error.retryable:
                self._breaker.record_failure()
            else:
                self._breaker.release()
            raise
        except BaseException:
            # Cancelled or closed early: the provider was not judged either way.
            self._breaker.release()
            raise
        self._breaker.record_success()
