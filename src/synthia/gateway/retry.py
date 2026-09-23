"""Retry transient model failures, with full-jitter exponential backoff.

Only an error whose ``retryable`` flag is set is retried, and never once part of
the answer has streamed: sending again then would repeat text already shown, or
already spoken. The wait before retry ``n`` is drawn uniformly from
``[0, min(cap, base * 2 ** (n - 1))]`` ("full jitter", from AWS's analysis of
backoff strategies), so many clients failing together do not retry in lockstep.

A provider's ``Retry-After`` is honoured instead of the drawn wait. If it asks
for longer than ``max_wait_s``, the error is raised at once: waiting that long
in a conversation is worse than letting the caller route elsewhere.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from synthia.gateway.errors import GatewayError, RateLimitedError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from synthia.gateway.protocol import ChatModel
    from synthia.gateway.types import ChatChunk, ChatRequest, ModelInfo

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How many times, and how patiently, a request is sent again.

    Raises:
        ValueError: If a value is out of range.
    """

    max_attempts: int = 3
    base_s: float = 0.5
    cap_s: float = 8.0
    max_wait_s: float = 30.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1 or min(self.base_s, self.cap_s, self.max_wait_s) < 0:
            message = "max_attempts must be at least 1 and every wait non-negative"
            raise ValueError(message)

    def ceiling(self, attempt: int) -> float:
        """Return the longest drawn wait after failed attempt number ``attempt``."""
        return min(self.cap_s, self.base_s * 2 ** (attempt - 1))


class RetryingModel:
    """A :class:`~synthia.gateway.protocol.ChatModel` that retries another one."""

    def __init__(
        self,
        inner: ChatModel,
        policy: RetryPolicy | None = None,
        rng: random.Random | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._inner = inner
        self._policy = policy or RetryPolicy()
        self._rng = rng or random.Random()  # noqa: S311 - jitter, not cryptography
        self._sleep = sleep

    @property
    def info(self) -> ModelInfo:
        """Return the wrapped model's capabilities."""
        return self._inner.info

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        """Yield the completion, sending the request again after a transient failure.

        Raises:
            GatewayError: The last failure, if it was not retryable, came after
                output had started, asked for too long a wait, or was the final
                attempt.
        """
        attempt = 1
        while True:
            started = False
            try:
                async for chunk in self._inner.stream(request):
                    started = True
                    yield chunk
            except GatewayError as error:
                wait = self._wait_before_retry(error, attempt, started=started)
                if wait is None:
                    raise
                logger.warning(
                    "retrying after a transient failure",
                    extra={
                        "attempt": attempt,
                        "wait_s": round(wait, 3),
                        "error": type(error).__name__,
                    },
                )
                await self._sleep(wait)
                attempt += 1
            else:
                return

    def _wait_before_retry(
        self, error: GatewayError, attempt: int, *, started: bool
    ) -> float | None:
        """Return how long to wait before the next attempt, or ``None`` to give up."""
        if started or not error.retryable or attempt >= self._policy.max_attempts:
            return None
        if isinstance(error, RateLimitedError) and error.retry_after_s is not None:
            asked = error.retry_after_s
            return asked if asked <= self._policy.max_wait_s else None
        return self._rng.uniform(0, self._policy.ceiling(attempt))
