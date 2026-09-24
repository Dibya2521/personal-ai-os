"""Pass the rate limiter and claim the daily budget before each remote send.

The order is deliberate. The limiter wait comes first because it can be
cancelled at no cost; the budget claim comes last, right before the request
leaves, because a claim is never given back: the provider counts a request
once it arrives, answered or not.

Wrap the adapter directly, inside any retry, so that every attempt is metered
the way the provider meters it.
"""

from __future__ import annotations

from contextlib import aclosing
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from synthia.gateway.budget import BudgetLedger
    from synthia.gateway.protocol import ChatModel
    from synthia.gateway.ratelimit import SlidingWindowLimiter
    from synthia.gateway.types import ChatChunk, ChatRequest, ModelInfo


class MeteredModel:
    """A :class:`~synthia.gateway.protocol.ChatModel` that meters another one."""

    def __init__(
        self,
        inner: ChatModel,
        provider: str,
        ledger: BudgetLedger,
        limiter: SlidingWindowLimiter,
    ) -> None:
        self._inner = inner
        self._provider = provider
        self._ledger = ledger
        self._limiter = limiter

    @property
    def info(self) -> ModelInfo:
        """Return the wrapped model's capabilities."""
        return self._inner.info

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        """Wait for the rate limit, claim one request, then yield the answer.

        Raises:
            BudgetExhaustedError: If today's budget is used; nothing is sent.
            GatewayError: Whatever the wrapped model raised.
        """
        await self._limiter.acquire()
        await self._ledger.claim(self._provider)
        async with aclosing(self._inner.stream(request)) as chunks:
            async for chunk in chunks:
                yield chunk
