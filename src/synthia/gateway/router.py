"""Choose, for each request, between the remote model and the local one.

Remote comes first, because the free remote models are far stronger than a
small local one. A request goes local instead when the first of these holds:

1. it is a background job, since the daily remote budget is kept for talking;
2. it needs images or tools the remote model cannot take;
3. the remote circuit is open, so the answer starts at once instead of after
   a timeout;
4. the remote budget is down to the reserve, which is kept for requests only
   the remote can serve;
5. the remote rate limit would hold the request longer than ``max_wait_s``.

Each rule sends a request local only if a local model exists, is ready, and can
serve it: a local server still loading its model is treated as absent.
With none, the remote is used while it has any budget at all.

If the remote fails before its first chunk, the request is sent local once.
After output has started nothing can switch, since those words are already on
screen or spoken. Every decision is published as a :class:`RouteDecided`.

A request asking for ``auto`` reasoning gets its level here, after the model
is chosen, so every caller has it decided the same way.
"""

from __future__ import annotations

import logging
from contextlib import aclosing
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from synthia.gateway.circuit import CircuitBreakerModel, CircuitState
from synthia.gateway.errors import GatewayError
from synthia.gateway.metered import MeteredModel
from synthia.gateway.reasoning import resolve
from synthia.gateway.retry import RetryingModel
from synthia.gateway.types import ModelInfo, Reasoning
from synthia.kernel.bus import Event

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from synthia.gateway.budget import BudgetLedger
    from synthia.gateway.circuit import CircuitBreaker
    from synthia.gateway.protocol import ChatModel
    from synthia.gateway.ratelimit import SlidingWindowLimiter
    from synthia.gateway.retry import RetryPolicy
    from synthia.gateway.types import ChatChunk, ChatRequest

logger = logging.getLogger(__name__)

DEFAULT_MAX_WAIT_S = 5.0
ROUTER_ID = "auto"


class Route(StrEnum):
    """Where a request is sent."""

    REMOTE = "remote"
    LOCAL = "local"


class RouteReason(StrEnum):
    """Why a request was sent where it was."""

    PREFERRED = "remote first"
    BACKGROUND = "background job"
    CAPABILITY = "remote cannot serve this request"
    CIRCUIT_OPEN = "remote circuit open"
    LOW_BUDGET = "remote budget at the reserve"
    RATE_LIMITED = "remote rate limit would wait too long"
    NO_LOCAL = "local preferred, but no local model can serve this request"
    FALLBACK = "remote failed before answering"


@dataclass(frozen=True, slots=True, kw_only=True)
class RouteDecided(Event):
    """A request was routed; ``model`` is the id of the model chosen.

    ``reasoning`` is the level sent, never ``AUTO``; ``None`` when the request
    set none.
    """

    route: Route
    reason: RouteReason
    model: str
    reasoning: Reasoning | None = None


@dataclass(frozen=True, slots=True)
class RemoteHealth:
    """The remote provider's guards, read by the router to judge it.

    Raises:
        ValueError: If ``reserve`` or ``max_wait_s`` is negative.
    """

    provider: str
    ledger: BudgetLedger
    limiter: SlidingWindowLimiter
    breaker: CircuitBreaker
    reserve: int
    max_wait_s: float = DEFAULT_MAX_WAIT_S

    def __post_init__(self) -> None:
        if self.reserve < 0 or self.max_wait_s < 0:
            message = "reserve and max_wait_s cannot be negative"
            raise ValueError(message)


def guard_remote(
    model: ChatModel, health: RemoteHealth, policy: RetryPolicy | None = None
) -> ChatModel:
    """Wrap a remote adapter in retry, circuit breaker and metering, in that order.

    Metering is innermost so every attempt is counted, and the circuit sits
    inside the retry so every failed attempt is evidence against the provider.
    """
    metered = MeteredModel(model, health.provider, health.ledger, health.limiter)
    return RetryingModel(CircuitBreakerModel(metered, health.breaker), policy)


def can_serve(info: ModelInfo, request: ChatRequest) -> bool:
    """Return whether a model with ``info`` can take ``request`` at all."""
    return (info.vision or not request.has_images) and (info.tools or not request.tools)


async def _discard(_: Event) -> None:
    return None


def _always() -> bool:
    return True


class Router:
    """A :class:`~synthia.gateway.protocol.ChatModel` that picks a model per request."""

    def __init__(
        self,
        remote: ChatModel,
        health: RemoteHealth,
        local: ChatModel | None = None,
        publish: Callable[[Event], Awaitable[None]] = _discard,
        local_ready: Callable[[], bool] = _always,
    ) -> None:
        """Route between ``remote`` (already guarded) and an optional ``local``."""
        self._remote = remote
        self._health = health
        self._local = local
        self._local_ready = local_ready
        self._publish = publish

    @property
    def info(self) -> ModelInfo:
        """Return what a request sent here may use: the union of both models."""
        models = [self._remote.info] + ([self._local.info] if self._local else [])
        return ModelInfo(
            ROUTER_ID,
            min(m.context_window for m in models),
            vision=any(m.vision for m in models),
            tools=any(m.tools for m in models),
        )

    async def decide(
        self, request: ChatRequest, *, background: bool = False
    ) -> tuple[Route, RouteReason]:
        """Return where ``request`` would be sent now, and why."""
        route, reason, _ = await self._choose(request, background=background)
        return route, reason

    async def stream(
        self, request: ChatRequest, *, background: bool = False
    ) -> AsyncGenerator[ChatChunk]:
        """Yield the answer from the model the rules choose.

        Raises:
            GatewayError: What the chosen model raised, when there was no
                fallback: output had started, or no local model could serve.
        """
        route, reason, model = await self._choose(request, background=background)
        request = resolve(request)
        await self._announce(route, reason, model, request)
        if route is Route.LOCAL:
            async with aclosing(model.stream(request)) as chunks:
                async for chunk in chunks:
                    yield chunk
            return
        started = False
        try:
            async with aclosing(self._remote.stream(request)) as chunks:
                async for chunk in chunks:
                    started = True
                    yield chunk
        except GatewayError as error:
            local = self._local_for(request)
            if started or local is None:
                raise
            logger.warning(
                "remote failed before answering; sending local",
                extra={"error": type(error).__name__},
            )
        else:
            return
        await self._announce(Route.LOCAL, RouteReason.FALLBACK, local, request)
        async with aclosing(local.stream(request)) as chunks:
            async for chunk in chunks:
                yield chunk

    async def _choose(
        self, request: ChatRequest, *, background: bool
    ) -> tuple[Route, RouteReason, ChatModel]:
        reason = await self._reason_for_local(request, background=background)
        if reason is None:
            return Route.REMOTE, RouteReason.PREFERRED, self._remote
        local = self._local_for(request)
        if local is not None:
            return Route.LOCAL, reason, local
        return Route.REMOTE, RouteReason.NO_LOCAL, self._remote

    def _local_for(self, request: ChatRequest) -> ChatModel | None:
        local = self._local
        if (
            local is None
            or not self._local_ready()
            or not can_serve(local.info, request)
        ):
            return None
        return local

    async def _reason_for_local(
        self, request: ChatRequest, *, background: bool
    ) -> RouteReason | None:
        health = self._health
        if background:
            return RouteReason.BACKGROUND
        if not can_serve(self._remote.info, request):
            return RouteReason.CAPABILITY
        if health.breaker.state is CircuitState.OPEN:
            return RouteReason.CIRCUIT_OPEN
        status = await health.ledger.status(health.provider)
        if status.remaining <= health.reserve:
            return RouteReason.LOW_BUDGET
        if health.limiter.delay() > health.max_wait_s:
            return RouteReason.RATE_LIMITED
        return None

    async def _announce(
        self, route: Route, reason: RouteReason, model: ChatModel, request: ChatRequest
    ) -> None:
        decided = RouteDecided(
            route=route,
            reason=reason,
            model=model.info.id,
            reasoning=request.reasoning,
        )
        logger.info(
            "routed",
            extra={
                "route": route,
                "reason": reason,
                "model": decided.model,
                "reasoning": decided.reasoning,
            },
        )
        await self._publish(decided)
