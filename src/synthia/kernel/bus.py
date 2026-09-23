"""A typed, asynchronous publish-subscribe event bus.

Subsystems never call each other across a boundary; they publish events here and
subscribe to the ones they need. That keeps them independent, and it is what
lets one component observe or interrupt another (a tracer that sees every
event, speech that cancels a reply in progress) without either knowing the other.

Delivery model:

- A subscriber receives every event that is an instance of the type it
  subscribed to, so subscribing to :class:`Event` receives everything.
- Each subscription owns a bounded queue and one consumer task. Events reach one
  subscriber in the order they were published; different subscribers run
  independently, so a slow one never delays another.
- When a queue is full, :attr:`Overflow.BLOCK` makes the publisher wait, and
  :attr:`Overflow.DROP_OLDEST` discards the oldest queued event, which suits
  live streams where a stale item is worthless.
- A handler that raises is logged and counted. The bus and the other
  subscribers carry on.

A handler must not publish to its own subscription with ``BLOCK`` while that
queue can be full: it would wait for itself.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from synthia.kernel.errors import SynthiaError
from synthia.kernel.logs import current_correlation_id

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

DEFAULT_CAPACITY = 256


def _new_event_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True, slots=True, kw_only=True)
class Event:
    """Base class for every event.

    ``correlation_id`` defaults to the one bound where the event is created, so
    an event raised while handling a turn is traceable to that turn.
    """

    event_id: str = field(default_factory=_new_event_id)
    created_at: float = field(default_factory=time.time)
    correlation_id: str | None = field(default_factory=current_correlation_id)


class Overflow(StrEnum):
    """What a full subscription queue does with a new event."""

    BLOCK = "block"
    DROP_OLDEST = "drop_oldest"


class BusClosedError(SynthiaError):
    """An event was published, or a subscription made, after the bus closed."""


@dataclass(slots=True)
class SubscriptionStats:
    """Counters for one subscription."""

    delivered: int = 0
    dropped: int = 0
    failed: int = 0


class Subscription[E: Event]:
    """One subscriber's queue and the task that feeds its handler."""

    def __init__(
        self,
        event_type: type[E],
        handler: Callable[[E], Awaitable[None]],
        *,
        name: str,
        capacity: int,
        overflow: Overflow,
    ) -> None:
        self.event_type = event_type
        self.name = name
        self.overflow = overflow
        self.stats = SubscriptionStats()
        self._handler = handler
        self._queue: asyncio.Queue[E] = asyncio.Queue(maxsize=capacity)
        self._task = asyncio.create_task(self._consume(), name=f"bus:{name}")

    def accepts(self, event: Event) -> bool:
        """Return whether this subscription receives ``event``."""
        return isinstance(event, self.event_type)

    async def offer(self, event: E) -> None:
        """Queue ``event``, waiting or dropping per the overflow policy."""
        if self.overflow is Overflow.BLOCK:
            await self._queue.put(event)
            return
        if self._queue.full():
            self._queue.get_nowait()
            self._queue.task_done()
            self.stats.dropped += 1
        self._queue.put_nowait(event)

    async def close(self, *, drain: bool) -> None:
        """Stop the consumer, first delivering what is queued if ``drain``."""
        if drain:
            await self._queue.join()
        self._task.cancel()
        # wait() neither re-raises the consumer's cancellation nor swallows our own.
        await asyncio.wait([self._task])

    async def _consume(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                await self._handler(event)
                self.stats.delivered += 1
            except Exception:
                self.stats.failed += 1
                logger.exception(
                    "event handler failed",
                    extra={"subscription": self.name, "event": type(event).__name__},
                )
            finally:
                self._queue.task_done()


class EventBus:
    """Route published events to every subscription that accepts them.

    Must be used from inside a running event loop.
    """

    def __init__(self, *, default_capacity: int = DEFAULT_CAPACITY) -> None:
        self._default_capacity = default_capacity
        self._subscriptions: list[Subscription[Event]] = []
        self._closed = False

    def subscribe[E: Event](
        self,
        event_type: type[E],
        handler: Callable[[E], Awaitable[None]],
        *,
        name: str | None = None,
        capacity: int | None = None,
        overflow: Overflow = Overflow.BLOCK,
    ) -> Subscription[E]:
        """Deliver every event of ``event_type``, or a subclass, to ``handler``.

        Raises:
            BusClosedError: If the bus has been closed.
        """
        if self._closed:
            message = "cannot subscribe to a closed bus"
            raise BusClosedError(message)
        subscription = Subscription(
            event_type,
            handler,
            name=name or getattr(handler, "__qualname__", repr(handler)),
            capacity=capacity or self._default_capacity,
            overflow=overflow,
        )
        # Invariant in E: the list holds subscriptions of different event types.
        self._subscriptions.append(subscription)  # pyright: ignore[reportArgumentType]
        return subscription

    async def unsubscribe[E: Event](
        self, subscription: Subscription[E], *, drain: bool
    ) -> None:
        """Stop delivering to ``subscription``."""
        self._subscriptions.remove(subscription)  # pyright: ignore[reportArgumentType]
        await subscription.close(drain=drain)

    async def publish(self, event: Event) -> None:
        """Offer ``event`` to every subscription that accepts it.

        Raises:
            BusClosedError: If the bus has been closed.
        """
        if self._closed:
            message = f"cannot publish {type(event).__name__} to a closed bus"
            raise BusClosedError(message)
        for subscription in list(self._subscriptions):
            if subscription.accepts(event):
                await subscription.offer(event)

    async def close(self, *, drain: bool = True) -> None:
        """Refuse new events, then stop every subscription.

        With ``drain``, what was queued before the call is still delivered. An
        event a handler publishes during the drain is refused like any other,
        so a handler that feeds itself cannot keep the bus open forever.
        """
        self._closed = True
        subscriptions, self._subscriptions = self._subscriptions, []
        await asyncio.gather(*(s.close(drain=drain) for s in subscriptions))
