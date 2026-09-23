"""A rate limiter that never exceeds N requests in any window of W seconds.

A token bucket is the usual choice, but it is not safe against an unknown
provider rule: a bucket of 20 per minute allows 20 at once and then one every
three seconds, which is 30 requests inside the first sixty seconds. If the
provider counts over a sliding window, that is rate limited. This limiter keeps
the time of each recent request and admits one only while fewer than the limit
fall inside the last window, which satisfies a fixed-window rule and a sliding
one alike.

Waiters are served in arrival order, and one cancelled while waiting takes no
slot.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

DEFAULT_WINDOW_S = 60.0


class SlidingWindowLimiter:
    """Admit at most ``limit`` acquisitions in any ``window_s`` seconds."""

    def __init__(
        self,
        limit: int,
        window_s: float = DEFAULT_WINDOW_S,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Create a limiter; ``clock`` and ``sleep`` are injectable for tests.

        Raises:
            ValueError: If ``limit`` or ``window_s`` is not positive.
        """
        if limit <= 0 or window_s <= 0:
            message = "limit and window must be positive"
            raise ValueError(message)
        self._limit = limit
        self._window = window_s
        self._clock = clock
        self._sleep = sleep
        self._admitted: deque[float] = deque()
        self._lock = asyncio.Lock()

    def delay(self) -> float:
        """Return how long an acquisition made now would wait, in seconds.

        Lets a caller choose another route instead of waiting.
        """
        now = self._clock()
        self._forget(now)
        if len(self._admitted) < self._limit:
            return 0.0
        return self._admitted[0] + self._window - now

    async def acquire(self) -> None:
        """Wait until a request may be sent, then count it."""
        # asyncio.Lock wakes waiters in the order they arrived, which keeps
        # admission first come, first served.
        async with self._lock:
            while (wait := self.delay()) > 0:
                await self._sleep(wait)
            self._admitted.append(self._clock())

    def _forget(self, now: float) -> None:
        # Written as t + window, the same form delay() uses: in floating point,
        # t <= now - window and t + window <= now can disagree in the last bit.
        while self._admitted and self._admitted[0] + self._window <= now:
            self._admitted.popleft()
