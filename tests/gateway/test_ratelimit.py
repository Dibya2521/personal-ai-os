import asyncio

import pytest
from hypothesis import given
from hypothesis import strategies as st

from synthia.gateway.ratelimit import SlidingWindowLimiter


class FakeTime:
    """A clock that only moves when something sleeps on it."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)


def limiter(
    limit: int = 3, window: float = 60.0
) -> tuple[SlidingWindowLimiter, FakeTime]:
    fake = FakeTime()
    return SlidingWindowLimiter(limit, window, fake.clock, fake.sleep), fake


async def test_the_limit_passes_at_once_and_the_next_waits_a_window() -> None:
    rate, fake = limiter(limit=20)
    for _ in range(20):
        await rate.acquire()
    assert fake.now == 0.0

    await rate.acquire()

    assert fake.now == 60.0
    assert fake.sleeps == [60.0]


async def test_a_token_bucket_would_break_this_but_the_window_does_not() -> None:
    """20 at t=0, more at t=30: a 20/min bucket admits 10 more, this admits none."""
    rate, fake = limiter(limit=20)
    for _ in range(20):
        await rate.acquire()
    fake.now = 30.0

    assert rate.delay() == 30.0
    await rate.acquire()
    assert fake.now == 60.0


async def test_delay_reports_the_wait_without_taking_a_slot() -> None:
    rate, fake = limiter(limit=2, window=10.0)
    assert rate.delay() == 0.0

    await rate.acquire()
    fake.now = 4.0
    await rate.acquire()

    assert rate.delay() == 6.0
    fake.now = 9.0
    assert rate.delay() == 1.0
    fake.now = 10.0  # the first request leaves the window exactly now
    assert rate.delay() == 0.0


async def test_waiters_are_admitted_in_arrival_order() -> None:
    rate, _ = limiter(limit=1, window=5.0)
    order: list[int] = []

    async def request(n: int) -> None:
        await rate.acquire()
        order.append(n)

    await asyncio.gather(*(request(n) for n in range(5)))

    assert order == [0, 1, 2, 3, 4]


async def test_a_cancelled_waiter_takes_no_slot() -> None:
    rate = SlidingWindowLimiter(1, window_s=60.0)
    await rate.acquire()
    waiter = asyncio.create_task(rate.acquire())
    await asyncio.sleep(0.01)

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    # Only the first acquisition holds a slot: the second one waits a full window.
    assert rate.delay() > 59


@given(
    st.integers(min_value=1, max_value=6),
    st.lists(st.floats(min_value=0, max_value=30), min_size=1, max_size=40),
)
def test_no_window_ever_holds_more_than_the_limit(
    limit: int, gaps: list[float]
) -> None:
    window = 10.0

    async def scenario() -> list[float]:
        rate, fake = limiter(limit, window)
        admitted: list[float] = []
        for gap in gaps:
            fake.now += gap
            await rate.acquire()
            admitted.append(fake.now)
        return admitted

    times = asyncio.run(scenario())

    for start in times:
        assert sum(start <= t < start + window for t in times) <= limit


@pytest.mark.parametrize(("limit", "window"), [(0, 60.0), (-1, 60.0), (5, 0.0)])
def test_nonsense_settings_are_rejected(limit: int, window: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        SlidingWindowLimiter(limit, window)
