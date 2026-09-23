import asyncio
import logging
from dataclasses import dataclass

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from synthia.kernel.bus import BusClosedError, Event, EventBus, Overflow
from synthia.kernel.logs import correlation


@dataclass(frozen=True, slots=True, kw_only=True)
class Heard(Event):
    text: str


@dataclass(frozen=True, slots=True, kw_only=True)
class HeardLoudly(Heard):
    volume: int = 11


@dataclass(frozen=True, slots=True, kw_only=True)
class Seen(Event):
    label: str


class Recorder:
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def __call__(self, event: Event) -> None:
        self.events.append(event)


async def test_an_event_reaches_subscribers_of_its_type_and_its_bases() -> None:
    bus = EventBus()
    heard, everything, seen = Recorder(), Recorder(), Recorder()
    bus.subscribe(Heard, heard)
    bus.subscribe(Event, everything)
    bus.subscribe(Seen, seen)

    await bus.publish(HeardLoudly(text="hi"))
    await bus.publish(Seen(label="cup"))
    await bus.close()

    assert [type(e) for e in heard.events] == [HeardLoudly]
    assert [type(e) for e in everything.events] == [HeardLoudly, Seen]
    assert [type(e) for e in seen.events] == [Seen]


@settings(deadline=None, max_examples=25)
@given(st.lists(st.text(max_size=5), max_size=40))
def test_one_subscriber_sees_events_in_publish_order(texts: list[str]) -> None:
    async def scenario() -> list[str]:
        bus = EventBus(default_capacity=4)
        recorder = Recorder()
        bus.subscribe(Heard, recorder)
        for text in texts:
            await bus.publish(Heard(text=text))
        await bus.close()
        return [e.text for e in recorder.events if isinstance(e, Heard)]

    assert asyncio.run(scenario()) == texts


async def test_a_failing_handler_is_isolated_and_counted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bus = EventBus()
    healthy = Recorder()

    async def broken(_: Event) -> None:
        message = "handler bug"
        raise RuntimeError(message)

    failing = bus.subscribe(Heard, broken, name="broken")
    bus.subscribe(Heard, healthy)

    with caplog.at_level(logging.ERROR):
        await bus.publish(Heard(text="one"))
        await bus.publish(Heard(text="two"))
        await bus.close()

    assert failing.stats.failed == 2
    assert failing.stats.delivered == 0
    assert len(healthy.events) == 2
    assert "event handler failed" in caplog.text


async def test_block_makes_the_publisher_wait_for_a_slow_subscriber() -> None:
    bus = EventBus()
    release = asyncio.Event()
    handled: list[str] = []

    async def slow(event: Heard) -> None:
        await release.wait()
        handled.append(event.text)

    bus.subscribe(Heard, slow, capacity=1)
    await bus.publish(Heard(text="a"))  # taken by the consumer
    await asyncio.sleep(0)
    await bus.publish(Heard(text="b"))  # fills the queue
    blocked = asyncio.create_task(bus.publish(Heard(text="c")))
    await asyncio.sleep(0.01)

    assert not blocked.done()
    release.set()
    await blocked
    await bus.close()
    assert handled == ["a", "b", "c"]


async def test_drop_oldest_keeps_the_newest_and_never_blocks() -> None:
    bus = EventBus()
    release = asyncio.Event()
    handled: list[str] = []

    async def slow(event: Heard) -> None:
        await release.wait()
        handled.append(event.text)

    subscription = bus.subscribe(Heard, slow, capacity=2, overflow=Overflow.DROP_OLDEST)
    await bus.publish(Heard(text="first"))
    await asyncio.sleep(0)  # the consumer holds "first"
    for text in ["a", "b", "c", "d"]:
        await asyncio.wait_for(bus.publish(Heard(text=text)), timeout=1)

    release.set()
    await bus.close()
    assert handled == ["first", "c", "d"]
    assert subscription.stats.dropped == 2


async def test_a_handler_may_publish_into_its_own_dropping_queue() -> None:
    bus = EventBus()
    seen: list[int] = []
    finished = asyncio.Event()

    async def echo(event: Heard) -> None:
        seen.append(len(event.text))
        if len(event.text) < 5:
            await bus.publish(Heard(text=event.text + "x"))
        else:
            finished.set()

    bus.subscribe(Heard, echo, capacity=1, overflow=Overflow.DROP_OLDEST)
    await bus.publish(Heard(text=""))
    await asyncio.wait_for(finished.wait(), timeout=1)
    await bus.close()

    assert seen == [0, 1, 2, 3, 4, 5]


async def test_close_refuses_events_published_while_draining(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bus = EventBus()
    seen: list[str] = []

    async def feeds_itself(event: Heard) -> None:
        seen.append(event.text)
        await bus.publish(Heard(text=event.text + "x"))

    subscription = bus.subscribe(Heard, feeds_itself)
    await bus.publish(Heard(text="a"))
    with caplog.at_level(logging.ERROR):
        await asyncio.wait_for(bus.close(), timeout=1)

    assert seen == ["a"]
    assert subscription.stats.failed == 1


async def test_close_without_drain_discards_what_is_queued() -> None:
    bus = EventBus()
    never = asyncio.Event()
    started: list[str] = []

    async def stuck(event: Heard) -> None:
        started.append(event.text)
        await never.wait()

    bus.subscribe(Heard, stuck)
    await bus.publish(Heard(text="a"))
    await bus.publish(Heard(text="b"))
    await asyncio.sleep(0)
    await asyncio.wait_for(bus.close(drain=False), timeout=1)

    assert started == ["a"]


async def test_nothing_is_accepted_after_close() -> None:
    bus = EventBus()
    await bus.close()

    with pytest.raises(BusClosedError):
        await bus.publish(Heard(text="late"))
    with pytest.raises(BusClosedError):
        bus.subscribe(Heard, Recorder())


async def test_unsubscribe_stops_delivery() -> None:
    bus = EventBus()
    recorder = Recorder()
    subscription = bus.subscribe(Heard, recorder)

    await bus.publish(Heard(text="kept"))
    await bus.unsubscribe(subscription, drain=True)
    await bus.publish(Heard(text="missed"))
    await bus.close()

    assert [e.text for e in recorder.events if isinstance(e, Heard)] == ["kept"]


async def test_an_event_carries_the_correlation_id_where_it_was_made() -> None:
    with correlation("turn-7"):
        inside = Heard(text="x")
    outside = Heard(text="y")

    assert inside.correlation_id == "turn-7"
    assert outside.correlation_id is None
    assert inside.event_id != outside.event_id


async def test_subscription_name_defaults_to_the_handler_name() -> None:
    bus = EventBus()

    async def on_heard(_: Heard) -> None:
        return None

    assert bus.subscribe(Heard, on_heard).name.endswith("on_heard")
    assert bus.subscribe(Heard, Recorder()).name.startswith("<")
    await bus.close()
