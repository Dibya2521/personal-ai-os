import asyncio
import logging

import pytest

from synthia.kernel.bus import Event, EventBus
from synthia.kernel.supervisor import (
    RestartMode,
    RestartPolicy,
    ServiceExited,
    ServiceStarted,
    ServiceStopped,
    Supervisor,
    SupervisorGaveUpError,
)

FAST = RestartPolicy(
    max_restarts=3, window_s=60, backoff_initial_s=0.001, backoff_max_s=0.01
)


class Flaky:
    """Crash ``failures`` times, then run until stopped."""

    def __init__(self, name: str, failures: int) -> None:
        self.name = name
        self.failures = failures
        self.starts = 0
        self.healthy = asyncio.Event()

    async def run(self, stop: asyncio.Event) -> None:
        self.starts += 1
        if self.starts <= self.failures:
            message = f"crash {self.starts}"
            raise RuntimeError(message)
        self.healthy.set()
        await stop.wait()


class Returns:
    """Return cleanly ``times`` times, then run until stopped."""

    def __init__(self, name: str, times: int = 1_000_000) -> None:
        self.name = name
        self.times = times
        self.starts = 0

    async def run(self, stop: asyncio.Event) -> None:
        self.starts += 1
        if self.starts > self.times:
            await stop.wait()


class Stubborn:
    name = "stubborn"

    def __init__(self) -> None:
        self.cancelled = False

    async def run(self, stop: asyncio.Event) -> None:  # noqa: ARG002 - ignores stop on purpose
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


async def stop_when(supervisor: Supervisor, condition: asyncio.Event) -> None:
    await asyncio.wait_for(condition.wait(), timeout=2)
    supervisor.stop()


async def test_a_crashing_service_restarts_without_touching_the_others() -> None:
    supervisor = Supervisor(default_policy=FAST)
    flaky, steady = Flaky("flaky", failures=2), Flaky("steady", failures=0)
    supervisor.add(flaky)
    supervisor.add(steady)

    await asyncio.gather(supervisor.run(), stop_when(supervisor, flaky.healthy))

    assert flaky.starts == 3
    assert steady.starts == 1


@pytest.mark.parametrize(
    ("mode", "expected_starts"),
    [
        (RestartMode.TRANSIENT, 1),
        (RestartMode.TEMPORARY, 1),
        (RestartMode.PERMANENT, 3),
    ],
)
async def test_a_clean_return_restarts_only_a_permanent_service(
    mode: RestartMode, expected_starts: int
) -> None:
    supervisor = Supervisor(default_policy=FAST)
    service = Returns("returns", times=2)
    supervisor.add(service, mode)

    run = asyncio.create_task(supervisor.run())
    await asyncio.sleep(0.1)
    supervisor.stop()
    await asyncio.wait_for(run, timeout=2)

    assert service.starts == expected_starts


async def test_a_temporary_service_is_not_restarted_after_a_crash() -> None:
    supervisor = Supervisor(default_policy=FAST)
    service = Flaky("once", failures=5)
    supervisor.add(service, RestartMode.TEMPORARY)

    await asyncio.wait_for(supervisor.run(), timeout=2)

    assert service.starts == 1


async def test_exceeding_the_intensity_stops_everything_and_escalates() -> None:
    supervisor = Supervisor(default_policy=FAST)
    doomed, bystander = Flaky("doomed", failures=100), Flaky("bystander", failures=0)
    supervisor.add(doomed)
    supervisor.add(bystander)

    with pytest.raises(SupervisorGaveUpError) as caught:
        await asyncio.wait_for(supervisor.run(), timeout=2)

    assert caught.value.service == "doomed"
    assert doomed.starts == FAST.max_restarts + 1
    assert bystander.healthy.is_set()


async def test_restarts_outside_the_window_are_forgotten() -> None:
    now = [0.0]
    policy = RestartPolicy(
        max_restarts=2, window_s=10, backoff_initial_s=0.001, backoff_max_s=0.001
    )
    supervisor = Supervisor(default_policy=policy, clock=lambda: now[0])

    class SlowFlaky(Flaky):
        async def run(self, stop: asyncio.Event) -> None:
            now[0] += 6  # every crash lands 6 s after the previous one
            await super().run(stop)

    service = SlowFlaky("slow", failures=6)
    supervisor.add(service)

    await asyncio.gather(supervisor.run(), stop_when(supervisor, service.healthy))

    assert service.starts == 7


def test_backoff_doubles_and_is_capped() -> None:
    policy = RestartPolicy(backoff_initial_s=0.5, backoff_factor=2, backoff_max_s=3)

    assert [policy.delay(n) for n in range(6)] == [0.5, 0.5, 1, 2, 3, 3]


async def test_stop_interrupts_a_long_backoff() -> None:
    policy = RestartPolicy(backoff_initial_s=60, backoff_max_s=60)
    supervisor = Supervisor(default_policy=policy)
    service = Flaky("sleepy", failures=100)
    supervisor.add(service)

    run = asyncio.create_task(supervisor.run())
    await asyncio.sleep(0.05)
    supervisor.stop()
    await asyncio.wait_for(run, timeout=1)

    assert service.starts == 1


async def test_a_service_that_ignores_stop_is_cancelled_after_the_timeout(
    caplog: pytest.LogCaptureFixture,
) -> None:
    supervisor = Supervisor(stop_timeout_s=0.05)
    stubborn = Stubborn()
    supervisor.add(stubborn)

    run = asyncio.create_task(supervisor.run())
    await asyncio.sleep(0.01)
    with caplog.at_level(logging.WARNING):
        supervisor.stop()
        await asyncio.wait_for(run, timeout=1)

    assert stubborn.cancelled
    assert "ignored stop" in caplog.text


async def test_cancelling_run_cancels_every_service() -> None:
    supervisor = Supervisor()
    stubborn = Stubborn()
    supervisor.add(stubborn)

    run = asyncio.create_task(supervisor.run())
    await asyncio.sleep(0.01)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert stubborn.cancelled


async def test_lifecycle_events_tell_the_whole_story() -> None:
    bus = EventBus()
    events: list[Event] = []

    async def record(event: Event) -> None:
        events.append(event)

    bus.subscribe(Event, record)
    supervisor = Supervisor(bus=bus, default_policy=FAST)
    service = Flaky("story", failures=1)
    supervisor.add(service)

    await asyncio.gather(supervisor.run(), stop_when(supervisor, service.healthy))
    await bus.close()

    shape = [
        (
            type(e).__name__,
            getattr(e, "error", None) is not None,
            getattr(e, "restart_in_s", None) is not None,
        )
        for e in events
    ]
    assert shape == [
        ("ServiceStarted", False, False),
        ("ServiceExited", True, True),
        ("ServiceStarted", False, False),
        ("ServiceExited", False, False),
        ("ServiceStopped", False, False),
    ]
    assert all(
        e.service == "story"
        for e in events
        if isinstance(e, ServiceStarted | ServiceExited | ServiceStopped)
    )


def test_two_services_cannot_share_a_name() -> None:
    supervisor = Supervisor()
    supervisor.add(Returns("same"))

    with pytest.raises(ValueError, match="already supervised"):
        supervisor.add(Returns("same"))


async def test_run_with_no_services_returns() -> None:
    await asyncio.wait_for(Supervisor().run(), timeout=1)
