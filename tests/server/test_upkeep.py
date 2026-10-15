import asyncio
import logging
import sqlite3
from pathlib import Path

import pytest

from synthia.knowledge.index import IndexReport, Pause
from synthia.server.host import in_background
from synthia.server.upkeep import CANCELLED, Activity, Indexed, Reindexing
from tests.knowledge.corpus import NOTES, opened, planted
from tests.timing import HANG_TIMEOUT_S


class Clock:
    """A clock the test moves by hand, which says each time it is read."""

    def __init__(self) -> None:
        self.now = 0.0
        self.reads = 0
        self._read = asyncio.Event()

    def __call__(self) -> float:
        self.reads += 1
        self._read.set()
        return self.now

    async def read_again(self, times: int) -> None:
        """Return once the clock has been read ``times`` more times."""
        wanted = self.reads + times
        async with asyncio.timeout(HANG_TIMEOUT_S):
            while self.reads < wanted:
                self._read.clear()
                await self._read.wait()


class Folders:
    """Knowledge whose passes the test watches, holds, or makes fail.

    ``write_waiting`` set says a write waits for the index; it clears once a
    pass has paused, as the write then runs.
    """

    def __init__(self, failure: Exception | None = None) -> None:
        self.pauses: list[Pause | None] = []
        self.began = asyncio.Event()
        self.held = asyncio.Event()
        self.held.set()
        self.write_waiting = False
        self._failure = failure

    async def index(self, pause: Pause | None = None) -> IndexReport:
        self.pauses.append(pause)
        self.began.set()
        await self.held.wait()
        if pause is not None:
            try:
                await pause()
            finally:
                self.write_waiting = False
        if self._failure is not None:
            raise self._failure
        return IndexReport(added=len(self.pauses))


class Reports:
    """Every pass reported, and a way to wait for the next ones."""

    def __init__(self) -> None:
        self.passes: list[Indexed] = []
        self._new = asyncio.Event()

    def __call__(self, done: Indexed) -> None:
        self.passes.append(done)
        self._new.set()

    async def until(self, count: int) -> None:
        async with asyncio.timeout(HANG_TIMEOUT_S):
            while len(self.passes) < count:
                self._new.clear()
                await self._new.wait()


def test_running_holds_until_every_turn_has_ended() -> None:
    activity = Activity()
    seen = [activity.running]

    with activity.turn():
        with activity.turn():
            seen.append(activity.running)
        seen.append(activity.running)
    seen.append(activity.running)

    assert seen == [False, True, True, False]


async def test_settled_waits_for_idle_time_after_the_last_turn() -> None:
    clock = Clock()
    activity = Activity(clock)
    settled = asyncio.create_task(activity.settled(0.001))

    await clock.read_again(2)
    assert not settled.done()
    with activity.turn():
        clock.now = 0.002
    clock.now = 0.0025
    await clock.read_again(2)
    assert not settled.done()
    clock.now = 0.003
    await asyncio.wait_for(settled, HANG_TIMEOUT_S)


async def test_settled_never_returns_while_a_turn_runs() -> None:
    clock = Clock()
    activity = Activity(clock)

    with activity.turn():
        clock.now = 100.0
        settled = asyncio.create_task(activity.settled(0.001))
        await asyncio.sleep(0)
        assert not settled.done()
    clock.now = 200.0
    await asyncio.wait_for(settled, HANG_TIMEOUT_S)


async def test_settled_waits_while_one_of_two_turns_still_runs() -> None:
    clock = Clock()
    activity = Activity(clock)

    with activity.turn():
        with activity.turn():
            pass
        clock.now = 100.0
        settled = asyncio.create_task(activity.settled(0.001))
        await asyncio.sleep(0)
        waited = not settled.done()
    clock.now = 200.0
    await asyncio.wait_for(settled, HANG_TIMEOUT_S)

    assert waited


async def test_a_scheduled_pass_waits_for_idle_and_comes_at_most_once_per_period() -> (
    None
):
    clock = Clock()
    activity = Activity(clock)
    folders = Folders()
    reports = Reports()
    reindexing = Reindexing(
        folders, activity, report=reports, idle_s=0.001, every_s=0.002, clock=clock
    )

    async with in_background(reindexing.run()):
        await clock.read_again(3)
        not_idle_yet = list(reports.passes)
        clock.now = 0.001
        await reports.until(1)
        await clock.read_again(3)
        within_the_period = list(reports.passes)
        clock.now = 0.003
        await reports.until(2)

    assert not_idle_yet == []
    assert len(within_the_period) == 1
    assert [p.scheduled for p in reports.passes] == [True, True]
    assert all(pause is not None for pause in folders.pauses)


async def test_a_scheduled_pass_stops_for_a_running_turn_and_is_reported() -> None:
    activity = Activity()
    reports = Reports()
    reindexing = Reindexing(Folders(), activity, report=reports)

    with activity.turn():
        done = await asyncio.wait_for(reindexing.index(scheduled=True), HANG_TIMEOUT_S)

    assert done == Indexed(None, scheduled=True, error=None, stopped=True)
    assert reports.passes == [done]


async def test_a_scheduled_pass_stops_for_a_write_waiting_on_the_index() -> None:
    folders = Folders()
    folders.write_waiting = True
    reindexing = Reindexing(folders, Activity(), report=Reports())

    done = await asyncio.wait_for(reindexing.index(scheduled=True), HANG_TIMEOUT_S)

    assert done == Indexed(None, scheduled=True, error=None, stopped=True)


async def test_a_stopped_pass_is_taken_up_again_without_waiting_for_the_period() -> (
    None
):
    folders = Folders()
    folders.write_waiting = True
    reports = Reports()
    reindexing = Reindexing(
        folders, Activity(), report=reports, idle_s=0.0, every_s=3600.0
    )

    async with in_background(reindexing.run()):
        await reports.until(2)

    assert [p.stopped for p in reports.passes[:2]] == [True, False]
    assert reports.passes[1].report == IndexReport(added=2)


async def test_the_period_is_counted_from_the_end_of_a_pass() -> None:
    clock = Clock()
    activity = Activity(clock)
    folders = Folders()
    folders.held.clear()
    reports = Reports()
    reindexing = Reindexing(
        folders, activity, report=reports, idle_s=0.001, every_s=0.002, clock=clock
    )

    async with in_background(reindexing.run()):
        clock.now = 0.001
        await asyncio.wait_for(folders.began.wait(), HANG_TIMEOUT_S)
        clock.now = 0.0025
        folders.held.set()
        await reports.until(1)
        clock.now = 0.0035
        await clock.read_again(3)
        within_the_period = list(reports.passes)
        clock.now = 0.005
        await reports.until(2)

    assert len(within_the_period) == 1
    assert len(reports.passes) == 2


async def test_a_turn_no_longer_holds_up_writes_to_a_real_index(
    tmp_path: Path,
) -> None:
    knowledge = await opened(tmp_path / "knowledge.db")
    root = planted(tmp_path / "notes", NOTES)
    await knowledge.add(root)
    activity = Activity()
    reindexing = Reindexing(knowledge, activity, report=Reports())

    with activity.turn():
        scheduled = await asyncio.wait_for(
            reindexing.index(scheduled=True), HANG_TIMEOUT_S
        )
        asked = await asyncio.wait_for(
            reindexing.index(scheduled=False), HANG_TIMEOUT_S
        )
        removed = await asyncio.wait_for(knowledge.remove(root), HANG_TIMEOUT_S)

    assert scheduled == Indexed(None, scheduled=True, error=None, stopped=True)
    assert asked.report is not None
    assert asked.report.added == len(NOTES)
    assert removed


async def test_an_asked_pass_runs_through_a_turn_and_is_reported() -> None:
    activity = Activity()
    folders = Folders()
    reports = Reports()
    reindexing = Reindexing(folders, activity, report=reports)

    with activity.turn():
        done = await asyncio.wait_for(reindexing.index(scheduled=False), HANG_TIMEOUT_S)

    assert folders.pauses == [None]
    assert reports.passes == [done]
    assert done == Indexed(IndexReport(added=1), scheduled=False, error=None)


@pytest.mark.parametrize(
    ("failure", "named"),
    [
        (sqlite3.OperationalError("disk I/O error"), "disk I/O error"),
        (RuntimeError(), "RuntimeError"),
    ],
    ids=["with-a-message", "without-one"],
)
async def test_a_failed_pass_is_logged_reported_and_never_raised(
    failure: Exception, named: str, caplog: pytest.LogCaptureFixture
) -> None:
    reports = Reports()
    reindexing = Reindexing(Folders(failure), Activity(), report=reports)

    with caplog.at_level(logging.ERROR, "synthia.server.upkeep"):
        done = await reindexing.index(scheduled=True)

    assert done == Indexed(None, scheduled=True, error=named)
    assert reports.passes == [done]
    assert "the knowledge index was not brought in step" in caplog.text


async def test_a_cancelled_pass_is_reported_and_the_cancellation_goes_on() -> None:
    folders = Folders()
    folders.held.clear()
    reports = Reports()
    reindexing = Reindexing(folders, Activity(), report=reports)

    running = asyncio.create_task(reindexing.index(scheduled=False))
    await asyncio.wait_for(folders.began.wait(), HANG_TIMEOUT_S)
    running.cancel()

    with pytest.raises(asyncio.CancelledError):
        await running
    assert reports.passes == [Indexed(None, scheduled=False, error=CANCELLED)]


async def test_a_report_that_fails_is_logged_and_the_pass_still_returns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def broken(_: Indexed) -> None:
        message = "the trace is gone"
        raise OSError(message)

    reindexing = Reindexing(Folders(), Activity(), report=broken)

    with caplog.at_level(logging.ERROR, "synthia.server.upkeep"):
        done = await reindexing.index(scheduled=False)

    assert done.report == IndexReport(added=1)
    assert "a pass of the knowledge index was not reported" in caplog.text
