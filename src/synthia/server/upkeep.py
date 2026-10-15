"""Upkeep the daemon does between turns: keeping the knowledge index in step.

:class:`Activity` counts the turns running. :class:`Reindexing` reads the
person's folders again once no turn has run for 10 minutes, at most once an
hour after the last pass ended: the rule facts are consolidated by, for the
same reason, so a busy laptop never walks folders in the middle of a
conversation. Such a pass stops before its next file when a turn starts or a
write (adding or removing a folder, a pass asked for) waits for the index, so
neither waits for it; what it read stays, and the next idle time goes on from
there. Everything runs on one event loop, so the counts need no lock.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from synthia.knowledge.index import IndexReport, Pause

INDEX_IDLE_S: Final = 600.0
INDEX_EVERY_S: Final = 3600.0
CANCELLED: Final = "cancelled"

logger = logging.getLogger(__name__)


class GaveWayError(Exception):
    """A scheduled pass stopped for a turn, or for a write waiting on the index."""


class Activity:
    """Counts the turns running; says whether one runs, or none has for a while."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        """Measure idle time on ``clock``, from now until the first turn."""
        self._clock = clock
        self._running = 0
        self._last = clock()
        self._free = asyncio.Event()
        self._free.set()

    @contextmanager
    def turn(self) -> Generator[None]:
        """Count a turn as running while the block runs."""
        self._running += 1
        self._free.clear()
        try:
            yield
        finally:
            self._running -= 1
            self._last = self._clock()
            if not self._running:
                self._free.set()

    @property
    def running(self) -> bool:
        """Whether a turn is running."""
        return self._running > 0

    async def settled(self, idle_s: float) -> None:
        """Return once no turn has run for ``idle_s``."""
        while True:
            await self._free.wait()
            left = self._last + idle_s - self._clock()
            if left <= 0:
                return
            await asyncio.sleep(left)


class Indexes(Protocol):
    """What re-indexing needs of knowledge."""

    @property
    def write_waiting(self) -> bool:
        """Whether a write waits for the index."""
        ...

    async def index(self, pause: Pause | None = None) -> IndexReport:
        """Bring every source in step with its files; ``pause`` comes between them."""
        ...


@dataclass(frozen=True, slots=True)
class Indexed:
    """One pass of the knowledge index: what it did, why it ran, and how it ended.

    ``report`` is None when the pass failed, was cancelled or ``stopped``;
    ``error`` says why it failed or that it was cancelled. A ``stopped``
    pass was scheduled and gave way to a turn or a write.
    """

    report: IndexReport | None
    scheduled: bool
    error: str | None
    stopped: bool = False


class Reindexing:
    """Keeps the knowledge index in step with its folders while the daemon is idle."""

    def __init__(  # noqa: PLR0913 - the schedule's parts are injected for tests
        self,
        knowledge: Indexes,
        activity: Activity,
        *,
        report: Callable[[Indexed], None],
        idle_s: float = INDEX_IDLE_S,
        every_s: float = INDEX_EVERY_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Index ``knowledge`` when ``activity`` idles; ``report`` hears every pass."""
        self._knowledge = knowledge
        self._activity = activity
        self._report = report
        self._idle_s = idle_s
        self._every_s = every_s
        self._clock = clock
        self._last: float | None = None

    async def run(self) -> None:
        """Index once no turn has run for ``idle_s``, then ``every_s`` after a pass.

        A pass that gave way does not count, so the next idle time takes it up.
        Runs until cancelled.
        """
        while True:
            await self._activity.settled(self._idle_s)
            left = self._until_due()
            if left > 0:
                await asyncio.sleep(left)
            else:
                await self.index(scheduled=True)

    def _until_due(self) -> float:
        if self._last is None:
            return 0.0
        return self._last + self._every_s - self._clock()

    async def index(self, *, scheduled: bool) -> Indexed:
        """Bring the index in step with its folders, and report the pass.

        A ``scheduled`` pass stops before its next file once a turn runs or a
        write waits for the index; an asked one runs through, since the
        person is waiting for it. Never fails: a failure is logged and named
        in :attr:`Indexed.error`. A cancelled pass is reported as such before
        the cancellation goes on.
        """
        report: IndexReport | None = None
        error: str | None = CANCELLED
        stopped = False
        try:
            report = await self._knowledge.index(self._give_way if scheduled else None)
            error = None
        except GaveWayError:
            error, stopped = None, True
        except Exception as failed:
            logger.exception("the knowledge index was not brought in step")
            error = str(failed) or type(failed).__name__
        finally:
            if not stopped:
                self._last = self._clock()
            done = Indexed(report, scheduled, error, stopped)
            self._reported(done)
        return done

    async def _give_way(self) -> None:
        if self._activity.running or self._knowledge.write_waiting:
            raise GaveWayError

    def _reported(self, done: Indexed) -> None:
        try:
            self._report(done)
        except Exception:
            logger.exception("a pass of the knowledge index was not reported")
