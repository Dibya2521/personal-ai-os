"""Supervision of long-running services, after Erlang/OTP.

Every long-running part of SYNTHIA (audio capture, the wake word, the camera, the
daemon's API) is a :class:`Service` under a :class:`Supervisor`. A service that
crashes is restarted on its own, one for one, so a failing peripheral never takes
the rest down.

Restarts are bounded by an intensity: more than ``max_restarts`` restarts of one
service within ``window_s`` seconds means the fault is not transient. The
supervisor then stops every service gracefully and raises
:class:`SupervisorGaveUpError`, handing the decision to whatever runs it.

The delay before a restart doubles with each restart still inside the window, and
falls back as old restarts age out of it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from synthia.kernel.bus import Event
from synthia.kernel.errors import SynthiaError

if TYPE_CHECKING:
    from collections.abc import Callable

    from synthia.kernel.bus import EventBus

logger = logging.getLogger(__name__)

DEFAULT_STOP_TIMEOUT_S = 5.0


class Service(Protocol):
    """Something that runs until asked to stop.

    ``run`` should return promptly once ``stop`` is set. Returning earlier means
    the service has finished; raising means it crashed.
    """

    @property
    def name(self) -> str:
        """Return a name unique within one supervisor."""
        ...

    async def run(self, stop: asyncio.Event) -> None:
        """Do the service's work until ``stop`` is set."""
        ...


class RestartMode(StrEnum):
    """When a service that ended is started again."""

    PERMANENT = "permanent"
    """Always: after a crash and after a clean return."""
    TRANSIENT = "transient"
    """Only after a crash. A clean return means the work is done."""
    TEMPORARY = "temporary"
    """Never."""


@dataclass(frozen=True, slots=True)
class RestartPolicy:
    """How often, and how soon, a service is restarted."""

    max_restarts: int = 5
    window_s: float = 60.0
    backoff_initial_s: float = 0.5
    backoff_factor: float = 2.0
    backoff_max_s: float = 30.0

    def delay(self, recent_restarts: int) -> float:
        """Return the wait before the restart that makes ``recent_restarts``."""
        exponent = max(recent_restarts - 1, 0)
        return min(
            self.backoff_initial_s * self.backoff_factor**exponent, self.backoff_max_s
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ServiceStarted(Event):
    """A service's ``run`` was entered."""

    service: str
    attempt: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ServiceExited(Event):
    """A service's ``run`` ended.

    ``error`` is the exception's repr after a crash and ``None`` after a clean
    return; ``restart_in_s`` is ``None`` when it will not be restarted.
    """

    service: str
    error: str | None
    restart_in_s: float | None


@dataclass(frozen=True, slots=True, kw_only=True)
class ServiceStopped(Event):
    """A service ended and will not be restarted."""

    service: str


class SupervisorGaveUpError(SynthiaError):
    """A service restarted more often than its policy allows."""

    def __init__(self, service: str, restarts: int, window_s: float) -> None:
        super().__init__(
            f"service {service!r} restarted {restarts} times within {window_s:g} s"
        )
        self.service = service


@dataclass(slots=True)
class _Child:
    service: Service
    mode: RestartMode
    policy: RestartPolicy
    restarts: deque[float] = field(default_factory=deque[float])


class Supervisor:
    """Run services concurrently and restart them one for one."""

    def __init__(
        self,
        *,
        bus: EventBus | None = None,
        default_policy: RestartPolicy | None = None,
        stop_timeout_s: float = DEFAULT_STOP_TIMEOUT_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._bus = bus
        self._default_policy = default_policy or RestartPolicy()
        self._stop_timeout_s = stop_timeout_s
        self._clock = clock
        self._children: dict[str, _Child] = {}
        self._stop = asyncio.Event()

    def add(
        self,
        service: Service,
        mode: RestartMode = RestartMode.TRANSIENT,
        policy: RestartPolicy | None = None,
    ) -> None:
        """Register ``service``. Takes effect at the next :meth:`run`.

        Raises:
            ValueError: If a service with the same name is already registered.
        """
        if service.name in self._children:
            message = f"a service named {service.name!r} is already supervised"
            raise ValueError(message)
        self._children[service.name] = _Child(
            service, mode, policy or self._default_policy
        )

    def stop(self) -> None:
        """Ask every service to stop. :meth:`run` returns once they have."""
        self._stop.set()

    async def run(self) -> None:
        """Run every service until all end, :meth:`stop` is called, or one fails.

        Raises:
            SupervisorGaveUpError: If a service exceeded its restart intensity.
                Every other service has been stopped by then.
        """
        tasks = [
            asyncio.create_task(self._supervise(child), name=f"service:{name}")
            for name, child in self._children.items()
        ]
        stop_requested = asyncio.create_task(self._stop.wait())
        try:
            await self._wait_for_end(set(tasks), stop_requested)
            self._stop.set()
            await self._shut_down(tasks)
        finally:
            # Reached with tasks still running only if run() itself was cancelled.
            everything: list[asyncio.Task[Any]] = [*tasks, stop_requested]
            for task in everything:
                task.cancel()
            await asyncio.gather(*everything, return_exceptions=True)
        for task in tasks:
            if not task.cancelled() and (error := task.exception()) is not None:
                raise error

    @staticmethod
    async def _wait_for_end(
        running: set[asyncio.Task[None]], stop_requested: asyncio.Task[bool]
    ) -> None:
        """Return once every task ended, one failed, or a stop was requested.

        Waiting on the tasks alone would never return for a service that ignores
        stop, and the shutdown timeout that deals with such a service would
        never be reached.
        """
        while running and not stop_requested.done():
            waiting: set[asyncio.Task[Any]] = {*running, stop_requested}
            done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
            running.difference_update(done)
            if any(t.exception() for t in done if t is not stop_requested):
                return

    async def _shut_down(self, tasks: list[asyncio.Task[None]]) -> None:
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=self._stop_timeout_s)
        for task in pending:
            logger.warning(
                "service ignored stop, cancelling it",
                extra={"task": task.get_name()},
            )
            task.cancel()
        if pending:
            await asyncio.wait(pending)

    async def _supervise(self, child: _Child) -> None:
        name = child.service.name
        attempt = 0
        try:
            while not self._stop.is_set():
                attempt += 1
                await self._publish(ServiceStarted(service=name, attempt=attempt))
                error = await self._run_once(child.service)
                described = repr(error) if error is not None else None
                if self._stop.is_set() or not self._should_restart(child, error):
                    await self._publish(
                        ServiceExited(service=name, error=described, restart_in_s=None)
                    )
                    break
                try:
                    delay = self._record_restart(child)
                except SupervisorGaveUpError:
                    await self._publish(
                        ServiceExited(service=name, error=described, restart_in_s=None)
                    )
                    raise
                await self._publish(
                    ServiceExited(service=name, error=described, restart_in_s=delay)
                )
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), timeout=delay)
        finally:
            await self._publish(ServiceStopped(service=name))

    async def _run_once(self, service: Service) -> Exception | None:
        try:
            await service.run(self._stop)
        except Exception as error:
            logger.exception("service crashed", extra={"service": service.name})
            return error
        return None

    @staticmethod
    def _should_restart(child: _Child, error: Exception | None) -> bool:
        if child.mode is RestartMode.TEMPORARY:
            return False
        return child.mode is RestartMode.PERMANENT or error is not None

    def _record_restart(self, child: _Child) -> float:
        """Count one restart and return its delay.

        Raises:
            SupervisorGaveUpError: If this restart exceeds the policy's intensity.
        """
        now = self._clock()
        policy = child.policy
        while child.restarts and now - child.restarts[0] > policy.window_s:
            child.restarts.popleft()
        child.restarts.append(now)
        if len(child.restarts) > policy.max_restarts:
            raise SupervisorGaveUpError(
                child.service.name, len(child.restarts), policy.window_s
            )
        return policy.delay(len(child.restarts))

    async def _publish(self, event: Event) -> None:
        if self._bus is not None:
            await self._bus.publish(event)
