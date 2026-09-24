"""Choose which installed llama.cpp build runs the local model, and fall back.

Builds are tried fastest first: Metal, CUDA, Vulkan, then the CPU build as
the floor. A build that fails to start, or never answers its health check, is
skipped for the rest of the process, and the next one is launched instead.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

from synthia.kernel.config import LocalBackend
from synthia.models.catalogue import Backend
from synthia.models.server import ServerError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from synthia.models.catalogue import Runtime
    from synthia.models.server import Launch

logger = logging.getLogger(__name__)

PRIORITY: Final = (Backend.METAL, Backend.CUDA, Backend.VULKAN, Backend.CPU)


class NoBackendError(ServerError):
    """No installed build can run the local model."""


def candidates(
    installed: Iterable[Runtime], choice: LocalBackend
) -> tuple[Runtime, ...]:
    """Return the builds to try, fastest first, limited to ``choice`` unless auto."""
    ordered = sorted(installed, key=lambda r: PRIORITY.index(r.backend))
    if choice is LocalBackend.AUTO:
        return tuple(ordered)
    return tuple(r for r in ordered if r.backend.value == choice.value)


class Fallback:
    """Hand out launches down the candidates, skipping builds that failed to start."""

    def __init__(
        self, runtimes: Iterable[Runtime], make: Callable[[Runtime], Launch]
    ) -> None:
        """Try ``runtimes`` in order; ``make`` builds a fresh launch of one."""
        self._runtimes = tuple(runtimes)
        self._make = make
        self._failures: dict[Backend, str] = {}

    @property
    def failures(self) -> dict[Backend, str]:
        """Return why each skipped build failed."""
        return dict(self._failures)

    def launch(self) -> Launch:
        """Return a launch of the fastest build that has not failed.

        Raises:
            NoBackendError: If none is installed or every one has failed.
        """
        for runtime in self._runtimes:
            if runtime.backend not in self._failures:
                logger.info(
                    "starting the local model on %s%s",
                    runtime.backend,
                    self._skipped(),
                )
                return self._make(runtime)
        if not self._runtimes:
            message = "no llama.cpp build is installed; run synthia models install"
            raise NoBackendError(message)
        message = f"no llama.cpp build could start the local model{self._skipped()}"
        raise NoBackendError(message)

    def failed(self, launch: Launch, error: ServerError) -> None:
        """Skip ``launch``'s build from now on, because it could not start."""
        self._failures[launch.backend] = str(error)
        logger.warning("the %s build could not start: %s", launch.backend, error)

    def _skipped(self) -> str:
        if not self._failures:
            return ""
        reasons = "; ".join(f"{b}: {why}" for b, why in self._failures.items())
        return f" (skipped {reasons})"
