"""A daemon for a chat test, in a thread of its own, with the fakes the test gives it.

The real daemon is another process; a thread with its own event loop is the
closest a test can come while still handing it a fake transport or a fake
local model. It stops the way the real one does, through its stop event.
"""

import asyncio
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager, suppress

import httpx

from synthia.gateway.providers import OPENROUTER_FREE, RemoteProvider
from synthia.kernel.config import Settings
from synthia.server.daemon import LocalModels, run_daemon
from synthia.server.discovery import DAEMON_FILE, DaemonInfo, read_info
from tests.timing import HANG_TIMEOUT_S


class _Running:
    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.stop: asyncio.Event | None = None
        self.error: BaseException | None = None


@contextmanager
def private_daemon(
    settings: Settings,
    transport: httpx.AsyncBaseTransport | None = None,
    *,
    local: LocalModels | None = None,
    remote: RemoteProvider = OPENROUTER_FREE,
) -> Generator[DaemonInfo]:
    """Yield a running daemon's file; stop the daemon when the block ends."""
    running = _Running()

    async def serve() -> None:
        running.loop = asyncio.get_running_loop()
        running.stop = asyncio.Event()
        await run_daemon(
            settings, local, transport=transport, remote=remote, stop=running.stop
        )

    def run() -> None:
        try:
            asyncio.run(serve())
        except BaseException as error:  # noqa: BLE001 - handed to the test
            running.error = error

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    info = _written(settings, thread, running)
    try:
        yield info
    finally:
        # A closed loop means the test already stopped the daemon.
        if running.loop is not None and running.stop is not None:
            with suppress(RuntimeError):
                running.loop.call_soon_threadsafe(running.stop.set)
        thread.join(HANG_TIMEOUT_S)
        if running.error is not None:
            raise running.error


def _written(
    settings: Settings, thread: threading.Thread, running: _Running
) -> DaemonInfo:
    deadline = time.monotonic() + HANG_TIMEOUT_S
    while (info := read_info(settings.home / DAEMON_FILE)) is None:
        if not thread.is_alive():
            raise running.error or AssertionError("the daemon stopped at start")
        assert time.monotonic() < deadline, "the daemon did not start"
        time.sleep(0.02)
    return info
