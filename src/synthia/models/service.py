"""Keep the local model's server running beside the chat.

The chat's event loop runs only while a turn does, so a server supervised on
it would load, and be checked, only during turns. It is supervised on a
thread and event loop of its own instead, with its own HTTP client; the chat
reads which launch is serving and sends its requests on its own loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import httpx

from synthia.kernel.supervisor import RestartPolicy, Supervisor, SupervisorGaveUpError
from synthia.models.backends import Fallback, candidates
from synthia.models.catalogue import MODELS, RUNTIMES, Model, Runtime
from synthia.models.local import LocalModel
from synthia.models.server import Launch, LlamaServer, free_port, new_key

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from synthia.kernel.config import LocalBackend, Settings
    from synthia.models.catalogue import Target
    from synthia.models.install import Installer

logger = logging.getLogger(__name__)

SERVER_LOG: Final = Path("logs") / "llama-server.log"
THREAD_NAME: Final = "local-model"
# Above the supervisor's 5 s shutdown limit, which already covers a kill.
JOIN_TIMEOUT_S: Final = 10.0


@dataclass(frozen=True, slots=True)
class LocalSetup:
    """An installed model, and the installed builds to run it on, fastest first."""

    model: Model
    runtimes: tuple[Runtime, ...]
    installer: Installer
    context: int

    def launch(self, runtime: Runtime) -> Launch:
        """Return a fresh launch of the model on ``runtime``."""
        return Launch.of(
            runtime.backend,
            self.installer.path_of(runtime),
            self.model,
            self.installer.path_of(self.model),
            port=free_port(),
            context=self.context,
            key=new_key(),
        )


def installed_builds(
    installer: Installer,
    target: Target | None,
    choice: LocalBackend,
    runtimes: Iterable[Runtime] = RUNTIMES,
) -> tuple[Runtime, ...]:
    """Return the installed builds for ``target``, fastest first, within ``choice``."""
    return candidates(
        (r for r in runtimes if r.target == target and installer.installed(r)), choice
    )


def find_local(
    settings: Settings,
    installer: Installer,
    target: Target | None,
    catalogue: Iterable[Runtime | Model] = (*RUNTIMES, *MODELS),
) -> LocalSetup | None:
    """Return the configured local model and its builds, or None if not installed."""
    items = tuple(catalogue)
    model = next(
        (m for m in items if isinstance(m, Model) and m.id == settings.local_model),
        None,
    )
    if model is None or not installer.installed(model):
        return None
    runtimes = installed_builds(
        installer,
        target,
        settings.local_backend,
        (r for r in items if isinstance(r, Runtime)),
    )
    if not runtimes:
        return None
    return LocalSetup(model, runtimes, installer, settings.local_context)


class LocalService:
    """The local model's server, supervised on a thread of its own."""

    def __init__(
        self,
        setup: LocalSetup,
        log: Path,
        *,
        command: Callable[[Launch], list[str]] = Launch.command,
        policy: RestartPolicy | None = None,
    ) -> None:
        self._setup = setup
        fallback = Fallback(setup.runtimes, setup.launch)
        self._client = httpx.AsyncClient()
        self.server = LlamaServer(
            fallback.launch,
            self._client,
            log,
            command=command,
            on_start_failure=fallback.failed,
        )
        self._supervisor = Supervisor(default_policy=policy)
        self._supervisor.add(self.server)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._looping = threading.Event()
        self._thread = threading.Thread(
            target=self._main, name=THREAD_NAME, daemon=True
        )

    def model(self, client: httpx.AsyncClient) -> LocalModel:
        """Return the chat model that sends requests on ``client``'s loop."""
        return LocalModel(self.server, client, self._setup.model, self._setup.context)

    def start(self) -> None:
        """Start the thread; the server loads in the background."""
        self._thread.start()
        self._looping.wait()

    def stop(self) -> None:
        """Stop the server and wait for the thread, at most ``JOIN_TIMEOUT_S``."""
        loop = self._loop
        if loop is None:
            return
        # The loop is closed already if the supervisor gave up.
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(self._supervisor.stop)
        self._thread.join(JOIN_TIMEOUT_S)

    def _main(self) -> None:
        with asyncio.Runner() as runner:
            self._loop = runner.get_loop()
            self._looping.set()
            try:
                runner.run(self._run())
            except SupervisorGaveUpError:
                logger.exception("the local model is not restarted again; remote only")

    async def _run(self) -> None:
        try:
            await self._supervisor.run()
        finally:
            await self._client.aclose()
