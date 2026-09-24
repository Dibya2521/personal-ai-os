"""Run llama.cpp's server for the local model.

The server listens on loopback only, on a port chosen per launch, and requires
a key made fresh for each launch, so no other program on the machine, and no
web page making requests to localhost, can use it.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import secrets
import socket
import subprocess
import time
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, BinaryIO, Final

import httpx
from pydantic import SecretStr

from synthia.kernel.errors import SynthiaError

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from pathlib import Path

    from synthia.models.catalogue import Backend, Model

SERVICE_NAME: Final = "llama-server"
SERVER_NAMES: Final = frozenset({"llama-server", "llama-server.exe"})
LOOPBACK: Final = "127.0.0.1"
API_KEY_VARIABLE: Final = "LLAMA_API_KEY"
KEY_BYTES: Final = 32
# Loading 3 GB of weights from a slow disk; C13 measures the real figure.
DEFAULT_START_TIMEOUT_S: Final = 180.0
# Below the supervisor's own 5 s shutdown limit, so a stop never gets cancelled.
DEFAULT_STOP_TIMEOUT_S: Final = 3.0
HEALTH_POLL_S: Final = 0.25


class ServerError(SynthiaError):
    """The local model server cannot be found, started or reached."""


def find_server(runtime: Path) -> Path:
    """Return the ``llama-server`` executable inside an installed runtime.

    The layout inside a release archive is not part of its contract, so the
    directory is searched; the shallowest match wins.

    Raises:
        ServerError: If there is none.
    """
    found = [p for p in runtime.rglob("llama-server*") if p.name in SERVER_NAMES]
    if not found:
        message = f"no llama-server in {runtime}; reinstall it"
        raise ServerError(message)
    return min(found, key=lambda p: (len(p.parts), str(p)))


def free_port() -> int:
    """Return a loopback port that was free a moment ago."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((LOOPBACK, 0))
        return int(probe.getsockname()[1])


def new_key() -> SecretStr:
    """Return a key for one launch."""
    return SecretStr(secrets.token_urlsafe(KEY_BYTES))


@dataclass(frozen=True, slots=True)
class Launch:
    """Everything needed to start the server once."""

    backend: Backend
    binary: Path
    weights: Path
    projector: Path | None
    port: int
    context: int
    key: SecretStr

    @classmethod
    def of(  # noqa: PLR0913
        cls,
        backend: Backend,
        runtime: Path,
        model: Model,
        model_dir: Path,
        *,
        port: int,
        context: int,
        key: SecretStr,
    ) -> Launch:
        """Return the launch of ``model``, installed in ``model_dir``, on ``runtime``.

        Raises:
            ServerError: If the runtime has no server executable.
        """
        projector = model_dir / model.projector.name if model.projector else None
        return cls(
            backend,
            find_server(runtime),
            model_dir / model.weights.name,
            projector,
            port,
            context,
            key,
        )

    @property
    def base_url(self) -> str:
        """Return the OpenAI-compatible API root of the running server."""
        return f"http://{LOOPBACK}:{self.port}/v1"

    @property
    def health_url(self) -> str:
        """Return the address that answers 200 once the model is loaded."""
        return f"http://{LOOPBACK}:{self.port}/health"

    def command(self) -> list[str]:
        """Return the command line; the key is passed in the environment instead."""
        args = [
            str(self.binary),
            "--model",
            str(self.weights),
            "--host",
            LOOPBACK,
            "--port",
            str(self.port),
            "--ctx-size",
            str(self.context),
            "--no-webui",
            "--offline",
        ]
        if self.projector is not None:
            args += ["--mmproj", str(self.projector)]
        return args

    def environment(self, base: dict[str, str]) -> dict[str, str]:
        """Return ``base`` with the key added, for the server's process."""
        return {**base, API_KEY_VARIABLE: self.key.get_secret_value()}


class ServerCrashedError(ServerError):
    """The server exited while it was serving."""


def _no_fallback(_launch: Launch, _error: ServerError) -> None:
    return None


class LlamaServer:
    """A :class:`~synthia.kernel.supervisor.Service` that keeps one server running.

    Each run is a new launch with its own port and key, so a restart after a
    crash never depends on a port another program may have taken meanwhile.
    ``ready`` is set while the server answers and cleared when it stops.
    ``on_start_failure`` hears of every start that never became ready, so the
    next launch can use another build.
    """

    def __init__(  # noqa: PLR0913
        self,
        launch: Callable[[], Launch],
        client: httpx.AsyncClient,
        log: Path,
        *,
        command: Callable[[Launch], list[str]] = Launch.command,
        start_timeout_s: float = DEFAULT_START_TIMEOUT_S,
        stop_timeout_s: float = DEFAULT_STOP_TIMEOUT_S,
        poll_s: float = HEALTH_POLL_S,
        on_start_failure: Callable[[Launch, ServerError], None] = _no_fallback,
    ) -> None:
        self._on_start_failure = on_start_failure
        self._launch = launch
        self._client = client
        self._log = log
        self._command = command
        self._start_timeout_s = start_timeout_s
        self._stop_timeout_s = stop_timeout_s
        self._poll_s = poll_s
        self._running: Launch | None = None
        self.ready = asyncio.Event()

    @property
    def name(self) -> str:
        """Return the service name."""
        return SERVICE_NAME

    @property
    def running(self) -> Launch | None:
        """Return the launch that is serving now, if any."""
        return self._running

    async def run(self, stop: asyncio.Event) -> None:
        """Start the server, serve until ``stop`` is set, then stop it.

        Raises:
            ServerError: If it exits or stays unready before it has served.
            ServerCrashedError: If it exits while serving.
        """
        launch = self._launch()
        self._log.parent.mkdir(parents=True, exist_ok=True)
        with self._log.open("ab") as log:
            process = await self._spawn(launch, log)
            try:
                await self._serve(launch, process, stop)
            finally:
                self.ready.clear()
                self._running = None
                await self._end(process)

    async def _spawn(self, launch: Launch, log: BinaryIO) -> asyncio.subprocess.Process:
        try:
            return await asyncio.create_subprocess_exec(
                *self._command(launch),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                env=launch.environment(dict(os.environ)),
                # Ctrl+C in the terminal stops an answer; in the terminal's own
                # process group it would also stop the server.
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                start_new_session=True,
            )
        except OSError as error:
            failure = ServerError(f"cannot run {launch.binary}: {error}")
            self._on_start_failure(launch, failure)
            raise failure from error

    async def _serve(
        self, launch: Launch, process: asyncio.subprocess.Process, stop: asyncio.Event
    ) -> None:
        try:
            healthy = await self._until_healthy(launch, process, stop)
        except ServerError as error:
            self._on_start_failure(launch, error)
            raise
        if not healthy:
            return
        self._running = launch
        self.ready.set()
        await _first_of(process.wait(), stop.wait())
        if not stop.is_set():
            message = (
                f"llama-server exited with code {process.returncode} "
                f"while serving; see {self._log}"
            )
            raise ServerCrashedError(message)

    async def _until_healthy(
        self, launch: Launch, process: asyncio.subprocess.Process, stop: asyncio.Event
    ) -> bool:
        """Return True once healthy, False if asked to stop first."""
        deadline = time.monotonic() + self._start_timeout_s
        while not stop.is_set():
            if process.returncode is not None:
                message = (
                    f"llama-server exited with code {process.returncode} "
                    f"before it was ready; see {self._log}"
                )
                raise ServerError(message)
            if await self._healthy(launch):
                return True
            if time.monotonic() >= deadline:
                message = (
                    f"llama-server was not ready within {self._start_timeout_s:g} s; "
                    f"see {self._log}"
                )
                raise ServerError(message)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), self._poll_s)
        return False

    async def _healthy(self, launch: Launch) -> bool:
        try:
            response = await self._client.get(launch.health_url, timeout=self._poll_s)
        except httpx.TransportError:
            return False
        return response.status_code == HTTPStatus.OK

    async def _end(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), self._stop_timeout_s)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
        await process.wait()


async def _first_of(*waits: Coroutine[object, object, object]) -> None:
    """Wait until one of ``waits`` finishes, then cancel the rest."""
    tasks = [asyncio.ensure_future(w) for w in waits]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
