import asyncio
import sys
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from synthia.kernel.config import LocalBackend
from synthia.kernel.supervisor import RestartPolicy, Supervisor, SupervisorGaveUpError
from synthia.models.backends import Fallback, NoBackendError, candidates
from synthia.models.catalogue import RUNTIMES, Backend, Runtime, Target
from synthia.models.server import (
    Launch,
    LlamaServer,
    ServerError,
    free_port,
    new_key,
)
from tests.models import fake_llama_server
from tests.timing import HANG_TIMEOUT_S

LINUX = {r.backend: r for r in RUNTIMES if r.target is Target.LINUX_X64}


def launch_of(runtime: Runtime) -> Launch:
    return Launch(
        runtime.backend,
        Path("llama-server"),
        Path("model.gguf"),
        None,
        port=1,
        context=1,
        key=SecretStr("k"),
    )


def test_builds_are_tried_fastest_first() -> None:
    installed = [LINUX[Backend.CPU], LINUX[Backend.VULKAN], LINUX[Backend.CUDA]]

    order = [r.backend for r in candidates(installed, LocalBackend.AUTO)]

    assert order == [Backend.CUDA, Backend.VULKAN, Backend.CPU]


def test_a_chosen_backend_is_the_only_candidate() -> None:
    installed = [LINUX[Backend.CPU], LINUX[Backend.VULKAN]]

    assert candidates(installed, LocalBackend.CPU) == (LINUX[Backend.CPU],)
    assert candidates(installed, LocalBackend.CUDA) == ()


def test_the_fastest_build_is_launched_while_it_works() -> None:
    fallback = Fallback([LINUX[Backend.VULKAN], LINUX[Backend.CPU]], launch_of)

    assert fallback.launch().backend is Backend.VULKAN
    assert fallback.launch().backend is Backend.VULKAN


def test_a_build_that_failed_to_start_is_skipped_with_its_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fallback = Fallback([LINUX[Backend.VULKAN], LINUX[Backend.CPU]], launch_of)
    fallback.failed(fallback.launch(), ServerError("exited with code 1"))

    with caplog.at_level("INFO", logger="synthia.models.backends"):
        launch = fallback.launch()

    assert launch.backend is Backend.CPU
    assert fallback.failures == {Backend.VULKAN: "exited with code 1"}
    assert "on cpu (skipped vulkan: exited with code 1)" in caplog.text


def test_when_every_build_failed_the_reasons_are_given() -> None:
    fallback = Fallback([LINUX[Backend.VULKAN], LINUX[Backend.CPU]], launch_of)
    fallback.failed(fallback.launch(), ServerError("no device"))
    fallback.failed(fallback.launch(), ServerError("not ready within 180 s"))

    with pytest.raises(NoBackendError, match="vulkan: no device; cpu: not ready"):
        fallback.launch()


def test_with_nothing_installed_the_error_says_how_to_install() -> None:
    with pytest.raises(NoBackendError, match="synthia models install"):
        Fallback([], launch_of).launch()


def serving_on(
    tmp_path: Path,
    client: httpx.AsyncClient,
    command: Callable[[Launch], list[str]],
) -> tuple[LlamaServer, Fallback]:
    def make(runtime: Runtime) -> Launch:
        return Launch(
            runtime.backend,
            tmp_path / "llama-server",
            tmp_path / "model.gguf",
            None,
            port=free_port(),
            context=512,
            key=new_key(),
        )

    fallback = Fallback([LINUX[Backend.VULKAN], LINUX[Backend.CPU]], make)
    server = LlamaServer(
        fallback.launch,
        client,
        tmp_path / "llama-server.log",
        command=command,
        poll_s=0.05,
        on_start_failure=fallback.failed,
    )
    return server, fallback


async def test_the_supervisor_falls_back_to_the_cpu_build(tmp_path: Path) -> None:
    def command(launch: Launch) -> list[str]:
        if launch.backend is Backend.VULKAN:
            return [sys.executable, "-c", "raise SystemExit(4)"]
        return fake_llama_server.command(launch)

    async with httpx.AsyncClient() as client:
        server, fallback = serving_on(tmp_path, client, command)
        supervisor = Supervisor(default_policy=RestartPolicy(backoff_initial_s=0.01))
        supervisor.add(server)
        supervised = asyncio.create_task(supervisor.run())
        await asyncio.wait_for(server.ready.wait(), HANG_TIMEOUT_S)
        running = server.running
        supervisor.stop()
        await asyncio.wait_for(supervised, HANG_TIMEOUT_S)

    assert running is not None
    assert running.backend is Backend.CPU
    assert list(fallback.failures) == [Backend.VULKAN]
    assert "code 4 before it was ready" in fallback.failures[Backend.VULKAN]


async def test_a_build_that_cannot_even_run_is_a_start_failure(tmp_path: Path) -> None:
    missing = [str(tmp_path / "missing" / "llama-server")]
    async with httpx.AsyncClient() as client:
        server, fallback = serving_on(tmp_path, client, lambda _: missing)
        with pytest.raises(ServerError, match="cannot run"):
            await server.run(asyncio.Event())

    assert list(fallback.failures) == [Backend.VULKAN]
    assert fallback.launch().backend is Backend.CPU


async def test_when_no_build_starts_the_supervisor_gives_up(tmp_path: Path) -> None:
    def command(launch: Launch) -> list[str]:
        return [sys.executable, "-c", f"raise SystemExit('{launch.backend} broken')"]

    policy = RestartPolicy(max_restarts=3, backoff_initial_s=0.01)
    async with httpx.AsyncClient() as client:
        server, fallback = serving_on(tmp_path, client, command)
        supervisor = Supervisor(default_policy=policy)
        supervisor.add(server)
        with pytest.raises(SupervisorGaveUpError):
            await asyncio.wait_for(supervisor.run(), HANG_TIMEOUT_S)

    assert list(fallback.failures) == [Backend.VULKAN, Backend.CPU]
    assert not server.ready.is_set()
