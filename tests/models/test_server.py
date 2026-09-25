import asyncio
import socket
import sys
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from synthia.kernel.config import DEFAULT_LOCAL_MODEL, LocalBackend
from synthia.kernel.supervisor import RestartPolicy, Supervisor
from synthia.models.catalogue import Backend, Download, Model, find
from synthia.models.server import (
    API_KEY_VARIABLE,
    DEFAULT_START_TIMEOUT_S,
    DEFAULT_STOP_TIMEOUT_S,
    LOOPBACK,
    Launch,
    LlamaServer,
    ServerCrashedError,
    ServerError,
    find_server,
    free_port,
    new_key,
)
from tests.models import fake_llama_server
from tests.timing import HANG_TIMEOUT_S

KEY = SecretStr("launch-test-key")
SECRET = KEY.get_secret_value()
TEXT_ONLY = Model(
    "tiny",
    "example/tiny",
    "0" * 40,
    "mit",
    Download("tiny.gguf", "https://example.com/tiny.gguf", 1, "0" * 64),
)


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


@pytest.mark.parametrize("name", ["llama-server.exe", "llama-server"])
def test_the_server_is_found_at_the_top_of_a_flat_archive(
    tmp_path: Path, name: str
) -> None:
    touch(tmp_path / "llama-cli.exe")
    server = touch(tmp_path / name)

    assert find_server(tmp_path) == server


def test_the_shallowest_server_wins_in_a_nested_archive(tmp_path: Path) -> None:
    touch(tmp_path / "build" / "bin" / "extra" / "llama-server")
    server = touch(tmp_path / "build" / "bin" / "llama-server")

    assert find_server(tmp_path) == server


def test_a_similar_name_is_not_the_server(tmp_path: Path) -> None:
    touch(tmp_path / "llama-server-helper.exe")
    touch(tmp_path / "llama-server.pdb")

    with pytest.raises(ServerError, match="no llama-server"):
        find_server(tmp_path)


def test_a_free_port_can_be_bound_on_loopback() -> None:
    port = free_port()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((LOOPBACK, port))


def test_each_launch_gets_its_own_key() -> None:
    assert new_key().get_secret_value() != new_key().get_secret_value()


def launch_of(tmp_path: Path, model: Model) -> Launch:
    touch(tmp_path / "runtime" / "llama-server")
    return Launch.of(
        Backend.VULKAN,
        tmp_path / "runtime",
        model,
        tmp_path / "model",
        port=41234,
        context=8192,
        key=KEY,
    )


def test_the_command_binds_loopback_offline_without_the_web_ui(tmp_path: Path) -> None:
    model = find(DEFAULT_LOCAL_MODEL)
    assert isinstance(model, Model)
    assert model.projector is not None
    launch = launch_of(tmp_path, model)

    command = launch.command()

    assert command[0] == str(tmp_path / "runtime" / "llama-server")
    pairs = dict(zip(command[1::2], command[2::2], strict=False))
    assert pairs["--model"] == str(tmp_path / "model" / model.weights.name)
    assert pairs["--mmproj"] == str(tmp_path / "model" / model.projector.name)
    assert (pairs["--host"], pairs["--port"]) == (LOOPBACK, "41234")
    assert pairs["--ctx-size"] == "8192"
    assert {"--no-webui", "--offline"} <= set(command)
    assert launch.base_url == f"http://{LOOPBACK}:41234/v1"
    assert launch.health_url == f"http://{LOOPBACK}:41234/health"


def test_a_text_only_model_is_launched_without_a_projector(tmp_path: Path) -> None:
    command = launch_of(tmp_path, TEXT_ONLY).command()

    assert "--mmproj" not in command


def test_the_key_goes_in_the_environment_never_on_the_command_line(
    tmp_path: Path,
) -> None:
    launch = launch_of(tmp_path, TEXT_ONLY)

    environment = launch.environment({"PATH": "somewhere"})

    assert environment == {"PATH": "somewhere", API_KEY_VARIABLE: SECRET}
    assert all(SECRET not in arg for arg in launch.command())
    assert SECRET not in repr(launch)


def test_every_local_backend_but_auto_names_a_catalogue_backend() -> None:
    names = {b.value for b in LocalBackend} - {LocalBackend.AUTO.value}

    assert names == {b.value for b in Backend}


def test_the_default_local_model_is_in_the_catalogue() -> None:
    assert isinstance(find(DEFAULT_LOCAL_MODEL), Model)


def fake_server(
    tmp_path: Path,
    client: httpx.AsyncClient,
    on_launch: Callable[[Launch], None] = lambda _: None,
    *,
    start_timeout_s: float = DEFAULT_START_TIMEOUT_S,
    stop_timeout_s: float = DEFAULT_STOP_TIMEOUT_S,
) -> LlamaServer:
    touch(tmp_path / "runtime" / "llama-server")

    def launch() -> Launch:
        made = Launch.of(
            Backend.CPU,
            tmp_path / "runtime",
            TEXT_ONLY,
            tmp_path / "model",
            port=free_port(),
            context=512,
            key=new_key(),
        )
        on_launch(made)
        return made

    return LlamaServer(
        launch,
        client,
        tmp_path / "logs" / "llama-server.log",
        command=fake_llama_server.command,
        poll_s=0.05,
        start_timeout_s=start_timeout_s,
        stop_timeout_s=stop_timeout_s,
    )


async def assert_refused(url: str) -> None:
    # A fresh client: a pooled connection from before the stop fails with a
    # ReadError instead, depending on timing.
    async with httpx.AsyncClient() as fresh:
        with pytest.raises(httpx.ConnectError):
            await fresh.get(url)


async def serve(server: LlamaServer) -> tuple[asyncio.Event, asyncio.Task[None]]:
    stop = asyncio.Event()
    task = asyncio.create_task(server.run(stop))
    await asyncio.wait_for(server.ready.wait(), HANG_TIMEOUT_S)
    return stop, task


async def test_the_server_is_ready_once_health_answers(tmp_path: Path) -> None:
    async with httpx.AsyncClient() as client:
        server = fake_server(tmp_path, client)
        stop, task = await serve(server)
        running = server.running
        assert running is not None
        health = await client.get(running.health_url)
        stop.set()
        await asyncio.wait_for(task, HANG_TIMEOUT_S)

    assert health.json() == {"status": "ok"}
    assert server.running is None
    assert not server.ready.is_set()


async def test_stopping_ends_the_process(tmp_path: Path) -> None:
    async with httpx.AsyncClient() as client:
        server = fake_server(tmp_path, client)
        stop, task = await serve(server)
        running = server.running
        assert running is not None
        stop.set()
        await asyncio.wait_for(task, HANG_TIMEOUT_S)
        await assert_refused(running.health_url)


# On Windows the same comes from a new process group, which has no query to
# test it by; a Ctrl+C sent to the console showed the server surviving it.
@pytest.mark.skipif(sys.platform == "win32", reason="sessions are POSIX")
async def test_the_server_leads_its_own_session_so_ctrl_c_misses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_REPORT_SESSION", "1")
    async with httpx.AsyncClient() as client:
        stop, task = await serve(fake_server(tmp_path, client))
        stop.set()
        await asyncio.wait_for(task, HANG_TIMEOUT_S)

    log = (tmp_path / "logs" / "llama-server.log").read_text()
    assert "own session: True" in log


async def test_ready_waits_while_the_model_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_LOAD_S", "0.6")
    async with httpx.AsyncClient() as client:
        server = fake_server(tmp_path, client)
        stop = asyncio.Event()
        task = asyncio.create_task(server.run(stop))
        await asyncio.sleep(0.3)
        loading = server.ready.is_set()
        await asyncio.wait_for(server.ready.wait(), HANG_TIMEOUT_S)
        stop.set()
        await asyncio.wait_for(task, HANG_TIMEOUT_S)

    assert not loading


async def test_an_exit_before_ready_is_an_error_naming_the_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_EXIT_CODE", "3")
    async with httpx.AsyncClient() as client:
        server = fake_server(tmp_path, client)
        with pytest.raises(ServerError, match="code 3 before it was ready"):
            await asyncio.wait_for(server.run(asyncio.Event()), HANG_TIMEOUT_S)


async def test_a_server_that_never_gets_ready_is_stopped_after_the_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_LOAD_S", "60")
    async with httpx.AsyncClient() as client:
        server = fake_server(tmp_path, client, start_timeout_s=0.5)
        with pytest.raises(ServerError, match=r"not ready within 0\.5 s"):
            await asyncio.wait_for(server.run(asyncio.Event()), HANG_TIMEOUT_S)


async def test_a_crash_while_serving_is_raised_for_the_supervisor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_EXIT_AFTER_S", "0.2")
    async with httpx.AsyncClient() as client:
        server = fake_server(tmp_path, client)
        stop, task = await serve(server)
        with pytest.raises(ServerCrashedError, match="code 9 while serving"):
            await asyncio.wait_for(task, HANG_TIMEOUT_S)
        stop.set()

    assert server.running is None


async def test_a_stop_during_loading_returns_without_serving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_LOAD_S", "60")
    async with httpx.AsyncClient() as client:
        server = fake_server(tmp_path, client)
        stop = asyncio.Event()
        task = asyncio.create_task(server.run(stop))
        await asyncio.sleep(0.3)
        stop.set()
        await asyncio.wait_for(task, HANG_TIMEOUT_S)

    assert not server.ready.is_set()


async def test_the_supervisor_restarts_a_crashed_server_on_a_new_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_EXIT_AFTER_S", "0.2")
    launches: list[Launch] = []
    relaunched = asyncio.Event()

    def record(launch: Launch) -> None:
        launches.append(launch)
        if len(launches) == 2:
            relaunched.set()

    async with httpx.AsyncClient() as client:
        server = fake_server(tmp_path, client, record)
        supervisor = Supervisor(default_policy=RestartPolicy(backoff_initial_s=0.01))
        supervisor.add(server)
        supervised = asyncio.create_task(supervisor.run())
        await asyncio.wait_for(relaunched.wait(), HANG_TIMEOUT_S)
        await asyncio.wait_for(server.ready.wait(), HANG_TIMEOUT_S)
        serving = server.running
        supervisor.stop()
        await asyncio.wait_for(supervised, HANG_TIMEOUT_S)

    first, second = launches
    assert serving is second
    assert first.port != second.port
    assert first.key.get_secret_value() != second.key.get_secret_value()


async def test_a_server_that_outlasts_the_stop_timeout_is_killed(
    tmp_path: Path,
) -> None:
    async with httpx.AsyncClient() as client:
        server = fake_server(tmp_path, client, stop_timeout_s=0)
        stop, task = await serve(server)
        running = server.running
        assert running is not None
        stop.set()
        await asyncio.wait_for(task, HANG_TIMEOUT_S)
        await assert_refused(running.health_url)
