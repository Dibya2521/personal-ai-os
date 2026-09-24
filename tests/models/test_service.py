import asyncio
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from synthia.gateway.protocol import collect
from synthia.gateway.types import ChatRequest, Message
from synthia.kernel.config import LocalBackend, Settings
from synthia.kernel.supervisor import RestartPolicy
from synthia.models.catalogue import Backend, Download, Model, Runtime, Target
from synthia.models.install import COMPLETE_MARKER, Installer
from synthia.models.service import THREAD_NAME, LocalService, LocalSetup, find_local
from tests.models import fake_llama_server

WAIT_S = 10
URL = "https://example.com/file"
TINY = Model(
    "tiny", "example/tiny", "0" * 40, "mit", Download("tiny.gguf", URL, 4, "0" * 64)
)
CPU = Runtime(Target.WINDOWS_X64, Backend.CPU, (Download("cpu.zip", URL, 1, "0" * 64),))
VULKAN = Runtime(
    Target.WINDOWS_X64, Backend.VULKAN, (Download("vulkan.zip", URL, 1, "0" * 64),)
)
LINUX_CPU = Runtime(
    Target.LINUX_X64, Backend.CPU, (Download("linux.zip", URL, 1, "0" * 64),)
)
CATALOGUE = (CPU, LINUX_CPU, VULKAN, TINY)
HELLO = ChatRequest((Message.user("hello"),))


def install(installer: Installer, *items: Runtime | Model) -> None:
    for item in items:
        path = installer.path_of(item)
        path.mkdir(parents=True, exist_ok=True)
        if isinstance(item, Model):
            for file in item.files:
                (path / file.name).write_bytes(b"x" * file.size)
        else:
            (path / "llama-server").write_bytes(b"")
            (path / COMPLETE_MARKER).write_bytes(b"")


def settings(tmp_path: Path, backend: LocalBackend = LocalBackend.AUTO) -> Settings:
    return Settings(
        home=tmp_path, local_model=TINY.id, local_backend=backend, local_context=512
    )


def setup_at(tmp_path: Path, *items: Runtime | Model) -> LocalSetup | None:
    installer = Installer(tmp_path, 1 << 30)
    install(installer, *items)
    return find_local(settings(tmp_path), installer, Target.WINDOWS_X64, CATALOGUE)


def test_the_installed_builds_for_this_machine_are_used_fastest_first(
    tmp_path: Path,
) -> None:
    setup = setup_at(tmp_path, CPU, LINUX_CPU, VULKAN, TINY)

    assert setup is not None
    assert setup.runtimes == (VULKAN, CPU)
    assert (setup.model, setup.context) == (TINY, 512)


@pytest.mark.parametrize(
    "items",
    [(CPU,), (LINUX_CPU, TINY), ()],
    ids=["no model", "no build for this machine", "nothing"],
)
def test_without_the_model_or_a_build_it_stays_remote_only(
    tmp_path: Path, items: tuple[Runtime | Model, ...]
) -> None:
    assert setup_at(tmp_path, *items) is None


def test_a_model_not_in_the_catalogue_stays_remote_only(tmp_path: Path) -> None:
    installer = Installer(tmp_path, 1 << 30)
    install(installer, CPU, TINY)
    unknown = Settings(home=tmp_path, local_model="not-catalogued")

    assert find_local(unknown, installer, Target.WINDOWS_X64, CATALOGUE) is None


def test_a_chosen_backend_that_is_not_installed_stays_remote_only(
    tmp_path: Path,
) -> None:
    installer = Installer(tmp_path, 1 << 30)
    install(installer, CPU, TINY)
    cuda = settings(tmp_path, LocalBackend.CUDA)

    assert find_local(cuda, installer, Target.WINDOWS_X64, CATALOGUE) is None


def test_a_launch_uses_the_installed_paths_and_a_fresh_port_and_key(
    tmp_path: Path,
) -> None:
    setup = setup_at(tmp_path, CPU, TINY)
    assert setup is not None

    first, second = setup.launch(CPU), setup.launch(CPU)

    assert first.binary == setup.installer.path_of(CPU) / "llama-server"
    assert first.weights == setup.installer.path_of(TINY) / "tiny.gguf"
    assert first.context == 512
    assert first.key.get_secret_value() != second.key.get_secret_value()


def wait_until(condition: Callable[[], bool]) -> None:
    # The server runs on another thread's loop; there is no event to await here.
    deadline = time.monotonic() + WAIT_S
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


def service_at(tmp_path: Path, policy: RestartPolicy | None = None) -> LocalService:
    setup = setup_at(tmp_path, CPU, TINY)
    assert setup is not None
    return LocalService(
        setup,
        tmp_path / "llama-server.log",
        command=fake_llama_server.command,
        policy=policy,
    )


def service_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == THREAD_NAME]


async def ask(service: LocalService) -> str:
    async with httpx.AsyncClient() as client:
        return (await collect(service.model(client).stream(HELLO))).text


def test_the_server_serves_from_its_own_thread_until_stopped(tmp_path: Path) -> None:
    service = service_at(tmp_path)
    service.start()
    wait_until(lambda: service.server.running is not None)
    answer = asyncio.run(ask(service))
    service.stop()

    assert answer == "echo: hello"
    assert service.server.running is None
    assert not service_threads()


def test_stopping_a_service_never_started_does_nothing(tmp_path: Path) -> None:
    service_at(tmp_path).stop()

    assert not service_threads()


def test_a_server_that_never_starts_is_given_up_and_the_thread_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("FAKE_EXIT_CODE", "3")
    quick = RestartPolicy(max_restarts=1, backoff_initial_s=0.01)
    service = service_at(tmp_path, quick)
    with caplog.at_level(logging.ERROR, "synthia.models.service"):
        service.start()
        wait_until(lambda: not service_threads())
        service.stop()

    assert "not restarted again" in caplog.text
    assert service.server.running is None
