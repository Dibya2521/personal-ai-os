import asyncio
import contextlib
import os
import re
import socket
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr
from typer.testing import CliRunner

from synthia import __version__
from synthia.agent.plan import PlanAnswerBegun, Planned, PlanStepBegun
from synthia.gateway.errors import GatewayError
from synthia.gateway.types import ChatChunk, ChatRequest, PromptProgress
from synthia.interfaces import cli, daemon_client, doctor
from synthia.interfaces.cli import app
from synthia.interfaces.daemon_client import (
    STOPPED_AT_START,
    DaemonError,
    conversation,
    current_daemon,
    find_or_start,
    healthy,
    running_daemon,
    start_daemon,
    stopped,
)
from synthia.kernel.config import Settings
from synthia.memory.store import MEMORY_FILE, MemoryStore
from synthia.retrieval.embed import TextEmbedder
from synthia.server import daemon as daemon_module
from synthia.server.api import NOT_LEARNING, mark_of
from synthia.server.discovery import (
    DAEMON_FILE,
    DaemonInfo,
    new_token,
    read_info,
    started_now,
)
from synthia.server.session import ImageError, PlanMark
from tests.agent.scripted import Reply
from tests.interfaces.daemons import private_daemon
from tests.interfaces.test_chat import KEY
from tests.memory.test_facts import SAME, held_facts
from tests.memory.test_hybrid import Network, embedder
from tests.server.test_daemon import Local, Service
from tests.timing import HANG_TIMEOUT_S


def keyed(home: Path) -> Settings:
    return Settings(home=home, openrouter_api_key=SecretStr(KEY))


class Process:
    """Stands in for a started daemon process: running until told otherwise."""

    def __init__(self, code: int | None = None) -> None:
        self.code = code

    def poll(self) -> int | None:
        return self.code


async def refuse(_tool: str, _arguments: str) -> bool:
    return False


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


async def test_a_running_daemon_is_found_and_none_is_started(tmp_path: Path) -> None:
    def start() -> Process:
        pytest.fail("a running daemon must not be started again")

    with private_daemon(keyed(tmp_path)) as info:
        found = await find_or_start(tmp_path, start=start)  # type: ignore[arg-type]

    assert found == info


async def test_a_missing_daemon_is_started_and_waited_for(tmp_path: Path) -> None:
    with contextlib.ExitStack() as stack:
        started: list[DaemonInfo] = []

        def start() -> Process:
            started.append(stack.enter_context(private_daemon(keyed(tmp_path))))
            return Process()

        found = await find_or_start(tmp_path, start=start)  # type: ignore[arg-type]

    assert [found] == started


@pytest.mark.parametrize(
    ("process", "timeout_s", "problem"),
    [
        (Process(code=1), HANG_TIMEOUT_S, f"^{re.escape(STOPPED_AT_START)}$"),
        (Process(), 0.3, r"^the daemon did not start within 0\.3 s"),
    ],
    ids=["stopped", "never-ready"],
)
async def test_a_daemon_that_never_serves_is_reported(
    tmp_path: Path, process: Process, timeout_s: float, problem: str
) -> None:
    with pytest.raises(DaemonError, match=problem):
        await find_or_start(tmp_path, start=lambda: process, timeout_s=timeout_s)  # type: ignore[arg-type,return-value]


async def test_a_file_whose_port_is_closed_is_no_daemon(tmp_path: Path) -> None:
    gone = DaemonInfo(
        port=free_port(),
        pid=os.getpid(),
        version=__version__,
        started=started_now(),
        token=new_token(),
    )
    (tmp_path / DAEMON_FILE).write_text(gone.model_dump_json())

    assert await running_daemon(tmp_path) is None


async def test_a_daemon_of_this_version_is_used_as_it_is(tmp_path: Path) -> None:
    with private_daemon(keyed(tmp_path)) as info:
        assert await current_daemon(tmp_path) == info


async def test_an_idle_daemon_of_another_version_is_restarted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(daemon_client, "__version__", "0.0.0-older")
    with contextlib.ExitStack() as stack:
        old = stack.enter_context(private_daemon(keyed(tmp_path)))
        new: list[DaemonInfo] = []

        def start() -> Process:
            new.append(stack.enter_context(private_daemon(keyed(tmp_path))))
            return Process()

        found = await current_daemon(tmp_path, start=start)  # type: ignore[arg-type]

    assert [found] == new
    assert found.token != old.token


async def test_a_busy_daemon_of_another_version_is_left_and_explained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(daemon_client, "__version__", "0.0.0-older")
    with private_daemon(keyed(tmp_path)) as info:
        async with conversation(info, refuse):
            with pytest.raises(DaemonError, match="run `synthia stop`"):
                await current_daemon(tmp_path)
        still = read_info(tmp_path / DAEMON_FILE)

    assert still == info


async def test_a_daemon_that_will_not_stop_is_reported(tmp_path: Path) -> None:
    with (
        private_daemon(keyed(tmp_path)) as info,
        pytest.raises(DaemonError, match=r"did not stop within 0\.2 s"),
    ):
        await stopped(tmp_path, info, 0.2)


async def test_an_image_the_daemon_cannot_send_is_an_image_error(
    tmp_path: Path,
) -> None:
    with private_daemon(keyed(tmp_path)) as info:
        async with conversation(info, refuse) as talk:
            with pytest.raises(ImageError, match=r"missing\.png"):
                async for _ in talk.turn("see", images=(Path("missing.png"),)):
                    pass


async def test_a_daemon_that_goes_away_mid_conversation_is_a_failed_answer(
    tmp_path: Path,
) -> None:
    with private_daemon(keyed(tmp_path)) as info:
        async with conversation(info, refuse) as talk:
            await talk.stop_daemon()
            await asyncio.wait_for(talk.closed.wait(), HANG_TIMEOUT_S)
            with pytest.raises(GatewayError, match="the daemon went away"):
                async for _ in talk.turn("hello"):
                    pass


async def test_commands_reach_the_daemon_and_come_back_as_replies(
    tmp_path: Path,
) -> None:
    with private_daemon(keyed(tmp_path)) as info:
        async with conversation(info, refuse) as talk:
            adjusted = await talk.adjust({"wit": 0.9})
            tools = await talk.tools()
            reset = await talk.reset()

    assert adjusted.notes[0].startswith("now SYNTHIA: ")
    assert "wit=0.9" in adjusted.notes[0]
    assert any("current_time" in note for note in tools.notes)
    assert reset.notes == ("conversation forgotten",)


@pytest.mark.parametrize(
    "mark",
    [
        Planned(("find a", "use a"), revised=False),
        Planned((), revised=True),
        PlanStepBegun(2, "use a"),
        PlanAnswerBegun(),
    ],
    ids=["planned", "replanned", "step", "answer"],
)
def test_a_plan_mark_comes_back_as_it_was_sent(mark: PlanMark) -> None:
    assert daemon_client.mark_from(mark_of(mark)) == mark


def test_streamed_pieces_are_rebuilt_as_the_terminal_shows_them() -> None:
    chunk = daemon_client.chunk_from(
        {"text": "", "reasoning": "", "tool_calls": True, "progress": [2048, 2723]}
    )

    assert chunk.tool_calls
    assert chunk.progress == PromptProgress(2048, 2723)
    assert daemon_client.chunk_from({"text": "hi"}) == ChatChunk(text="hi")


def test_status_and_stop_from_the_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SYNTHIA_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    with private_daemon(keyed(tmp_path)) as info:
        running = runner.invoke(app, ["status"]).output.splitlines()
        stopping = runner.invoke(app, ["stop"])
        gone = not (tmp_path / DAEMON_FILE).exists()

    assert running[0].startswith(f"daemon running: version {__version__}, since ")
    assert running[0].endswith(f", pid {info.pid}, port {info.port}")
    assert running[1:] == ["other conversations open: 0", "local model: none"]
    assert (stopping.exit_code, stopping.output, gone) == (0, "daemon stopped\n", True)
    assert runner.invoke(app, ["status"]).output == "no daemon running\n"
    assert runner.invoke(app, ["stop"]).output == "no daemon running\n"


class Judging(Service):
    """The local model's service, with a model that answers ``replies``."""

    def __init__(self, *replies: Reply) -> None:
        super().__init__()
        self.replies = replies

    def model(self, client: httpx.AsyncClient) -> Local:
        del client
        return Local(*self.replies)


@pytest.fixture
def learns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A home whose daemon learns facts, holding two that say the same."""
    monkeypatch.setenv("SYNTHIA_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    def loaded(*_: object) -> TextEmbedder:
        return embedder(Network())

    monkeypatch.setattr(daemon_module, "open_embedder", loaded)
    store = MemoryStore(tmp_path / MEMORY_FILE)
    asyncio.run(held_facts(store, "likes green tea", "likes green tea"))
    return tmp_path


def test_facts_are_consolidated_from_the_command_line_and_traced(
    learns: Path,
) -> None:
    runner = CliRunner()
    with private_daemon(keyed(learns), local=Judging(SAME, SAME)):
        merged = runner.invoke(app, ["memory", "consolidate"])
        again = runner.invoke(app, ["memory", "consolidate"])
    listed = runner.invoke(app, ["trace", "--list"])

    assert merged.exit_code == 0
    assert re.fullmatch(
        r"1 pair of facts asked about, 1 merged, in \d+\.\d s\n", merged.stdout
    )
    assert again.stdout.startswith("0 pairs of facts asked about, 0 merged, in ")
    assert listed.stdout.rstrip().endswith("| service restarts 0 | consolidations 2")


def test_a_consolidation_the_local_model_fails_says_so_and_exits_2(
    learns: Path,
) -> None:
    def down(_: ChatRequest) -> list[ChatChunk]:
        message = "the local model is not running"
        raise GatewayError(message)

    with private_daemon(keyed(learns), local=Judging(down)):
        failed = CliRunner().invoke(app, ["memory", "consolidate"])

    assert failed.exit_code == doctor.Status.FAIL
    assert failed.stdout.startswith("1 pair of facts asked about, 0 merged, in ")
    assert "error: " in failed.stderr
    assert "the local model is not running" in failed.stderr


def test_consolidating_needs_a_running_daemon_that_learns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SYNTHIA_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    none_running = runner.invoke(app, ["memory", "consolidate"])
    with private_daemon(keyed(tmp_path)):
        not_learning = runner.invoke(app, ["memory", "consolidate"])

    assert (none_running.exit_code, not_learning.exit_code) == (2, 2)
    assert none_running.stderr == (
        "error: no daemon running; it holds the local model: "
        "start it with `synthia serve` or `synthia chat`\n"
    )
    assert not_learning.stderr == f"error: {NOT_LEARNING}\n"


def test_a_consolidation_stopped_with_ctrl_c_says_what_stays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def interrupted(_: Path) -> None:
        raise KeyboardInterrupt

    monkeypatch.setenv("SYNTHIA_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "consolidate_memory", interrupted)

    stopped = CliRunner().invoke(app, ["memory", "consolidate"])

    assert stopped.exit_code == doctor.Status.FAIL
    assert stopped.stdout == "stopped; facts merged so far stay merged\n"


async def test_the_real_command_starts_a_daemon_in_the_background(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SYNTHIA_HOME", str(tmp_path))
    monkeypatch.setenv("SYNTHIA_OPENROUTER_API_KEY", KEY)
    monkeypatch.chdir(tmp_path)

    process = start_daemon()
    try:
        async with asyncio.timeout(HANG_TIMEOUT_S):
            while (info := await running_daemon(tmp_path)) is None:
                assert process.poll() is None, "the daemon stopped while starting"
                await asyncio.sleep(0.1)
        async with conversation(info, refuse) as talk:
            await talk.stop_daemon()
        process.wait(HANG_TIMEOUT_S)
    finally:
        if process.poll() is None:
            process.kill()

    assert info.version == __version__
    assert not (tmp_path / DAEMON_FILE).exists()
    assert not await healthy(info)
