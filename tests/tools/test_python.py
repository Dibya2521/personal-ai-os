import asyncio
import json
import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from synthia.agent.tools import Effect, InvalidArgumentsError, Reach, ToolError
from synthia.tools.python import (
    MAX_CODE_CHARS,
    MAX_OUTPUT_BYTES,
    PYTHON_TIMEOUT_S,
    child_environment,
    python_tool,
    run_code,
)
from tests.timing import HANG_TIMEOUT_S


async def ran(
    code: str,
    *,
    timeout_s: float = PYTHON_TIMEOUT_S,
    max_output_bytes: int = MAX_OUTPUT_BYTES,
) -> str:
    tool = python_tool(timeout_s=timeout_s, max_output_bytes=max_output_bytes)
    return await asyncio.wait_for(tool.run(json.dumps({"code": code})), HANG_TIMEOUT_S)


async def test_what_the_code_prints_is_the_result() -> None:
    assert await ran("print(6 * 7)\nprint('done')") == "42\ndone\n"


async def test_an_error_comes_back_with_its_traceback() -> None:
    with pytest.raises(ToolError) as caught:
        await ran("print('before')\nprint(1 / 0)")

    text = str(caught.value)
    assert text.startswith("exit code 1:\nbefore\nTraceback (most recent call last):\n")
    assert text.endswith("\nZeroDivisionError: division by zero\n")


async def test_a_quiet_failure_says_so() -> None:
    with pytest.raises(ToolError) as caught:
        await ran("raise SystemExit(3)")

    assert str(caught.value) == "exit code 3:\n(nothing printed)"


async def test_a_quiet_success_says_so() -> None:
    assert await ran("x = 1") == "(nothing printed)"


async def test_output_is_utf8_whatever_the_console() -> None:
    assert await ran("print('snow \\N{SNOWMAN}')") == "snow \N{SNOWMAN}\n"


async def test_code_running_too_long_is_stopped_with_what_it_printed() -> None:
    code = "import time\nprint('started', flush=True)\ntime.sleep(60)"

    with pytest.raises(ToolError) as caught:
        await ran(code, timeout_s=3)

    assert str(caught.value) in {
        "stopped after 3 s; printed so far:\nstarted\n",
        # On a loaded machine the interpreter may not even start within 3 s.
        "stopped after 3 s; printed so far:\n(nothing printed)",
    }


async def test_output_past_the_limit_stops_the_code() -> None:
    line = "x" * 999 + os.linesep
    expected = (line * 6).encode()[:5000].decode().replace("\r\n", "\n")

    with pytest.raises(ToolError) as caught:
        await ran(f"while True:\n    print({'x' * 999!r})", max_output_bytes=5000)

    assert str(caught.value) == (
        f"stopped: the output passed 5,000 bytes; the first 5,000:\n{expected}"
    )


async def test_code_starts_in_a_new_empty_directory_deleted_after() -> None:
    printed = await ran("import os\nprint(os.listdir('.'))\nprint(os.getcwd())")

    listing, where, _ = printed.split("\n")
    assert listing == "[]"
    assert not Path(where).exists()


async def test_the_code_sees_none_of_the_persons_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYNTHIA_OPENROUTER_API_KEY", "not-a-real-key")
    code = "import os, json\nprint(json.dumps(dict(os.environ)))"

    seen = json.loads(await ran(code))

    # Python itself adds LC_CTYPE on POSIX when the locale is C.
    seen.pop("LC_CTYPE", None)
    assert seen == child_environment(Path(seen["TEMP"]))


async def test_temporary_files_go_into_the_working_directory() -> None:
    code = "import os, tempfile\nprint(tempfile.gettempdir() == os.getcwd())"

    assert await ran(code) == "True\n"


type Connections = asyncio.Queue[asyncio.StreamReader]


@asynccontextmanager
async def _listening() -> AsyncGenerator[tuple[int, Connections]]:
    """Yield a local port and a queue of the connections made to it."""
    connected: Connections = asyncio.Queue()
    writers: list[asyncio.StreamWriter] = []

    async def on_connect(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        writers.append(writer)
        await connected.put(reader)

    async with await asyncio.start_server(on_connect, "127.0.0.1", 0) as server:
        try:
            yield server.sockets[0].getsockname()[1], connected
        finally:
            # Leaving the server waits for every connection it accepted to close.
            for writer in writers:
                writer.close()
            await asyncio.gather(
                *(writer.wait_closed() for writer in writers), return_exceptions=True
            )


def _with_grandchild(port: int, then: str) -> str:
    """Return code that starts a process holding a connection, then runs ``then``."""
    holder = (
        "import socket, time; "
        f"s = socket.create_connection(('127.0.0.1', {port})); "
        "print('up', flush=True); time.sleep(60)"
    )
    return (
        "import subprocess, sys\n"
        f"command = [sys.executable, '-c', {holder!r}]\n"
        "p = subprocess.Popen(command, stdout=subprocess.PIPE)\n"
        "p.stdout.readline()\n"
        f"{then}\n"
    )


async def _closed(reader: asyncio.StreamReader) -> bool:
    try:
        return await asyncio.wait_for(reader.read(), HANG_TIMEOUT_S) == b""
    except ConnectionError:
        return True


async def test_a_process_the_code_left_running_is_ended_when_it_exits() -> None:
    async with _listening() as (port, connected):
        run = await asyncio.wait_for(
            run_code(_with_grandchild(port, "print('leaving')")), HANG_TIMEOUT_S
        )
        reader = await asyncio.wait_for(connected.get(), HANG_TIMEOUT_S)

        assert (run.output, run.exit_code) == ("leaving\n", 0)
        assert await _closed(reader)


async def test_cancelling_a_run_ends_every_process_it_started() -> None:
    async with _listening() as (port, connected):
        code = _with_grandchild(port, "import time\ntime.sleep(60)")
        task = asyncio.create_task(run_code(code, timeout_s=HANG_TIMEOUT_S))
        reader = await asyncio.wait_for(connected.get(), HANG_TIMEOUT_S)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert await _closed(reader)


def test_the_tool_asks_every_call_and_takes_one_program() -> None:
    tool = python_tool()

    assert (tool.spec.name, tool.reach, tool.effect) == (
        "run_python",
        Reach.LOCAL,
        Effect.CHANGE,
    )
    assert tool.spec.parameters == {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_CODE_CHARS,
                "description": "a whole Python 3 program; print what you need back",
            }
        },
        "required": ["code"],
        "additionalProperties": False,
    }


@pytest.mark.parametrize(
    "code", ["", "x" * (MAX_CODE_CHARS + 1)], ids=["empty", "oversized"]
)
async def test_empty_or_oversized_code_is_refused_before_anything_starts(
    code: str, tmp_path: Path
) -> None:
    # A start would fail differently: this executable does not exist.
    tool = python_tool(executable=str(tmp_path / "no-python"))

    with pytest.raises(InvalidArgumentsError, match="code"):
        await tool.run(json.dumps({"code": code}))


@pytest.mark.skipif(sys.platform != "win32", reason="SYSTEMROOT is a Windows variable")
def test_windows_code_gets_systemroot_or_python_cannot_start(tmp_path: Path) -> None:
    assert child_environment(tmp_path)["SYSTEMROOT"] == os.environ["SYSTEMROOT"]
