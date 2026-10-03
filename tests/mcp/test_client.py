import asyncio
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from synthia.agent.tools import Effect, Reach, ToolError
from synthia.kernel.errors import ConfigError
from synthia.mcp.client import (
    MAX_RESULT_CHARS,
    McpError,
    McpServer,
    McpServers,
    McpTool,
    ServerConfig,
    load_config,
    result_text,
)
from tests.timing import HANG_TIMEOUT_S

FAKE = Path(__file__).with_name("fake_mcp_server.py")


def fake(*flags: str, **config: object) -> ServerConfig:
    return ServerConfig.model_validate(
        {"command": [sys.executable, str(FAKE), *flags], **config}
    )


@pytest.fixture
async def server(tmp_path: Path) -> AsyncIterator[McpServer]:
    started = await asyncio.wait_for(
        McpServer.start("fake", fake(), tmp_path), HANG_TIMEOUT_S
    )
    yield started
    await asyncio.wait_for(started.stop(), HANG_TIMEOUT_S)


def tool(server: McpServer, name: str) -> McpTool:
    return next(t for t in server.tools if t.spec.name == f"fake__{name}")


async def call(server: McpServer, name: str, arguments: object = None) -> str:
    return await asyncio.wait_for(
        tool(server, name).run(json.dumps(arguments or {})), HANG_TIMEOUT_S
    )


async def test_the_servers_tools_join_under_its_name_and_ask_every_call(
    server: McpServer,
) -> None:
    assert [(t.spec.name, t.reach, t.effect) for t in server.tools] == [
        (f"fake__{name}", Reach.OUTSIDE, Effect.CHANGE)
        for name in (
            "echo",
            "add",
            "fail",
            "die",
            "slow",
            "cancelled",
            "ask_back",
            "env",
            "big",
            "mixed",
        )
    ]
    echo = tool(server, "echo").spec
    assert (echo.description, echo.parameters) == (
        "Say the text back.",
        {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )
    assert tool(server, "fail").spec.description == "fail from fake"


async def test_a_call_returns_the_tools_text(server: McpServer) -> None:
    assert await call(server, "echo", {"text": "hello"}) == "hello"
    assert await call(server, "add", {"a": 2, "b": 3}) == "5"


async def test_an_error_result_and_a_refused_call_become_tool_errors(
    server: McpServer,
) -> None:
    with pytest.raises(ToolError, match=r"^it failed$"):
        await call(server, "fail")
    with pytest.raises(ToolError, match=r"^fake refused: unknown tool$"):
        await server.call("missing", {})


@pytest.mark.parametrize("arguments", ["[1]", "not json"])
async def test_arguments_must_be_a_json_object(
    server: McpServer, arguments: str
) -> None:
    with pytest.raises(ToolError, match="arguments"):
        await tool(server, "echo").run(arguments)


async def test_the_servers_own_requests_are_answered(server: McpServer) -> None:
    assert await call(server, "ask_back") == "ping {} roots -32601"


async def test_a_cancelled_call_is_cancelled_on_the_server_too(
    server: McpServer,
) -> None:
    slow = asyncio.create_task(call(server, "slow"))
    await asyncio.sleep(0.5)
    slow.cancel()
    with pytest.raises(asyncio.CancelledError):
        await slow

    cancelled = json.loads(await call(server, "cancelled"))
    assert len(cancelled) == 1


async def test_a_server_that_dies_fails_the_call_and_every_later_one(
    server: McpServer,
) -> None:
    with pytest.raises(
        ToolError, match=r"^fake is not running: fake exited with code 3$"
    ):
        await call(server, "die")
    with pytest.raises(ToolError, match="not running"):
        await call(server, "echo", {"text": "again"})


async def test_the_server_sees_none_of_synthias_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SYNTHIA_OPENROUTER_API_KEY", "not-a-real-key")
    started = await McpServer.start("fake", fake(env={"FAKE_EXTRA": "given"}), tmp_path)
    try:
        assert json.loads(await call(started, "env")) == [[], "given"]
    finally:
        await started.stop()


async def test_long_results_are_cut_and_other_parts_are_named(
    server: McpServer,
) -> None:
    big = await call(server, "big")
    assert big == "x" * MAX_RESULT_CHARS + "\n[cut: 30,000 characters in all]"
    assert await call(server, "mixed") == "[image image/png]\na caption"


def test_result_text_names_every_kind_of_part() -> None:
    result = {
        "content": [
            {"type": "audio", "mimeType": "audio/wav"},
            {"type": "resource", "resource": {"uri": "file:///a", "text": "inline"}},
            {"type": "resource", "resource": {"uri": "file:///b"}},
            {"type": "resource_link", "uri": "file:///c"},
            {"type": "video"},
            "not a part",
        ]
    }

    assert result_text(result).split("\n") == [
        "[audio audio/wav]",
        "inline",
        "[resource file:///b]",
        "[link file:///c]",
        "[content of type 'video']",
    ]
    assert result_text({"structuredContent": {"n": 1}}) == '{"n": 1}'
    assert result_text({}) == "(no content)"


async def test_stderr_goes_to_the_log_and_lines_that_are_not_messages_are_skipped(
    tmp_path: Path,
) -> None:
    started = await McpServer.start("fake", fake("--garbage"), tmp_path)
    try:
        assert await call(started, "echo", {"text": "still here"}) == "still here"
    finally:
        await started.stop()
    assert (tmp_path / "fake.log").read_text() == "fake mcp server started\n"


async def test_odd_listings_cost_only_the_odd_entries(tmp_path: Path) -> None:
    started = await McpServer.start("fake", fake("--odd"), tmp_path)
    try:
        assert [(t.spec.name, t.spec.description) for t in started.tools] == [
            ("fake__echo", "Say the text back.")
        ]
    finally:
        await started.stop()


async def test_a_file_that_is_not_a_program_is_refused(tmp_path: Path) -> None:
    junk = tmp_path / "junk.exe"
    junk.write_bytes(b"not a program")
    junk.chmod(0o755)

    with pytest.raises(McpError, match=r"^junk: could not start: "):
        await McpServer.start("junk", ServerConfig(command=[str(junk)]), tmp_path)


class BugError(Exception):
    pass


async def test_an_unexpected_error_starting_a_server_is_not_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken(*_: object, **__: object) -> McpServer:
        raise BugError

    monkeypatch.setattr(McpServer, "start", broken)

    with pytest.raises(BugError):
        await McpServers.start({"fake": fake()}, tmp_path)


async def test_tools_listed_in_pages_are_all_read(tmp_path: Path) -> None:
    started = await McpServer.start("fake", fake("--pages"), tmp_path)
    try:
        assert len(started.tools) == 10
    finally:
        await started.stop()


@pytest.mark.parametrize(
    ("flags", "reason"),
    [
        (
            ("--version", "1999-01-01"),
            "^fake: protocol version '1999-01-01' is not supported$",
        ),
        (("--exit-at-start",), "^fake: fake exited with code 2$"),
    ],
    ids=["old-version", "exits"],
)
async def test_a_server_that_cannot_complete_the_handshake_is_refused(
    tmp_path: Path, flags: tuple[str, ...], reason: str
) -> None:
    with pytest.raises(McpError, match=reason):
        await asyncio.wait_for(
            McpServer.start("fake", fake(*flags), tmp_path), HANG_TIMEOUT_S
        )


async def test_a_silent_server_is_stopped_after_the_start_timeout(
    tmp_path: Path,
) -> None:
    with pytest.raises(McpError, match=r"^fake: no handshake within 1 s$"):
        await asyncio.wait_for(
            McpServer.start("fake", fake("--silent"), tmp_path, start_timeout_s=1),
            HANG_TIMEOUT_S,
        )


async def test_a_missing_program_is_named(tmp_path: Path) -> None:
    config = ServerConfig(command=["no-such-mcp-server-anywhere"])

    with pytest.raises(
        McpError, match=r"^gone: no-such-mcp-server-anywhere was not found$"
    ):
        await McpServer.start("gone", config, tmp_path)


async def test_a_log_that_cannot_be_opened_is_a_failure_not_a_crash(
    tmp_path: Path,
) -> None:
    not_a_folder = tmp_path / "logs"
    not_a_folder.write_text("a file where the log folder should be")

    servers = await asyncio.wait_for(
        McpServers.start({"fake": fake()}, not_a_folder), HANG_TIMEOUT_S
    )

    assert servers.servers == []
    assert len(servers.failures) == 1
    assert servers.failures[0].startswith("fake: cannot open its log: ")


async def test_servers_that_fail_leave_the_others_working(tmp_path: Path) -> None:
    configs = {
        "good": fake(),
        "old": fake("--version", "1999-01-01"),
        "gone": ServerConfig(command=["no-such-mcp-server-anywhere"]),
    }
    servers = await asyncio.wait_for(
        McpServers.start(configs, tmp_path), HANG_TIMEOUT_S
    )
    try:
        assert [s.name for s in servers.servers] == ["good"]
        assert sorted(servers.failures) == [
            "gone: no-such-mcp-server-anywhere was not found",
            "old: protocol version '1999-01-01' is not supported",
        ]
        assert [t.spec.name for t in servers.tools()][:2] == ["good__echo", "good__add"]
    finally:
        await servers.stop()


def test_the_config_file_lists_servers_with_their_reach(tmp_path: Path) -> None:
    path = tmp_path / "mcp.toml"
    path.write_text(
        '[servers.notes]\ncommand = ["notes-server", "--root", "x"]\n'
        'reach = "local"\neffect = "read"\nenv = { NOTES = "1" }\n',
        encoding="utf-8",
    )

    assert load_config(path) == {
        "notes": ServerConfig(
            command=["notes-server", "--root", "x"],
            env={"NOTES": "1"},
            reach=Reach.LOCAL,
            effect=Effect.READ,
        )
    }
    assert load_config(tmp_path / "absent.toml") == {}


@pytest.mark.parametrize(
    "text",
    [
        "[servers.notes\n",
        "[servers.notes]\ncommand = []\n",
        '[servers.notes]\ncommand = ["x"]\nreach = "everywhere"\n',
        '[servers.no__double]\ncommand = ["x"]\n',
        '[servers.notes]\ncommand = ["x"]\ncolour = "red"\n',
    ],
    ids=["toml", "empty-command", "reach", "name", "unknown-key"],
)
def test_an_invalid_config_file_is_refused_with_its_path(
    tmp_path: Path, text: str
) -> None:
    path = tmp_path / "mcp.toml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="is not a valid MCP server list"):
        load_config(path)
