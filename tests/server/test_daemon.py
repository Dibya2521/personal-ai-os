import asyncio
import contextlib
import json
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

import httpx
import pytest
from pydantic import SecretStr
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import InvalidStatus

from synthia import __version__
from synthia.agent.tools import Effect, FunctionTool, Reach, Toolbox
from synthia.agent.trace import TRACES, read_trace, sessions
from synthia.gateway.assemble import build_gateway
from synthia.gateway.errors import ProviderError
from synthia.gateway.types import ChatChunk, ChatRequest
from synthia.kernel.bus import Event
from synthia.kernel.config import Settings
from synthia.kernel.jsonrpc import Method, Notice, Params, Peer, RpcError
from synthia.kernel.supervisor import ServiceEvent, ServiceStarted
from synthia.persona.library import PersonaLibrary
from synthia.server.api import ANSWER_FAILED, IMAGE_REFUSED, INVALID_PARAMS
from synthia.server.daemon import run_daemon, serve
from synthia.server.discovery import DAEMON_FILE, DaemonInfo, read_info
from synthia.server.host import Host, publish_route
from tests.agent.scripted import Scripted, calls, says
from tests.interfaces.test_chat import KEY
from tests.timing import HANG_TIMEOUT_S


class Local(Scripted):
    """A scripted model the gateway accepts as the local one."""

    def ready(self) -> bool:
        return True


class Failing(Local):
    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        del request
        message = "the model broke"
        raise ProviderError(message)
        yield ChatChunk()  # pragma: no cover


def edit_tool(ran: list[str]) -> FunctionTool:
    def edit(text: str) -> str:
        """Change something."""
        ran.append(text)
        return f"edited {text}"

    return FunctionTool.of(edit, reach=Reach.LOCAL, effect=Effect.CHANGE)


@asynccontextmanager
async def daemon(
    tmp_path: Path, model: Local, tools: Toolbox | None = None
) -> AsyncGenerator[DaemonInfo]:
    settings = Settings(home=tmp_path)
    async with httpx.AsyncClient() as client:
        gateway = build_gateway(settings, client, publish_route, model)
        host = Host(gateway, PersonaLibrary(), "synthia", tools or Toolbox())
        stop = asyncio.Event()
        task = asyncio.create_task(serve(host, tmp_path, stop=stop))
        info = await written(tmp_path, task)
        try:
            yield info
        finally:
            stop.set()
            await asyncio.wait_for(task, HANG_TIMEOUT_S)


async def written(home: Path, serving: asyncio.Task[None]) -> DaemonInfo:
    """Return the daemon's file once written; fail at once if serving ended."""
    async with asyncio.timeout(HANG_TIMEOUT_S):
        while (info := read_info(home / DAEMON_FILE)) is None:
            if serving.done():
                serving.result()
                pytest.fail("the daemon stopped before it wrote its file")
            await asyncio.sleep(0.02)
    return info


class Client:
    """A test's side of one conversation."""

    def __init__(
        self, socket: ClientConnection, methods: dict[str, Method] | None
    ) -> None:
        self.notes: list[tuple[str, Params]] = []
        self._outgoing: asyncio.Queue[bytes] = asyncio.Queue()
        notices: dict[str, Notice] = {
            name: self._keeping(name) for name in ("chunk", "tool", "plan")
        }
        self.peer = Peer(self._outgoing.put_nowait, methods=methods, notices=notices)
        self._socket = socket

    def _keeping(self, name: str) -> Notice:
        def keep(params: Params) -> None:
            self.notes.append((name, params))

        return keep

    async def run(self) -> None:
        async def write() -> None:
            while True:
                await self._socket.send((await self._outgoing.get()).decode())

        writer = asyncio.create_task(write())
        try:
            async for message in self._socket:
                self.peer.receive(message)
        finally:
            writer.cancel()
            self.peer.close("closed")

    async def call(self, method: str, params: Params | None = None) -> object:
        return await asyncio.wait_for(
            self.peer.request(method, params or {}), HANG_TIMEOUT_S
        )


@asynccontextmanager
async def client(
    info: DaemonInfo, methods: dict[str, Method] | None = None
) -> AsyncGenerator[Client]:
    headers = {"Authorization": f"Bearer {info.token}"}
    async with connect(info.socket_url, additional_headers=headers) as socket:
        talking = Client(socket, methods)
        reader = asyncio.create_task(talking.run())
        try:
            yield talking
        finally:
            await socket.close()
            await asyncio.gather(reader, return_exceptions=True)


async def test_health_answers_anyone_and_the_file_says_where(tmp_path: Path) -> None:
    async with daemon(tmp_path, Local()) as info, httpx.AsyncClient() as web:
        answer = await web.get(f"{info.base_url}/health")
        docs = await web.get(f"{info.base_url}/docs")

    assert answer.json() == {"status": "up"}
    assert docs.status_code == httpx.codes.NOT_FOUND
    assert (info.version, len(info.token) > 30) == (__version__, True)


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer wrong"}, {"Origin": "https://example.org"}],
    ids=["no-token", "wrong-token", "from-a-web-page"],
)
async def test_a_conversation_needs_the_token_and_no_web_page_origin(
    tmp_path: Path, headers: dict[str, str]
) -> None:
    async with daemon(tmp_path, Local()) as info:
        sent = {"Authorization": f"Bearer {info.token}"} | headers
        if "Authorization" not in headers and "Origin" not in headers:
            del sent["Authorization"]
        with pytest.raises(InvalidStatus):
            async with connect(info.socket_url, additional_headers=sent):
                pass


async def test_a_turn_streams_its_answer_and_reports_where_it_went(
    tmp_path: Path,
) -> None:
    async with daemon(tmp_path, Local(says("Hello."))) as info, client(info) as talk:
        hello = await talk.call("hello")
        report = await talk.call("turn", {"text": "hi"})

    assert hello == {
        "version": __version__,
        "persona": "SYNTHIA",
        "sessions": 1,
        "warnings": [],
    }
    assert isinstance(report, dict)
    assert (report["route"], report["model"]) == ("local", "scripted")
    texts = [p["text"] for name, p in talk.notes if name == "chunk"]
    assert "".join(str(t) for t in texts) == "Hello."


async def test_a_call_that_needs_a_yes_asks_the_client_that_asked_the_turn(
    tmp_path: Path,
) -> None:
    ran: list[str] = []
    asked: list[Params] = []

    async def approve(params: Params) -> object:
        asked.append(params)
        return params["arguments"] == '{"text": "yes"}'

    model = Local(
        calls(("edit", '{"text": "yes"}')),
        says("done"),
        calls(("edit", '{"text": "no"}')),
        says("refused"),
    )
    tools = Toolbox([edit_tool(ran)])
    async with (
        daemon(tmp_path, model, tools) as info,
        client(info, {"approve": approve}) as talk,
    ):
        await talk.call("turn", {"text": "edit yes"})
        await talk.call("turn", {"text": "edit no"})

    assert ran == ["yes"]
    assert asked == [
        {"tool": "edit", "arguments": '{"text": "yes"}'},
        {"tool": "edit", "arguments": '{"text": "no"}'},
    ]
    results = [p["result"] for name, p in talk.notes if name == "tool"]
    assert results == ["edited yes", "running edit was not approved"]


async def test_a_client_that_goes_away_mid_question_never_gets_the_call_run(
    tmp_path: Path,
) -> None:
    ran: list[str] = []
    waiting = asyncio.Event()

    async def approve(_: Params) -> object:
        waiting.set()
        await asyncio.Event().wait()
        return True  # pragma: no cover

    model = Local(calls(("edit", '{"text": "x"}')), says("done"))
    tools = Toolbox([edit_tool(ran)])
    async with daemon(tmp_path, model, tools) as info:
        async with client(info, {"approve": approve}) as talk:
            turn = asyncio.create_task(talk.peer.request("turn", {"text": "go"}))
            await asyncio.wait_for(waiting.wait(), HANG_TIMEOUT_S)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(turn, HANG_TIMEOUT_S)
        await asyncio.sleep(0.2)

    assert ran == []


async def test_each_client_has_its_own_conversation(tmp_path: Path) -> None:
    async with (
        daemon(tmp_path, Local()) as info,
        client(info) as first,
        client(info) as second,
    ):
        set_off = await first.call("think", {"level": "off"})
        unchanged = await second.call("think", {})

    assert set_off == {"notes": ["thinking: off"], "errors": []}
    assert unchanged == {"notes": ["thinking: auto"], "errors": []}


async def test_commands_answer_with_lines_to_show(tmp_path: Path) -> None:
    async with daemon(tmp_path, Local()) as info, client(info) as talk:
        replies = [
            await talk.call("remote", {"on": True}),
            await talk.call("persona", {"key": "nobody"}),
            await talk.call("tools"),
            await talk.call("reset"),
            await talk.call("private", {"on": True}),
            await talk.call("forget"),
        ]

    assert replies[4:] == [
        {"notes": ["nothing is remembered in this conversation"], "errors": []},
        {"notes": ["no turn to forget"], "errors": []},
    ]
    assert replies[:4] == [
        {
            "notes": [],
            "errors": [
                "no remote model is configured: set SYNTHIA_OPENROUTER_API_KEY in .env"
            ],
        },
        {
            "notes": [],
            "errors": [
                (
                    "no persona 'nobody'; there are glacier, horizon, minato, neon, "
                    "nova, starlight, synthia, yume, zenith"
                )
            ],
        },
        {"notes": ["no tools"], "errors": []},
        {"notes": ["conversation forgotten"], "errors": []},
    ]


@pytest.mark.parametrize(
    ("method", "params", "problem"),
    [
        ("turn", {}, "text: Field required"),
        ("turn", {"text": "hi", "colour": "red"}, "colour: Extra inputs"),
        ("think", {"level": "max"}, "level: Input should be"),
        ("budget", {"now": True}, "now: Extra inputs"),
    ],
    ids=["no-text", "unknown-field", "bad-level", "params-for-none"],
)
async def test_params_that_do_not_fit_are_refused_by_name(
    tmp_path: Path, method: str, params: Params, problem: str
) -> None:
    async with daemon(tmp_path, Local()) as info, client(info) as talk:
        with pytest.raises(RpcError, match=problem) as refused:
            await talk.call(method, params)

    assert refused.value.code == INVALID_PARAMS


async def test_an_image_that_cannot_be_sent_is_refused_with_its_own_code(
    tmp_path: Path,
) -> None:
    async with daemon(tmp_path, Local()) as info, client(info) as talk:
        with pytest.raises(RpcError, match=r"missing\.png") as refused:
            await talk.call("turn", {"text": "see", "images": ["missing.png"]})

    assert refused.value.code == IMAGE_REFUSED


async def test_a_failed_answer_is_an_error_with_the_reason(tmp_path: Path) -> None:
    async with daemon(tmp_path, Failing()) as info, client(info) as talk:
        with pytest.raises(RpcError, match=r"^no answer: the model broke$") as failed:
            await talk.call("turn", {"text": "hi"})

    assert failed.value.code == ANSWER_FAILED


async def test_stop_ends_the_daemon_and_removes_its_file(tmp_path: Path) -> None:
    settings = Settings(home=tmp_path)
    async with httpx.AsyncClient() as web:
        gateway = build_gateway(settings, web, publish_route, Local())
        host = Host(gateway, PersonaLibrary(), "synthia", Toolbox())
        task = asyncio.create_task(serve(host, tmp_path))
        info = await written(tmp_path, task)
        async with client(info) as talk:
            assert await talk.call("stop") == {}
        await asyncio.wait_for(task, HANG_TIMEOUT_S)

    assert not (tmp_path / DAEMON_FILE).exists()


async def test_a_planned_turn_tells_the_plan_each_step_and_the_answer(
    tmp_path: Path,
) -> None:
    model = Local(
        says(json.dumps({"steps": ["find a", "use a"]})),
        says("found a"),
        says("used a"),
        says("Done."),
    )
    async with daemon(tmp_path, model) as info, client(info) as talk:
        await talk.call("turn", {"text": "do it", "plan": True})

    assert [p for name, p in talk.notes if name == "plan"] == [
        {"kind": "planned", "steps": ["find a", "use a"], "revised": False},
        {"kind": "step", "number": 1, "text": "find a"},
        {"kind": "step", "number": 2, "text": "use a"},
        {"kind": "answer"},
    ]


async def test_each_conversation_knows_where_its_own_last_turn_went(
    tmp_path: Path,
) -> None:
    async with (
        daemon(tmp_path, Local(says("hi"))) as info,
        client(info) as first,
        client(info) as second,
    ):
        budget = await first.call("budget")
        await first.call("turn", {"text": "hello"})
        after = await first.call("model")
        untouched = await second.call("model")

    assert budget == {
        "notes": ["no remote model is configured, so there is no budget"],
        "errors": [],
    }
    notes = cast("dict[str, list[str]]", after)["notes"]
    assert notes[0] == "context 4096 tokens, images no, tools yes"
    assert notes[1].startswith("last turn: local to scripted (")
    assert untouched == {
        "notes": ["context 4096 tokens, images no, tools yes", "no turn yet"],
        "errors": [],
    }


async def test_the_daemon_starts_what_it_shares_serves_and_stops(
    tmp_path: Path,
) -> None:
    settings = Settings(home=tmp_path, openrouter_api_key=SecretStr(KEY))
    stop = asyncio.Event()
    running = asyncio.create_task(
        run_daemon(
            settings,
            None,
            transport=httpx.MockTransport(lambda _: httpx.Response(404)),
            stop=stop,
        )
    )
    info = await written(tmp_path, running)
    async with client(info) as talk:
        hello = await talk.call("hello")
        tools = await talk.call("tools")
        status = await talk.call("status")
    stop.set()
    await asyncio.wait_for(running, HANG_TIMEOUT_S)

    assert hello == {
        "version": __version__,
        "persona": "SYNTHIA",
        "sessions": 1,
        "warnings": [],
    }
    assert status == {"sessions": 1, "local": "none"}
    names = [
        line.split(" | ")[0] for line in cast("dict[str, list[str]]", tools)["notes"]
    ]
    assert names[:2] == ["current_time", "calculate"]
    assert not (tmp_path / DAEMON_FILE).exists()


class Service:
    """Stands in for the local model's service: hands out a model, counts starts."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def model(self, client: httpx.AsyncClient) -> Local:
        del client
        return Local(says("from the local model"))

    def state(self) -> str:
        return "ready"

    def start(self, observe: Callable[[ServiceEvent], None] | None = None) -> None:
        self.events.append("start")
        if observe is not None:
            observe(ServiceStarted(service="llama-server", attempt=1))

    def stop(self) -> None:
        self.events.append("stop")


async def test_the_local_model_starts_with_the_daemon_and_stops_with_it(
    tmp_path: Path,
) -> None:
    service = Service()
    stop = asyncio.Event()
    running = asyncio.create_task(
        run_daemon(Settings(home=tmp_path), service, stop=stop)
    )
    info = await written(tmp_path, running)
    async with client(info) as talk:
        report = await talk.call("turn", {"text": "hi"})
        adjusted = await talk.call("adjust", {"values": {"wit": 0.9}})
        status = await talk.call("status")
    stop.set()
    await asyncio.wait_for(running, HANG_TIMEOUT_S)

    assert service.events == ["start", "stop"]
    assert cast("dict[str, object]", report)["route"] == "local"
    assert cast("dict[str, list[str]]", adjusted)["notes"][0].startswith("now ")
    assert status == {"sessions": 1, "local": "ready"}
    kinds = [
        [r.kind for r in read_trace(path)[0]] for path in sessions(tmp_path / TRACES)
    ]
    assert sorted(kinds) == [["service"], ["turn", "model", "finished"]]


async def test_a_routing_decision_outside_any_turn_goes_nowhere() -> None:
    await publish_route(Event())
