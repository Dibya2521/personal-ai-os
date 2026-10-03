import asyncio
import json
from collections.abc import AsyncGenerator, Callable
from dataclasses import replace
from http import HTTPStatus
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from synthia.gateway.budget import BudgetLedger
from synthia.gateway.circuit import CircuitBreaker
from synthia.gateway.errors import ConnectionFailedError
from synthia.gateway.openai_compat import Dialect, Endpoint
from synthia.gateway.protocol import collect
from synthia.gateway.ratelimit import SlidingWindowLimiter
from synthia.gateway.reasoning import ALLOWANCE_S
from synthia.gateway.router import Remote, RemoteHealth, Route, Router, RouteReason
from synthia.gateway.types import (
    ChatChunk,
    ChatRequest,
    FinishReason,
    ImagePart,
    Message,
    ModelInfo,
    Reasoning,
    Role,
    TextPart,
)
from synthia.models.catalogue import Backend, Model, find
from synthia.models.local import LocalModel, end_thinking
from synthia.models.server import Launch, LlamaServer, free_port, new_key
from tests.models import fake_llama_server
from tests.timing import HANG_TIMEOUT_S

HELLO = ChatRequest((Message.user("hello"),))
HELLO_BODY = {"messages": [{"role": "user", "content": "hello"}]}
QWEN_ID = "qwen3.5-4b"


def catalogued(name: str) -> Model:
    model = find(name)
    assert isinstance(model, Model)
    return model


QWEN = catalogued(QWEN_ID)


def server_at(tmp_path: Path, client: httpx.AsyncClient, key: SecretStr) -> LlamaServer:
    def launch() -> Launch:
        return Launch(
            Backend.CPU,
            tmp_path / "llama-server",
            tmp_path / "model.gguf",
            None,
            port=free_port(),
            context=512,
            key=key,
            name="m",
        )

    return LlamaServer(
        launch,
        client,
        tmp_path / "llama-server.log",
        command=fake_llama_server.command,
        poll_s=0.05,
    )


async def test_a_request_is_answered_by_the_running_server(tmp_path: Path) -> None:
    async with httpx.AsyncClient() as client:
        server = server_at(tmp_path, client, new_key())
        local = LocalModel(server, client, QWEN, 4096)
        stop = asyncio.Event()
        task = asyncio.create_task(server.run(stop))
        await asyncio.wait_for(server.ready.wait(), HANG_TIMEOUT_S)
        ready = local.ready()
        reply = await collect(local.stream(HELLO))
        stop.set()
        await asyncio.wait_for(task, HANG_TIMEOUT_S)

    assert ready
    assert not local.ready()
    assert reply.text == "echo: hello"
    assert reply.finish_reason is FinishReason.STOP


async def test_a_long_silence_while_the_server_reads_the_prompt_is_waited_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_FIRST_BYTE_S", "1")
    async with httpx.AsyncClient(timeout=httpx.Timeout(0.3)) as client:
        server = server_at(tmp_path, client, new_key())
        local = LocalModel(server, client, QWEN, 4096)
        stop = asyncio.Event()
        task = asyncio.create_task(server.run(stop))
        await asyncio.wait_for(server.ready.wait(), HANG_TIMEOUT_S)
        reply = await asyncio.wait_for(collect(local.stream(HELLO)), HANG_TIMEOUT_S)
        stop.set()
        await asyncio.wait_for(task, HANG_TIMEOUT_S)

    assert reply.text == "echo: hello"


async def test_the_reasoning_level_reaches_the_server_as_its_thinking_switch(
    tmp_path: Path,
) -> None:
    async with httpx.AsyncClient() as client:
        server = server_at(tmp_path, client, new_key())
        local = LocalModel(server, client, QWEN, 4096)
        stop = asyncio.Event()
        task = asyncio.create_task(server.run(stop))
        await asyncio.wait_for(server.ready.wait(), HANG_TIMEOUT_S)
        replies = {
            level: await collect(local.stream(replace(HELLO, reasoning=level)))
            for level in (Reasoning.LOW, Reasoning.OFF)
        }
        stop.set()
        await asyncio.wait_for(task, HANG_TIMEOUT_S)

    assert replies[Reasoning.LOW].reasoning == fake_llama_server.THOUGHT
    assert replies[Reasoning.OFF].reasoning == ""
    assert replies[Reasoning.LOW].text == replies[Reasoning.OFF].text == "echo: hello"


async def controls_seen(client: httpx.AsyncClient, server: LlamaServer) -> object:
    running = server.running
    assert running is not None
    url = running.base_url.removesuffix("/v1") + fake_llama_server.CONTROLS_PATH
    return (await client.get(url)).json()


async def test_thinking_past_its_allowance_is_ended_and_the_answer_follows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # This fake thinks until the control call ends it, or for 10 s at most.
    monkeypatch.setenv("FAKE_THINK_UNTIL_ENDED", "1")
    waited: list[float] = []

    async def no_wait(seconds: float) -> None:
        waited.append(seconds)

    async with httpx.AsyncClient() as client:
        server = server_at(tmp_path, client, new_key())
        local = LocalModel(server, client, QWEN, 4096, sleep=no_wait)
        stop = asyncio.Event()
        task = asyncio.create_task(server.run(stop))
        await asyncio.wait_for(server.ready.wait(), HANG_TIMEOUT_S)
        request = replace(HELLO, reasoning=Reasoning.MEDIUM)
        reply = await asyncio.wait_for(collect(local.stream(request)), HANG_TIMEOUT_S)
        controls = await controls_seen(client, server)
        stop.set()
        await asyncio.wait_for(task, HANG_TIMEOUT_S)

    assert reply.text == "echo: hello"
    assert reply.reasoning.startswith(fake_llama_server.THOUGHT)
    assert waited == [ALLOWANCE_S[Reasoning.MEDIUM]]
    assert controls == [{"id": "chatcmpl-fake-1", "action": "reasoning_end"}]


@pytest.mark.parametrize(
    ("level", "timed"), [(Reasoning.LOW, True), (Reasoning.OFF, False), (None, False)]
)
async def test_thinking_that_ends_in_time_is_never_interrupted(
    tmp_path: Path, level: Reasoning | None, *, timed: bool
) -> None:
    waited: list[float] = []

    async def never(seconds: float) -> None:
        waited.append(seconds)
        await asyncio.Event().wait()

    async with httpx.AsyncClient() as client:
        server = server_at(tmp_path, client, new_key())
        local = LocalModel(server, client, QWEN, 4096, sleep=never)
        stop = asyncio.Event()
        task = asyncio.create_task(server.run(stop))
        await asyncio.wait_for(server.ready.wait(), HANG_TIMEOUT_S)
        request = replace(HELLO, reasoning=level)
        reply = await asyncio.wait_for(collect(local.stream(request)), HANG_TIMEOUT_S)
        controls = await controls_seen(client, server)
        stop.set()
        await asyncio.wait_for(task, HANG_TIMEOUT_S)

    assert reply.text == "echo: hello"
    assert controls == []
    # With a timed level the timer may be cancelled before it ever runs, so
    # only an untimed level can be held to "never asked to wait".
    if not timed:
        assert waited == []


CONTROL_URL = "http://127.0.0.1:8080/v1/chat/completions/control"
UNKNOWN_ID = b'{"success":false,"message":"no active completion for this id"}'

type Handler = Callable[[httpx.Request], httpx.Response]


def control_client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_the_end_command_names_the_completion_and_carries_the_key() -> None:
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"success": True})

    key = new_key()
    endpoint = Endpoint("http://127.0.0.1:8080/v1", "m", key, dialect=Dialect.LLAMA_CPP)
    async with control_client(handler) as client:
        ended = await end_thinking(client, endpoint, "chatcmpl-1")

    assert ended
    assert str(sent[0].url) == CONTROL_URL
    assert sent[0].headers["Authorization"] == f"Bearer {key.get_secret_value()}"
    assert json.loads(sent[0].content) == {
        "id": "chatcmpl-1",
        "action": "reasoning_end",
    }


@pytest.mark.parametrize(
    ("status", "body", "logged"),
    [
        # The real server answers an unknown id with 200, not an error status.
        (200, UNKNOWN_ID, "200"),
        (400, b'{"error":{"message":"unknown control action"}}', "400"),
        (200, b"not json", "200"),
        (500, b"", "500"),
    ],
)
async def test_a_refused_end_command_is_logged_not_raised(
    status: int, body: bytes, logged: str, caplog: pytest.LogCaptureFixture
) -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(status, content=body)

    endpoint = Endpoint("http://127.0.0.1:8080/v1", "m", dialect=Dialect.LLAMA_CPP)
    async with control_client(answer) as client:
        ended = await end_thinking(client, endpoint, "chatcmpl-1")

    assert not ended
    assert f"did not end its thinking: HTTP {logged}" in caplog.text


async def test_an_unreachable_server_for_the_end_command_is_logged_not_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        message = "refused"
        raise httpx.ConnectError(message, request=request)

    endpoint = Endpoint("http://127.0.0.1:8080/v1", "m", dialect=Dialect.LLAMA_CPP)
    async with control_client(refuse) as client:
        ended = await end_thinking(client, endpoint, "chatcmpl-1")

    assert not ended
    assert "could not end the local thinking: ConnectError" in caplog.text


async def test_a_request_with_an_image_is_sent_as_parts(tmp_path: Path) -> None:
    request = ChatRequest(
        (
            Message(
                Role.USER, (TextPart("what is this"), ImagePart(b"png", "image/png"))
            ),
        )
    )
    async with httpx.AsyncClient() as client:
        server = server_at(tmp_path, client, new_key())
        stop = asyncio.Event()
        task = asyncio.create_task(server.run(stop))
        await asyncio.wait_for(server.ready.wait(), HANG_TIMEOUT_S)
        reply = await collect(LocalModel(server, client, QWEN, 4096).stream(request))
        stop.set()
        await asyncio.wait_for(task, HANG_TIMEOUT_S)

    assert reply.text == "echo: what is this"


async def test_the_server_refuses_a_request_without_its_key(tmp_path: Path) -> None:
    async with httpx.AsyncClient() as client:
        server = server_at(tmp_path, client, new_key())
        stop = asyncio.Event()
        task = asyncio.create_task(server.run(stop))
        await asyncio.wait_for(server.ready.wait(), HANG_TIMEOUT_S)
        running = server.running
        assert running is not None
        refused = await client.post(
            f"{running.base_url}/chat/completions", json=HELLO_BODY
        )
        stop.set()
        await asyncio.wait_for(task, HANG_TIMEOUT_S)

    assert refused.status_code == HTTPStatus.UNAUTHORIZED


async def test_a_server_that_never_starts_is_waited_for_then_refused(
    tmp_path: Path,
) -> None:
    waits: list[float] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)

    async with httpx.AsyncClient() as client:
        server = server_at(tmp_path, client, new_key())
        local = LocalModel(server, client, QWEN, 4096, sleep=sleep)

        with pytest.raises(ConnectionFailedError, match="did not start within"):
            await collect(local.stream(HELLO))

    assert sum(waits) == server.start_timeout_s
    assert not local.ready()
    assert (local.info.id, local.info.context_window) == (QWEN_ID, 4096)
    assert local.info.vision
    assert local.info.tools


class Unused:
    info = ModelInfo("remote", 32_768, vision=True, tools=True)

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        del request
        message = "the remote must not be asked"
        raise AssertionError(message)
        yield ChatChunk()


async def test_a_request_sent_while_the_server_starts_is_answered_once_it_serves(
    tmp_path: Path,
) -> None:
    async with httpx.AsyncClient() as client:
        server = server_at(tmp_path, client, new_key())
        local = LocalModel(server, client, QWEN, 4096)
        asked = asyncio.create_task(collect(local.stream(HELLO)))
        await asyncio.sleep(0)
        stop = asyncio.Event()
        task = asyncio.create_task(server.run(stop))
        reply = await asyncio.wait_for(asked, HANG_TIMEOUT_S)
        stop.set()
        await asyncio.wait_for(task, HANG_TIMEOUT_S)

    assert reply.text == "echo: hello"


async def test_the_router_sends_a_local_request_to_the_served_model(
    tmp_path: Path,
) -> None:
    health = RemoteHealth(
        "openrouter",
        BudgetLedger(tmp_path / "gateway.db", {"openrouter": 50}),
        SlidingWindowLimiter(20),
        CircuitBreaker("openrouter"),
        reserve=10,
    )
    async with httpx.AsyncClient() as client:
        server = server_at(tmp_path, client, new_key())
        local = LocalModel(server, client, QWEN, 4096)
        router = Router(local, Remote(Unused(), health), local_ready=local.ready)
        stop = asyncio.Event()
        task = asyncio.create_task(server.run(stop))
        await asyncio.wait_for(server.ready.wait(), HANG_TIMEOUT_S)
        route = await router.decide(HELLO)
        reply = await collect(router.stream(HELLO))
        stop.set()
        await asyncio.wait_for(task, HANG_TIMEOUT_S)

    assert route == (Route.LOCAL, RouteReason.LOCAL_FIRST)
    assert reply.text == "echo: hello"
