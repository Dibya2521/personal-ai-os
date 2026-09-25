import asyncio
from collections.abc import AsyncGenerator
from http import HTTPStatus
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from synthia.gateway.budget import BudgetLedger
from synthia.gateway.circuit import CircuitBreaker
from synthia.gateway.errors import ConnectionFailedError
from synthia.gateway.protocol import collect
from synthia.gateway.ratelimit import SlidingWindowLimiter
from synthia.gateway.router import RemoteHealth, Route, Router, RouteReason
from synthia.gateway.types import (
    ChatChunk,
    ChatRequest,
    FinishReason,
    ImagePart,
    Message,
    ModelInfo,
    Role,
    TextPart,
)
from synthia.models.catalogue import Backend, Model, find
from synthia.models.local import LocalModel
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


async def test_a_stopped_server_is_not_ready_and_refuses_requests(
    tmp_path: Path,
) -> None:
    async with httpx.AsyncClient() as client:
        local = LocalModel(server_at(tmp_path, client, new_key()), client, QWEN, 4096)

        with pytest.raises(ConnectionFailedError, match="not running"):
            await collect(local.stream(HELLO))

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


async def test_the_router_sends_a_background_job_to_the_served_model(
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
        router = Router(Unused(), health, local, local_ready=local.ready)
        stop = asyncio.Event()
        task = asyncio.create_task(server.run(stop))
        await asyncio.wait_for(server.ready.wait(), HANG_TIMEOUT_S)
        route = await router.decide(HELLO, background=True)
        reply = await collect(router.stream(HELLO, background=True))
        stop.set()
        await asyncio.wait_for(task, HANG_TIMEOUT_S)

    assert route == (Route.LOCAL, RouteReason.BACKGROUND)
    assert reply.text == "echo: hello"
