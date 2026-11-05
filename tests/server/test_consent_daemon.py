import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

import httpx
from pydantic import SecretStr

from synthia.agent.tools import Toolbox
from synthia.gateway.assemble import build_gateway
from synthia.interfaces.daemon_client import conversation
from synthia.kernel.config import Settings
from synthia.kernel.consent import CONSENT_FILE, ConsentBook, Scope
from synthia.persona.library import PersonaLibrary
from synthia.server.daemon import serve
from synthia.server.discovery import DaemonInfo
from synthia.server.host import Host, publish_route
from tests.agent.scripted import says
from tests.consenting import OPENROUTER
from tests.server.test_daemon import Local, written
from tests.timing import HANG_TIMEOUT_S


def _nothing_goes_out(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(401, json={"error": {"message": "offline test"}})


@asynccontextmanager
async def daemon(tmp_path: Path) -> AsyncGenerator[DaemonInfo]:
    settings = Settings(home=tmp_path, openrouter_api_key=SecretStr("sk-or-test"))
    transport = httpx.MockTransport(_nothing_goes_out)
    async with httpx.AsyncClient(transport=transport) as client:
        gateway = build_gateway(settings, client, publish_route, Local(says("hi")))
        host = Host(gateway, PersonaLibrary(), "synthia", Toolbox())
        stop = asyncio.Event()
        task = asyncio.create_task(serve(host, tmp_path, stop=stop))
        info = await written(tmp_path, task)
        try:
            yield info
        finally:
            stop.set()
            await asyncio.wait_for(task, HANG_TIMEOUT_S)


async def _no_tools(_tool: str, _arguments: str, *, confined: bool) -> bool:
    del confined
    return False


async def test_remote_on_asks_the_client_and_saves_its_yes(tmp_path: Path) -> None:
    asked: list[tuple[str, str]] = []

    async def yes(provider: str, url: str) -> bool:
        asked.append((provider, url))
        return True

    async with daemon(tmp_path) as info, conversation(info, _no_tools, yes) as talk:
        reply = await talk.remote(on=True)

    assert reply.notes == ("remote: on, turns may leave this machine",)
    assert asked == [(OPENROUTER.name, OPENROUTER.url)]
    assert ConsentBook(tmp_path).allows(OPENROUTER, Scope.CHAT)


async def test_a_client_that_cannot_ask_leaves_nothing_decided(
    tmp_path: Path,
) -> None:
    async with daemon(tmp_path) as info, conversation(info, _no_tools) as talk:
        reply = await talk.remote(on=True)

    (problem,) = reply.errors
    assert "this client cannot ask" in problem
    assert not (tmp_path / CONSENT_FILE).exists()


async def test_a_reply_that_is_not_a_boolean_leaves_nothing_decided(
    tmp_path: Path,
) -> None:
    async def in_words(_provider: str, _url: str) -> bool:
        return cast("bool", "yes")

    async with (
        daemon(tmp_path) as info,
        conversation(info, _no_tools, in_words) as talk,
    ):
        reply = await talk.remote(on=True)

    (problem,) = reply.errors
    assert "this client cannot ask" in problem
    assert not (tmp_path / CONSENT_FILE).exists()
