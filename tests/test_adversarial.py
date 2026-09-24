"""Inputs the design did not anticipate, taken from how these things fail in use.

Each test is a situation someone will hit: a hotel Wi-Fi page instead of an
answer, a connection dropped mid tool call, a persona saved by Notepad, a model
that writes markup. The unit tests of each module test the design; these test
the world.
"""

import io
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr
from rich.console import Console

from synthia.gateway.assemble import build_gateway
from synthia.gateway.errors import ConnectionFailedError, MalformedStreamError
from synthia.gateway.protocol import collect
from synthia.gateway.providers import OPENROUTER
from synthia.gateway.types import ChatRequest, Message
from synthia.interfaces.chat import ChatApp
from synthia.interfaces.commands import parse
from synthia.interfaces.session import ChatSession, LastRoute
from synthia.kernel.bus import Event
from synthia.kernel.config import Settings
from synthia.persona.library import PersonaLibrary
from synthia.persona.model import PersonaError

KEY = "sk-or-v1-adversarial-test-key-000"  # pragma: allowlist secret
HELLO = ChatRequest((Message.user("hello"),))
PORTAL = b"<html><body><h1>Welcome to Hotel Wi-Fi</h1><form>Log in</form></body></html>"


def event(data: dict[str, object]) -> bytes:
    return f"data: {json.dumps(data)}\n\n".encode()


def finish() -> bytes:
    return (
        event({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        + b"data: [DONE]\n\n"
    )


async def _ignore(_: Event) -> None:
    return None


def settings(tmp_path: Path, cap: int = 50) -> Settings:
    return Settings(
        home=tmp_path, openrouter_api_key=SecretStr(KEY), remote_daily_cap=cap
    )


async def test_a_login_page_instead_of_a_stream_says_what_arrived(
    tmp_path: Path,
) -> None:
    portal = httpx.MockTransport(
        lambda _: httpx.Response(
            200, content=PORTAL, headers={"content-type": "text/html"}
        )
    )
    async with httpx.AsyncClient(transport=portal) as client:
        gateway = build_gateway(settings(tmp_path), client, _ignore)
        with pytest.raises(
            MalformedStreamError, match="expected an event stream, got text/html"
        ):
            await collect(gateway.model.stream(HELLO))


class DroppedMidToolCall(httpx.AsyncByteStream):
    """Start a tool call, then lose the connection, as a sleeping laptop does."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        call = {
            "index": 0,
            "id": "call_1",
            "function": {"name": "weather", "arguments": '{"ci'},
        }
        yield event({"choices": [{"delta": {"tool_calls": [call]}}]})
        message = "connection reset"
        raise httpx.ReadError(message)


async def test_a_connection_lost_mid_tool_call_is_not_sent_again(
    tmp_path: Path,
) -> None:
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, stream=DroppedMidToolCall())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway = build_gateway(settings(tmp_path), client, _ignore)
        with pytest.raises(ConnectionFailedError):
            await collect(gateway.model.stream(HELLO))
        status = await gateway.health.ledger.status(OPENROUTER)

    assert len(sent) == 1  # output had started, so a retry would repeat it
    assert status.used == 1
    assert await gateway.usage.today() == ()


async def test_invalid_utf8_in_a_delta_becomes_a_replacement_character(
    tmp_path: Path,
) -> None:
    broken = b'data: {"choices": [{"delta": {"content": "caf\xe9 au lait"}}]}\n\n'
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, content=broken + finish())
    )
    async with httpx.AsyncClient(transport=transport) as client:
        gateway = build_gateway(settings(tmp_path), client, _ignore)
        answer = await collect(gateway.model.stream(HELLO))

    assert answer.text == "caf\N{REPLACEMENT CHARACTER} au lait"


def test_a_persona_saved_by_notepad_with_a_byte_order_mark_loads(
    tmp_path: Path,
) -> None:
    text = 'name = "Notepad"\ndescription = "d."\n[blend]\njarvis = 1.0\n'
    (tmp_path / "notepad.toml").write_bytes(b"\xef\xbb\xbf" + text.encode())

    assert PersonaLibrary(tmp_path).get("notepad").name == "Notepad"


def test_a_persona_file_that_is_not_utf8_names_the_file(tmp_path: Path) -> None:
    (tmp_path / "latin.toml").write_bytes('name = "Caf\xe9"\n'.encode("latin-1"))

    with pytest.raises(PersonaError, match=r"latin\.toml is not UTF-8 text"):
        PersonaLibrary(tmp_path)


async def chat_with(
    tmp_path: Path, transport: httpx.MockTransport, cap: int = 50
) -> tuple[ChatApp, io.StringIO, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=transport)
    routes = LastRoute()
    gateway = build_gateway(settings(tmp_path, cap), client, routes)
    session = ChatSession(gateway.model, PersonaLibrary(), "synthia", routes)
    out = io.StringIO()
    return (
        ChatApp(session, gateway, Console(file=out, width=200, color_system=None)),
        out,
        client,
    )


async def test_markup_in_an_answer_is_printed_as_text(tmp_path: Path) -> None:
    body = (
        event({"choices": [{"delta": {"content": "Use [bold]x[/bold] or [red]"}}]})
        + finish()
    )
    chat, out, client = await chat_with(
        tmp_path, httpx.MockTransport(lambda _: httpx.Response(200, content=body))
    )
    async with client:
        await chat.handle(parse("show me markup"))

    assert "Use [bold]x[/bold] or [red]" in out.getvalue()


async def test_the_used_up_budget_says_when_it_comes_back(tmp_path: Path) -> None:
    body = event({"choices": [{"delta": {"content": "one"}}]}) + finish()
    chat, out, client = await chat_with(
        tmp_path,
        httpx.MockTransport(lambda _: httpx.Response(200, content=body)),
        cap=1,
    )
    async with client:
        await chat.handle(parse("first"))
        await chat.handle(parse("second"))

    text = out.getvalue()
    assert (
        "no answer: all 1 of today's openrouter requests are used; the budget resets at"
        in text
    )
    assert text.startswith("one\nremote | ")
    assert "\n\n" not in text  # a failure before any output adds no blank line
    assert len(chat.session.history) == 2


class DroppedMidSentence(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield event({"choices": [{"delta": {"content": "The answer is"}}]})
        message = "connection reset"
        raise httpx.ReadError(message)


async def test_a_connection_lost_mid_answer_ends_the_line_then_says_why(
    tmp_path: Path,
) -> None:
    chat, out, client = await chat_with(
        tmp_path,
        httpx.MockTransport(lambda _: httpx.Response(200, stream=DroppedMidSentence())),
    )
    async with client:
        assert await chat.handle(parse("what is it?"))

    assert out.getvalue().startswith("The answer is\nno answer: no response from ")
    assert chat.session.history == []


async def test_an_answer_that_is_only_done_says_so_and_is_not_kept(
    tmp_path: Path,
) -> None:
    chat, out, client = await chat_with(
        tmp_path,
        httpx.MockTransport(lambda _: httpx.Response(200, content=b"data: [DONE]\n\n")),
    )
    async with client:
        await chat.handle(parse("hello?"))

    assert out.getvalue() == "no answer came back\n"
    assert chat.session.history == []
