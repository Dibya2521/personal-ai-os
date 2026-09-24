import asyncio
import io
import json
import signal
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr
from rich.console import Console
from typer.testing import CliRunner

from synthia.gateway.assemble import build_gateway
from synthia.interfaces.chat import (
    MORE,
    PROMPT,
    ChatApp,
    describe,
    read_message,
    run_chat,
)
from synthia.interfaces.cli import app
from synthia.interfaces.commands import parse
from synthia.interfaces.session import ChatSession, LastRoute, TurnReport
from synthia.kernel.config import Settings
from synthia.persona.library import PersonaLibrary

KEY = "sk-or-v1-chat-test-key-000"  # pragma: allowlist secret


def event(data: dict[str, object]) -> bytes:
    return f"data: {json.dumps(data)}\n\n".encode()


def answer(text: str) -> bytes:
    return (
        event({"model": "vendor/free", "choices": [{"delta": {"content": text}}]})
        + event(
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 30, "completion_tokens": 2},
            }
        )
        + b"data: [DONE]\n\n"
    )


def replying(text: str = "Hello, Dibya.") -> httpx.MockTransport:
    return httpx.MockTransport(lambda _: httpx.Response(200, content=answer(text)))


def scripted(*lines: str) -> Callable[[str], str]:
    """Answer prompts with ``lines`` in order, then end the input."""
    queue: Iterator[str] = iter(lines)
    prompts: list[str] = []

    def read(prompt: str) -> str:
        prompts.append(prompt)
        try:
            return next(queue)
        except StopIteration:
            raise EOFError from None

    read.prompts = prompts  # type: ignore[attr-defined]
    return read


def console() -> tuple[Console, io.StringIO]:
    out = io.StringIO()
    return Console(file=out, width=200, color_system=None), out


def settings(tmp_path: Path) -> Settings:
    return Settings(home=tmp_path, openrouter_api_key=SecretStr(KEY))


def test_a_line_ending_in_a_backslash_continues_on_the_next() -> None:
    read = scripted("first \\", "second\\", "third", "unread")

    assert read_message(read) == "first \nsecond\nthird"
    assert read.prompts == [PROMPT, MORE, MORE]  # type: ignore[attr-defined]


def test_the_end_of_input_ends_a_message() -> None:
    with pytest.raises(EOFError):
        read_message(scripted())


def test_the_report_line_says_route_model_tokens_and_time() -> None:
    known = TurnReport("remote", "vendor/free", 30, 2, 1.26)
    unknown = TurnReport("local", "qwen", None, 2, 0.5)

    assert describe(known) == "remote | vendor/free | 30 in, 2 out | 1.3 s"
    assert describe(unknown) == "local | qwen | tokens not reported | 0.5 s"


def test_a_whole_chat_answers_reports_and_leaves_on_exit(tmp_path: Path) -> None:
    screen, out = console()
    read = scripted("hello", "/budget", "/model", "/exit", "never read")

    run_chat(settings(tmp_path), "synthia", screen, read, replying())

    text = out.getvalue()
    assert text.startswith("SYNTHIA is listening.")
    assert "Hello, Dibya.\nremote | vendor/free | 30 in, 2 out |" in text
    assert (
        "1 of 50 remote requests used today (UTC); 49 left, 10 kept in reserve" in text
    )
    assert "last turn: remote to openrouter/free (remote first)" in text


@pytest.mark.parametrize("ending", [EOFError, KeyboardInterrupt])
def test_the_end_of_input_or_ctrl_c_at_the_prompt_leaves(
    tmp_path: Path, ending: type[BaseException]
) -> None:
    def read(_: str) -> str:
        raise ending

    screen, out = console()
    run_chat(settings(tmp_path), "synthia", screen, read, replying())

    assert out.getvalue().startswith("SYNTHIA is listening.")


class InterruptedBody(httpx.AsyncByteStream):
    """Send the first words, then press Ctrl+C while the rest is awaited."""

    def __init__(self) -> None:
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield event({"choices": [{"delta": {"content": "Once upon"}}]})
        signal.raise_signal(signal.SIGINT)
        await asyncio.sleep(30)  # cancelled by the Ctrl+C before it ends

    async def aclose(self) -> None:
        self.closed = True


def test_ctrl_c_mid_answer_stops_it_closes_the_stream_and_keeps_chatting(
    tmp_path: Path,
) -> None:
    body = InterruptedBody()
    transport = httpx.MockTransport(lambda _: httpx.Response(200, stream=body))
    screen, out = console()
    read = scripted("tell me a story", "/model")

    run_chat(settings(tmp_path), "synthia", screen, read, transport)

    text = out.getvalue()
    assert "Once upon\nstopped; that turn is not kept" in text
    assert "last turn: remote" in text  # the chat went on after the stop
    assert body.closed


async def app_for(
    tmp_path: Path, client: httpx.AsyncClient
) -> tuple[ChatApp, io.StringIO]:
    routes = LastRoute()
    gateway = build_gateway(settings(tmp_path), client, routes)
    session = ChatSession(gateway.model, PersonaLibrary(), "synthia", routes)
    screen, out = console()
    return ChatApp(session, gateway, screen), out


async def test_persona_commands_switch_list_adjust_and_report_mistakes(
    tmp_path: Path,
) -> None:
    async with httpx.AsyncClient(transport=replying()) as client:
        chat, out = await app_for(tmp_path, client)
        for line in [
            "/persona",
            "/persona edith",
            "/persona set wit=0.9",
            "/persona friday",
            "/persona set charm=1",
        ]:
            assert await chat.handle(parse(line))

    text = out.getvalue()
    assert "personas: companion, edith, jarvis, synthia" in text
    assert "now EDITH: warmth=0.25, formality=0.6, wit=0.15" in text
    assert "now EDITH: warmth=0.25, formality=0.6, wit=0.9" in text
    assert "no persona 'friday'" in text
    assert "no such trait: charm" in text
    assert chat.session.persona.traits.wit == 0.9


async def test_an_image_is_sent_and_a_bad_one_is_refused_without_a_request(
    tmp_path: Path,
) -> None:
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, content=answer("A cat."))

    picture = tmp_path / "cat.png"
    picture.write_bytes(b"\x89PNG")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        chat, out = await app_for(tmp_path, client)
        await chat.handle(parse(f'/image "{tmp_path / "missing.png"}"'))
        await chat.handle(parse(f'/image "{picture}" what animal?'))

    assert "cannot read" in out.getvalue()
    assert "A cat." in out.getvalue()
    (request,) = sent
    content = json.loads(request.content)["messages"][-1]["content"]
    assert [part["type"] for part in content] == ["text", "image_url"]


async def test_a_failed_answer_is_reported_and_the_chat_goes_on(tmp_path: Path) -> None:
    refusing = httpx.MockTransport(
        lambda _: httpx.Response(401, json={"error": {"message": "User not found."}})
    )
    async with httpx.AsyncClient(transport=refusing) as client:
        chat, out = await app_for(tmp_path, client)
        assert await chat.handle(parse("hello"))
        assert await chat.handle(parse(""))
        assert await chat.handle(parse("/reset"))
        assert await chat.handle(parse("/help"))
        assert await chat.handle(parse("/dance"))
        assert await chat.handle(parse("/model"))

    text = out.getvalue()
    assert "no answer: " in text
    assert "User not found." in text
    assert "conversation forgotten" in text
    assert "/persona set wit=0.3" in text
    assert "unknown command /dance" in text
    assert "no turn yet" not in text  # the failed turn was still routed
    assert chat.session.history == []


async def test_an_answer_that_never_finished_is_shown_without_a_report(
    tmp_path: Path,
) -> None:
    unfinished = (
        event({"choices": [{"delta": {"content": "Half"}}]}) + b"data: [DONE]\n\n"
    )
    transport = httpx.MockTransport(lambda _: httpx.Response(200, content=unfinished))
    async with httpx.AsyncClient(transport=transport) as client:
        chat, out = await app_for(tmp_path, client)
        await chat.handle(parse("hello"))

    assert out.getvalue() == "Half\n"
    assert chat.session.history == []


async def test_model_before_any_turn_says_so(tmp_path: Path) -> None:
    async with httpx.AsyncClient(transport=replying()) as client:
        chat, out = await app_for(tmp_path, client)
        await chat.handle(parse("/model"))

    assert "context 32768 tokens, images yes, tools yes\nno turn yet" in out.getvalue()


def test_the_command_refuses_to_start_without_a_key_or_with_an_unknown_persona(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SYNTHIA_HOME", str(tmp_path / "home"))
    cli = CliRunner()

    no_key = cli.invoke(app, ["chat"])
    monkeypatch.setenv("SYNTHIA_OPENROUTER_API_KEY", KEY)
    no_persona = cli.invoke(app, ["chat", "--persona", "friday"])

    assert (no_key.exit_code, no_persona.exit_code) == (2, 2)
    assert "SYNTHIA_OPENROUTER_API_KEY" in no_key.stderr
    assert "no persona 'friday'" in no_persona.stderr
    assert KEY not in no_persona.stderr + no_persona.stdout
