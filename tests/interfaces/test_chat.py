import asyncio
import io
import json
import logging
import signal
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr
from rich.console import Console
from typer.testing import CliRunner

from synthia.gateway.assemble import build_gateway
from synthia.gateway.providers import OPENROUTER_FREE
from synthia.gateway.types import ChatChunk, Reasoning
from synthia.interfaces import cli
from synthia.interfaces.chat import (
    MORE,
    PROMPT,
    ChatApp,
    ThinkingLine,
    ThinkingTimer,
    describe,
    read_message,
    run_chat,
)
from synthia.interfaces.cli import app
from synthia.interfaces.commands import parse
from synthia.interfaces.session import ChatSession, LastRoute, TurnReport
from synthia.kernel.config import Settings
from synthia.kernel.errors import ConfigError
from synthia.models.service import LocalService, LocalSetup
from synthia.persona.library import PersonaLibrary
from synthia.persona.model import PersonaError
from tests.models.test_service import (
    CPU,
    TINY,
    service_at,
    service_threads,
    setup_at,
    wait_until,
)
from tests.timing import HANG_TIMEOUT_S

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


def test_the_report_line_says_route_model_level_tokens_and_time() -> None:
    known = TurnReport("remote", "vendor/free", 30, 2, 1.26, Reasoning.MEDIUM)
    unknown = TurnReport("local", "qwen", None, 2, 0.5)

    assert describe(known) == (
        "remote | vendor/free | thinking medium | 30 in, 2 out | 1.3 s"
    )
    assert describe(unknown) == "local | qwen | tokens not reported | 0.5 s"


def test_the_thinking_line_counts_seconds_from_when_thinking_began() -> None:
    now = iter([100.0, 100.4, 103.6])
    timer = ThinkingTimer(lambda: next(now))

    assert [timer.__rich__().plain for _ in range(2)] == [
        "thinking 0 s",
        "thinking 4 s",
    ]


def test_a_thought_after_the_answer_began_does_not_bring_the_line_back() -> None:
    out = io.StringIO()
    terminal = Console(file=out, width=80, force_terminal=True, color_system=None)
    line = ThinkingLine(terminal, lambda: 0.0)

    line.see(ChatChunk(text="an answer that did not think first"))
    line.see(ChatChunk(reasoning="a late thought"))
    line.close()

    assert out.getvalue() == ""


def thinking_then(text: str) -> bytes:
    thought: dict[str, object] = {
        "model": "vendor/free",
        "choices": [{"delta": {"reasoning": "hmm"}}],
    }
    return event(thought) + answer(text)


def test_a_level_set_with_think_is_sent_and_reported(tmp_path: Path) -> None:
    bodies: list[dict[str, object]] = []

    def reply(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(404)
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=thinking_then("Done."))

    screen, out = console()
    read = scripted(
        "/think", "/think high", "hello", "/think auto", "hello", "/think max"
    )

    run_chat(
        settings(tmp_path),
        "synthia",
        screen,
        read,
        httpx.MockTransport(reply),
        use_remote=True,
    )

    text = out.getvalue()
    assert text.count("thinking: auto") == 2
    assert "thinking: high" in text
    assert "'max' is not a thinking level" in text
    # High first, while no speed is measured: 60 s at the assumed 25 tokens/s.
    # Then auto decides a greeting needs no thinking.
    reasoning = [b["reasoning"] for b in bodies]
    assert reasoning == [{"max_tokens": 1500}, {"effort": "none"}]
    assert "remote | vendor/free | thinking high | 30 in, 2 out |" in text
    assert "remote | vendor/free | thinking off | 30 in, 2 out |" in text


def test_the_chat_asks_for_the_keys_daily_cap_while_the_first_answer_streams(
    tmp_path: Path,
) -> None:
    asked: list[str] = []
    key_asked = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            asked.append(request.url.path)
            key_asked.set()
            return httpx.Response(404)
        await asyncio.wait_for(key_asked.wait(), HANG_TIMEOUT_S)
        return httpx.Response(200, content=answer("Hello."))

    screen, out = console()
    run_chat(
        settings(tmp_path),
        "synthia",
        screen,
        scripted("hello"),
        httpx.MockTransport(handler),
        use_remote=True,
    )

    assert asked == ["/api/v1/key"]
    assert "Hello." in out.getvalue()


def test_a_cap_question_that_never_answers_does_not_hold_the_chat_open(
    tmp_path: Path,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            await asyncio.Event().wait()
        return httpx.Response(200, content=answer("Hello."))

    screen, out = console()
    chat = threading.Thread(
        target=run_chat,
        args=(settings(tmp_path), "synthia", screen, scripted("hello")),
        kwargs={"transport": httpx.MockTransport(handler), "use_remote": True},
    )
    chat.start()
    chat.join(HANG_TIMEOUT_S)

    assert not chat.is_alive()
    assert "Hello." in out.getvalue()


def test_the_thinking_line_shows_while_thinking_and_leaves_only_the_answer(
    tmp_path: Path,
) -> None:
    out = io.StringIO()
    terminal = Console(file=out, width=80, force_terminal=True, color_system=None)
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, content=thinking_then("Done."))
    )

    run_chat(
        settings(tmp_path),
        "synthia",
        terminal,
        scripted("why?"),
        transport,
        use_remote=True,
    )

    text = out.getvalue()
    assert "thinking 0 s" in text
    assert text.index("thinking 0 s") < text.index("Done.")
    assert "hmm" not in text


def test_a_whole_chat_answers_reports_and_leaves_on_exit(tmp_path: Path) -> None:
    screen, out = console()
    read = scripted(
        "/remote", "/remote on", "hello", "/budget", "/model", "/exit", "never read"
    )

    run_chat(settings(tmp_path), "synthia", screen, read, replying())

    text = out.getvalue()
    assert text.startswith("SYNTHIA is listening.")
    assert "remote: off, every turn stays on this machine" in text
    assert "remote: on, turns may leave this machine" in text
    assert "Hello, Dibya.\nremote | vendor/free | thinking off | 30 in, 2 out |" in text
    assert (
        "1 of 50 remote requests used today (UTC); 49 left, 10 kept in reserve" in text
    )
    assert "last turn: remote to openrouter/free (remote asked for)" in text


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
    transport = httpx.MockTransport(
        lambda request: (
            httpx.Response(404)
            if request.method == "GET"
            else httpx.Response(200, stream=body)
        )
    )
    screen, out = console()
    read = scripted("tell me a story", "/model")

    run_chat(settings(tmp_path), "synthia", screen, read, transport, use_remote=True)

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
    session.use_remote = True
    screen, out = console()
    return ChatApp(session, gateway, screen), out


async def test_persona_commands_switch_list_adjust_and_report_mistakes(
    tmp_path: Path,
) -> None:
    async with httpx.AsyncClient(transport=replying()) as client:
        chat, out = await app_for(tmp_path, client)
        for line in [
            "/persona",
            "/persona horizon",
            "/persona set wit=0.9",
            "/persona nobody",
            "/persona set charm=1",
        ]:
            assert await chat.handle(parse(line))

    text = out.getvalue()
    personas = "glacier, horizon, minato, neon, nova, starlight, synthia, yume, zenith"
    assert f"personas: {personas}\n" in text
    assert "now SYNTHIA Horizon: warmth=0.25, formality=0.6, wit=0.15" in text
    assert "now SYNTHIA Horizon: warmth=0.25, formality=0.6, wit=0.9" in text
    assert "no persona 'nobody'" in text
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
    no_persona = cli.invoke(app, ["chat", "--persona", "nobody"])

    assert (no_key.exit_code, no_persona.exit_code) == (2, 2)
    assert "SYNTHIA_OPENROUTER_API_KEY" in no_key.stderr
    assert "no persona 'nobody'" in no_persona.stderr
    assert KEY not in no_persona.stderr + no_persona.stdout


# No budget, so every request goes local; and nothing listens on port 9, so a
# request wrongly sent remote fails fast.
NO_REMOTE = replace(OPENROUTER_FREE, base_url="http://127.0.0.1:9/api/v1", daily_cap=0)


def remote_spent(tmp_path: Path, **values: object) -> Settings:
    return Settings.model_validate({"home": tmp_path} | values)


def test_with_the_remote_budget_spent_the_local_model_answers(tmp_path: Path) -> None:
    service = service_at(tmp_path)
    lines = scripted("hello", "/model", "/exit")

    def read(prompt: str) -> str:
        wait_until(lambda: service.server.running is not None)
        return lines(prompt)

    screen, out = console()
    run_chat(
        remote_spent(tmp_path, openrouter_api_key=KEY),
        "synthia",
        screen,
        read,
        local=service,
        remote=NO_REMOTE,
        use_remote=True,
    )

    text = out.getvalue()
    assert "echo: hello\nlocal | local | thinking off | 3 in, 2 out |" in text
    assert "context 512 tokens" in text
    assert "last turn: local to tiny (remote budget at the reserve)" in text
    assert not service_threads()


def test_without_a_key_the_local_model_answers_and_remote_is_refused(
    tmp_path: Path,
) -> None:
    service = service_at(tmp_path)
    screen, out = console()

    run_chat(
        remote_spent(tmp_path),
        "synthia",
        screen,
        scripted("hello", "/model", "/remote on", "/budget", "/exit"),
        local=service,
    )

    text = out.getvalue()
    assert "echo: hello\nlocal | local | thinking off | 3 in, 2 out |" in text
    assert "last turn: local to tiny (local first)" in text
    assert "no remote model is configured: set SYNTHIA_OPENROUTER_API_KEY" in text
    assert "no remote model is configured, so there is no budget" in text
    assert not service_threads()


def test_a_first_turn_waits_for_the_starting_local_model_and_never_goes_out(
    tmp_path: Path,
) -> None:
    service = service_at(tmp_path)
    screen, out = console()

    run_chat(
        remote_spent(tmp_path, openrouter_api_key=KEY),
        "synthia",
        screen,
        scripted("hello", "/model", "/exit"),
        local=service,
        remote=NO_REMOTE,
    )

    text = out.getvalue()
    assert "echo: hello\nlocal | local |" in text
    assert "last turn: local to tiny (local first)" in text


def test_with_a_key_but_remote_off_nothing_leaves_the_machine(tmp_path: Path) -> None:
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, content=answer("from the remote"))

    screen, out = console()
    run_chat(
        settings(tmp_path),
        "synthia",
        screen,
        scripted("hello", "/budget", "/exit"),
        httpx.MockTransport(handler),
    )

    assert "no answer: no local model is installed" in out.getvalue()
    assert sent == []


def test_the_local_model_is_not_started_when_the_chat_cannot_begin(
    tmp_path: Path,
) -> None:
    service = service_at(tmp_path)

    with pytest.raises(PersonaError):
        run_chat(
            remote_spent(tmp_path),
            "nobody",
            console()[0],
            local=service,
            remote=NO_REMOTE,
        )

    assert service.server.running is None
    assert not service_threads()


def test_without_a_local_model_or_a_key_the_chat_cannot_begin(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="synthia models install"):
        run_chat(remote_spent(tmp_path), "synthia", console()[0])


def test_the_command_hands_an_installed_local_model_to_the_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = setup_at(tmp_path, CPU, TINY)
    given: list[tuple[LocalService | None, bool]] = []

    def chat(
        *_: object, local: LocalService | None = None, use_remote: bool = False
    ) -> None:
        given.append((local, use_remote))

    def find_local(*_: object, **__: object) -> LocalSetup | None:
        return setup

    monkeypatch.setattr(cli, "find_local", find_local)
    monkeypatch.setattr(cli, "run_chat", chat)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SYNTHIA_HOME", str(tmp_path))

    plain = CliRunner().invoke(app, ["chat"])
    remote = CliRunner().invoke(app, ["chat", "--remote"])

    assert (plain.exit_code, remote.exit_code) == (0, 0)
    assert [type(local) for local, _ in given] == [LocalService, LocalService]
    assert [use_remote for _, use_remote in given] == [False, True]


def test_the_command_logs_to_a_file_not_over_the_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def chat(*_: object, **__: object) -> None:
        logging.getLogger("synthia.test").warning("the vulkan build could not start")

    monkeypatch.setattr(cli, "run_chat", chat)
    # As in a real run: with no handler at all, logging prints to stderr.
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SYNTHIA_HOME", str(tmp_path))

    result = CliRunner().invoke(app, ["chat"])

    assert result.exit_code == 0
    assert "could not start" not in result.stdout + result.stderr
    assert "could not start" in (tmp_path / cli.CHAT_LOG).read_text(encoding="utf-8")
