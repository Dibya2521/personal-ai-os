import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import httpx
import pytest

from synthia.gateway.assemble import build_gateway
from synthia.interfaces.chat import run_chat
from synthia.memory.hybrid import Recall
from synthia.memory.remembering import PINNED_HEADER, Remembering
from synthia.memory.store import MEMORY_FILE, FactKind, MemoryStore, Turn
from synthia.persona.library import PersonaLibrary
from synthia.server.conversation import Conversation
from synthia.server.session import ChatSession, LastRoute
from tests.interfaces.daemons import private_daemon
from tests.interfaces.test_chat import console, scripted, settings, thinking_then

AT = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)


def remembered(home: Path) -> list[str]:
    found = asyncio.run(MemoryStore(home / MEMORY_FILE).search(""))
    return [turn.question for turn in reversed(found)]


def learned(home: Path, *taught: tuple[str, str, FactKind]) -> None:
    """Keep each (question, fact, kind) as a turn and the fact learned from it."""

    async def keep() -> None:
        store = MemoryStore(home / MEMORY_FILE)
        conversation = await store.begin_conversation("SYNTHIA", AT)
        for minute, (question, fact, kind) in enumerate(taught):
            at = AT + timedelta(minutes=minute)
            asked = Turn(question, "Noted.", at, "SYNTHIA", "local", "qwen")
            source = await store.add_turn(conversation, asked)
            await store.add_fact(source, fact, kind, at)

    asyncio.run(keep())


def system_prompt(body: dict[str, object]) -> str:
    messages = cast("list[dict[str, object]]", body["messages"])
    return str(messages[0]["content"])


def test_learned_facts_are_listed_pinned_and_forgotten_on_request(
    tmp_path: Path,
) -> None:
    learned(
        tmp_path,
        ("my cat is called Miso", "Has a cat named Miso.", FactKind.FACT),
        ("keep it short please", "Prefers short answers.", FactKind.PREFERENCE),
        ("my sister lives in Pune", "Has a sister in Pune.", FactKind.FACT),
    )
    bodies: list[dict[str, object]] = []

    def reply(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(404)
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=thinking_then("Noted."))

    screen, out = console()
    read = scripted(
        "/facts",
        "hello",
        "/forget fact 3",
        "/forget fact 9",
        "how are you",
        "/forget about miso",
        "/forget about zebra",
        "fine",
        "/forget fact 2",
        "/facts",
        "bye",
    )

    with private_daemon(settings(tmp_path), httpx.MockTransport(reply)) as daemon:
        run_chat(daemon, screen, read, use_remote=True)

    text = out.getvalue()
    listed = (
        "3  Has a sister in Pune.",
        "2  Prefers short answers. (preference)",
        "1  Has a cat named Miso.",
    )
    assert [text.index(line) for line in listed] == sorted(
        text.index(line) for line in listed
    )
    assert "forgot fact 3: Has a sister in Pune." in text
    assert "there is no fact 9; /facts lists them" in text
    assert "forgot 1 fact and 1 turn holding all of: miso" in text
    assert "forgot 0 facts and 0 turns holding all of: zebra" in text
    assert text.count("this conversation keeps what was said in it until /reset") == 1
    assert "forgot fact 2: Prefers short answers." in text
    assert "no facts learned yet" in text
    first, second, third, fourth = (system_prompt(body) for body in bodies)
    assert first.endswith(
        f"\n\n{PINNED_HEADER}\n- Has a sister in Pune.\n"
        "- Prefers short answers.\n- Has a cat named Miso."
    )
    assert second.endswith(
        f"\n\n{PINNED_HEADER}\n- Prefers short answers.\n- Has a cat named Miso."
    )
    assert third.endswith(f"\n\n{PINNED_HEADER}\n- Prefers short answers.")
    assert PINNED_HEADER not in fourth
    assert remembered(tmp_path) == [
        "keep it short please",
        "my sister lives in Pune",
        "hello",
        "how are you",
        "fine",
        "bye",
    ]


async def test_a_memory_that_fails_turns_each_fact_command_into_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def locked(*args: object, **kwargs: object) -> None:
        del args, kwargs
        message = "database is locked"
        raise sqlite3.OperationalError(message)

    store = MemoryStore(tmp_path / MEMORY_FILE)
    monkeypatch.setattr(store, "facts", locked)
    monkeypatch.setattr(store, "forget_about", locked)
    routes = LastRoute()
    async with httpx.AsyncClient() as client:
        gateway = build_gateway(settings(tmp_path), client, routes)
        memory = Remembering(Recall(store))
        session = ChatSession(
            gateway.model, PersonaLibrary(), "synthia", routes, memory=memory
        )
        conversation = Conversation(session, gateway)
        replies = [
            await conversation.facts(),
            await conversation.forget_fact(1),
            await conversation.forget_about("miso"),
        ]

    assert [(r.notes, r.errors) for r in replies] == [
        ((), ("the facts could not be read: database is locked",)),
        ((), ("fact 1 could not be forgotten: database is locked",)),
        ((), ("'miso' could not be forgotten: database is locked",)),
    ]


def test_finished_turns_are_remembered_and_forget_takes_one_back(
    tmp_path: Path,
) -> None:
    bodies: list[dict[str, object]] = []

    def reply(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(404)
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=thinking_then("Noted."))

    screen, out = console()
    read = scripted(
        "/forget",
        "my cat is Miso",
        "/forget",
        "my dog is Rex",
        "/private on",
        "my secret plan",
        "/private",
        "/private off",
        "what are my pets called",
    )

    with private_daemon(settings(tmp_path), httpx.MockTransport(reply)) as daemon:
        run_chat(daemon, screen, read, use_remote=True)

    text = out.getvalue()
    assert "no turn to forget" in text
    assert "the last turn is forgotten" in text
    assert text.count("private: on, nothing from here on is remembered") == 2
    assert "private: off, each finished turn is remembered" in text
    assert remembered(tmp_path) == ["my dog is Rex", "what are my pets called"]
    # The forgotten turn left the conversation too: no later request holds it.
    assert "Miso" not in json.dumps(bodies[1:])
    assert "my secret plan" in json.dumps(bodies[-1])


def test_a_memory_that_cannot_open_is_a_warning_and_the_chat_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(self: MemoryStore, path: Path) -> None:
        del self, path
        message = "unable to open database file"
        raise sqlite3.OperationalError(message)

    monkeypatch.setattr(MemoryStore, "__init__", broken)
    screen, out = console()

    with private_daemon(settings(tmp_path)) as daemon:
        run_chat(daemon, screen, scripted("/private"))

    text = out.getvalue()
    assert "nothing will be remembered: unable to open database file" in text
    assert "nothing is remembered in this conversation" in text
