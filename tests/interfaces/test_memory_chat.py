import asyncio
import json
import sqlite3
from pathlib import Path

import httpx
import pytest

from synthia.interfaces.chat import run_chat
from synthia.memory.store import MEMORY_FILE, MemoryStore
from tests.interfaces.daemons import private_daemon
from tests.interfaces.test_chat import console, scripted, settings, thinking_then


def remembered(home: Path) -> list[str]:
    found = asyncio.run(MemoryStore(home / MEMORY_FILE).search(""))
    return [turn.question for turn in reversed(found)]


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
