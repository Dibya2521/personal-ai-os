import asyncio
import datetime as dt
from pathlib import Path
from typing import cast

import pytest

from synthia.agent.loop import Limits
from synthia.interfaces.daemon_client import ask_daemon
from synthia.memory.store import MEMORY_FILE, MemoryStore, Turn
from synthia.server import host
from synthia.server.api import ANSWER_FAILED, INVALID_PARAMS, MAX_CHAT_CHARS
from tests.agent.scripted import calls, says
from tests.interfaces.daemons import private_daemon, toy_embedder
from tests.knowledge.corpus import planted
from tests.server.test_confinement import Serving
from tests.server.test_daemon import Failing, Local
from tests.server.test_knowledge_api import Gated, keyed, refused
from tests.timing import HANG_TIMEOUT_S

SEARCH = ("search_knowledge", '{"query": "kestrel"}')


async def chat_in_daemon(
    home: Path, model: Local, params: dict[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    """Share three folders, chat as another agent, then search memory for the chat."""
    shared = planted(home / "public", {"bird.txt": "a kestrel over the field\n"})
    diary = planted(home / "diary", {"nest.txt": "the kestrel nests behind my house\n"})
    other = planted(home / "other", {"facts.txt": "kestrel facts nobody shared\n"})
    with private_daemon(keyed(home), local=Serving(model), embedder=toy_embedder):
        await ask_daemon(home, "knowledge_add", {"root": str(shared)})
        await ask_daemon(home, "knowledge_add", {"root": str(diary), "private": True})
        await ask_daemon(home, "knowledge_add", {"root": str(other)})
        await ask_daemon(home, "knowledge_index", {})
        answered = await ask_daemon(home, "host_chat", params)
        remembered = await ask_daemon(home, "memory_search", {"query": "kestrel"})
    return cast("dict[str, object]", answered), cast("dict[str, object]", remembered)


def offered(model: Local) -> list[str]:
    return sorted(tool.name for tool in model.requests[0].tools)


async def test_another_agents_chat_reads_only_its_sources_and_is_not_remembered(
    tmp_path: Path,
) -> None:
    model = Local(calls(SEARCH), says("A kestrel hovers over the field."))

    answered, remembered = await chat_in_daemon(
        tmp_path,
        model,
        {"text": "where is the kestrel?", "sources": ["public", "diary"]},
    )

    assert answered == {
        "answer": "A kestrel hovers over the field.",
        "sources": ["public/bird.txt:1"],
    }
    assert offered(model) == ["calculate", "current_time", "search_knowledge"]
    assert not any(request.use_remote for request in model.requests)
    shown = repr(model.requests[1].messages)
    assert "over the field" in shown
    assert "behind my house" not in shown
    assert "nobody shared" not in shown
    assert remembered["turns"] == []


async def test_memory_is_offered_only_when_shared_and_no_source_means_no_search(
    tmp_path: Path,
) -> None:
    model = Local(says("Hello."))

    answered, _ = await chat_in_daemon(
        tmp_path, model, {"text": "hello", "memory": True}
    )

    assert answered == {"answer": "Hello.", "sources": []}
    assert offered(model) == ["calculate", "current_time", "search_memory"]


async def test_a_private_grant_shows_private_passages_of_shared_sources(
    tmp_path: Path,
) -> None:
    model = Local(calls(SEARCH), says("It nests behind the house."))

    answered, _ = await chat_in_daemon(
        tmp_path,
        model,
        {"text": "where does it nest?", "sources": ["diary"], "private": True},
    )

    assert answered["sources"] == ["diary/nest.txt:1"]
    assert "behind my house" in repr(model.requests[1].messages)


async def test_shared_memory_still_hides_turns_of_private_chats(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / MEMORY_FILE)
    for day, confined in ((1, False), (5, True)):
        at = dt.datetime(2026, 11, day, 12, tzinfo=dt.UTC)
        said = f"the kestrel was here on day {day}"
        turn = Turn("kestrel?", said, at, "synthia", "local", "m", confined=confined)
        await store.add_turn(await store.begin_conversation("synthia", at), turn)
    model = Local(calls(("search_memory", '{"query": "kestrel"}')), says("Day 1."))

    await chat_in_daemon(tmp_path, model, {"text": "kestrel?", "memory": True})

    shown = repr(model.requests[1].messages)
    assert "on day 1" in shown
    assert "on day 5" not in shown


async def test_a_chat_that_runs_past_the_deadline_ends_with_what_it_has(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(host, "HOST_CHAT_LIMITS", Limits(deadline_s=0.5))
    model = Gated()

    answered, _ = await asyncio.wait_for(
        chat_in_daemon(tmp_path, model, {"text": "hello"}), HANG_TIMEOUT_S
    )

    assert answered == {"answer": "", "sources": []}


async def test_a_chat_the_model_cannot_answer_is_an_error(tmp_path: Path) -> None:
    with private_daemon(
        keyed(tmp_path), local=Serving(Failing()), embedder=toy_embedder
    ):
        failed = await refused(tmp_path, "host_chat", {"text": "hello"})
    with private_daemon(keyed(tmp_path), embedder=toy_embedder):
        no_local = await refused(tmp_path, "host_chat", {"text": "hello"})

    assert failed.code == ANSWER_FAILED
    assert str(failed).startswith("no answer: ")
    assert "the model broke" in str(failed)
    assert no_local.code == ANSWER_FAILED
    assert str(no_local).startswith(
        "no answer: chat runs only on this machine's local model: "
    )


async def test_a_message_that_does_not_fit_is_refused(tmp_path: Path) -> None:
    with private_daemon(keyed(tmp_path), embedder=toy_embedder):
        empty = await refused(tmp_path, "host_chat", {"text": ""})
        long = await refused(
            tmp_path, "host_chat", {"text": "x" * (MAX_CHAT_CHARS + 1)}
        )
        tools = await refused(tmp_path, "host_chat", {"text": "hi", "tools": ["x"]})

    assert [e.code for e in (empty, long, tools)] == [INVALID_PARAMS] * 3
