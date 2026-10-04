import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from synthia.agent.loop import ToolFinished
from synthia.agent.tools import Effect, InvalidArgumentsError, Reach, Toolbox
from synthia.gateway.types import ChatChunk, ChatRequest
from synthia.memory.recall import MAX_PREVIEW_CHARS, NOTHING_FOUND, memory_tool
from synthia.memory.remembering import Remembering
from synthia.memory.store import MemoryStore, ToolUse, Turn
from synthia.persona.library import PersonaLibrary
from synthia.server.session import ChatSession
from synthia.tools.basic import clock_tool
from tests.agent.scripted import Scripted, calls, says

IST = timezone(timedelta(hours=5, minutes=30))


def at(day: int, hour: int) -> datetime:
    return datetime(2026, 9, day, hour, tzinfo=IST)


async def seeded(path: Path) -> MemoryStore:
    store = MemoryStore(path)
    chat = await store.begin_conversation("SYNTHIA", at(21, 9))
    for question, answer, when in (
        ("my cat is called Miso", "Miso is a lovely name.", at(22, 23)),
        ("the cat knocked a cup over", "Classic Miso.", at(28, 1)),
        ("lunch ideas?", "Dal and rice.", at(29, 13)),
    ):
        await store.add_turn(
            chat, Turn(question, answer, when, "SYNTHIA", "local", "qwen")
        )
    return store


async def search(store: MemoryStore, **arguments: object) -> str:
    tool = memory_tool(store, zone=lambda: IST)
    return await tool.run(json.dumps(arguments))


async def test_it_reads_this_machine_and_never_asks(tmp_path: Path) -> None:
    tool = memory_tool(await seeded(tmp_path / "m.db"))

    found = await tool.run('{"query": "lunch", "since": "2026-09-01"}')

    assert (tool.spec.name, tool.reach, tool.effect) == (
        "search_memory",
        Reach.LOCAL,
        Effect.READ,
    )
    assert "lunch ideas?" in found


async def test_days_are_whole_days_on_this_machines_clock(tmp_path: Path) -> None:
    store = await seeded(tmp_path / "m.db")

    # 22 September 23:00 here is still 22 September, though in UTC it is 17:30.
    one_day = await search(store, query="cat", since="2026-09-22", until="2026-09-22")
    the_week = await search(store, query="cat", since="2026-09-28", until="2026-10-04")

    assert "my cat is called Miso" in one_day
    assert "knocked" not in one_day
    assert "Monday 2026-09-28 01:00" in the_week
    assert "my cat is called Miso" not in the_week


async def test_with_no_words_the_newest_turns_come_back(tmp_path: Path) -> None:
    found = await search(await seeded(tmp_path / "m.db"), query="")

    assert found.index("lunch ideas?") < found.index("knocked a cup")


async def test_nothing_found_says_so(tmp_path: Path) -> None:
    found = await search(await seeded(tmp_path / "m.db"), query="dog")

    assert found == NOTHING_FOUND


async def test_a_long_answer_is_cut_and_its_tools_are_named(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "m.db")
    chat = await store.begin_conversation("SYNTHIA", at(21, 9))
    read = ToolUse("read_file", "{}", "...", ok=True)
    long = "word " * 200
    await store.add_turn(
        chat, Turn("summarise notes", long, at(21, 9), "SYNTHIA", "local", "q", (read,))
    )

    found = await search(store, query="notes")

    answer_line = found.splitlines()[2]
    assert answer_line.endswith("...")
    assert len(answer_line) == len("  SYNTHIA: ") + MAX_PREVIEW_CHARS + 3
    assert found.splitlines()[3] == "  tools used: read_file"


async def test_a_date_that_is_not_a_date_goes_back_to_the_model(
    tmp_path: Path,
) -> None:
    with pytest.raises(InvalidArgumentsError, match="since"):
        await search(await seeded(tmp_path / "m.db"), query="x", since="last week")


async def test_what_did_i_tell_you_last_week_is_answered_from_memory(
    tmp_path: Path,
) -> None:
    store = await seeded(tmp_path / "m.db")

    def answer_from_result(request: ChatRequest) -> list[ChatChunk]:
        result = request.messages[-1].text
        assert "my cat is called Miso" in result
        return says("You told me your cat is called Miso.")

    model = Scripted(
        calls(("current_time", "{}")),
        calls(
            (
                "search_memory",
                '{"query": "cat", "since": "2026-09-21", "until": "2026-09-27"}',
            )
        ),
        answer_from_result,
    )
    morning = datetime(2026, 10, 3, 10, tzinfo=IST)
    tools = Toolbox([clock_tool(lambda: morning), memory_tool(store, lambda: IST)])
    session = ChatSession(
        model,
        PersonaLibrary(),
        "synthia",
        tools=tools,
        memory=Remembering(store),
    )

    question = "what did I tell you about my cat last week?"
    items = [item async for item in session.turn(question)]

    ran = [i.call.name for i in items if isinstance(i, ToolFinished)]
    assert ran == ["current_time", "search_memory"]
    assert session.history[-1].text == "You told me your cat is called Miso."
    kept = (await store.search("tell you about my cat last week"))[0]
    assert kept.question == question
    assert [use.name for use in kept.tools] == ["current_time", "search_memory"]
