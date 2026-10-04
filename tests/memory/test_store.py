import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from synthia.agent.trace import MAX_TRACED_CHARS
from synthia.memory.store import (
    MemoryStore,
    Remembered,
    ToolUse,
    Turn,
    match_expression,
)

MONDAY = datetime(2026, 9, 28, 9, 0, tzinfo=UTC)


def said(question: str, answer: str, at: datetime, *tools: ToolUse) -> Turn:
    return Turn(question, answer, at, "SYNTHIA", "local", "qwen", tools)


def asked(found: list[Remembered]) -> list[str]:
    return [turn.question for turn in found]


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory.db")


async def test_a_conversation_comes_back_in_order_with_its_tool_calls(
    store: MemoryStore,
) -> None:
    read = ToolUse("read_file", '{"path": "notes.md"}', "buy milk", ok=True)
    chat = await store.begin_conversation("SYNTHIA", MONDAY)
    await store.add_turn(chat, said("what is in my notes?", "Buy milk.", MONDAY, read))
    later = MONDAY + timedelta(minutes=1)
    await store.add_turn(chat, said("thanks", "Any time.", later))

    first, second = await store.conversation(chat)

    assert (first.question, first.answer, first.at) == (
        "what is in my notes?",
        "Buy milk.",
        MONDAY,
    )
    assert first.tools == (read,)
    assert (second.question, second.tools) == ("thanks", ())


async def test_words_find_turns_best_match_first_within_the_dates(
    store: MemoryStore,
) -> None:
    chat = await store.begin_conversation("SYNTHIA", MONDAY)
    week = [MONDAY + timedelta(days=day) for day in range(7)]
    await store.add_turn(chat, said("my sister Asha lives in Pune", "Noted.", week[0]))
    await store.add_turn(chat, said("Asha called", "How is Asha?", week[2]))
    await store.add_turn(chat, said("the weather", "Sunny.", week[3]))
    await store.add_turn(chat, said("Asha again", "Say hi.", week[6]))

    everything = await store.search("Asha")
    midweek = await store.search("Asha", since=week[1], until=week[6])

    assert asked(everything)[0] == "Asha called"
    assert set(asked(everything)) == {
        "my sister Asha lives in Pune",
        "Asha called",
        "Asha again",
    }
    assert asked(midweek) == ["Asha called"]


async def test_a_word_finds_its_other_forms(store: MemoryStore) -> None:
    chat = await store.begin_conversation("SYNTHIA", MONDAY)
    await store.add_turn(chat, said("I went running today", "Nice.", MONDAY))

    assert asked(await store.search("run")) == ["I went running today"]


async def test_a_tool_call_is_found_by_what_it_was_given(store: MemoryStore) -> None:
    read = ToolUse("read_file", '{"path": "taxes-2026.md"}', "...", ok=True)
    chat = await store.begin_conversation("SYNTHIA", MONDAY)
    await store.add_turn(chat, said("check that file", "Done.", MONDAY, read))

    (found,) = await store.search("which file had the taxes")

    assert found.tools == (read,)


@pytest.mark.parametrize(
    "query",
    ['"', "NEAR(a b) OR *", "a AND (", "-x", "^start", "col:value", "\U0001f600 ok"],
)
async def test_what_a_person_types_is_never_search_syntax(
    store: MemoryStore, query: str
) -> None:
    chat = await store.begin_conversation("SYNTHIA", MONDAY)
    await store.add_turn(chat, said("ok then", "fine", MONDAY))

    await store.search(query)


def test_each_word_is_quoted_and_any_may_match() -> None:
    assert match_expression('tell me "about" Asha!') == (
        '"tell" OR "me" OR "about" OR "Asha"'
    )
    assert match_expression("?!") == ""


async def test_with_no_words_the_newest_turns_in_the_range_come_back(
    store: MemoryStore,
) -> None:
    chat = await store.begin_conversation("SYNTHIA", MONDAY)
    for day in range(3):
        at = MONDAY + timedelta(days=day)
        await store.add_turn(chat, said(f"day {day}", "ok", at))

    found = await store.search("", since=MONDAY, limit=2)

    assert asked(found) == ["day 2", "day 1"]


async def test_a_forgotten_turn_is_gone_from_search_and_its_conversation(
    store: MemoryStore,
) -> None:
    read = ToolUse("read_file", "{}", "secret", ok=True)
    chat = await store.begin_conversation("SYNTHIA", MONDAY)
    turn = await store.add_turn(chat, said("my password hint", "Kept.", MONDAY, read))

    assert await store.forget_turn(turn)
    assert not await store.forget_turn(turn)
    assert await store.search("password") == []
    assert await store.conversation(chat) == []


async def test_a_forgotten_conversation_takes_every_turn_with_it(
    store: MemoryStore,
) -> None:
    gone = await store.begin_conversation("SYNTHIA", MONDAY)
    kept = await store.begin_conversation("Nova", MONDAY)
    await store.add_turn(gone, said("first", "a", MONDAY))
    await store.add_turn(gone, said("second", "b", MONDAY))
    await store.add_turn(kept, said("third", "c", MONDAY))

    assert await store.forget_conversation(gone) == 2
    assert asked(await store.search("first second third")) == ["third"]


async def test_a_turn_for_no_conversation_is_refused_and_nothing_is_kept(
    store: MemoryStore,
) -> None:
    read = ToolUse("read_file", "{}", "x", ok=True)

    with pytest.raises(sqlite3.IntegrityError):
        await store.add_turn(404, said("lost", "answer", MONDAY, read))

    assert await store.search("lost") == []
    assert await store.forget_conversation(404) == 0


async def test_another_store_on_the_same_file_sees_what_was_written(
    tmp_path: Path,
) -> None:
    path = tmp_path / "home" / "memory.db"
    writer = MemoryStore(path)
    chat = await writer.begin_conversation("SYNTHIA", MONDAY)
    await writer.add_turn(chat, said("remember the blue door", "I will.", MONDAY))

    found = await MemoryStore(path).search("door")

    assert asked(found) == ["remember the blue door"]


async def test_text_that_cannot_be_utf8_is_kept_as_its_escape(
    store: MemoryStore,
) -> None:
    chat = await store.begin_conversation("SYNTHIA", MONDAY)
    await store.add_turn(chat, said("half an emoji", "here \ud83d it is", MONDAY))

    (found,) = await store.search("emoji")

    assert found.answer == "here \\ud83d it is"


async def test_a_long_tool_result_is_cut_as_the_trace_cuts_it(
    store: MemoryStore,
) -> None:
    read = ToolUse("read_file", "{}", "x" * (MAX_TRACED_CHARS + 50), ok=False)
    chat = await store.begin_conversation("SYNTHIA", MONDAY)
    await store.add_turn(chat, said("read it", "It was long.", MONDAY, read))

    (found,) = await store.conversation(chat)

    assert found.tools[0].result == "x" * MAX_TRACED_CHARS
    assert found.tools[0].ok is False
