import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from synthia.agent.trace import MAX_TRACED_CHARS
from synthia.memory.bm25 import Bm25Index
from synthia.memory.store import (
    Asked,
    Fact,
    FactKind,
    FactVector,
    Forgotten,
    MemoryStore,
    Remembered,
    StoredVector,
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


async def test_word_scores_are_bm25_over_the_whole_turn_best_first(
    store: MemoryStore,
) -> None:
    read = ToolUse("read_file", '{"path": "pune.md"}', "x", ok=True)
    texts = [
        ("my sister Asha lives in Pune", "Noted, Asha in Pune.", ()),
        ("the weather today", "Sunny in Pune.", ()),
        ("open my notes", "Done.", (read,)),
        ("nothing related", "Fine.", ()),
    ]
    chat = await store.begin_conversation("SYNTHIA", MONDAY)
    ids = [
        await store.add_turn(chat, said(q, a, MONDAY, *tools)) for q, a, tools in texts
    ]
    words = Bm25Index()
    for id_, (q, a, tools) in zip(ids, texts, strict=True):
        calls = "\n".join(f"{t.name} {t.arguments}" for t in tools)
        words.add(id_, f"{q} {a} {calls}")

    found = await store.word_scores("Asha Pune", limit=10)

    assert found.turns == len(texts)
    assert list(found.scores) == ids[:3]
    assert found.scores == pytest.approx(words.scores("Asha Pune"), rel=1e-12)


async def test_word_scores_keep_to_the_dates_and_the_limit(store: MemoryStore) -> None:
    chat = await store.begin_conversation("SYNTHIA", MONDAY)
    first = await store.add_turn(chat, said("Asha Asha", "a", MONDAY))
    later = MONDAY + timedelta(days=2)
    await store.add_turn(chat, said("Asha", "b", later))

    best = await store.word_scores("Asha", limit=1)
    early = await store.word_scores("Asha", until=MONDAY + timedelta(days=1))
    none = await store.word_scores("  ?! ")

    assert list(best.scores) == [first]
    assert list(early.scores) == [first]
    assert (none.scores, none.turns) == ({}, 2)


async def test_turns_come_back_in_the_order_asked_without_the_forgotten(
    store: MemoryStore,
) -> None:
    chat = await store.begin_conversation("SYNTHIA", MONDAY)
    a, b, c = [await store.add_turn(chat, said(q, "x", MONDAY)) for q in "abc"]
    await store.forget_turn(b)

    found = await store.turns([c, b, a, 404])

    assert asked(found) == ["c", "a"]


async def test_vectors_are_kept_per_model_with_the_time_of_their_turn(
    store: MemoryStore,
) -> None:
    chat = await store.begin_conversation("SYNTHIA", MONDAY)
    later = MONDAY + timedelta(hours=1)
    first = await store.add_turn(chat, said("first", "x", MONDAY))
    second = await store.add_turn(chat, said("second", "x", later))

    kept = await store.put_vectors("mini", [(first, b"\x01\x02"), (second, b"\x03")])

    assert kept == [first, second]
    assert await store.vectors("mini") == [
        StoredVector(first, MONDAY, b"\x01\x02"),
        StoredVector(second, later, b"\x03"),
    ]
    assert await store.vectors("bge") == []


async def test_turns_without_a_vector_from_the_model_in_use_are_listed(
    store: MemoryStore,
) -> None:
    chat = await store.begin_conversation("SYNTHIA", MONDAY)
    first = await store.add_turn(chat, said("first", "x", MONDAY))
    second = await store.add_turn(chat, said("second", "x", MONDAY))
    await store.put_vectors("mini", [(first, b"\x01")])

    assert await store.unembedded("mini", 10) == [Asked(second, MONDAY, "second")]
    assert [a.turn for a in await store.unembedded("bge", 1)] == [first]

    await store.put_vectors("bge", [(first, b"\x02")])

    assert [a.turn for a in await store.unembedded("mini", 10)] == [first, second]
    assert await store.vectors("bge") == [StoredVector(first, MONDAY, b"\x02")]


async def test_a_forgotten_turn_takes_its_vector_and_gets_no_new_one(
    store: MemoryStore,
) -> None:
    chat = await store.begin_conversation("SYNTHIA", MONDAY)
    turn = await store.add_turn(chat, said("forget me", "x", MONDAY))
    await store.put_vectors("mini", [(turn, b"\x01")])

    await store.forget_turn(turn)

    assert await store.vectors("mini") == []
    assert await store.put_vectors("mini", [(turn, b"\x01")]) == []
    assert await store.vectors("mini") == []


async def a_turn(store: MemoryStore, question: str = "q") -> int:
    chat = await store.begin_conversation("SYNTHIA", MONDAY)
    return await store.add_turn(chat, said(question, "x", MONDAY))


async def test_facts_come_back_newest_first_with_their_kind(store: MemoryStore) -> None:
    turn = await a_turn(store)
    later = MONDAY + timedelta(hours=1)
    cat = await store.add_fact(turn, "The cat is Miso.", FactKind.FACT, MONDAY)
    tea = await store.add_fact(turn, "Likes green tea.", FactKind.PREFERENCE, later)

    assert await store.facts() == [
        Fact(tea or 0, "Likes green tea.", FactKind.PREFERENCE, turn, later),
        Fact(cat or 0, "The cat is Miso.", FactKind.FACT, turn, MONDAY),
    ]


async def test_nothing_is_learned_from_a_forgotten_turn(store: MemoryStore) -> None:
    turn = await a_turn(store)
    await store.forget_turn(turn)

    assert await store.add_fact(turn, "Lost.", FactKind.FACT, MONDAY) is None
    assert await store.facts(held=False) == []


async def test_a_changed_fact_is_kept_with_its_end_and_what_changed_it(
    store: MemoryStore,
) -> None:
    turn = await a_turn(store)
    later = MONDAY + timedelta(days=3)
    pune = await store.add_fact(turn, "Sister lives in Pune.", FactKind.FACT, MONDAY)
    mumbai = await store.add_fact(turn, "Sister lives in Mumbai.", FactKind.FACT, later)
    assert pune is not None
    assert mumbai is not None

    assert await store.end_fact(pune, later, mumbai)
    assert not await store.end_fact(pune, later, mumbai)

    assert [f.text for f in await store.facts()] == ["Sister lives in Mumbai."]
    old = (await store.facts(held=False))[1]
    assert (old.id, old.until, old.replaced_by) == (pune, later, mumbai)


async def test_a_forgotten_turn_takes_its_facts_and_their_vectors(
    store: MemoryStore,
) -> None:
    kept = await a_turn(store, "kept")
    gone = await a_turn(store, "gone")
    stays = await store.add_fact(kept, "Stays.", FactKind.FACT, MONDAY)
    leaves = await store.add_fact(gone, "Leaves.", FactKind.FACT, MONDAY)
    assert stays is not None
    assert leaves is not None
    await store.put_fact_vectors("mini", [(stays, b"\x01"), (leaves, b"\x02")])

    await store.forget_turn(gone)

    assert [f.text for f in await store.facts(held=False)] == ["Stays."]
    assert await store.fact_vectors("mini") == [FactVector(stays, b"\x01")]


async def test_a_forgotten_fact_is_gone_and_gets_no_new_vector(
    store: MemoryStore,
) -> None:
    fact = await store.add_fact(await a_turn(store), "Gone.", FactKind.FACT, MONDAY)
    assert fact is not None

    assert await store.forget_fact(fact)
    assert not await store.forget_fact(fact)
    assert await store.facts(held=False) == []
    assert await store.put_fact_vectors("mini", [(fact, b"\x01")]) == []


async def test_vectors_of_changed_facts_are_not_loaded(store: MemoryStore) -> None:
    turn = await a_turn(store)
    old = await store.add_fact(turn, "Old.", FactKind.FACT, MONDAY)
    new = await store.add_fact(turn, "New.", FactKind.FACT, MONDAY)
    assert old is not None
    assert new is not None
    await store.put_fact_vectors("mini", [(old, b"\x01"), (new, b"\x02")])
    await store.end_fact(old, MONDAY, new)

    assert await store.fact_vectors("mini") == [FactVector(new, b"\x02")]
    assert await store.fact_vectors("bge") == []


async def test_forgetting_a_topic_takes_what_holds_all_its_words(
    store: MemoryStore,
) -> None:
    sister = await a_turn(store, "my sister Asha lives in Pune")
    other = await a_turn(store, "Asha from work called")
    await store.add_fact(sister, "Sister is a doctor.", FactKind.FACT, MONDAY)
    await store.add_fact(other, "Asha's sister plays chess.", FactKind.FACT, MONDAY)
    await store.add_fact(other, "Works with Asha.", FactKind.FACT, MONDAY)

    forgotten = await store.forget_about("Asha sister")

    assert forgotten == Forgotten(facts=2, turns=(sister,))
    assert asked(await store.search("Asha")) == ["Asha from work called"]
    assert [f.text for f in await store.facts()] == ["Works with Asha."]
    assert await store.forget_about(" ?! ") == Forgotten(0, ())


async def test_turns_to_learn_from_wait_oldest_first_until_learned(
    store: MemoryStore,
) -> None:
    chat = await store.begin_conversation("SYNTHIA", MONDAY)
    first = await store.add_turn(chat, said("first", "a", MONDAY), to_learn=True)
    await store.add_turn(chat, said("not asked", "b", MONDAY))
    second = await store.add_turn(chat, said("second", "c", MONDAY), to_learn=True)
    third = await store.add_turn(chat, said("third", "d", MONDAY), to_learn=True)

    waiting = await store.to_learn(2)

    assert [(t.id, t.question, t.answer) for t in waiting] == [
        (first, "first", "a"),
        (second, "second", "c"),
    ]
    assert await store.learned(first)
    assert not await store.learned(first)
    assert [t.id for t in await store.to_learn(10)] == [second, third]


async def test_a_turn_stops_waiting_at_its_last_failed_try(store: MemoryStore) -> None:
    chat = await store.begin_conversation("SYNTHIA", MONDAY)
    turn = await store.add_turn(chat, said("q", "a", MONDAY), to_learn=True)

    tries = [await store.tried(turn, most=3) for _ in range(3)]

    assert tries == [False, False, True]
    assert await store.to_learn(10) == []
    assert not await store.tried(turn, most=3)


async def test_a_forgotten_turn_no_longer_waits_to_be_learned_from(
    store: MemoryStore,
) -> None:
    chat = await store.begin_conversation("SYNTHIA", MONDAY)
    turn = await store.add_turn(chat, said("q", "a", MONDAY), to_learn=True)

    await store.forget_turn(turn)

    assert await store.to_learn(10) == []


async def test_a_store_made_before_the_learning_queue_gains_it_empty(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.db"
    older = MemoryStore(path)
    await older.add_turn(
        await older.begin_conversation("SYNTHIA", MONDAY), said("q", "a", MONDAY)
    )
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE facts_pending")
    connection.close()

    reopened = MemoryStore(path)

    assert await reopened.to_learn(10) == []
