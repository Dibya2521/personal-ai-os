import logging
import sqlite3
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path

import numpy as np
import pytest

from synthia.memory.embed import Feeds
from synthia.memory.facts import FactKeeper, FactLearning
from synthia.memory.hybrid import Recall
from synthia.memory.remembering import (
    PINNED_HEADER,
    REMINDED,
    REMINDER_HEADER,
    Remembering,
)
from synthia.memory.store import Fact, FactKind, Forgotten, MemoryStore, Turn
from tests.agent.scripted import Scripted
from tests.memory.test_hybrid import Network, asked, embedder
from tests.models.fakes import EMBEDDER

AT = datetime(2026, 10, 3, 12, 0, tzinfo=UTC)


def turn(question: str) -> Turn:
    return Turn(question, "ok", AT, "SYNTHIA", "local", "qwen")


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory.db")


async def test_turns_join_one_conversation_until_it_begins_again(
    store: MemoryStore,
) -> None:
    memory = Remembering(Recall(store))
    await memory.remember(turn("first"))
    await memory.remember(turn("second"))
    memory.begin_again()
    await memory.remember(turn("third"))

    found = {t.question: t.conversation for t in await store.search("")}

    assert found["first"] == found["second"] != found["third"]


async def test_a_private_conversation_keeps_nothing(store: MemoryStore) -> None:
    memory = Remembering(Recall(store))
    memory.private = True
    await memory.remember(turn("secret"))

    assert await store.search("") == []
    assert not await memory.forget_last()


async def test_only_the_last_turn_is_forgotten_and_only_once(
    store: MemoryStore,
) -> None:
    memory = Remembering(Recall(store))
    await memory.remember(turn("keep this"))
    await memory.remember(turn("drop this"))

    assert await memory.forget_last()
    assert not await memory.forget_last()
    assert [t.question for t in await store.search("")] == ["keep this"]


async def test_a_turn_that_cannot_be_kept_is_logged_and_the_chat_goes_on(
    store: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def broken(conversation: int, kept: Turn, *, to_learn: bool = False) -> int:
        del conversation, kept, to_learn
        message = "database is locked"
        raise sqlite3.OperationalError(message)

    monkeypatch.setattr(store, "add_turn", broken)
    memory = Remembering(Recall(store))

    with caplog.at_level(logging.ERROR, "synthia.memory.remembering"):
        await memory.remember(turn("lost"))

    assert "a turn was not remembered" in caplog.text
    assert not await memory.forget_last()


class FailsAfterStart(Network):
    """A network that answers the width check at start, then fails."""

    def __call__(self, feeds: Feeds) -> np.ndarray:
        if self.texts:
            message = "the network failed"
            raise ValueError(message)
        return super().__call__(feeds)


async def test_a_remembered_turn_is_found_by_meaning_until_it_is_forgotten(
    store: MemoryStore,
) -> None:
    recall = Recall(store, embedder(Network()))
    memory = Remembering(recall)
    await memory.remember(turn("my cat is called miso"))

    found = await recall.find("pet", limit=5)

    assert [r.turn.question for r in found] == ["my cat is called miso"]
    assert await memory.forget_last()
    assert await recall.find("pet", limit=5) == []


async def test_a_turn_whose_meaning_cannot_be_made_is_still_remembered(
    store: MemoryStore, caplog: pytest.LogCaptureFixture
) -> None:
    memory = Remembering(Recall(store, embedder(FailsAfterStart())))

    with caplog.at_level(logging.ERROR, "synthia.memory.remembering"):
        await memory.remember(turn("my cat"))

    assert "a turn's meaning was not kept" in caplog.text
    assert [t.question for t in await store.search("cat")] == ["my cat"]
    assert [a.question for a in await store.unembedded(EMBEDDER.id, 5)] == ["my cat"]


def utc() -> tzinfo:
    return UTC


async def test_a_reminder_holds_close_turns_of_other_conversations_only(
    store: MemoryStore,
) -> None:
    recall = Recall(store, embedder(Network()))
    earlier = Remembering(recall)
    await earlier.remember(turn("my cat is called miso"))
    await earlier.remember(turn("my sister lives in pune"))
    now = Remembering(recall)
    await now.remember(turn("the pet cat is asleep"))

    reminder = await now.reminder("what is the name of the pet", zone=utc)

    assert reminder == (
        f"{REMINDER_HEADER}\n"
        "Saturday 2026-10-03 12:00 (conversation 1, turn 1)\n"
        "  person: my cat is called miso\n"
        "  SYNTHIA: ok\n\n"
    )


async def test_a_reminder_holds_at_most_three_turns(store: MemoryStore) -> None:
    recall = Recall(store, embedder(Network()))
    earlier = Remembering(recall)
    for question in ("my cat", "the cat", "a pet cat", "cat miso", "my pet"):
        await earlier.remember(turn(question))

    reminder = await Remembering(recall).reminder("my pet cat", zone=utc)

    assert reminder.count("  person: ") == REMINDED == 3


async def test_nothing_is_recalled_before_a_question_without_an_embedding_model(
    store: MemoryStore,
) -> None:
    recall = Recall(store)
    await Remembering(recall).remember(turn("my cat is called miso"))

    assert await Remembering(recall).reminder("my cat is called miso") == ""


async def test_a_search_that_fails_leaves_the_question_alone(
    store: MemoryStore, caplog: pytest.LogCaptureFixture
) -> None:
    memory = Remembering(Recall(store, embedder(FailsAfterStart())))

    with caplog.at_level(logging.ERROR, "synthia.memory.remembering"):
        reminder = await memory.reminder("what is the name of the pet")

    assert reminder == ""
    assert "earlier conversations were not searched" in caplog.text


async def test_with_facts_learned_each_kept_turn_waits_for_the_learner(
    store: MemoryStore,
) -> None:
    recall = Recall(store, embedder(Network()))
    learning = FactLearning(FactKeeper(store, embedder(Network()), Scripted()), store)
    memory = Remembering(recall, learning)
    await memory.remember(turn("my cat is called miso"))
    memory.private = True
    await memory.remember(turn("a secret"))

    waiting = await store.to_learn(5)

    assert [t.question for t in waiting] == ["my cat is called miso"]


async def test_without_facts_learned_no_turn_waits(store: MemoryStore) -> None:
    await Remembering(Recall(store)).remember(turn("my cat is called miso"))

    assert await store.to_learn(5) == []


async def learned(store: MemoryStore, *texts: str) -> list[int]:
    """Keep ``texts`` as facts learned a minute apart, oldest first."""
    conversation = await store.begin_conversation("SYNTHIA", AT)
    source = await store.add_turn(conversation, turn("about me"))
    ids: list[int] = []
    for minute, text in enumerate(texts):
        at = AT + timedelta(minutes=minute)
        fact = await store.add_fact(source, text, FactKind.FACT, at)
        assert fact is not None
        ids.append(fact)
    return ids


async def source_of(store: MemoryStore) -> int:
    """Return the turn the facts were learned from."""
    return (await store.facts())[0].turn


async def test_pinned_facts_are_the_ones_held_newest_first(store: MemoryStore) -> None:
    pune, _, mumbai = await learned(
        store, "Lives in Pune.", "Has a cat named Miso.", "Lives in Mumbai."
    )
    await store.end_fact(pune, AT + timedelta(minutes=3), mumbai)

    pinned = await Remembering(Recall(store)).pinned(1000)

    assert pinned == f"{PINNED_HEADER}\n- Lives in Mumbai.\n- Has a cat named Miso."


async def test_pinned_facts_end_at_the_last_whole_fact_that_fits(
    store: MemoryStore,
) -> None:
    await learned(store, "Has a cat named Miso.", "Lives in Mumbai.")
    newest = f"{PINNED_HEADER}\n- Lives in Mumbai."
    both = f"{newest}\n- Has a cat named Miso."

    async def pinned(max_chars: int) -> str:
        return await Remembering(Recall(store)).pinned(max_chars)

    assert await pinned(len(both)) == both
    assert await pinned(len(both) - 1) == newest
    assert await pinned(len(newest)) == newest
    assert await pinned(len(newest) - 1) == ""


async def test_nothing_is_pinned_before_any_fact_is_learned(
    store: MemoryStore,
) -> None:
    assert await Remembering(Recall(store)).pinned(1000) == ""


async def test_facts_that_cannot_be_read_pin_nothing_and_are_logged(
    store: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def broken(*, held: bool = True) -> list[Fact]:
        del held
        message = "database is locked"
        raise sqlite3.OperationalError(message)

    await learned(store, "Has a cat named Miso.")
    memory = Remembering(Recall(store))
    monkeypatch.setattr(store, "facts", broken)

    with caplog.at_level(logging.ERROR, "synthia.memory.remembering"):
        pinned = await memory.pinned(1000)
    monkeypatch.undo()

    assert pinned == ""
    assert "facts about the person were not read" in caplog.text
    assert await memory.pinned(1000) == f"{PINNED_HEADER}\n- Has a cat named Miso."


async def test_facts_that_cannot_be_read_later_pin_nothing_for_that_turn_only(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken(*, held: bool = True) -> list[Fact]:
        del held
        message = "database is locked"
        raise sqlite3.OperationalError(message)

    await learned(store, "Has a cat named Miso.")
    memory = Remembering(Recall(store))
    first = await memory.pinned(1000)
    monkeypatch.setattr(store, "facts", broken)
    unread = await memory.pinned(1000)
    monkeypatch.undo()
    await store.add_fact(await source_of(store), "Lives in Mumbai.", FactKind.FACT, AT)

    assert unread == ""
    assert (
        await memory.pinned(1000)
        == first
        == (f"{PINNED_HEADER}\n- Has a cat named Miso.")
    )


async def test_a_pinned_fact_the_learner_ends_stays_pinned_until_the_next_conversation(
    store: MemoryStore,
) -> None:
    (pune,) = await learned(store, "Lives in Pune.")
    memory = Remembering(Recall(store))
    first = await memory.pinned(1000)
    later = AT + timedelta(minutes=1)
    mumbai = await store.add_fact(
        await source_of(store), "Lives in Mumbai.", FactKind.FACT, later
    )
    assert mumbai is not None
    await store.end_fact(pune, later, mumbai)

    assert await memory.pinned(1000) == first == f"{PINNED_HEADER}\n- Lives in Pune."
    memory.begin_again()
    assert await memory.pinned(1000) == f"{PINNED_HEADER}\n- Lives in Mumbai."


async def test_pinned_facts_are_taken_once_until_one_of_them_is_forgotten(
    store: MemoryStore,
) -> None:
    (miso,) = await learned(store, "Has a cat named Miso.")
    here, elsewhere = Remembering(Recall(store)), Remembering(Recall(store))
    first = await here.pinned(1000)
    await store.add_fact(await source_of(store), "Lives in Mumbai.", FactKind.FACT, AT)

    unchanged = await here.pinned(1000)
    await elsewhere.forget_fact(miso)

    assert unchanged == first == f"{PINNED_HEADER}\n- Has a cat named Miso."
    assert await here.pinned(1000) == f"{PINNED_HEADER}\n- Lives in Mumbai."


async def test_a_pinned_fact_whose_id_went_to_a_new_fact_is_taken_afresh(
    store: MemoryStore,
) -> None:
    _, pune = await learned(store, "Has a cat named Miso.", "Lives in Pune.")
    memory = Remembering(Recall(store))
    await memory.pinned(1000)
    source = await source_of(store)
    await store.forget_fact(pune)
    later = AT + timedelta(minutes=5)

    reused = await store.add_fact(source, "Lives in Mumbai.", FactKind.FACT, later)

    assert reused == pune
    assert await memory.pinned(1000) == (
        f"{PINNED_HEADER}\n- Lives in Mumbai.\n- Has a cat named Miso."
    )


async def test_a_new_conversation_takes_the_facts_afresh(store: MemoryStore) -> None:
    await learned(store, "Has a cat named Miso.")
    memory = Remembering(Recall(store))
    await memory.pinned(1000)
    await store.add_fact(await source_of(store), "Lives in Mumbai.", FactKind.FACT, AT)

    memory.begin_again()

    assert await memory.pinned(1000) == (
        f"{PINNED_HEADER}\n- Lives in Mumbai.\n- Has a cat named Miso."
    )


async def test_a_forgotten_fact_comes_back_once_and_is_held_no_more(
    store: MemoryStore,
) -> None:
    miso, mumbai = await learned(store, "Has a cat named Miso.", "Lives in Mumbai.")
    memory = Remembering(Recall(store))

    forgotten = await memory.forget_fact(miso)

    assert forgotten is not None
    assert forgotten.text == "Has a cat named Miso."
    assert await memory.forget_fact(miso) is None
    assert [f.id for f in await memory.facts()] == [mumbai]


async def test_forgetting_the_newer_version_of_a_fact_leaves_the_older_ended(
    store: MemoryStore,
) -> None:
    pune, mumbai = await learned(store, "Lives in Pune.", "Lives in Mumbai.")
    moved = AT + timedelta(minutes=2)
    await store.end_fact(pune, moved, mumbai)
    memory = Remembering(Recall(store))

    await memory.forget_fact(mumbai)

    assert await memory.facts() == []
    (older,) = await store.facts(held=False)
    assert (older.text, older.until, older.replaced_by) == (
        "Lives in Pune.",
        moved,
        None,
    )


async def test_a_forgotten_topic_leaves_recall_and_the_last_turn_stays(
    store: MemoryStore,
) -> None:
    recall = Recall(store, embedder(Network()))
    memory = Remembering(recall)
    await memory.remember(turn("my cat is called miso"))
    await memory.remember(turn("my sister lives in pune"))

    forgotten = await memory.forget_about("miso")

    assert forgotten == Forgotten(0, (1,))
    # A vector left behind would take the one place and fetch nothing.
    assert asked(await recall.find("pet", limit=1)) == ["my sister lives in pune"]
    assert await memory.forget_last()


async def test_forget_never_takes_a_turn_of_another_conversation(
    store: MemoryStore,
) -> None:
    recall = Recall(store, embedder(Network()))
    here, elsewhere = Remembering(recall), Remembering(recall)
    await here.remember(turn("my cat is called miso"))
    await elsewhere.forget_about("miso")
    # SQLite gives the freed id to the next turn, wherever it is said.
    await elsewhere.remember(turn("my sister lives in pune"))

    assert not await here.forget_last()
    assert [t.id for t in await store.search("")] == [1]
    assert asked(await recall.find("pet", limit=1)) == ["my sister lives in pune"]
