import logging
import sqlite3
from datetime import UTC, datetime, tzinfo
from pathlib import Path

import numpy as np
import pytest

from synthia.memory.embed import Feeds
from synthia.memory.facts import FactKeeper, FactLearning
from synthia.memory.hybrid import Recall
from synthia.memory.remembering import REMINDED, REMINDER_HEADER, Remembering
from synthia.memory.store import MemoryStore, Turn
from tests.agent.scripted import Scripted
from tests.memory.test_hybrid import Network, embedder
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
