import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from synthia.memory.embed import Feeds
from synthia.memory.hybrid import Recall
from synthia.memory.remembering import Remembering
from synthia.memory.store import MemoryStore, Turn
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
    async def broken(conversation: int, kept: Turn) -> int:
        del conversation, kept
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
