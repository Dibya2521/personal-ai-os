import json
import sqlite3
from contextlib import closing
from contextvars import copy_context
from datetime import timedelta
from pathlib import Path

import pytest

from synthia.gateway.confinement import CONFINEMENT, Confinement
from synthia.memory.hybrid import Recall
from synthia.memory.recall import (
    NOTHING_FOUND,
    RECALLED_PRIVATE,
    memory_tool,
    surfaced,
)
from synthia.memory.remembering import Remembering
from synthia.memory.store import MemoryStore, Remembered, Turn
from tests.memory.test_hybrid import MONDAY
from tests.retrieval.toy import Network, embedder

# The turns table as files made before turns could be confined hold it.
OLD_TURNS = """
CREATE TABLE turns (
    id INTEGER PRIMARY KEY,
    conversation INTEGER NOT NULL REFERENCES conversations(id),
    at TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    calls TEXT NOT NULL,
    persona TEXT NOT NULL,
    route TEXT NOT NULL,
    model TEXT NOT NULL
)
"""
OLD_CONVERSATIONS = """
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY,
    started TEXT NOT NULL,
    persona TEXT NOT NULL
)
"""
PRIVATE = "my door code is 4021"
OPEN = "my cat is called miso"


def turn(question: str, *, confined: bool = False, minutes: int = 0) -> Turn:
    at = MONDAY + timedelta(minutes=minutes)
    return Turn(question, "Noted.", at, "SYNTHIA", "local", "m", confined=confined)


async def kept(store: MemoryStore, *turns: Turn, to_learn: bool = False) -> int:
    chat = await store.begin_conversation("SYNTHIA", MONDAY)
    for each in turns:
        await store.add_turn(chat, each, to_learn=to_learn)
    return chat


def remembered(question: str, *, confined: bool) -> Remembered:
    return Remembered(1, 1, MONDAY, question, "Noted.", confined=confined)


async def test_an_old_file_gains_the_mark_and_its_turns_read_as_not_confined(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.db"
    with closing(sqlite3.connect(path)) as old:
        old.execute(OLD_CONVERSATIONS)
        old.execute(OLD_TURNS)
        old.execute("INSERT INTO conversations VALUES (1, '2026-09-28', 'SYNTHIA')")
        old.execute(
            "INSERT INTO turns VALUES (1, 1, '2026-09-28T09:00:00+00:00',"
            " 'hello', 'hi', '', 'SYNTHIA', 'local', 'm')"
        )
        old.commit()

    MemoryStore(path)
    store = MemoryStore(path)

    with closing(sqlite3.connect(path)) as opened:
        columns = [row[1] for row in opened.execute("PRAGMA table_info(turns)")]
    assert columns.count("confined") == 1
    assert [t.confined for t in await store.conversation(1)] == [False]
    chat = await kept(store, turn(PRIVATE, confined=True))
    assert [t.confined for t in await store.conversation(chat)] == [True]


async def test_a_confined_turn_is_kept_marked_and_never_learned_from(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    chat = await kept(
        store,
        turn(PRIVATE, confined=True),
        turn(OPEN, minutes=1),
        to_learn=True,
    )

    waiting = await store.to_learn(10)
    turns = await store.conversation(chat)

    assert [t.question for t in waiting] == [OPEN]
    assert [(t.question, t.confined) for t in turns] == [
        (PRIVATE, True),
        (OPEN, False),
    ]


def test_surfacing_keeps_confined_turns_only_for_a_latch_that_can_stay_local() -> None:
    turns = [remembered(OPEN, confined=False), remembered(PRIVATE, confined=True)]

    def shown(latch: Confinement | None, *, stays: bool, most: int) -> list[str]:
        CONFINEMENT.set(latch)
        return [t.question for t in surfaced(turns, most, lambda: stays)]

    open_latch, beyond, refused = Confinement(), Confinement(), Confinement()

    assert copy_context().run(shown, open_latch, stays=True, most=2) == [OPEN, PRIVATE]
    assert copy_context().run(shown, beyond, stays=True, most=1) == [OPEN]
    assert copy_context().run(shown, refused, stays=False, most=2) == [OPEN]
    assert copy_context().run(shown, None, stays=True, most=2) == [OPEN]
    assert (open_latch.reason, beyond.reason, refused.reason) == (
        RECALLED_PRIVATE,
        None,
        None,
    )


def test_a_left_out_turn_frees_its_place_for_the_next() -> None:
    turns = [remembered(PRIVATE, confined=True), remembered(OPEN, confined=False)]

    assert [t.question for t in surfaced(turns, 1, lambda: True)] == [OPEN]


async def search(
    store: MemoryStore, latch: Confinement | None, *, stays: bool | None
) -> str:
    recall = Recall(store)
    tool = (
        memory_tool(recall)
        if stays is None
        else memory_tool(recall, can_stay_local=lambda: stays)
    )
    CONFINEMENT.set(latch)
    return await tool.run(json.dumps({"query": "door code"}))


CASES = [
    pytest.param(True, True, id="latch-and-local"),
    pytest.param(True, False, id="no-local-model"),
    pytest.param(True, None, id="no-router-to-ask"),
    pytest.param(False, True, id="no-latch"),
]


@pytest.mark.parametrize(("in_chat", "stays"), CASES)
async def test_search_memory_shows_a_private_turn_only_to_a_chat_that_stays_local(
    tmp_path: Path, in_chat: bool, stays: bool | None
) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    await kept(store, turn(PRIVATE, confined=True))
    latch = Confinement() if in_chat else None

    found = await search(store, latch, stays=stays)

    shown = in_chat and stays is True
    assert (PRIVATE in found) is shown
    assert (found == NOTHING_FOUND) is not shown
    if latch is not None:
        assert latch.confined is shown


@pytest.mark.parametrize(("in_chat", "stays"), CASES)
async def test_a_reminder_holds_a_private_turn_only_in_a_chat_that_stays_local(
    tmp_path: Path, in_chat: bool, stays: bool | None
) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    await kept(store, turn(OPEN, confined=True))
    recall = Recall(store, embedder(Network()))
    await recall.backfill()
    memory = (
        Remembering(recall)
        if stays is None
        else Remembering(recall, can_stay_local=lambda: stays)
    )
    latch = Confinement() if in_chat else None
    CONFINEMENT.set(latch)

    reminder = await memory.reminder("what is the name of the pet")

    shown = in_chat and stays is True
    assert (OPEN in reminder) is shown
    assert (reminder == "") is not shown
    if latch is not None:
        assert latch.reason == (RECALLED_PRIVATE if shown else None)
