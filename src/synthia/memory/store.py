"""SYNTHIA's episodic memory: every finished turn, kept and found by its words.

Each conversation is a row, each finished turn another, with the tool calls it
made; their arguments and results are cut as the trace cuts them, while the
question and the answer are kept whole, since they are what is remembered. A
full-text index over each turn's question, answer and tool calls, kept in step
by triggers, finds turns by their words, ranked by BM25, within an optional
time range.

The file follows the budget ledger's rules: a connection per call, write-ahead
logging so a reader never waits for a writer, the work done off the event
loop. Times are stored as UTC ISO text, which sorts as time does.
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

from synthia.agent.trace import MAX_TRACED_CHARS
from synthia.kernel.text import mended

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

MEMORY_FILE: Final = Path("memory.db")
BUSY_TIMEOUT_S: Final = 5.0
_WORD: Final = re.compile(r"\w+")
_EARLIEST: Final = "0000"
_LATEST: Final = "9999"
FTS_TOKENIZER: Final = "porter unicode61"

_SCHEMA: Final = (
    """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY,
    started TEXT NOT NULL,
    persona TEXT NOT NULL
)
""",
    """
CREATE TABLE IF NOT EXISTS turns (
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
""",
    "CREATE INDEX IF NOT EXISTS turns_by_time ON turns(at)",
    "CREATE INDEX IF NOT EXISTS turns_by_conversation ON turns(conversation)",
    """
CREATE TABLE IF NOT EXISTS tool_calls (
    turn INTEGER NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    name TEXT NOT NULL,
    arguments TEXT NOT NULL,
    result TEXT NOT NULL,
    ok INTEGER NOT NULL,
    PRIMARY KEY (turn, position)
)
""",
    f"""
CREATE VIRTUAL TABLE IF NOT EXISTS turns_text USING fts5(
    question, answer, calls,
    content='turns', content_rowid='id', tokenize='{FTS_TOKENIZER}'
)
""",
    """
CREATE TRIGGER IF NOT EXISTS turn_indexed AFTER INSERT ON turns BEGIN
    INSERT INTO turns_text(rowid, question, answer, calls)
    VALUES (new.id, new.question, new.answer, new.calls);
END
""",
    """
CREATE TRIGGER IF NOT EXISTS turn_unindexed AFTER DELETE ON turns BEGIN
    INSERT INTO turns_text(turns_text, rowid, question, answer, calls)
    VALUES ('delete', old.id, old.question, old.answer, old.calls);
END
""",
)
_COLUMNS: Final = "t.id, t.conversation, t.at, t.question, t.answer"


@dataclass(frozen=True, slots=True)
class ToolUse:
    """One tool call a turn made, and what came back."""

    name: str
    arguments: str
    result: str
    ok: bool


@dataclass(frozen=True, slots=True)
class Turn:
    """A finished turn, as it is remembered."""

    question: str
    answer: str
    at: datetime
    persona: str
    route: str
    model: str
    tools: tuple[ToolUse, ...] = ()


@dataclass(frozen=True, slots=True)
class Remembered:
    """A remembered turn: where and when it was, what was said, and its tool calls."""

    id: int
    conversation: int
    at: datetime
    question: str
    answer: str
    tools: tuple[ToolUse, ...] = ()


def _stamp(at: datetime) -> str:
    return at.astimezone(UTC).isoformat(timespec="microseconds")


def _clip(text: str) -> str:
    return mended(text[:MAX_TRACED_CHARS])


def match_expression(query: str) -> str:
    """Return ``query`` as an FTS5 expression matching any of its words.

    Each word is quoted, so nothing a person types is read as FTS5 syntax.
    """
    return " OR ".join(f'"{word}"' for word in _WORD.findall(query))


class MemoryStore:
    """The episodic store in one SQLite file."""

    def __init__(self, path: Path) -> None:
        """Open ``path``, creating the file and its tables if needed."""
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._opened() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            for statement in _SCHEMA:
                connection.execute(statement)

    async def begin_conversation(self, persona: str, at: datetime) -> int:
        """Record a conversation starting now and return its id."""
        return await asyncio.to_thread(self._begin, persona, at)

    async def add_turn(self, conversation: int, turn: Turn) -> int:
        """Remember ``turn`` in ``conversation`` and return its id."""
        return await asyncio.to_thread(self._add, conversation, turn)

    async def search(
        self,
        query: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 10,
    ) -> list[Remembered]:
        """Return up to ``limit`` turns from ``since`` up to ``until``.

        With words in ``query``, the turns holding any of them, best match
        first; with none, the newest turns in the range.
        """
        return await asyncio.to_thread(self._search, query, since, until, limit)

    async def conversation(self, conversation: int) -> list[Remembered]:
        """Return every turn of ``conversation``, oldest first."""
        return await asyncio.to_thread(self._conversation, conversation)

    async def forget_turn(self, turn: int) -> bool:
        """Forget ``turn`` and its tool calls; False if there was no such turn."""
        return await asyncio.to_thread(self._forget_turn, turn)

    async def forget_conversation(self, conversation: int) -> int:
        """Forget ``conversation`` and every turn in it; return how many turns."""
        return await asyncio.to_thread(self._forget_conversation, conversation)

    @contextmanager
    def _opened(self) -> Generator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self._path, timeout=BUSY_TIMEOUT_S, isolation_level=None
        )
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection]:
        with self._opened() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            connection.execute("COMMIT")

    def _begin(self, persona: str, at: datetime) -> int:
        with self._opened() as connection:
            cursor = connection.execute(
                "INSERT INTO conversations (started, persona) VALUES (?, ?)",
                (_stamp(at), mended(persona)),
            )
            return int(cursor.lastrowid or 0)

    def _add(self, conversation: int, turn: Turn) -> int:
        calls = "\n".join(f"{use.name} {use.arguments}" for use in turn.tools)
        with self._transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO turns (conversation, at, question, answer, calls,"
                " persona, route, model) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    conversation,
                    _stamp(turn.at),
                    mended(turn.question),
                    mended(turn.answer),
                    _clip(calls),
                    mended(turn.persona),
                    mended(turn.route),
                    mended(turn.model),
                ),
            )
            turn_id = int(cursor.lastrowid or 0)
            connection.executemany(
                "INSERT INTO tool_calls (turn, position, name, arguments, result, ok)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        turn_id,
                        position,
                        mended(use.name),
                        _clip(use.arguments),
                        _clip(use.result),
                        int(use.ok),
                    )
                    for position, use in enumerate(turn.tools)
                ],
            )
            return turn_id

    def _search(
        self,
        query: str,
        since: datetime | None,
        until: datetime | None,
        limit: int,
    ) -> list[Remembered]:
        window = (
            _EARLIEST if since is None else _stamp(since),
            _LATEST if until is None else _stamp(until),
        )
        expression = match_expression(query)
        with self._opened() as connection:
            if expression:
                rows = connection.execute(
                    f"SELECT {_COLUMNS} FROM turns_text"  # noqa: S608 - constant columns
                    " JOIN turns t ON t.id = turns_text.rowid"
                    " WHERE turns_text MATCH ? AND t.at >= ? AND t.at < ?"
                    " ORDER BY bm25(turns_text), t.at DESC LIMIT ?",
                    (expression, *window, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    f"SELECT {_COLUMNS} FROM turns t"  # noqa: S608 - constant columns
                    " WHERE t.at >= ? AND t.at < ? ORDER BY t.at DESC LIMIT ?",
                    (*window, limit),
                ).fetchall()
            return self._remembered(connection, rows)

    def _conversation(self, conversation: int) -> list[Remembered]:
        with self._opened() as connection:
            rows = connection.execute(
                f"SELECT {_COLUMNS} FROM turns t"  # noqa: S608 - constant columns
                " WHERE t.conversation = ? ORDER BY t.at, t.id",
                (conversation,),
            ).fetchall()
            return self._remembered(connection, rows)

    @staticmethod
    def _remembered(
        connection: sqlite3.Connection,
        rows: Iterable[tuple[int, int, str, str, str]],
    ) -> list[Remembered]:
        found: list[Remembered] = []
        for turn_id, conversation, at, question, answer in rows:
            uses = connection.execute(
                "SELECT name, arguments, result, ok FROM tool_calls"
                " WHERE turn = ? ORDER BY position",
                (turn_id,),
            ).fetchall()
            found.append(
                Remembered(
                    turn_id,
                    conversation,
                    datetime.fromisoformat(at),
                    question,
                    answer,
                    tuple(ToolUse(n, a, r, bool(ok)) for n, a, r, ok in uses),
                )
            )
        return found

    def _forget_turn(self, turn: int) -> bool:
        with self._opened() as connection:
            cursor = connection.execute("DELETE FROM turns WHERE id = ?", (turn,))
            return cursor.rowcount > 0

    def _forget_conversation(self, conversation: int) -> int:
        with self._transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM turns WHERE conversation = ?", (conversation,)
            )
            connection.execute(
                "DELETE FROM conversations WHERE id = ?", (conversation,)
            )
            return cursor.rowcount
