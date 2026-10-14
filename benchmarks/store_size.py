"""Measure how much disk SYNTHIA's memory takes per 1,000 remembered turns.

    uv run python benchmarks/store_size.py

For each pair of lengths, a fresh store remembers the turns through
:class:`synthia.memory.store.MemoryStore`, the code the daemon writes with:
each turn's question and answer cut to those many characters from English
sentences taken from the public benchmark sets, so the full-text index sees
real words. The file is measured with words alone, again after each turn
gets a vector of the default embedding model's size, and again after the
index upkeep a consolidation ends with. That upkeep writes each full-text
index anew as one piece, and the pages of the old one stay in the file as
free pages that later turns fill before it grows, so they are reported too.
The vectors are random, since a vector takes the same bytes whatever its
numbers. Tool calls
and learned facts are left out: each tool call adds its name and up to
2,000 characters each of arguments and result, each fact its sentence and a
vector. No model, network or file outside a temporary folder is used.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import tempfile
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

from synthia.memory.store import MEMORY_FILE, MemoryStore, Turn
from synthia.models.catalogue import DEFAULT_EMBEDDER

if TYPE_CHECKING:
    from collections.abc import Iterator

DATA: Final = Path(__file__).parent / "data"
# (question, answer) characters: a short exchange, a typical one, a long one.
LENGTHS: Final = ((100, 500), (200, 1500), (400, 4000))
FIRST_TURN_AT: Final = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)
BYTES_PER_MB: Final = 1024**2
VECTOR_BATCH: Final = 500


def sentences() -> list[str]:
    """Return every sentence a person or SYNTHIA says in the public sets."""
    recall = json.loads((DATA / "recall_set.json").read_text(encoding="utf-8"))
    facts = json.loads((DATA / "facts_set.json").read_text(encoding="utf-8"))
    said = [
        p[key]
        for p in (*recall["pairs"], *recall["named"])
        for key in ("said", "asked")
    ]
    said += [*recall["other"], *recall["near"]]
    said += [e[key] for e in facts["exchanges"] for key in ("person", "synthia")]
    said += [c[key] for c in facts["changes"] for key in ("person", "synthia")]
    return said


def texts(pool: list[str], seed: int) -> Iterator[str]:
    """Yield the pool's sentences without end, in a seeded order."""
    rng = np.random.default_rng(seed)
    while True:
        yield from (pool[i] for i in rng.permutation(len(pool)))


def cut(source: Iterator[str], chars: int) -> str:
    """Return sentences from ``source`` joined and cut to exactly ``chars``."""
    taken: list[str] = []
    while sum(len(t) + 1 for t in taken) < chars:
        taken.append(next(source))
    return " ".join(taken)[:chars]


def size_of(path: Path) -> int:
    """Return the bytes of the database file with its write-ahead log, if any."""
    files = (path, path.with_name(path.name + "-wal"))
    return sum(f.stat().st_size for f in files if f.exists())


def free_bytes(path: Path) -> int:
    """Return the bytes of the pages the database holds free for later writes."""
    with closing(sqlite3.connect(path)) as connection:
        (page_size,) = connection.execute("PRAGMA page_size").fetchone()
        (free,) = connection.execute("PRAGMA freelist_count").fetchone()
    return int(page_size) * int(free)


async def measure(
    turns: int, question: int, answer: int, seed: int
) -> dict[str, object]:
    """Remember ``turns`` turns of the given lengths; return the file's sizes."""
    source = texts(sentences(), seed)
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory() as home:
        path = Path(home) / MEMORY_FILE
        store = MemoryStore(path)
        empty = size_of(path)
        chat = await store.begin_conversation("SYNTHIA", FIRST_TURN_AT)
        ids: list[int] = []
        for n in range(turns):
            at = FIRST_TURN_AT + timedelta(minutes=n)
            turn = Turn(
                cut(source, question), cut(source, answer), at, "SYNTHIA", "local", "-"
            )
            ids.append(await store.add_turn(chat, turn))
        words_only = size_of(path)
        dimensions = DEFAULT_EMBEDDER.dimensions
        for start in range(0, turns, VECTOR_BATCH):
            batch = ids[start : start + VECTOR_BATCH]
            vectors = rng.standard_normal((len(batch), dimensions)).astype(np.float32)
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            await store.put_vectors(
                DEFAULT_EMBEDDER.id,
                [(t, v.tobytes()) for t, v in zip(batch, vectors, strict=True)],
            )
        with_vectors = size_of(path)
        await store.optimize()
        optimized = size_of(path)
        free = free_bytes(path)
    per_thousand = 1000 / turns
    return {
        "turns": turns,
        "question_chars": question,
        "answer_chars": answer,
        "empty_mb": round(empty / BYTES_PER_MB, 3),
        "words_only_mb_per_1000": round(words_only * per_thousand / BYTES_PER_MB, 2),
        "with_vectors_mb_per_1000": round(
            with_vectors * per_thousand / BYTES_PER_MB, 2
        ),
        "after_optimize_mb_per_1000": round(optimized * per_thousand / BYTES_PER_MB, 2),
        "free_after_optimize_mb_per_1000": round(free * per_thousand / BYTES_PER_MB, 2),
        "text_mb_per_1000": round((question + answer) * 1000 / BYTES_PER_MB, 2),
        "vector_bytes": dimensions * np.dtype(np.float32).itemsize,
    }


async def main() -> None:
    """Print one JSON line per pair of lengths."""
    parser = argparse.ArgumentParser(
        description="Disk taken per 1,000 remembered turns"
    )
    parser.add_argument("--turns", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    options = parser.parse_args()
    for question, answer in LENGTHS:
        result = await measure(options.turns, question, answer, options.seed)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
