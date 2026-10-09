"""Measure how well the local model learns facts about the person, as SYNTHIA does.

    uv run python benchmarks/fact_extraction.py [--threads N]

The facts set (``benchmarks/data/facts_set.json``) holds made-up exchanges,
each with the lasting facts a reader takes from it (none, for a question or
a thank-you), facts that change, and pairs of statements that are or are
not the same fact, all written before any extraction was run. Everything
goes through :class:`synthia.memory.facts.FactKeeper`, the code SYNTHIA
learns with, on the installed local model and the default embedding model,
in a throwaway store:

1. The exchanges in order through one keeper, so facts learned earlier are
   shown as known, as in a chat. A kept fact is right when its cosine to an
   expected fact of its exchange reaches the match line. Precision is right
   kept facts over kept facts; recall is expected facts reached by some kept
   fact, so a fact split in two true halves is not punished. Reported at
   three match lines, with every kept fact beside its closest expected one
   for a person to judge.
2. Each change in a fresh store: the old fact is held first, then the
   exchange is learned. Right when the old fact was ended and a new held
   fact matches the expected one; "both kept" counts a missed change.
3. How many same and different pairs each duplicate line would merge.
4. Seconds per exchange learned.

It reaches no outside service and spends no budget.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import httpx
import numpy as np

from synthia.interfaces.cli import local_service
from synthia.kernel.config import default_home, load_settings
from synthia.memory.embed import TextEmbedder
from synthia.memory.facts import FactKeeper
from synthia.memory.store import MEMORY_FILE, FactKind, MemoryStore, Turn
from synthia.models.catalogue import DEFAULT_EMBEDDER
from synthia.models.install import BYTES_PER_GB, Installer

if TYPE_CHECKING:
    from synthia.gateway.protocol import ChatModel
    from synthia.models.local import LocalModel
    from synthia.models.server import Launch
    from synthia.models.service import LocalService

DESCRIPTION: Final = "Measure how well the local model learns facts about the person."
FACTS_SET: Final = Path(__file__).parent / "data" / "facts_set.json"
MATCH_LINES: Final = (0.6, 0.7, 0.8)
DUPLICATE_LINES: Final = (0.80, 0.85, 0.90, 0.95)
SAID_AT: Final = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)
APART: Final = timedelta(minutes=1)
SAMPLE_EVERY_S: Final = 0.2
# Loading on a busy machine has taken over 200 s on the CPU build.
LOAD_TIMEOUT_S: Final = 900.0
TIMEOUT: Final = httpx.Timeout(LOAD_TIMEOUT_S, connect=10.0)

# The facts set is this script's own data file, read as plain JSON.
type Item = dict[str, Any]


class Bench:
    """The embedder, the local model and the throwaway folder one run shares."""

    def __init__(self, embedder: TextEmbedder, model: ChatModel, folder: Path) -> None:
        self.embedder = embedder
        self.model = model
        self.folder = folder
        self.seconds: list[float] = []
        self._stores = 0

    async def cosines(self, rows: list[str], columns: list[str]) -> np.ndarray:
        """Return the cosine of each of ``rows`` to each of ``columns``."""
        if not rows or not columns:
            return np.zeros((len(rows), len(columns)), np.float32)
        return await self.embedder.embed(rows) @ (await self.embedder.embed(columns)).T

    async def keeper(self) -> tuple[MemoryStore, FactKeeper, int]:
        """Return a fresh store, a keeper on it, and a conversation in it."""
        self._stores += 1
        store = MemoryStore(self.folder / f"{self._stores}-{MEMORY_FILE}")
        keeper = FactKeeper(store, self.embedder, self.model)
        await keeper.load()
        return store, keeper, await store.begin_conversation("SYNTHIA", SAID_AT)

    async def learn(
        self,
        keeper: FactKeeper,
        store: MemoryStore,
        chat: int,
        item: Item,
        at: datetime,
    ) -> list[int]:
        """Remember ``item``'s exchange as a turn at ``at`` and learn from it, timed."""
        person, synthia = item["person"], item["synthia"]
        turn = await store.add_turn(
            chat, Turn(person, synthia, at, "SYNTHIA", "local", "-")
        )
        started = time.perf_counter()
        kept = await keeper.learn(turn, at, person, synthia)
        self.seconds.append(time.perf_counter() - started)
        return kept


async def exchanges(bench: Bench, items: list[Item]) -> list[dict[str, object]]:
    """Learn every exchange in order through one keeper; one row each."""
    store, keeper, chat = await bench.keeper()
    rows: list[dict[str, object]] = []
    for n, item in enumerate(items):
        ids = await bench.learn(keeper, store, chat, item, SAID_AT + n * APART)
        texts = {f.id: f.text for f in await store.facts(held=False)}
        kept = [texts[i] for i in ids]
        expected: list[str] = item["expected"]
        near = await bench.cosines(kept, expected)
        row: dict[str, object] = {
            "person": item["person"],
            "kept": [
                {
                    "text": text,
                    "closest": expected[int(near[k].argmax())] if expected else None,
                    "cosine": round(float(near[k].max()), 3) if expected else None,
                }
                for k, text in enumerate(kept)
            ],
            "expected_reached": [
                round(float(near[:, e].max()), 3) if kept else 0.0
                for e in range(len(expected))
            ],
        }
        # Printed at once: a model error later in the run keeps what came before.
        out(row)
        rows.append(row)
    return rows


def scored(rows: list[dict[str, Any]], line: float) -> dict[str, object]:
    """Return precision and recall of the exchanges at match line ``line``."""
    kept = [k["cosine"] or 0.0 for row in rows for k in row["kept"]]
    reached = [c for row in rows for c in row["expected_reached"]]
    return {
        "match_line": line,
        "kept": len(kept),
        "right": sum(c >= line for c in kept),
        "precision": round(sum(c >= line for c in kept) / len(kept), 3)
        if kept
        else None,
        "expected": len(reached),
        "recall": round(sum(c >= line for c in reached) / len(reached), 3),
    }


async def change(bench: Bench, item: Item) -> dict[str, object]:
    """Hold ``item``'s known fact, learn its exchange, and report what is held."""
    store, keeper, chat = await bench.keeper()
    known: str = item["known"]
    first = await store.add_turn(
        chat, Turn(known, "Noted.", SAID_AT, "SYNTHIA", "local", "-")
    )
    old = await store.add_fact(first, known, FactKind.FACT, SAID_AT)
    if old is None:
        message = "the known fact's turn was not kept"
        raise RuntimeError(message)
    (vector,) = await bench.embedder.embed([known])
    await store.put_fact_vectors(bench.embedder.spec.id, [(old, vector.tobytes())])
    await keeper.load()
    await bench.learn(keeper, store, chat, item, SAID_AT + APART)
    held = await store.facts()
    new = [f.text for f in held if f.id != old]
    near = await bench.cosines(new, [item["expected"]])
    return {
        "person": item["person"],
        "expected": item["expected"],
        "held": [f.text for f in held],
        "ended": all(f.id != old for f in held),
        "best_new_cosine": round(float(near.max()), 3) if new else 0.0,
    }


def changes_scored(rows: list[dict[str, Any]], line: float) -> dict[str, object]:
    """Return how many changes were right, and missed, at match line ``line``."""
    matched = [r["best_new_cosine"] >= line for r in rows]
    return {
        "match_line": line,
        "changes": len(rows),
        "replaced_right": sum(
            r["ended"] and m for r, m in zip(rows, matched, strict=True)
        ),
        "both_kept": sum(
            not r["ended"] and m for r, m in zip(rows, matched, strict=True)
        ),
    }


async def duplicates(bench: Bench, data: Item) -> list[dict[str, object]]:
    """Return how many same and different pairs each duplicate line merges."""
    cosines: dict[str, list[float]] = {}
    for kind in ("same", "different"):
        left = await bench.embedder.embed([a for a, _ in data[kind]])
        right = await bench.embedder.embed([b for _, b in data[kind]])
        cosines[kind] = (left * right).sum(axis=1).tolist()
    return [
        {
            "duplicate_line": line,
            **{
                f"{kind}_merged": f"{sum(c >= line for c in found)} of {len(found)}"
                for kind, found in cosines.items()
            },
        }
        for line in DUPLICATE_LINES
    ]


def installed(threads: int | None) -> LocalService:
    """Return the installed local model's service, on ``threads`` if given."""

    def command(launch: Launch) -> list[str]:
        extra = [] if threads is None else ["--threads", str(threads)]
        return [*launch.command(), *extra]

    service = local_service(load_settings(), command=command)
    if service is None:
        message = "no local model is installed: run `synthia models install`"
        raise RuntimeError(message)
    return service


async def ready(model: LocalModel) -> None:
    """Wait until ``model``'s server answers."""
    async with asyncio.timeout(LOAD_TIMEOUT_S):
        # The server's state lives on the service's own thread and loop, so
        # there is no event on this loop to wait for.
        while not model.ready():  # noqa: ASYNC110
            await asyncio.sleep(SAMPLE_EVERY_S)


async def measured(bench: Bench, data: Item) -> None:
    """Print every line of the benchmark."""
    for line in await duplicates(bench, data):
        out(line)
    rows = await exchanges(bench, data["exchanges"])
    for line in MATCH_LINES:
        out(scored(rows, line))
    moved: list[dict[str, object]] = []
    for item in data["changes"]:
        moved.append(await change(bench, item))
        out(moved[-1])
    for line in MATCH_LINES:
        out(changes_scored(moved, line))
    out(
        {
            "seconds_per_learn": {
                "min": round(min(bench.seconds), 1),
                "median": round(statistics.median(bench.seconds), 1),
                "max": round(max(bench.seconds), 1),
            },
            "runs": len(bench.seconds),
        }
    )


def out(record: object) -> None:
    """Print ``record`` as one JSON line, at once."""
    sys.stdout.write(json.dumps(record) + "\n")
    sys.stdout.flush()


async def main(arguments: list[str]) -> int:
    """Print one JSON line per exchange, change and summary."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--threads", type=int, default=None)
    options = parser.parse_args(arguments)
    data: Item = json.loads(FACTS_SET.read_text(encoding="utf-8"))
    models = Installer(default_home(), BYTES_PER_GB)
    embedder = TextEmbedder.load(models.path_of(DEFAULT_EMBEDDER), DEFAULT_EMBEDDER)
    service = installed(options.threads)
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        model = service.model(client)
        service.start()
        try:
            await ready(model)
            with tempfile.TemporaryDirectory() as folder:
                await measured(Bench(embedder, model, Path(folder)), data)
        finally:
            service.stop()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
