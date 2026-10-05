"""Measure SYNTHIA's recall by words, by meaning and by both, as it runs.

    uv run python benchmarks/hybrid_recall.py

Every turn of the recall set (``benchmarks/data/recall_set.json``) is
remembered in a fresh memory store, as a question with no answer, and every
question is asked through :class:`synthia.memory.hybrid.Recall`, the code
SYNTHIA searches with. The set has two kinds of question: 40 ``pairs`` asked
in other words than what was said ("What's the name of my pet?" for "My cat
is called Miso"), and 20 ``named`` ones that reuse a rare name or code
("What is SYN-482 about?"). Besides the 100 turns those questions are about
or that are about nothing they ask, 20 ``near`` turns share a topic with a
named one under another name (a different ticket, a different flight).

Per kind and method: recall at 1, 3 and 5 and the mean reciprocal rank.
Then, for the fused score, how many questions' right turn reaches each
floor, against how many questions would show some other turn at that floor
if the right one had never been said: what a floor on injected memories
trades. Needs the default embedding model (``synthia models install``).
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from synthia.kernel.config import default_home
from synthia.memory.embed import TextEmbedder
from synthia.memory.hybrid import Recall
from synthia.memory.store import MEMORY_FILE, MemoryStore, Turn
from synthia.models.catalogue import DEFAULT_EMBEDDER
from synthia.models.install import BYTES_PER_GB, Installer

RECALL_SET: Final = Path(__file__).parent / "data" / "recall_set.json"
DEPTHS: Final = (1, 3, 5)
FLOORS: Final = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
SAID_AT: Final = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)


def scores(ranks: list[int]) -> dict[str, float]:
    """Return recall at each depth and the mean reciprocal rank, from 1-based ranks."""
    found = {f"recall@{k}": sum(r <= k for r in ranks) / len(ranks) for k in DEPTHS}
    return {**found, "mrr": round(sum(1 / r for r in ranks) / len(ranks), 3)}


async def remembered(store: MemoryStore, turns: list[str]) -> list[int]:
    """Remember each of ``turns`` as a question with no answer; return their ids."""
    chat = await store.begin_conversation("SYNTHIA", SAID_AT)
    return [
        await store.add_turn(chat, Turn(text, "", SAID_AT, "SYNTHIA", "local", "-"))
        for text in turns
    ]


async def measured(
    recall: Recall, questions: list[tuple[str, str]], ids: list[int]
) -> tuple[dict[str, list[int]], list[tuple[float, float]]]:
    """Return each kind's ranks of the right turn, and each question's two scores.

    The two scores are the right turn's and the best other turn's.
    """
    last = len(ids)
    ranks: dict[str, list[int]] = {"pairs": [], "named": []}
    pairs: list[tuple[float, float]] = []
    for n, (question, kind) in enumerate(questions):
        found = await recall.find(question, limit=last)
        order = [r.turn.id for r in found]
        # A turn not found at all comes last.
        ranks[kind].append(order.index(ids[n]) + 1 if ids[n] in order else last)
        right = next((r.score for r in found if r.turn.id == ids[n]), 0.0)
        other = max((r.score for r in found if r.turn.id != ids[n]), default=0.0)
        pairs.append((right, other))
    return ranks, pairs


def floor_line(floor: float, pairs: list[tuple[float, float]]) -> str:
    """Return how often the right turn, and some other turn, reach ``floor``."""
    reached = sum(right >= floor for right, _ in pairs) / len(pairs)
    unrelated = sum(other >= floor for _, other in pairs) / len(pairs)
    return json.dumps(
        {
            "floor": floor,
            "right_turn_reaches": round(reached, 3),
            "other_turn_if_none_said": round(unrelated, 3),
        }
    )


async def main() -> None:
    """Print one JSON line per kind and method, then one per floor."""
    data = json.loads(RECALL_SET.read_text(encoding="utf-8"))
    questions = [(p["asked"], "pairs") for p in data["pairs"]]
    questions += [(p["asked"], "named") for p in data["named"]]
    said = [p["said"] for p in (*data["pairs"], *data["named"])]
    turns = [*said, *data["other"], *data["near"]]
    models = Installer(default_home(), BYTES_PER_GB)
    embedder = TextEmbedder.load(models.path_of(DEFAULT_EMBEDDER), DEFAULT_EMBEDDER)
    with tempfile.TemporaryDirectory() as home:
        store = MemoryStore(Path(home) / MEMORY_FILE)
        ids = await remembered(store, turns)
        print(json.dumps({"turns": len(ids), "questions": len(questions)}))
        methods = {
            "words": Recall(store),
            "meaning": Recall(store, embedder, word_weight=0.0),
            "hybrid": Recall(store, embedder),
        }
        await methods["meaning"].backfill()
        await methods["hybrid"].load()
        floors: list[tuple[float, float]] = []
        for name, recall in methods.items():
            ranks, pairs = await measured(recall, questions, ids)
            if name == "hybrid":
                floors = pairs
            for kind, kind_ranks in ranks.items():
                print(json.dumps({"kind": kind, "method": name, **scores(kind_ranks)}))
    for floor in FLOORS:
        print(floor_line(floor, floors))


if __name__ == "__main__":
    asyncio.run(main())
