"""Measure how well each way of recalling finds what the person said before.

    uv run python benchmarks/embedding_recall.py

This benchmark reads two parts of the recall set
(``benchmarks/data/recall_set.json``): 40 things a person tells SYNTHIA,
each with a question asked later in other words (``pairs``), and 120 turns
about other things (``other``). Every question is searched against all 160
turns by each method: SYNTHIA's own BM25 (words), and each installed
embedding model (meaning, cosine similarity). A question counts as found at
k when the turn it asks about is among the first k. Reported per method:
recall@1, recall@5, mean reciprocal rank, and the milliseconds to embed one
turn. Embedding models that are not installed are left out.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Final

import numpy as np
import psutil

from synthia.kernel.config import default_home
from synthia.memory.bm25 import Bm25Index
from synthia.memory.embed import TextEmbedder
from synthia.models.catalogue import EMBEDDERS
from synthia.models.install import BYTES_PER_GB, Installer

RECALL_SET: Final = Path(__file__).parent / "data" / "recall_set.json"
DEPTHS: Final = (1, 5)


def scores(ranks: list[int]) -> dict[str, float]:
    """Return recall at each depth and the mean reciprocal rank, from 1-based ranks."""
    found = {f"recall@{k}": sum(r <= k for r in ranks) / len(ranks) for k in DEPTHS}
    return {**found, "mrr": sum(1 / r for r in ranks) / len(ranks)}


def rank_of(order: list[int], target: int) -> int:
    """Return where ``target`` comes in ``order``, counting from 1."""
    return order.index(target) + 1


def main() -> None:
    """Print one JSON line per method."""
    data = json.loads(RECALL_SET.read_text(encoding="utf-8"))
    turns = [p["said"] for p in data["pairs"]] + list(data["other"])
    questions = [p["asked"] for p in data["pairs"]]
    print(json.dumps({"turns": len(turns), "questions": len(questions)}))

    words = Bm25Index()
    for id_, turn in enumerate(turns):
        words.add(id_, turn)
    ranks: list[int] = []
    for target, question in enumerate(questions):
        found = [hit.document for hit in words.search(question, len(turns))]
        # A turn sharing no word with the question is not found at all: last.
        ranks.append(rank_of(found, target) if target in found else len(turns))
    print(json.dumps({"method": "bm25", **scores(ranks)}))

    models = Installer(default_home(), BYTES_PER_GB)
    for spec in EMBEDDERS:
        if not models.installed(spec):
            continue
        embedding = TextEmbedder.load(models.path_of(spec), spec)
        load = psutil.cpu_percent(interval=1)
        started = time.perf_counter()
        stored = embedding.vectors(turns)
        per_turn_ms = (time.perf_counter() - started) * 1000 / len(turns)
        asked = embedding.vectors(questions)
        order = np.argsort(-(asked @ stored.T), axis=1)
        ranks = [rank_of(order[t].tolist(), t) for t in range(len(questions))]
        print(
            json.dumps(
                {
                    "method": spec.id,
                    **scores(ranks),
                    "embed_ms_per_turn": round(per_turn_ms, 2),
                    "cpu_percent_before": load,
                }
            )
        )


if __name__ == "__main__":
    main()
