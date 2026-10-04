"""Measure SYNTHIA's own HNSW index against exact search, on generated vectors.

    uv run python benchmarks/hnsw_recall.py --size 1000 --size 10000 --size 50000

For each size, unit vectors of 384 numbers (the size bge-small-en-v1.5
returns) are drawn in tight clusters around random centres, the shape
sentence embeddings take; queries are drawn the same way and are not in the
index. The index is built once, then each ``ef_search`` is run for every
query. recall@10 is the share of the exact ten nearest (by brute-force
cosine over all vectors) that the index returns. Build time per vector and
query time are wall-clock on this machine, so the load is printed with them.
No model, file or network is used.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Final

import numpy as np
import psutil

from synthia.memory.hnsw import EF_CONSTRUCTION, Hnsw, M

DIM: Final = 384
QUERIES: Final = 200
K: Final = 10
POINTS_PER_CLUSTER: Final = 100
SPREAD: Final = 0.6
EF_SEARCH: Final = (10, 20, 40, 80, 160)


@dataclass(frozen=True, slots=True)
class Row:
    """One size and ef_search: how much was found and how fast."""

    size: int
    ef_search: int
    recall_at_10: float
    query_ms: float
    exact_query_ms: float
    build_ms_per_vector: float


def clustered(rng: np.random.Generator, count: int, centres: np.ndarray) -> np.ndarray:
    """Return ``count`` unit vectors scattered around ``centres``."""
    picks = rng.integers(0, len(centres), count)
    points = centres[picks] + SPREAD * rng.standard_normal((count, DIM))
    return (points / np.linalg.norm(points, axis=1, keepdims=True)).astype(np.float32)


def measure(size: int, seed: int) -> list[Row]:
    """Build an index of ``size`` vectors and search it at every ef_search."""
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((max(size // POINTS_PER_CLUSTER, 2), DIM))
    points = clustered(rng, size, centres)
    queries = clustered(rng, QUERIES, centres)
    started = time.perf_counter()
    exact = [set(np.argpartition(-(points @ q), K)[:K].tolist()) for q in queries]
    exact_ms = (time.perf_counter() - started) * 1000 / QUERIES
    index = Hnsw(DIM, seed=seed)
    started = time.perf_counter()
    for id_, point in enumerate(points):
        index.add(id_, point)
    build_ms = (time.perf_counter() - started) * 1000 / size
    rows: list[Row] = []
    for ef in EF_SEARCH:
        started = time.perf_counter()
        found = [{n.id for n in index.search(q, K, ef)} for q in queries]
        query_ms = (time.perf_counter() - started) * 1000 / QUERIES
        hits = sum(len(f & e) for f, e in zip(found, exact, strict=True))
        rows.append(Row(size, ef, hits / (K * QUERIES), query_ms, exact_ms, build_ms))
    return rows


def main() -> None:
    """Measure each size asked for and print one JSON line per row."""
    parser = argparse.ArgumentParser(description="HNSW recall@10 against exact search")
    parser.add_argument("--size", type=int, action="append", required=True)
    parser.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args()
    print(json.dumps({"m": M, "ef_construction": EF_CONSTRUCTION, "dim": DIM}))
    for size in arguments.size:
        load = psutil.cpu_percent(interval=1)
        for row in measure(size, arguments.seed):
            print(json.dumps({**asdict(row), "cpu_percent_before": load}), flush=True)


if __name__ == "__main__":
    main()
