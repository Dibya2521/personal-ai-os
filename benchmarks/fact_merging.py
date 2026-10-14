"""Measure how well the local model tells a repeated fact from a different one.

    uv run python benchmarks/fact_merging.py [--threads N]

Consolidation merges two held facts only when the local model says they are
the same fact. This asks it, through :func:`synthia.memory.facts.same_request`
(the request SYNTHIA sends), about every pair in two sets: the pairs of
``benchmarks/data/facts_set.json`` and a held-out set,
``benchmarks/data/merge_set.json``, written before any run of the judge.
Each pair is asked in both orders, since a small model can answer by
position, and only a "same" in both orders counts as a merge.

For each candidate line (a pair is asked only when its cosine reaches it),
it prints how many same pairs would be merged and how many different pairs
would be merged by mistake. A wrong merge loses a true fact; a missed one
only shows a fact twice. It reaches no outside service and spends no budget.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import httpx
from fact_extraction import FACTS_SET, TIMEOUT, Item, installed, out, ready

from synthia.gateway.structured import StructuredOutputError, generate
from synthia.kernel.config import default_home
from synthia.memory.facts import Same, same_request
from synthia.models.catalogue import DEFAULT_EMBEDDER
from synthia.models.install import BYTES_PER_GB, Installer
from synthia.retrieval.embed import TextEmbedder

if TYPE_CHECKING:
    from synthia.gateway.protocol import ChatModel

DESCRIPTION: Final = "Measure how well the local model tells repeated facts apart."
MERGE_SET: Final = Path(__file__).parent / "data" / "merge_set.json"
CANDIDATE_LINES: Final = (0.0, 0.70, 0.75, 0.80, 0.85)


async def judged(
    model: ChatModel, first: str, second: str
) -> tuple[bool | None, float]:
    """Return the model's answer for one order (None if never valid) and seconds."""
    started = time.perf_counter()
    try:
        same = (await generate(model, same_request(first, second), Same)).same
    except StructuredOutputError:
        same = None
    return same, time.perf_counter() - started


async def pairs(
    embedder: TextEmbedder, model: ChatModel, name: str, data: Item
) -> list[dict[str, object]]:
    """Return one row per pair of ``data``: kind, cosine, both answers."""
    rows: list[dict[str, object]] = []
    for kind in ("same", "different"):
        found = data[kind]
        left = await embedder.embed([a for a, _ in found])
        right = await embedder.embed([b for _, b in found])
        for (first, second), cosine in zip(
            found, (left * right).sum(axis=1).tolist(), strict=True
        ):
            forward, took_forward = await judged(model, first, second)
            backward, took_backward = await judged(model, second, first)
            rows.append(
                {
                    "set": name,
                    "kind": kind,
                    "cosine": round(cosine, 3),
                    "forward": forward,
                    "backward": backward,
                    "seconds": [round(took_forward, 1), round(took_backward, 1)],
                    "first": first,
                    "second": second,
                }
            )
            out(rows[-1])
    return rows


def scored(rows: list[dict[str, object]], name: str, line: float) -> dict[str, object]:
    """Return the merges a candidate line and the judge together would make."""
    asked = [r for r in rows if r["set"] == name and float(str(r["cosine"])) >= line]
    merged = [r for r in asked if r["forward"] is True and r["backward"] is True]
    total = {
        kind: sum(r["kind"] == kind for r in rows if r["set"] == name)
        for kind in ("same", "different")
    }
    return {
        "set": name,
        "candidate_line": line,
        "asked": len(asked),
        "same_merged": f"{sum(r['kind'] == 'same' for r in merged)} of {total['same']}",
        "different_merged": (
            f"{sum(r['kind'] == 'different' for r in merged)} of {total['different']}"
        ),
        "orders_disagree": sum(r["forward"] != r["backward"] for r in asked),
        "never_valid": sum(None in (r["forward"], r["backward"]) for r in asked),
    }


async def main(arguments: list[str]) -> int:
    """Print one JSON line per pair, then one per set and candidate line."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--threads", type=int, default=None)
    options = parser.parse_args(arguments)
    sets = {
        "facts_set": json.loads(FACTS_SET.read_text(encoding="utf-8")),
        "merge_set": json.loads(MERGE_SET.read_text(encoding="utf-8")),
    }
    models = Installer(default_home(), BYTES_PER_GB)
    embedder = TextEmbedder.load(models.path_of(DEFAULT_EMBEDDER), DEFAULT_EMBEDDER)
    service = installed(options.threads)
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        model = service.model(client)
        service.start()
        try:
            await ready(model)
            rows: list[dict[str, object]] = []
            for name, data in sets.items():
                rows.extend(await pairs(embedder, model, name, data))
        finally:
            service.stop()
    for name in sets:
        for line in CANDIDATE_LINES:
            out(scored(rows, name, line))
    seconds = [float(took) for r in rows for took in cast("list[float]", r["seconds"])]
    out(
        {
            "seconds_per_question": {
                "min": round(min(seconds), 1),
                "median": round(statistics.median(seconds), 1),
                "max": round(max(seconds), 1),
            },
            "questions": 2 * len(rows),
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
