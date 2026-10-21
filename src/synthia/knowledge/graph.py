"""Follow links from the best passages to the documents they name.

A note that links to another says the two belong together, so a document
linked from a passage that matches a query well is likely to hold more of
the answer, even where its own words and meaning match the query poorly.
Each hop takes the best :data:`SEEDS` chunks, follows their resolved links,
and scores every chunk of each document reached:

    its own score for the query + link weight * the best score among the
    seeds whose links reach it

A chunk keeps the higher of that and any score it already had, so a linked
chunk already found is lifted by the same rule as one found only through
the link. A second hop starts from the best chunks the first one reached,
with their lifted scores, so a chain a -> b -> c carries a's score to c.
A cycle carries a lift back too: two seeds linking to each other lift each
other in the first hop, and again by the square of the weight in the
second, the same order as a chain's.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

from synthia.knowledge.model import ChunkId

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Collection, Mapping

SEEDS: Final = 5
"""Chunks each hop follows links from: the passages a person reads first."""

type Scorer = Callable[[Collection[ChunkId] | None], Awaitable[dict[int, float]]]
"""Scores a query against the chunks given, or finds the best when given None."""
type Reach = Callable[
    [Collection[ChunkId]], Awaitable[Mapping[ChunkId, Collection[ChunkId]]]
]
"""Returns, per chunk, every chunk of the documents its links resolve to."""


def best(scores: Mapping[int, float], count: int) -> list[int]:
    """Return the ``count`` chunks with the highest scores, ties to the lower id."""
    return sorted(scores, key=lambda chunk: (-scores[chunk], chunk))[:count]


async def hopped(
    scores: Mapping[int, float],
    *,
    hops: int,
    link_weight: float,
    reach: Reach,
    score: Scorer,
) -> dict[int, float]:
    """Return ``scores`` with the chunks ``hops`` links away from the best scored."""
    result = dict(scores)
    seeds = best(result, SEEDS)
    for _ in range(hops):
        reached = await reach([ChunkId(seed) for seed in seeds])
        prior: dict[int, float] = {}
        for seed, chunks in reached.items():
            for chunk in chunks:
                prior[chunk] = max(prior.get(chunk, -math.inf), result[seed])
        if not prior:
            break
        own = await score([ChunkId(chunk) for chunk in prior])
        for chunk, seed_score in prior.items():
            lifted = own.get(chunk, 0.0) + link_weight * seed_score
            result[chunk] = max(result.get(chunk, -math.inf), lifted)
        seeds = best({chunk: result[chunk] for chunk in prior}, SEEDS)
    return result
