"""Recall: earlier turns found by their words and by their meaning together.

Words find what meaning blurs: a name, a ticket number, a host. Meaning
finds what words miss: "what's my pet called?" for "my cat is Miso". Each
turn found by either gets one score::

    cosine similarity + 0.1 * bm25 / (ln(2N + 1) * (k1 + 1))

The divisor is a ceiling on one query word's BM25 score among N turns: the
highest idf a word can get, ln((N + 0.5) / 0.5), times the most a repeated
word can add, k1 + 1. It is fixed for the store rather than stretched per
query, so a question whose only shared words are "my" and "what" keeps the
small word score it deserves. Fusing by rank instead (reciprocal rank
fusion, k 60) gives that weak list's first turn as much weight as the
meaning's first turn; on 40 questions asked in other words than the answer
it cut how often the right turn came first from 0.500 to 0.125.

The weight 0.1 was measured with ``benchmarks/hybrid_recall.py``: from 0.05
to 0.15 the fused ranking was at least as good as meaning alone at 1, 3 and
5 turns and in mean reciprocal rank, on questions in other words and on
questions naming a rare word alike, and better on some (the right turn first
for 19 of 20 questions naming a rare word, against 18); from 0.25 up it put
the right turn first less often for questions in other words. 0.1 is the
middle of the range that never did worse.

Each turn's vector is of the person's question alone: what they said is
what "what did I tell you" asks about, and a long answer averaged in would
make the vector about the answer. The words cover the question, the answer
and the tool calls.

Errors from the embedding model's runtime pass through as it raises them.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from synthia.memory.bm25 import K1
from synthia.memory.store import match_expression
from synthia.memory.vectors import VectorSet

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from synthia.memory.embed import TextEmbedder
    from synthia.memory.store import Asked, MemoryStore, Remembered, WordScores

WORD_WEIGHT: Final = 0.1
# Turns taken from each side before fusing. A turn outside the meaning's
# first 50 and without a word scores below all 50, so the first 50 fused are
# exact.
CANDIDATES: Final = 50
EMBED_BATCH: Final = 64


def word_scale(turns: int) -> float:
    """Return a ceiling on one query word's BM25 score among ``turns`` turns."""
    return math.log(2 * turns + 1) * (K1 + 1)


def fused(
    similarities: Mapping[int, float], words: WordScores, word_weight: float
) -> dict[int, float]:
    """Return each turn's similarity plus its scaled word score."""
    scale = word_scale(words.turns)
    return {
        turn: similarities.get(turn, 0.0)
        + word_weight * words.scores.get(turn, 0.0) / scale
        for turn in similarities.keys() | words.scores.keys()
    }


@dataclass(frozen=True, slots=True)
class Recalled:
    """A turn recalled for a query, and its fused score (higher is closer)."""

    turn: Remembered
    score: float


class Recall:
    """The store's turns, found by words and, with an embedding model, by meaning."""

    def __init__(
        self,
        store: MemoryStore,
        embedder: TextEmbedder | None = None,
        *,
        word_weight: float = WORD_WEIGHT,
    ) -> None:
        """Search ``store``; without ``embedder``, by words alone.

        :meth:`load` reads the vectors already kept before the first search.
        """
        self.store = store
        self.embedder = embedder
        self.word_weight = word_weight
        self._vectors = (
            None if embedder is None else VectorSet(embedder.spec.dimensions)
        )

    async def load(self) -> None:
        """Hold every vector the store keeps from the embedding model."""
        if self.embedder is None or self._vectors is None:
            return
        for kept in await self.store.vectors(self.embedder.spec.id):
            vector = np.frombuffer(kept.vector, np.float32)
            self._vectors.add(kept.turn, kept.at, vector)

    async def keep(self, *asked: Asked) -> None:
        """Embed each turn's question, keep the vector and search it from now on.

        A turn forgotten meanwhile is skipped.
        """
        if self.embedder is None or self._vectors is None or not asked:
            return
        vectors = await self.embedder.embed([a.question for a in asked])
        pairs = list(zip(asked, vectors, strict=True))
        model = self.embedder.spec.id
        kept = set(
            await self.store.put_vectors(
                model, [(a.turn, vector.tobytes()) for a, vector in pairs]
            )
        )
        for a, vector in pairs:
            if a.turn in kept:
                self._vectors.add(a.turn, a.at, vector)

    async def backfill(self) -> int:
        """Embed every turn the model has no vector for yet; return how many."""
        if self.embedder is None:
            return 0
        model = self.embedder.spec.id
        done = 0
        while waiting := await self.store.unembedded(model, EMBED_BATCH):
            await self.keep(*waiting)
            done += len(waiting)
        return done

    def forget(self, turn: int) -> None:
        """Stop finding ``turn`` by meaning; the store drops its vector itself."""
        if self._vectors is not None:
            self._vectors.remove(turn)

    async def find(
        self,
        query: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int,
        at_least: float = -math.inf,
    ) -> list[Recalled]:
        """Return up to ``limit`` turns from ``since`` up to ``until``, best first.

        Only turns scoring ``at_least`` are returned. With no words in
        ``query``, the newest turns in the range, scored 0.
        """
        if not match_expression(query):
            if at_least > 0.0:
                return []
            newest = await self.store.search("", since=since, until=until, limit=limit)
            return [Recalled(turn, 0.0) for turn in newest]
        words = await self.store.word_scores(
            query, since=since, until=until, limit=CANDIDATES
        )
        similar: dict[int, float] = {}
        if self.embedder is not None and self._vectors is not None:
            (vector,) = await self.embedder.embed([query])
            near = self._vectors.similarities(vector, since=since, until=until)
            similar = near.nearest(CANDIDATES) | near.of(words.scores.keys())
        scores = fused(similar, words, self.word_weight)
        enough = [(turn, score) for turn, score in scores.items() if score >= at_least]
        best = heapq.nlargest(limit, enough, key=lambda s: (s[1], s[0]))
        turns = await self.store.turns([turn for turn, _ in best])
        return [Recalled(turn, scores[turn.id]) for turn in turns]
