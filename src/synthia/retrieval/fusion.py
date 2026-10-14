"""One score for text found by its words, its meaning, or both.

The score is::

    cosine similarity + word weight * bm25 / (ln(2N + 1) * (k1 + 1))

The divisor is a ceiling on one query word's BM25 score among N indexed texts:
the highest idf a word can get, ln((N + 0.5) / 0.5), times the most a repeated
word can add, k1 + 1. It is fixed for the index rather than stretched per
query, so a question whose only shared words are "my" and "what" keeps the
small word score it deserves. Fusing by rank instead (reciprocal rank fusion,
k 60) gives that weak list's first text as much weight as the meaning's first
text; on 40 questions to memory asked in other words than the answer it cut
how often the right turn came first from 0.500 to 0.125.

The word weight is measured per store, on that store's own questions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from synthia.retrieval.bm25 import K1

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class WordScores:
    """The texts holding a query's words, each with its BM25 score (higher is better).

    ``indexed`` is how many texts the index holds, which BM25's scores depend on.
    """

    scores: dict[int, float]
    indexed: int


def word_scale(indexed: int) -> float:
    """Return a ceiling on one query word's BM25 score among ``indexed`` texts."""
    return math.log(2 * indexed + 1) * (K1 + 1)


def fused(
    similarities: Mapping[int, float], words: WordScores, word_weight: float
) -> dict[int, float]:
    """Return each text's similarity plus its scaled word score."""
    scale = word_scale(words.indexed)
    return {
        text: similarities.get(text, 0.0)
        + word_weight * words.scores.get(text, 0.0) / scale
        for text in similarities.keys() | words.scores.keys()
    }
