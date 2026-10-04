"""SYNTHIA's own BM25 index: the ranking SQLite's FTS5 gives, written out.

Text becomes terms the way the memory store's ``porter unicode61`` index makes
them: lower case, accents removed, split on anything that is not a letter or
a digit, then Porter-stemmed. For text in the Latin script the terms are the
same; other scripts may split differently.

A document scores, for each query term (a repeated term counts again)::

    idf * f * (k1 + 1) / (f + k1 * (1 - b + b * length / average_length))

where ``f`` is how often the term occurs in the document, ``length`` its
terms, and ``idf = ln((N - n + 0.5) / (n + 0.5))`` over ``N`` documents, ``n``
of them holding the term. A term in more than half of them would get a
negative ``idf``, so it is floored at a millionth, as FTS5 does. Each term
is computed in FTS5's order of operations, so the scores are its scores
with the sign turned: higher is better.
"""

from __future__ import annotations

import functools
import heapq
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Final

from synthia.memory.porter import stem

K1: Final = 1.2
B: Final = 0.75
MIN_IDF: Final = 1e-6
STEM_CACHE: Final = 65_536
_TOKEN: Final = re.compile(r"[^\W_]+")


def terms(text: str) -> list[bytes]:
    """Return the terms of ``text``, in order."""
    plain = text.lower()
    if not plain.isascii():
        folded = unicodedata.normalize("NFD", plain)
        plain = "".join(c for c in folded if not unicodedata.combining(c))
    return [_term(word) for word in _TOKEN.findall(plain)]


# Words repeat far more than they vary, so most are stemmed once.
@functools.lru_cache(maxsize=STEM_CACHE)
def _term(word: str) -> bytes:
    return stem(word.encode())


@dataclass(frozen=True, slots=True)
class Hit:
    """A document that matched, and its score."""

    document: int
    score: float


class Bm25Index:
    """An inverted index over documents known by integer ids."""

    def __init__(self, k1: float = K1, b: float = B) -> None:
        """Start empty, with BM25's ``k1`` and ``b``."""
        self.k1 = k1
        self.b = b
        self._postings: dict[bytes, dict[int, int]] = {}
        self._lengths: dict[int, int] = {}
        self._terms: dict[int, tuple[bytes, ...]] = {}
        self._total = 0

    def __len__(self) -> int:
        """Return how many documents are indexed."""
        return len(self._lengths)

    def __contains__(self, document: object) -> bool:
        """Return whether ``document`` is indexed."""
        return document in self._lengths

    def add(self, document: int, text: str) -> None:
        """Index ``text`` as ``document``, replacing what it held before."""
        self.remove(document)
        found = terms(text)
        counts = Counter(found)
        for term, count in counts.items():
            self._postings.setdefault(term, {})[document] = count
        self._terms[document] = tuple(counts)
        self._lengths[document] = len(found)
        self._total += len(found)

    def remove(self, document: int) -> bool:
        """Take ``document`` out; False if it was not there."""
        if document not in self._lengths:
            return False
        self._total -= self._lengths.pop(document)
        for term in self._terms.pop(document):
            docs = self._postings[term]
            del docs[document]
            if not docs:
                del self._postings[term]
        return True

    def scores(self, query: str) -> dict[int, float]:
        """Return the score of every document holding a term of ``query``."""
        if not self._lengths:
            return {}
        count = len(self._lengths)
        average = self._total / count
        k1, b = self.k1, self.b
        # Many documents share a length, so each length's part is worked out once.
        norms: dict[int, float] = {}
        found: dict[int, float] = {}
        for term in terms(query):
            docs = self._postings.get(term)
            if not docs:
                continue
            idf = math.log((count - len(docs) + 0.5) / (len(docs) + 0.5))
            if idf <= 0.0:
                idf = MIN_IDF
            for document, f in docs.items():
                length = self._lengths[document]
                norm = norms.get(length)
                if norm is None:
                    norm = norms[length] = k1 * (1 - b + b * length / average)
                part = idf * ((f * (k1 + 1.0)) / (f + norm))
                found[document] = found.get(document, 0.0) + part
        return found

    def search(self, query: str, limit: int) -> list[Hit]:
        """Return the ``limit`` best documents for ``query``, newest first on ties."""
        best = heapq.nsmallest(
            limit, self.scores(query).items(), key=lambda s: (-s[1], -s[0])
        )
        return [Hit(document, score) for document, score in best]
