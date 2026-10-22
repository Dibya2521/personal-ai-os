"""Passages for a query, found by their words, their meaning, or both.

Words find what meaning blurs: a name, an error code, a symbol only one
file holds. Meaning finds what words miss: "pet" for a note that says
"cat". One :class:`Variant` value says which ranks, so a flag, a table of
results and the default all name a ranking the same way.

- Words: the chunks' BM25 scores from the full-text index (higher is better).
- Meaning: the cosine similarity of the query's vector to each chunk's.
- Fused: :func:`synthia.retrieval.fusion.fused` of the two, the cosine plus
  the BM25 score scaled by a ceiling fixed for the index and weighted.

Each side gives its best :data:`CANDIDATES` chunks; fused adds the cosine of
every chunk the words found, as memory's search does. Ties go to the lower
chunk id, so a ranking never depends on the order a set was read in. A
variant may then follow links from the best chunks one or two hops
(:mod:`synthia.knowledge.graph`), scoring what they reach the same way.

A search may rank more queries for the same question, such as rewordings of
it, and :class:`Merge` their rankings into one.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto, unique
from functools import partial
from typing import TYPE_CHECKING, Final, Self

from synthia.knowledge.graph import hopped
from synthia.knowledge.model import ChunkId
from synthia.retrieval.fts import match_expression
from synthia.retrieval.fusion import fused

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping, Sequence

    from synthia.knowledge.graph import Scorer
    from synthia.knowledge.meaning import ChunkVectors
    from synthia.knowledge.model import Chunk, Document
    from synthia.knowledge.store import KnowledgeStore, Row
    from synthia.retrieval.rerank import CrossEncoder
    from synthia.retrieval.vectors import Similarities

CANDIDATES: Final = 50
"""Chunks each side gives before fusing, as memory's search takes; a larger
limit takes that many instead."""
WORD_WEIGHT: Final = 0.2
"""The weight measured on knowledge's own questions (``benchmarks/rag_eval.py``):
61 questions about this repository at commit ffd814e, 49 of them answerable,
each searched 10 passages deep. 0.2 scored a higher nDCG@10 than 0.1 in 99.7 %
of 2,000 paired bootstrap resamples over the questions with sections and a
header, and in 86.5 % with windows of 40 lines, where it reached 0.461, the
best of 18 (3 chunkings by 6 variants). Memory measured 0.1 on its own questions
and keeps it (:mod:`synthia.memory.hybrid`); here that weight is ``fused:0.1``."""
MAX_HOPS: Final = 2
RRF_K: Final = 60
"""Reciprocal rank fusion's constant, the one Cormack, Clarke and Buettcher
proposed it with (SIGIR 2009): it damps how far a ranking's first places lead."""


class VariantError(ValueError):
    """A variant name that does not parse, or one needing a reranker not loaded."""


@unique
class Side(Enum):
    """Which evidence ranks the chunks: words, meaning, or both fused."""

    WORDS = auto()
    MEANING = auto()
    FUSED = auto()


_SIDES: Final = {side.name.lower(): side for side in Side}
_FOLLOW: Final = "+hop"
"""Not a bare "+", which a weight such as 1e+16 holds."""
_HOPS: Final = re.compile(r"(?P<hops>[0-9]+)x(?P<weight>.+)")
_RERANK: Final = "+rerank:"
_RERANKED: Final = re.compile(r"(?P<model>.+)@(?P<top>[0-9]+)")
_RERANKER_ID: Final = re.compile(r"[a-z0-9][a-z0-9.-]*")


def _checked_weight(label: str, weight: float) -> None:
    if not math.isfinite(weight) or math.copysign(1.0, weight) < 0:
        message = f"{label}={weight!r} is not a finite number of at least 0"
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class Rerank:
    """Reorder the first ``top`` chunks with the reranker catalogued as ``model``.

    Raises:
        ValueError: If ``model`` is not a catalogue id (lower-case letters,
            digits, dots and hyphens) or ``top`` is below 1.
    """

    model: str
    top: int

    def __post_init__(self) -> None:
        if not _RERANKER_ID.fullmatch(self.model):
            message = f"model={self.model!r} is not a reranker's catalogue id"
            raise ValueError(message)
        if self.top < 1:
            message = f"top={self.top} is below 1"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class Variant:
    """One way to rank chunks, with a name a flag or a table can spell.

    Names are ``words``, ``meaning``, ``fused`` and ``fused:<weight>``: the
    weight as Python's shortest round-trip float, left out when it is
    :data:`WORD_WEIGHT`. Following links adds ``+hop<hops>x<link weight>``,
    as in ``fused+hop1x0.3``. Reranking the first chunks adds
    ``+rerank:<reranker>@<top>``, last, as in
    ``fused+rerank:ms-marco-minilm-l6-int8@10``. Only fused takes a word
    weight and only hops take a link weight, so each variant has one name and
    :meth:`parse` reads back exactly what :meth:`name` writes.

    Raises:
        ValueError: If a weight is not finite or is below 0 (negative zero
            included), ``word_weight`` is not the default on a side that does
            not fuse, ``hops`` is not 0 to :data:`MAX_HOPS`, or
            ``link_weight`` is not 0 without hops.
    """

    side: Side = Side.FUSED
    word_weight: float = WORD_WEIGHT
    hops: int = 0
    link_weight: float = 0.0
    rerank: Rerank | None = None

    def __post_init__(self) -> None:
        _checked_weight("word_weight", self.word_weight)
        _checked_weight("link_weight", self.link_weight)
        if self.side is not Side.FUSED and self.word_weight != WORD_WEIGHT:
            side = self.side.name.lower()
            message = f"word_weight={self.word_weight!r} on {side}, which does not fuse"
            raise ValueError(message)
        if not 0 <= self.hops <= MAX_HOPS:
            message = f"hops={self.hops} is not 0 to {MAX_HOPS}"
            raise ValueError(message)
        if not self.hops and self.link_weight:
            message = f"link_weight={self.link_weight!r} with no hops to weigh"
            raise ValueError(message)

    def name(self) -> str:
        """Return the variant's name, which :meth:`parse` reads back."""
        name = self.side.name.lower()
        if self.word_weight != WORD_WEIGHT:
            name += f":{float(self.word_weight)!r}"
        if self.hops:
            name += f"{_FOLLOW}{self.hops}x{float(self.link_weight)!r}"
        if self.rerank is not None:
            name += f"{_RERANK}{self.rerank.model}@{self.rerank.top}"
        return name

    @classmethod
    def parse(cls, name: str) -> Self:
        """Return the variant ``name`` spells.

        Raises:
            VariantError: If ``name`` is not a variant's name as :meth:`name`
                writes it.
        """
        before, reranked, after = name.partition(_RERANK)
        rerank = _RERANKED.fullmatch(after)
        if reranked and rerank is None:
            message = f"{name!r} is not a variant: rerank as +rerank:<reranker>@<top>"
            raise VariantError(message)
        ranking, plus, following = before.partition(_FOLLOW)
        label, colon, weight = ranking.partition(":")
        side = _SIDES.get(label)
        if side is None:
            message = f"{name!r} is not a variant: start with words, meaning or fused"
            raise VariantError(message)
        hops = _HOPS.fullmatch(following)
        if plus and hops is None:
            message = f"{name!r} is not a variant: follow links as +hop<N>x<weight>"
            raise VariantError(message)
        try:
            variant = cls(
                side,
                float(weight) if colon else WORD_WEIGHT,
                int(hops["hops"]) if hops else 0,
                float(hops["weight"]) if hops else 0.0,
                Rerank(rerank["model"], int(rerank["top"])) if rerank else None,
            )
        except ValueError as error:
            message = f"{name!r} is not a variant: {error}"
            raise VariantError(message) from error
        if variant.name() != name:
            message = f"{name!r} is not a variant: it is spelled {variant.name()!r}"
            raise VariantError(message)
        return variant


DEFAULT_VARIANT: Final = Variant()


def _ranked(scores: Mapping[int, float]) -> list[int]:
    return sorted(scores, key=lambda chunk: (-scores[chunk], chunk))


@unique
class Merge(Enum):
    """How the rankings of several queries for one question become one.

    - Best: each chunk keeps its highest score over the queries that found it.
    - RRF, reciprocal rank fusion: each chunk scores the sum, over the queries
      that found it, of 1 / (:data:`RRF_K` + rank), rank 1 being a query's
      best chunk and ties going to the lower chunk id.

    Either way the merged scores do not depend on the order of the rankings.
    """

    BEST = auto()
    RRF = auto()

    def of(self, rankings: Iterable[Mapping[int, float]]) -> dict[int, float]:
        """Return a score for every chunk in any of ``rankings``, merged this way."""
        if self is Merge.BEST:
            best: dict[int, float] = {}
            for scores in rankings:
                for chunk, score in scores.items():
                    best[chunk] = max(best.get(chunk, -math.inf), score)
            return best
        terms: defaultdict[int, list[float]] = defaultdict(list)
        for scores in rankings:
            for rank, chunk in enumerate(_ranked(scores), 1):
                terms[chunk].append(1 / (RRF_K + rank))
        # fsum rounds the exact sum once, so the order of the terms cannot
        # change a score, nor therefore a tie.
        return {chunk: math.fsum(parts) for chunk, parts in terms.items()}


@dataclass(frozen=True, slots=True)
class Passage:
    """A chunk found for a query, where it is from, and how well it matched.

    ``source`` is the source's name. ``score`` is the variant's: BM25 for
    words, cosine similarity for meaning, the fused score for both.
    """

    chunk: Chunk
    document: Document
    source: str
    score: float
    private: bool

    def citation(self) -> str:
        """Return where the passage is: ``source/path:start-end``.

        A single line is ``source/path:N``; a page comes first, as in
        ``source/path:p.2:4-9``.
        """
        start, end = self.chunk.start_line, self.chunk.end_line
        lines = str(start) if start == end else f"{start}-{end}"
        page = "" if self.chunk.page is None else f"p.{self.chunk.page}:"
        return f"{self.source}/{self.document.path.as_posix()}:{page}{lines}"


@dataclass(frozen=True, slots=True)
class Found:
    """The passages for a query, best first, and how many private ones were left out.

    ``withheld`` counts the private passages among the best ``limit`` of the
    whole ranking, so a question only private text answers says so.
    """

    passages: tuple[Passage, ...]
    withheld: int


def _shown(
    rows: Sequence[Row], scores: Mapping[int, float], *, limit: int, private: bool
) -> Found:
    allowed = [row for row in rows if private or not row.private]
    withheld = 0 if private else sum(row.private for row in rows[:limit])
    return Found(
        tuple(
            Passage(
                row.chunk,
                row.document,
                row.source_name,
                scores[row.chunk.id],
                row.private,
            )
            for row in allowed[:limit]
        ),
        withheld,
    )


class Ranker:
    """Ranks a store's chunks for a query, as ``Knowledge.search`` asks."""

    def __init__(
        self,
        store: KnowledgeStore,
        vectors: ChunkVectors,
        rerankers: Mapping[str, CrossEncoder] | None = None,
    ) -> None:
        """Rank ``store``'s chunks by their words and by ``vectors``.

        ``rerankers`` are the loaded rerankers, by catalogue id.
        """
        self._store = store
        self._vectors = vectors
        self._rerankers = dict(rerankers or {})

    def reranker(self, variant: Variant) -> CrossEncoder | None:
        """Return the loaded reranker ``variant`` reorders with; None if it does not.

        Raises:
            VariantError: If ``variant`` reranks with a reranker not loaded.
        """
        if variant.rerank is None:
            return None
        encoder = self._rerankers.get(variant.rerank.model)
        if encoder is None:
            message = f"{variant.name()!r} needs the reranker {variant.rerank.model}"
            raise VariantError(message)
        return encoder

    async def find(  # noqa: PLR0913 - one search's whole request
        self,
        query: str,
        *,
        limit: int,
        variant: Variant,
        private: bool,
        also: Sequence[str] = (),
        merge: Merge = Merge.BEST,
    ) -> Found:
        """Return up to ``limit`` passages for ``query`` as ``variant`` ranks them.

        A query with no word the full-text index can match finds nothing,
        whatever the variant. A chunk removed since it was scored is left
        out. With ``private`` False, private passages are left out and
        counted in :attr:`Found.withheld`. A reranked variant reorders the
        first ``top`` chunks, private ones included, by the reranker's
        scores, which then are their passages' scores.

        Each query in ``also`` is ranked by ``variant`` as ``query`` is, hops
        included, from as many candidates, and ``merge`` joins every ranking
        into the passages' scores; with ``also`` empty nothing is merged. A
        query in ``also`` with no word the full-text index can match still
        ranks by its meaning, so it adds nothing to words alone; only
        ``query`` decides whether anything is found. The reranker judges the
        passages against ``query`` alone.

        Raises:
            VariantError: If ``variant`` reranks with a reranker not loaded.
        """
        encoder = self.reranker(variant)
        if not match_expression(query):
            return Found((), 0)
        count = max(CANDIDATES, limit)
        scores = await self._scores(query, variant, count)
        if also:
            more = [await self._scores(other, variant, count) for other in also]
            scores = merge.of([scores, *more])
        rows = await self._store.rows([ChunkId(chunk) for chunk in _ranked(scores)])
        if encoder is not None and variant.rerank is not None:
            rows = await _reranked(query, rows, scores, encoder, variant.rerank.top)
        return _shown(rows, scores, limit=limit, private=private)

    async def _scores(
        self, query: str, variant: Variant, count: int
    ) -> dict[int, float]:
        score = await self._scorer(query, variant, count)
        scores = await score(None)
        if not variant.hops:
            return scores
        return await hopped(
            scores,
            hops=variant.hops,
            link_weight=variant.link_weight,
            reach=self._store.linked_chunks,
            score=score,
        )

    async def _scorer(self, query: str, variant: Variant, count: int) -> Scorer:
        if variant.side is Side.WORDS:
            return partial(_words, store=self._store, query=query, count=count)
        near = await self._vectors.similarities(query)
        if variant.side is Side.MEANING:
            return partial(_meaning, near=near, count=count)
        return partial(
            _fused,
            store=self._store,
            query=query,
            near=near,
            word_weight=variant.word_weight,
            count=count,
        )


async def _reranked(
    query: str,
    rows: Sequence[Row],
    scores: dict[int, float],
    encoder: CrossEncoder,
    top: int,
) -> list[Row]:
    """Return ``rows`` with the first ``top`` in the reranker's order.

    Each reordered chunk's entry in ``scores`` becomes the reranker's score.
    """
    first = rows[:top]
    judged = await encoder.rank(query, [_text(row.chunk) for row in first])
    for row, score in zip(first, judged, strict=True):
        scores[row.chunk.id] = score
    order = sorted(range(len(first)), key=lambda i: (-judged[i], i))
    return [first[i] for i in order] + list(rows[top:])


def _text(chunk: Chunk) -> str:
    return f"{chunk.header}\n{chunk.body}" if chunk.header else chunk.body


def _similar(
    within: Collection[ChunkId] | None, near: Similarities, count: int
) -> dict[int, float]:
    return near.nearest(count) if within is None else near.of(within)


async def _words(
    within: Collection[ChunkId] | None,
    /,
    *,
    store: KnowledgeStore,
    query: str,
    count: int,
) -> dict[int, float]:
    limit = count if within is None else len(within)
    return (await store.word_scores(query, limit=limit, within=within)).scores


async def _meaning(
    within: Collection[ChunkId] | None, /, *, near: Similarities, count: int
) -> dict[int, float]:
    return _similar(within, near, count)


async def _fused(  # noqa: PLR0913 - one query's evidence, bound once by partial
    within: Collection[ChunkId] | None,
    /,
    *,
    store: KnowledgeStore,
    query: str,
    near: Similarities,
    word_weight: float,
    count: int,
) -> dict[int, float]:
    limit = count if within is None else len(within)
    words = await store.word_scores(query, limit=limit, within=within)
    similar = _similar(within, near, count) | near.of(words.scores.keys())
    return fused(similar, words, word_weight)
