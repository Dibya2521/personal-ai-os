"""How well a ranking of passages finds the lines that answer a question.

A question's answer is one or more gold spans: a file's path relative to the
source root, a page for a PDF, and a run of lines. A passage covers a gold
span when it is in that file, on the same page, and its lines overlap the
span's (both ends inclusive). Each span counts once, at the first passage
that covers it, so a second passage over the same lines adds nothing.

- recall@k: the share of gold spans covered in the first k passages.
- Reciprocal rank: 1 / the rank of the first passage that covers a span.
- nDCG@k, binary: gain 1 at each rank that covers a new span, discounted by
  1 / log2(rank + 1); the ideal covers one span at each of the first
  min(k, spans) ranks. At most min(k, spans) ranks cover a new span, and the
  m-th of them is at rank m or later, so nDCG is at most 1.

The eval indexes one source, so a passage is matched by its path alone.

A question with no gold span is unanswerable and has no rank metrics; every
question keeps the score of its best passage, so the floor reached on
unanswerable questions can be set against the answerable ones.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean
from typing import TYPE_CHECKING, Final, Self

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import PurePosixPath

    from synthia.knowledge.retrieve import Passage

DEPTHS: Final = (1, 5, 10)
"""The depths recall is reported at."""
NDCG_DEPTH: Final = 10
"""The depth nDCG is reported at."""


@dataclass(frozen=True, slots=True)
class Gold:
    """Lines that answer a question: a file, its page for a PDF, and a line range.

    ``path`` is relative to the source root, as a document's path is. Lines
    are 1-based and inclusive; ``page`` is None for a file without pages.

    Raises:
        ValueError: If ``start`` is below 1 or ``end`` is below ``start``.
    """

    path: PurePosixPath
    start: int
    end: int
    page: int | None = None

    def __post_init__(self) -> None:
        if self.start < 1 or self.end < self.start:
            message = (
                f"start={self.start}, end={self.end} is not a line range: "
                "lines start at 1 and end at or after the start"
            )
            raise ValueError(message)


def covers(passage: Passage, gold: Gold) -> bool:
    """Return whether ``passage`` is in ``gold``'s file and page and shares a line."""
    chunk = passage.chunk
    return (
        passage.document.path == gold.path
        and chunk.page == gold.page
        and chunk.start_line <= gold.end
        and gold.start <= chunk.end_line
    )


def hits(ranked: Sequence[Passage], gold: Sequence[Gold]) -> tuple[int, ...]:
    """Return, per rank, how many gold spans that passage is the first to cover."""
    uncovered = set(range(len(gold)))
    counts: list[int] = []
    for passage in ranked:
        new = {span for span in uncovered if covers(passage, gold[span])}
        uncovered -= new
        counts.append(len(new))
    return tuple(counts)


def _check_depth(hits: Sequence[int], k: int, golds: int) -> None:
    if k < 1 or golds < 1:
        message = (
            f"k={k}, golds={golds}: both must be at least 1, since a question "
            "with no gold span has no rank metrics"
        )
        raise ValueError(message)
    if sum(hits) > golds:
        message = f"hits={tuple(hits)} cover more than golds={golds} spans"
        raise ValueError(message)


def recall_at(hits: Sequence[int], k: int, golds: int) -> float:
    """Return the share of the ``golds`` spans covered in the first ``k`` ranks.

    ``hits`` is what :func:`hits` gives for the same gold spans.

    Raises:
        ValueError: If ``k`` or ``golds`` is below 1, or ``hits`` covers
            more than ``golds`` spans.
    """
    _check_depth(hits, k, golds)
    return sum(hits[:k]) / golds


def reciprocal_rank(hits: Sequence[int]) -> float:
    """Return 1 / the first rank that covers a span, or 0.0 when none does."""
    return next((1 / rank for rank, hit in enumerate(hits, start=1) if hit), 0.0)


def _gain(rank: int) -> float:
    return 1 / math.log2(rank + 1)


def ndcg_at(hits: Sequence[int], k: int, golds: int) -> float:
    """Return binary nDCG over the first ``k`` ranks for ``golds`` spans.

    ``hits`` is what :func:`hits` gives for the same gold spans.

    Raises:
        ValueError: If ``k`` or ``golds`` is below 1, or ``hits`` covers
            more than ``golds`` spans.
    """
    _check_depth(hits, k, golds)
    found = sum(_gain(rank) for rank, hit in enumerate(hits[:k], start=1) if hit)
    ideal = sum(_gain(rank) for rank in range(1, min(k, golds) + 1))
    return found / ideal


@dataclass(frozen=True, slots=True)
class RankMetrics:
    """One question's rank metrics, or their means over several questions.

    ``recall`` maps each of :data:`DEPTHS` to recall at that depth; ``ndcg``
    is at :data:`NDCG_DEPTH`.
    """

    recall: Mapping[int, float]
    reciprocal_rank: float
    ndcg: float

    @classmethod
    def mean(cls, metrics: Sequence[RankMetrics]) -> Self:
        """Return the mean of each metric over ``metrics``.

        Raises:
            statistics.StatisticsError: If ``metrics`` is empty.
        """
        return cls(
            {k: fmean(m.recall[k] for m in metrics) for k in DEPTHS},
            fmean(m.reciprocal_rank for m in metrics),
            fmean(m.ndcg for m in metrics),
        )


@dataclass(frozen=True, slots=True)
class Answerable:
    """A question with gold spans: its rank metrics and its best passage's score."""

    ranks: RankMetrics
    top_score: float


@dataclass(frozen=True, slots=True)
class Unanswerable:
    """A question with no gold span: only its best passage's score."""

    top_score: float


type Scored = Answerable | Unanswerable


def scored(ranked: Sequence[Passage], gold: Sequence[Gold]) -> Scored:
    """Return one question's metrics for ``ranked``, best first.

    The top score is the first passage's, 0.0 when nothing was found. With
    no ``gold`` span the question is :class:`Unanswerable`.
    """
    top_score = ranked[0].score if ranked else 0.0
    if not gold:
        return Unanswerable(top_score)
    counts = hits(ranked, gold)
    ranks = RankMetrics(
        {k: recall_at(counts, k, len(gold)) for k in DEPTHS},
        reciprocal_rank(counts),
        ndcg_at(counts, NDCG_DEPTH, len(gold)),
    )
    return Answerable(ranks, top_score)


@dataclass(frozen=True, slots=True)
class Means:
    """Means over a group of questions.

    ``top_score`` is over every question; ``ranks`` over the answerable
    ones only, and None when the group has none.
    """

    questions: int
    answerable: int
    top_score: float
    ranks: RankMetrics | None

    @classmethod
    def of(cls, results: Sequence[Scored]) -> Self:
        """Return the means of ``results``.

        Raises:
            statistics.StatisticsError: If ``results`` is empty.
        """
        ranks = [result.ranks for result in results if isinstance(result, Answerable)]
        return cls(
            len(results),
            len(ranks),
            fmean(result.top_score for result in results),
            RankMetrics.mean(ranks) if ranks else None,
        )


@dataclass(frozen=True, slots=True)
class Report:
    """Means per question category, in the order first seen, and over all."""

    categories: Mapping[str, Means]
    overall: Means

    @classmethod
    def of(cls, results: Iterable[tuple[str, Scored]]) -> Self:
        """Return the report over each ``(category, result)`` pair.

        Raises:
            ValueError: If ``results`` is empty.
        """
        grouped: defaultdict[str, list[Scored]] = defaultdict(list)
        every: list[Scored] = []
        for category, result in results:
            grouped[category].append(result)
            every.append(result)
        if not every:
            message = "a report needs at least one question"
            raise ValueError(message)
        categories = {name: Means.of(group) for name, group in grouped.items()}
        return cls(categories, Means.of(every))
