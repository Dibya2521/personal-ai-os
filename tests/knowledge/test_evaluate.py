import math
from collections.abc import Callable, Sequence
from pathlib import PurePosixPath

import pytest
from hypothesis import given
from hypothesis import strategies as st

from synthia.knowledge.evaluate import (
    DEPTHS,
    Answerable,
    Gold,
    RankMetrics,
    Report,
    Unanswerable,
    covers,
    hits,
    ndcg_at,
    recall_at,
    reciprocal_rank,
    scored,
)
from synthia.knowledge.model import (
    Chunk,
    ChunkId,
    Document,
    DocumentId,
    DocumentKind,
    SourceId,
)
from synthia.knowledge.retrieve import Passage

CAT = PurePosixPath("notes/cat.md")
DOG = PurePosixPath("notes/dog.md")


def passage(
    path: PurePosixPath,
    start: int,
    end: int,
    page: int | None = None,
    score: float = 1.0,
) -> Passage:
    chunk = Chunk(ChunkId(1), DocumentId(1), 0, page, start, end, "", "text")
    document = Document(
        DocumentId(1),
        SourceId(1),
        path,
        DocumentKind.MARKDOWN,
        0,
        0,
        "sha",
        "recipe",
        path.stem,
    )
    return Passage(chunk, document, "notes", score, private=False)


def ranks(recall: tuple[float, ...], rr: float, ndcg: float) -> RankMetrics:
    return RankMetrics(dict(zip(DEPTHS, recall, strict=True)), rr, ndcg)


@pytest.mark.parametrize(
    ("found", "gold", "covered"),
    [
        (passage(CAT, 3, 6), Gold(CAT, 5, 9), True),
        (passage(DOG, 3, 6), Gold(CAT, 5, 9), False),
        (passage(CAT, 3, 6, page=2), Gold(CAT, 3, 6, page=1), False),
        (passage(CAT, 3, 6, page=1), Gold(CAT, 3, 6), False),
        (passage(CAT, 3, 6, page=2), Gold(CAT, 3, 6, page=2), True),
        (passage(CAT, 3, 6), Gold(CAT, 7, 9), False),
        (passage(CAT, 7, 9), Gold(CAT, 3, 6), False),
        (passage(CAT, 3, 6), Gold(CAT, 6, 9), True),
        (passage(CAT, 6, 9), Gold(CAT, 3, 6), True),
        (passage(CAT, 1, 20), Gold(CAT, 8, 8), True),
    ],
    ids=[
        "overlap",
        "wrong_file",
        "wrong_page",
        "page_against_none",
        "same_page",
        "adjacent_after",
        "adjacent_before",
        "touching_end_on_start",
        "touching_start_on_end",
        "inside",
    ],
)
def test_covers_needs_file_page_and_a_shared_line(
    found: Passage, gold: Gold, covered: bool
) -> None:
    assert covers(found, gold) is covered


def test_ndcg_at_new_hits_at_ranks_one_and_three() -> None:
    # (1 + 1/log2 4) / (1 + 1/log2 3) = 1.5 / 1.6309 = 0.9197
    expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at((1, 0, 1), 3, 2) == pytest.approx(expected)
    assert expected == pytest.approx(0.9197, abs=1e-4)


def test_ndcg_at_a_depth_below_the_gold_spans_is_perfect_when_every_rank_hits() -> None:
    # Five spans but only three ranks counted: the ideal also fills three.
    assert ndcg_at((1, 1, 1, 1, 1), 3, 5) == 1.0


def test_ndcg_at_counts_a_hit_at_rank_k_and_none_after_it() -> None:
    assert ndcg_at((0, 1), 1, 1) == 0.0
    assert ndcg_at((0, 1), 2, 1) == pytest.approx(1 / math.log2(3))


@pytest.mark.parametrize("metric", [recall_at, ndcg_at])
def test_rank_metrics_refuse_hits_covering_more_than_the_gold_spans(
    metric: Callable[[Sequence[int], int, int], float],
) -> None:
    with pytest.raises(
        ValueError, match=r"^hits=\(1, 1\) cover more than golds=1 spans$"
    ):
        metric((1, 1), 2, 1)


def test_hits_second_passage_over_one_span_adds_nothing() -> None:
    ranked = [passage(CAT, 1, 4), passage(CAT, 3, 8)]
    counts = hits(ranked, [Gold(CAT, 2, 5)])
    assert counts == (1, 0)
    assert recall_at(counts, 10, 1) == 1.0
    assert ndcg_at(counts, 10, 1) == 1.0


def test_hits_one_passage_over_two_spans() -> None:
    counts = hits([passage(CAT, 1, 10)], [Gold(CAT, 2, 3), Gold(CAT, 8, 9)])
    assert counts == (2,)
    assert recall_at(counts, 1, 2) == 1.0
    # One new-hit rank, gain 1; the ideal covers a span at ranks 1 and 2.
    assert ndcg_at(counts, 10, 2) == pytest.approx(1 / (1 + 1 / math.log2(3)))
    # At k=1 the ideal has room for one span only, so one hit at rank 1 is ideal.
    assert ndcg_at(counts, 1, 2) == 1.0


def test_hits_counts_a_span_at_the_first_passage_to_cover_it() -> None:
    ranked = [passage(DOG, 1, 2), passage(CAT, 1, 3), passage(CAT, 2, 9)]
    counts = hits(ranked, [Gold(CAT, 3, 3), Gold(CAT, 8, 8), Gold(DOG, 9, 9)])
    assert counts == (0, 1, 1)


def test_reciprocal_rank_of_the_first_hit() -> None:
    assert reciprocal_rank((0, 0, 2, 1)) == pytest.approx(1 / 3)
    assert reciprocal_rank((0, 0)) == 0.0
    assert reciprocal_rank(()) == 0.0


def test_recall_at_a_depth_beyond_the_ranking() -> None:
    assert recall_at((0, 1), 10, 2) == 0.5
    assert recall_at((0, 1), 1, 2) == 0.0


@pytest.mark.parametrize(("k", "golds"), [(1, 0), (0, 1), (-1, 3)])
def test_rank_metrics_refuse_no_gold_or_no_depth(k: int, golds: int) -> None:
    with pytest.raises(ValueError, match=rf"k={k}, golds={golds}"):
        recall_at((1,), k, golds)
    with pytest.raises(ValueError, match=rf"k={k}, golds={golds}"):
        ndcg_at((1,), k, golds)


@pytest.mark.parametrize(("start", "end"), [(5, 4), (0, 3)])
def test_gold_refuses_a_line_range_that_is_not_one(start: int, end: int) -> None:
    with pytest.raises(ValueError, match=rf"start={start}, end={end}"):
        Gold(CAT, start, end)


SPANS = st.tuples(
    st.sampled_from((CAT, DOG)),
    st.sampled_from((None, 1)),
    st.integers(1, 12),
    st.integers(0, 4),
)


@given(
    found=st.lists(SPANS, max_size=12),
    golds=st.lists(SPANS, min_size=1, max_size=6),
)
def test_rank_metrics_stay_in_bounds_and_recall_grows_with_depth(
    found: list[tuple[PurePosixPath, int | None, int, int]],
    golds: list[tuple[PurePosixPath, int | None, int, int]],
) -> None:
    ranked = [
        passage(path, start, start + size, page) for path, page, start, size in found
    ]
    gold = [Gold(path, start, start + size, page) for path, page, start, size in golds]
    counts = hits(ranked, gold)
    assert sum(counts) <= len(gold)
    depths = range(1, len(ranked) + 3)
    recalls = [recall_at(counts, k, len(gold)) for k in depths]
    assert all(0.0 <= recall <= 1.0 for recall in recalls)
    assert recalls == sorted(recalls)
    assert all(0.0 <= ndcg_at(counts, k, len(gold)) <= 1.0 for k in depths)


def test_scored_answerable_question() -> None:
    ranked = [passage(DOG, 1, 2, score=0.7), passage(CAT, 4, 6, score=0.6)]
    result = scored(ranked, [Gold(CAT, 5, 5)])
    assert isinstance(result, Answerable)
    assert result.top_score == 0.7
    # The only hit is at rank 2: recall 0 at depth 1, then 1; 1/log2 3 = 0.6309.
    assert result.ranks.recall == {1: 0.0, 5: 1.0, 10: 1.0}
    assert result.ranks.reciprocal_rank == 0.5
    assert result.ranks.ndcg == pytest.approx(0.6309, abs=1e-4)


def test_scored_answerable_question_with_nothing_found() -> None:
    assert scored([], [Gold(CAT, 5, 5)]) == Answerable(
        ranks((0.0, 0.0, 0.0), 0.0, 0.0), 0.0
    )


def test_scored_unanswerable_question_keeps_the_top_score() -> None:
    assert scored([passage(DOG, 1, 2, score=0.4)], []) == Unanswerable(0.4)
    assert scored([], []) == Unanswerable(0.0)


def test_report_means_per_category_and_over_all() -> None:
    perfect = Answerable(ranks((1.0, 1.0, 1.0), 1.0, 1.0), 0.9)
    second = Answerable(ranks((0.0, 1.0, 1.0), 0.5, 0.5), 0.7)
    report = Report.of(
        [
            ("named", perfect),
            ("none", Unanswerable(0.4)),
            ("named", second),
            ("none", Unanswerable(0.0)),
        ]
    )
    named, none, overall = (
        report.categories["named"],
        report.categories["none"],
        report.overall,
    )
    named_ranks = ranks((0.5, 1.0, 1.0), 0.75, 0.75)
    assert list(report.categories) == ["named", "none"]
    assert (named.questions, named.answerable, named.ranks) == (2, 2, named_ranks)
    assert named.top_score == pytest.approx(0.8)
    assert (none.questions, none.answerable, none.ranks) == (2, 0, None)
    assert none.top_score == pytest.approx(0.2)
    # Rank means over the two answerable questions only; top score over all four.
    assert (overall.questions, overall.answerable, overall.ranks) == (4, 2, named_ranks)
    assert overall.top_score == pytest.approx(0.5)


def test_report_refuses_no_question() -> None:
    with pytest.raises(ValueError, match="at least one question"):
        Report.of([])
