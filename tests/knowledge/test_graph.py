from collections.abc import Collection, Mapping

import pytest

from synthia.knowledge.graph import SEEDS, best, hopped
from synthia.knowledge.model import ChunkId


class Graph:
    """Links between chunks and each chunk's own score, recording every call."""

    def __init__(
        self, links: Mapping[int, tuple[int, ...]], own: Mapping[int, float]
    ) -> None:
        self.links = links
        self.own = own
        self.reached: list[list[int]] = []
        self.scored: list[list[int]] = []

    async def reach(
        self, chunks: Collection[ChunkId]
    ) -> dict[ChunkId, tuple[ChunkId, ...]]:
        self.reached.append(sorted(chunks))
        return {
            ChunkId(c): tuple(ChunkId(t) for t in self.links[c])
            for c in chunks
            if c in self.links
        }

    async def score(self, within: Collection[ChunkId] | None) -> dict[int, float]:
        assert within is not None
        self.scored.append(sorted(within))
        return {c: self.own[c] for c in within if c in self.own}


async def run(
    graph: Graph, scores: dict[int, float], *, hops: int = 1, weight: float = 0.5
) -> dict[int, float]:
    return await hopped(
        scores, hops=hops, link_weight=weight, reach=graph.reach, score=graph.score
    )


def test_best_takes_the_highest_scores_and_breaks_ties_by_the_lower_id() -> None:
    assert best({3: 0.5, 1: 0.9, 2: 0.5, 4: 0.1}, 3) == [1, 2, 3]


async def test_a_linked_chunk_scores_its_own_score_plus_the_weighted_seed() -> None:
    graph = Graph({1: (9,)}, {9: 0.1})

    assert await run(graph, {1: 0.8, 2: 0.4}) == {1: 0.8, 2: 0.4, 9: 0.1 + 0.5 * 0.8}
    assert graph.scored == [[9]]


async def test_a_chunk_reached_by_two_seeds_takes_the_best_seed() -> None:
    graph = Graph({1: (9,), 2: (9,)}, {9: 0.0})

    assert (await run(graph, {1: 0.4, 2: 0.8}))[9] == pytest.approx(0.4)


async def test_a_linked_chunk_already_scored_is_lifted_by_the_same_rule() -> None:
    graph = Graph({1: (2, 3)}, {2: 0.3, 3: 0.3})

    lifted = await run(graph, {1: 0.8, 2: 0.3, 3: 0.9})

    assert lifted[2] == pytest.approx(0.3 + 0.5 * 0.8)
    assert lifted[3] == 0.9


async def test_a_chunk_with_no_score_of_its_own_gets_the_lift_alone() -> None:
    graph = Graph({1: (9,)}, {})

    assert (await run(graph, {1: 0.8}))[9] == pytest.approx(0.4)


async def test_links_are_followed_from_the_best_seeds_only() -> None:
    scores = {chunk: 1.0 - chunk / 10 for chunk in range(1, SEEDS + 2)}
    graph = Graph({SEEDS + 1: (99,)}, {99: 5.0})

    assert await run(graph, scores) == scores
    assert graph.reached == [list(range(1, SEEDS + 1))]


async def test_a_second_hop_starts_from_the_chunks_the_first_reached() -> None:
    graph = Graph({1: (5,), 5: (9,)}, {5: 0.2, 9: 0.1})

    one = await run(graph, {1: 0.8})
    two = await run(graph, {1: 0.8}, hops=2)

    assert 9 not in one
    assert two[5] == pytest.approx(0.2 + 0.5 * 0.8)
    assert two[9] == pytest.approx(0.1 + 0.5 * (0.2 + 0.5 * 0.8))
    assert graph.reached[-2:] == [[1], [5]]


async def test_hops_stop_when_no_seed_links_anywhere() -> None:
    graph = Graph({}, {})

    assert await run(graph, {1: 0.8}, hops=2) == {1: 0.8}
    assert graph.reached == [[1]]
    assert graph.scored == []


async def test_a_cycle_carries_a_lift_back_by_the_square_of_the_weight() -> None:
    graph = Graph({1: (2,), 2: (1,)}, {1: 0.5, 2: 0.4})

    lifted = await run(graph, {1: 0.5, 2: 0.4}, hops=2)

    assert lifted[1] == pytest.approx(0.5 + 0.5 * 0.4 + 0.25 * 0.5)
    assert lifted[2] == pytest.approx(0.4 + 0.5 * 0.5 + 0.25 * 0.4)


async def test_a_second_hop_back_to_the_first_seed_lifts_it_by_its_own_chain() -> None:
    graph = Graph({1: (5,), 5: (1, 9)}, {1: 0.8, 5: 0.2, 9: 0.1})

    lifted = await run(graph, {1: 0.8}, hops=2)

    assert lifted[1] == pytest.approx(0.8 + 0.5 * (0.2 + 0.5 * 0.8))
    assert lifted[9] == pytest.approx(0.1 + 0.5 * (0.2 + 0.5 * 0.8))


async def test_nothing_scored_reaches_nothing() -> None:
    graph = Graph({}, {})

    assert await run(graph, {}) == {}
    assert graph.scored == []
