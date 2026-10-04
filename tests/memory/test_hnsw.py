from pathlib import Path

import numpy as np
import pytest

from synthia.memory.hnsw import Hnsw, HnswError

DIM = 64


def clustered(seed: int, count: int) -> np.ndarray:
    """Unit vectors in tight groups, the shape sentence embeddings take."""
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((max(count // 50, 2), DIM))
    points = centres[rng.integers(0, len(centres), count)]
    points += 0.6 * rng.standard_normal((count, DIM))
    return (points / np.linalg.norm(points, axis=1, keepdims=True)).astype(np.float32)


def built(points: np.ndarray, **options: int) -> Hnsw:
    index = Hnsw(DIM, **options)
    for id_, point in enumerate(points):
        index.add(id_, point)
    return index


def recall(
    index: Hnsw, points: np.ndarray, live: list[int], queries: np.ndarray
) -> float:
    hits = 0
    for query in queries:
        nearest: list[int] = np.argsort(-(points[live] @ query))[:10].tolist()
        truth = {live[i] for i in nearest}
        hits += len(truth & {n.id for n in index.search(query, 10)})
    return hits / (10 * len(queries))


@pytest.fixture(scope="module")
def points() -> np.ndarray:
    return clustered(1, 1500)


@pytest.fixture(scope="module")
def index(points: np.ndarray) -> Hnsw:
    return built(points)


def test_search_finds_the_true_nearest_ten(index: Hnsw, points: np.ndarray) -> None:
    queries = clustered(2, 50)
    assert recall(index, points, list(range(len(points))), queries) >= 0.98


def test_the_nearest_comes_first_with_its_distance() -> None:
    index = Hnsw(3)
    index.add(7, np.array([1, 0, 0], np.float32))
    index.add(8, np.array([0, 1, 0], np.float32))
    index.add(9, np.array([1, 1, 0], np.float32))

    found = index.search(np.array([2, 0, 0], np.float32), 2)

    assert [n.id for n in found] == [7, 9]
    assert found[0].distance == pytest.approx(0.0, abs=1e-6)
    assert found[1].distance == pytest.approx(1 - 2**-0.5, abs=1e-6)


def test_removed_vectors_are_never_found_and_the_rest_still_are(
    points: np.ndarray,
) -> None:
    index = built(points)
    gone = list(range(0, len(points), 2))
    for id_ in gone:
        assert index.remove(id_)
    live = list(range(1, len(points), 2))

    assert not index.remove(0)
    assert len(index) == len(live)
    assert 0 not in index
    queries = clustered(3, 50)
    found = {n.id for q in queries for n in index.search(q, 10)}
    assert found.isdisjoint(gone)
    assert recall(index, points, live, queries) >= 0.98


def test_a_compacted_index_holds_only_what_can_be_found(points: np.ndarray) -> None:
    index = built(points[:300])
    for id_ in range(100):
        index.remove(id_)

    fresh = index.compacted()

    assert len(fresh) == 200
    assert 0 not in fresh
    assert 150 in fresh
    assert fresh.search(points[150], 1)[0].id == 150


def test_adding_an_id_again_replaces_its_vector() -> None:
    index = Hnsw(3)
    index.add(1, np.array([1, 0, 0], np.float32))
    index.add(1, np.array([0, 0, 1], np.float32))

    assert len(index) == 1
    [found] = index.search(np.array([0, 0, 1], np.float32), 5)
    assert found.id == 1
    assert found.distance == pytest.approx(0.0, abs=1e-6)


def test_an_empty_index_finds_nothing() -> None:
    assert Hnsw(3).search(np.array([1, 0, 0], np.float32), 5) == []


@pytest.mark.parametrize(
    "vector",
    [np.zeros(3), np.array([1.0, np.nan, 0.0]), np.ones(4)],
    ids=["zero", "nan", "wrong size"],
)
def test_vectors_that_cannot_be_compared_are_refused(vector: np.ndarray) -> None:
    index = Hnsw(3)
    with pytest.raises(HnswError):
        index.add(1, vector)
    with pytest.raises(HnswError):
        index.search(vector, 1)


def test_the_same_seed_builds_the_same_index(points: np.ndarray) -> None:
    a, b = built(points[:400], seed=5), built(points[:400], seed=5)
    query = clustered(4, 1)[0]
    assert a.search(query, 10) == b.search(query, 10)


def test_a_saved_index_answers_as_before(points: np.ndarray, tmp_path: Path) -> None:
    path = tmp_path / "vectors.npz"
    index = built(points[:500])
    index.remove(3)
    index.save(path)

    loaded = Hnsw.load(path)

    assert len(loaded) == len(index)
    assert 3 not in loaded
    for query in clustered(5, 20):
        assert loaded.search(query, 10) == index.search(query, 10)
    loaded.add(10_000, points[3])
    assert loaded.search(points[3], 1)[0].id == 10_000
    assert not (tmp_path / "vectors.npz.part").exists()


def test_an_empty_index_saves_and_loads(tmp_path: Path) -> None:
    path = tmp_path / "vectors.npz"
    Hnsw(3).save(path)
    loaded = Hnsw.load(path)
    loaded.add(1, np.array([1, 0, 0], np.float32))
    assert [n.id for n in loaded.search(np.array([1, 0, 0], np.float32), 1)] == [1]


def test_a_file_that_is_not_a_saved_index_is_refused(tmp_path: Path) -> None:
    other = tmp_path / "other.npz"
    np.savez(other, numbers=np.arange(3))
    with pytest.raises(HnswError):
        Hnsw.load(other)
    newer = tmp_path / "newer.npz"
    np.savez(newer, header=np.array('{"version": 99}'))
    with pytest.raises(HnswError):
        Hnsw.load(newer)
