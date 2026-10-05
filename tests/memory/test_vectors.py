import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from synthia.memory.vectors import VectorError, VectorSet

MONDAY = datetime(2026, 9, 28, 9, 0, tzinfo=UTC)
DAY = timedelta(days=1)
HALF = 1 / math.sqrt(2)


def unit(*numbers: float) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
    vector = np.array(numbers, np.float32)
    return vector / np.linalg.norm(vector)


@pytest.fixture
def three() -> VectorSet:
    vectors = VectorSet(3)
    vectors.add(1, MONDAY, unit(1, 0, 0))
    vectors.add(2, MONDAY + DAY, unit(0, 1, 0))
    vectors.add(3, MONDAY + 2 * DAY, unit(1, 1, 0))
    return vectors


def test_the_nearest_come_with_their_cosine_similarity(three: VectorSet) -> None:
    found = three.similarities(unit(1, 0, 0)).nearest(2)

    assert found == pytest.approx({1: 1.0, 3: HALF})


def test_asking_for_more_than_are_held_returns_them_all(three: VectorSet) -> None:
    found = three.similarities(unit(0, 0, 1)).nearest(10)

    assert found == pytest.approx({1: 0.0, 2: 0.0, 3: 0.0})


def test_only_turns_inside_the_dates_are_compared(three: VectorSet) -> None:
    middle = three.similarities(
        unit(1, 0, 0), since=MONDAY + DAY, until=MONDAY + 2 * DAY
    )
    from_tuesday = three.similarities(unit(1, 0, 0), since=MONDAY + DAY)

    assert middle.nearest(3) == pytest.approx({2: 0.0})
    assert from_tuesday.nearest(3) == pytest.approx({2: 0.0, 3: HALF})


def test_named_turns_are_scored_when_they_are_in_range(three: VectorSet) -> None:
    found = three.similarities(unit(0, 1, 0), until=MONDAY + 2 * DAY).of({2, 3, 99})

    assert found == pytest.approx({2: 1.0})


def test_a_removed_vector_is_never_found_and_the_rest_still_are(
    three: VectorSet,
) -> None:
    assert three.remove(1)
    assert not three.remove(1)

    found = three.similarities(unit(1, 0, 0)).nearest(3)

    assert found == pytest.approx({2: 0.0, 3: HALF})
    assert (len(three), 1 in three, 3 in three) == (2, False, True)


def test_the_vector_moved_into_a_removed_place_can_be_removed_too(
    three: VectorSet,
) -> None:
    three.remove(1)
    three.remove(3)

    assert three.similarities(unit(0, 1, 0)).nearest(3) == pytest.approx({2: 1.0})


def test_the_newest_vector_can_be_removed_and_a_new_one_takes_its_place(
    three: VectorSet,
) -> None:
    three.remove(3)
    three.add(4, MONDAY, unit(0, 0, 1))

    found = three.similarities(unit(0, 0, 1)).nearest(3)

    assert found == pytest.approx({1: 0.0, 2: 0.0, 4: 1.0})


def test_adding_a_turn_again_replaces_its_vector_and_time(three: VectorSet) -> None:
    three.add(1, MONDAY + 5 * DAY, unit(0, 0, 1))

    found = three.similarities(unit(0, 0, 1), since=MONDAY + 3 * DAY).nearest(3)

    assert found == pytest.approx({1: 1.0})
    assert len(three) == 3


def test_many_vectors_match_a_plain_product_after_growing_and_removing() -> None:
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(600, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    held = VectorSet(8)
    for id_, vector in enumerate(vectors):
        held.add(id_, MONDAY, vector)
    for id_ in range(0, 600, 3):
        held.remove(id_)
    query = vectors[1]

    found = held.similarities(query).nearest(600)

    expected = {i: float(vectors[i] @ query) for i in range(600) if i % 3}
    # One matrix product sums float32 in another order than 600 dot products.
    assert found == pytest.approx(expected, abs=1e-6)


def test_a_vector_of_the_wrong_size_is_refused() -> None:
    with pytest.raises(VectorError, match=r"\(2,\), not \(3,\)"):
        VectorSet(3).add(1, MONDAY, unit(1, 0))
