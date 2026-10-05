"""Remembered turns as vectors of their meaning, searched exactly.

Every vector is compared with the query in one matrix product. On a laptop
CPU that took about a millisecond at 10,000 vectors of 384 numbers and
about seven at 50,000, while embedding the query itself takes 3 to 12, so
until a store holds tens of thousands of turns an exact search costs less
than the question it answers, and it is exactly right. The HNSW index in
:mod:`synthia.memory.hnsw` is the way past that size.

Each vector keeps the time of its turn, so a search can be limited to a
range of days as the word search is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Collection
    from datetime import datetime

    from numpy.typing import NDArray

type Vectors = NDArray[np.float32]

_FIRST_CAPACITY: Final = 256
_MICROSECONDS: Final = 1_000_000
_EARLIEST: Final = np.iinfo(np.int64).min
_LATEST: Final = np.iinfo(np.int64).max


class VectorError(ValueError):
    """A vector that does not have the set's number of dimensions."""


def _microseconds(at: datetime) -> int:
    return round(at.timestamp() * _MICROSECONDS)


@dataclass(frozen=True, slots=True)
class Similarities:
    """The cosine similarity of a query to each vector in a range, by turn id."""

    ids: NDArray[np.int64]
    cosines: Vectors

    def nearest(self, count: int) -> dict[int, float]:
        """Return the ``count`` most similar turns and their similarity."""
        if count >= len(self.ids):
            chosen = np.arange(len(self.ids))
        else:
            chosen = np.argpartition(-self.cosines, count)[:count]
        return self._of(chosen)

    def of(self, ids: Collection[int]) -> dict[int, float]:
        """Return the similarity of each turn in ``ids`` that is in the range."""
        wanted = np.fromiter(ids, np.int64, len(ids))
        return self._of(np.flatnonzero(np.isin(self.ids, wanted)))

    def _of(self, rows: NDArray[np.intp]) -> dict[int, float]:
        return dict(
            zip(self.ids[rows].tolist(), self.cosines[rows].tolist(), strict=True)
        )


class VectorSet:
    """Unit vectors known by turn id, each with the time of its turn."""

    def __init__(self, dimensions: int) -> None:
        """Start empty, for vectors of ``dimensions`` numbers."""
        self.dimensions = dimensions
        self._ids = np.zeros(_FIRST_CAPACITY, np.int64)
        self._ats = np.zeros(_FIRST_CAPACITY, np.int64)
        self._vectors: Vectors = np.zeros((_FIRST_CAPACITY, dimensions), np.float32)
        self._rows: dict[int, int] = {}

    def __len__(self) -> int:
        """Return how many vectors are held."""
        return len(self._rows)

    def __contains__(self, id_: object) -> bool:
        """Return whether a vector is held for ``id_``."""
        return id_ in self._rows

    def add(self, id_: int, at: datetime, vector: Vectors) -> None:
        """Hold ``vector`` for the turn ``id_`` from ``at``, replacing any before.

        Raises:
            VectorError: If ``vector`` does not have :attr:`dimensions` numbers.
        """
        if vector.shape != (self.dimensions,):
            message = f"a vector of shape {vector.shape}, not ({self.dimensions},)"
            raise VectorError(message)
        row = self._rows.get(id_)
        if row is None:
            row = len(self._rows)
            if row == len(self._ids):
                self._grow()
            self._rows[id_] = row
        self._ids[row] = id_
        self._ats[row] = _microseconds(at)
        self._vectors[row] = vector

    def remove(self, id_: int) -> bool:
        """Let go of the vector for ``id_``; False if none was held."""
        row = self._rows.pop(id_, None)
        if row is None:
            return False
        last = len(self._rows)
        if row != last:
            moved = int(self._ids[last])
            self._ids[row] = moved
            self._ats[row] = self._ats[last]
            self._vectors[row] = self._vectors[last]
            self._rows[moved] = row
        return True

    def similarities(
        self,
        query: Vectors,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> Similarities:
        """Return how similar ``query`` is to every turn from ``since`` up to ``until``.

        Both are unit vectors, so their dot product is their cosine similarity.
        """
        count = len(self._rows)
        start = _EARLIEST if since is None else _microseconds(since)
        end = _LATEST if until is None else _microseconds(until)
        ats = self._ats[:count]
        rows = np.flatnonzero((ats >= start) & (ats < end))
        return Similarities(self._ids[rows], self._vectors[rows] @ query)

    def _grow(self) -> None:
        self._ids = np.concatenate([self._ids, np.zeros_like(self._ids)])
        self._ats = np.concatenate([self._ats, np.zeros_like(self._ats)])
        self._vectors = np.concatenate([self._vectors, np.zeros_like(self._vectors)])
