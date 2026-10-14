"""SYNTHIA's own nearest-neighbour index: a hierarchical navigable small world.

Exact search compares a query with every stored vector. An HNSW graph
(Malkov and Yashunin, 2016) links each vector to a few near ones and searches
by walking: from an entry point, move to whichever neighbour is closer to the
query, until none is. Layers make the walk short: every vector is on layer
0, and each layer above holds a random, geometrically shrinking share of the
one below (a vector reaches layer l with probability 1 / m**l), so the top
layers cross the space in long jumps and the lower ones refine. A search
keeps the ``ef`` best candidates seen, not just one, which is what trades
speed for recall.

Vectors are compared by cosine distance (1 minus the dot product of unit
vectors), so each is normalised when it is added. A new vector is linked to
up to ``m`` neighbours on each of its layers (``2 * m`` allowed on layer 0),
chosen by the paper's heuristic: a candidate is kept only if it is closer to
the new vector than to every neighbour already kept, so the links spread out
in different directions instead of bunching in one cluster.

A removed vector is marked, not unlinked: it still carries searches across
the graph but is never returned, and never takes a place among the ``ef``
kept: with half of 1,500 vectors removed, recall@10 was 0.942 when they took
places and 0.998 when they do not. :meth:`Hnsw.compacted` builds a fresh index
of the vectors still present, for when the marks pile up.
"""

from __future__ import annotations

import heapq
import itertools
import json
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

M: Final = 16
# Measured at 10,000 clustered vectors: 200 gave the same recall@10 as 100 at
# every ef_search and took 1.6 times as long to build.
EF_CONSTRUCTION: Final = 100
EF_SEARCH: Final = 64
FORMAT_VERSION: Final = 1
_FIRST_CAPACITY: Final = 1024

type Vectors = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class Neighbour:
    """A stored vector near the query: its id and its cosine distance."""

    id: int
    distance: float


class HnswError(ValueError):
    """A vector of the wrong size, or a file that is not a saved index."""


def _unit(vector: Vectors) -> Vectors:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or not math.isfinite(norm):
        message = "a vector must be finite and not all zeros"
        raise HnswError(message)
    return (vector / norm).astype(np.float32)


def _unflattened(
    levels: list[int], sizes: list[int], flat: list[int]
) -> list[list[list[int]]]:
    """Rebuild each slot's links per layer from their saved, flattened form."""
    links: list[list[list[int]]] = []
    layer_sizes = iter(sizes)
    position = 0
    for level_count in levels:
        layers: list[list[int]] = []
        for size in itertools.islice(layer_sizes, level_count):
            layers.append(flat[position : position + size])
            position += size
        links.append(layers)
    return links


class Hnsw:
    """An HNSW graph over unit vectors known by integer ids."""

    def __init__(
        self,
        dim: int,
        *,
        m: int = M,
        ef_construction: int = EF_CONSTRUCTION,
        ef_search: int = EF_SEARCH,
        seed: int = 0,
    ) -> None:
        """Start empty for vectors of ``dim`` numbers.

        ``m`` links per vector and layer, ``ef_construction`` candidates kept
        while linking a new vector, ``ef_search`` while searching; ``seed``
        makes the layers each vector reaches repeatable.
        """
        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._level_scale = 1 / math.log(m)
        self._rng = random.Random(seed)  # noqa: S311 - layers need spread, not secrecy
        self._vectors: Vectors = np.zeros((_FIRST_CAPACITY, dim), np.float32)
        self._ids: list[int] = []
        self._slots: dict[int, int] = {}
        self._links: list[list[list[int]]] = []
        self._removed: set[int] = set()
        self._entry: int | None = None

    def __len__(self) -> int:
        """Return how many vectors can be found."""
        return len(self._slots)

    def __contains__(self, id_: object) -> bool:
        """Return whether a vector is stored under ``id_``."""
        return id_ in self._slots

    def add(self, id_: int, vector: Vectors) -> None:
        """Store ``vector`` under ``id_``, replacing what was stored there.

        Raises:
            HnswError: If the vector has the wrong size, or is zero or not finite.
        """
        query = self._checked(vector)
        self.remove(id_)
        slot = len(self._ids)
        if slot == len(self._vectors):
            self._vectors = np.concatenate(
                [self._vectors, np.zeros_like(self._vectors)]
            )
        self._vectors[slot] = query
        self._ids.append(id_)
        self._slots[id_] = slot
        level = int(-math.log(1.0 - self._rng.random()) * self._level_scale)
        self._links.append([[] for _ in range(level + 1)])
        if self._entry is None:
            self._entry = slot
        else:
            self._link(slot, query, level, self._entry)

    def remove(self, id_: int) -> bool:
        """Stop finding ``id_``; False if nothing is stored under it."""
        slot = self._slots.pop(id_, None)
        if slot is None:
            return False
        self._removed.add(slot)
        return True

    def search(self, vector: Vectors, k: int, ef: int | None = None) -> list[Neighbour]:
        """Return up to ``k`` stored vectors nearest ``vector``, nearest first.

        Raises:
            HnswError: If the vector has the wrong size, or is zero or not finite.
        """
        query = self._checked(vector)
        if self._entry is None or not self._slots:
            return []
        entry = self._descend(query, self._entry, 0)
        ef = max(ef or self.ef_search, k)
        found = self._search_layer(query, [entry], ef, 0, self._removed)
        return [Neighbour(self._ids[s], d) for d, s in found[:k]]

    def compacted(self) -> Hnsw:
        """Return a new index holding only the vectors that can be found."""
        fresh = Hnsw(
            self.dim,
            m=self.m,
            ef_construction=self.ef_construction,
            ef_search=self.ef_search,
        )
        for id_, slot in self._slots.items():
            fresh.add(id_, self._vectors[slot])
        return fresh

    def save(self, path: Path) -> None:
        """Write the index to ``path`` in one step, so a reader never sees half."""
        count = len(self._ids)
        flat = [n for links in self._links for layer in links for n in layer]
        sizes = [len(layer) for links in self._links for layer in links]
        header = {
            "version": FORMAT_VERSION,
            "dim": self.dim,
            "m": self.m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "entry": self._entry,
        }
        part = path.with_name(path.name + ".part")
        with part.open("wb") as file:
            np.savez(
                file,
                header=np.array(json.dumps(header)),
                vectors=self._vectors[:count],
                ids=np.array(self._ids, np.int64),
                levels=np.array([len(links) for links in self._links], np.int32),
                sizes=np.array(sizes, np.int32),
                links=np.array(flat, np.int32),
                removed=np.array(sorted(self._removed), np.int64),
            )
        part.replace(path)

    @classmethod
    def load(cls, path: Path, *, seed: int = 0) -> Hnsw:
        """Read an index written by :meth:`save`.

        Raises:
            HnswError: If the file is not a saved index of this format.
            OSError: If it cannot be read.
        """
        try:
            with np.load(path, allow_pickle=False) as data:
                header = json.loads(str(data["header"]))
                if header.get("version") != FORMAT_VERSION:
                    message = f"{path} is not a saved index of format {FORMAT_VERSION}"
                    raise HnswError(message)
                index = cls(
                    header["dim"],
                    m=header["m"],
                    ef_construction=header["ef_construction"],
                    ef_search=header["ef_search"],
                    seed=seed,
                )
                vectors = data["vectors"].astype(np.float32)
                ids = [int(i) for i in data["ids"]]
                levels = data["levels"].tolist()
                sizes = data["sizes"].tolist()
                flat = data["links"].tolist()
                removed = {int(s) for s in data["removed"]}
        except (KeyError, ValueError) as error:
            message = f"{path} is not a saved index"
            raise HnswError(message) from error
        index._vectors = vectors if len(vectors) else index._vectors
        index._ids = ids
        index._removed = removed
        index._slots = {i: s for s, i in enumerate(ids) if s not in removed}
        index._entry = header["entry"]
        index._links = _unflattened(levels, sizes, flat)
        return index

    def _checked(self, vector: Vectors) -> Vectors:
        array = np.asarray(vector, np.float32)
        if array.shape != (self.dim,):
            message = (
                f"expected a vector of {self.dim} numbers, got shape {array.shape}"
            )
            raise HnswError(message)
        return _unit(array)

    def _distances(self, query: Vectors, slots: list[int]) -> list[float]:
        return (1.0 - self._vectors[slots] @ query).tolist()

    def _descend(self, query: Vectors, entry: int, down_to: int) -> int:
        top = len(self._links[entry]) - 1
        for level in range(top, down_to, -1):
            entry = self._search_layer(query, [entry], 1, level)[0][1]
        return entry

    def _search_layer(
        self,
        query: Vectors,
        entries: list[int],
        ef: int,
        level: int,
        skip: frozenset[int] | set[int] = frozenset(),
    ) -> list[tuple[float, int]]:
        """Return up to ``ef`` (distance, slot) nearest ``query``, nearest first.

        Slots in ``skip`` are walked through but never kept, and while fewer
        than ``ef`` are kept the walk goes on, so removed vectors cannot crowd
        out the ones still present.
        """
        visited = set(entries)
        candidates = list(zip(self._distances(query, entries), entries, strict=True))
        heapq.heapify(candidates)
        best = [(-d, s) for d, s in candidates if s not in skip]
        heapq.heapify(best)
        while candidates:
            distance, slot = heapq.heappop(candidates)
            bound = -best[0][0] if best else math.inf
            if distance > bound and (len(best) >= ef or not skip):
                break
            fresh = [n for n in self._links[slot][level] if n not in visited]
            visited.update(fresh)
            for d, n in zip(self._distances(query, fresh), fresh, strict=True):
                if len(best) < ef or d < -best[0][0]:
                    heapq.heappush(candidates, (d, n))
                    if n not in skip:
                        heapq.heappush(best, (-d, n))
                        if len(best) > ef:
                            heapq.heappop(best)
        return sorted((-d, s) for d, s in best)

    def _chosen(self, candidates: list[tuple[float, int]], limit: int) -> list[int]:
        """Keep, nearest first, each candidate nearer the vector than any kept one."""
        kept: list[int] = []
        for distance, slot in candidates:
            if len(kept) == limit:
                break
            if not kept or min(self._distances(self._vectors[slot], kept)) >= distance:
                kept.append(slot)
        return kept

    def _link(self, slot: int, query: Vectors, level: int, entry: int) -> None:
        top = len(self._links[entry]) - 1
        entries = [self._descend(query, entry, level)]
        for layer in range(min(top, level), -1, -1):
            found = self._search_layer(query, entries, self.ef_construction, layer)
            chosen = self._chosen(found, self.m)
            self._links[slot][layer] = chosen
            most = 2 * self.m if layer == 0 else self.m
            for other in chosen:
                links = self._links[other][layer]
                links.append(slot)
                if len(links) > most:
                    near = self._vectors[other]
                    ranked = sorted(
                        zip(self._distances(near, links), links, strict=True)
                    )
                    self._links[other][layer] = self._chosen(ranked, most)
            entries = [s for _, s in found]
        if level > top:
            self._entry = slot
