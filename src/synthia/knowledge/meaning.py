"""The chunks' vectors, held in memory for search by meaning.

A search compares the query with every vector held, in one matrix product:
on a laptop CPU about a millisecond at 10,000 vectors of 384 numbers and
about seven at 50,000, less than embedding the query itself (3 to 12), and
exactly right (see :mod:`synthia.retrieval.vectors`). Holding the vectors
spares every search reading and decoding them all from knowledge.db.

:class:`ChunkVectors` is the only code that changes the held copy: it loads
what the store keeps when knowledge opens, holds what the indexer embeds,
and forgets what the indexer or a removal deletes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

import numpy as np

from synthia.retrieval.vectors import VectorSet

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from synthia.knowledge.model import ChunkId
    from synthia.knowledge.store import KnowledgeStore
    from synthia.retrieval.embed import TextEmbedder
    from synthia.retrieval.vectors import Similarities

UNDATED: Final = datetime(1970, 1, 1, tzinfo=UTC)
"""Every chunk vector's time: chunks have none, and searches name no range."""


class ChunkVectors:
    """Each chunk's vector from one embedder, held for search by meaning."""

    def __init__(self, store: KnowledgeStore, embedder: TextEmbedder) -> None:
        """Hold nothing yet; :meth:`load` reads what ``store`` keeps."""
        self._store = store
        self._embedder = embedder
        self._held = VectorSet(embedder.spec.dimensions)

    def __len__(self) -> int:
        """Return how many chunk vectors are held."""
        return len(self._held)

    async def load(self) -> None:
        """Hold every vector the store keeps from this embedder."""
        for chunk, vector in await self._store.vectors(self._embedder.spec.id):
            self._held.add(chunk, UNDATED, np.frombuffer(vector, np.float32))

    def hold(self, pairs: Sequence[tuple[ChunkId, NDArray[np.float32]]]) -> None:
        """Hold each chunk's vector, replacing any held before."""
        for chunk, vector in pairs:
            self._held.add(chunk, UNDATED, vector)

    def forget(self, chunks: Sequence[ChunkId]) -> None:
        """Stop finding ``chunks`` by meaning; one not held is passed over."""
        for chunk in chunks:
            self._held.remove(chunk)

    async def similarities(self, text: str) -> Similarities:
        """Return each held chunk's cosine similarity to ``text``.

        ``text`` is embedded off the event loop.
        """
        (vector,) = await self._embedder.embed([text])
        return self._held.similarities(vector)
