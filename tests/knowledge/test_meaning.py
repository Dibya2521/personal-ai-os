from pathlib import Path, PurePosixPath

import numpy as np
import pytest

from synthia.knowledge.index import Indexer
from synthia.knowledge.meaning import ChunkVectors
from synthia.knowledge.model import ChunkId, SourceKind
from synthia.knowledge.store import KnowledgeStore
from tests.knowledge.corpus import (
    CAT,
    HEADLESS,
    NOTES,
    PET,
    SERVER,
    SISTER,
    cosine,
    planted,
)
from tests.retrieval.toy import Network, embedder


@pytest.fixture
def store(tmp_path: Path) -> KnowledgeStore:
    return KnowledgeStore(tmp_path / "home" / "knowledge.db")


@pytest.fixture
async def indexed(store: KnowledgeStore, tmp_path: Path) -> KnowledgeStore:
    """The notes indexed and embedded with the toy model: chunks 1, 2 and 3."""
    root = planted(tmp_path / "notes", NOTES)
    await store.put_source(root, "notes", SourceKind.FILES, private=False, excludes=[])
    model = embedder(Network())
    await Indexer(store, HEADLESS, model, model).run()
    return store


async def test_load_holds_every_vector_kept_from_its_embedder(
    indexed: KnowledgeStore,
) -> None:
    vectors = ChunkVectors(indexed, embedder(Network()))

    await vectors.load()

    assert len(vectors) == 3
    near = await vectors.similarities("pet")
    assert near.nearest(3) == pytest.approx(
        {1: cosine(PET, CAT), 2: cosine(PET, SERVER), 3: cosine(PET, SISTER)}
    )


async def test_load_passes_over_vectors_from_another_embedder(
    indexed: KnowledgeStore,
) -> None:
    # Three numbers, not the toy's five: holding it would raise.
    other = np.ones(3, np.float32)
    await indexed.put_vectors("another-model", [(ChunkId(1), other.tobytes())])
    vectors = ChunkVectors(indexed, embedder(Network()))

    await vectors.load()

    assert len(vectors) == 2
    assert set((await vectors.similarities("pet")).nearest(3)) == {2, 3}


async def test_a_new_holder_holds_nothing_until_it_loads(
    indexed: KnowledgeStore,
) -> None:
    vectors = ChunkVectors(indexed, embedder(Network()))

    assert len(vectors) == 0
    assert (await vectors.similarities("pet")).nearest(3) == {}


async def test_hold_adds_and_replaces_and_forget_lets_go(
    store: KnowledgeStore,
) -> None:
    vectors = ChunkVectors(store, embedder(Network()))
    unit = np.eye(5, dtype=np.float32)

    vectors.hold([(ChunkId(7), unit[0]), (ChunkId(8), unit[3])])
    vectors.hold([(ChunkId(8), unit[0])])
    vectors.forget([ChunkId(7), ChunkId(99)])

    assert len(vectors) == 1
    assert (await vectors.similarities("pet")).nearest(5) == {8: 1.0}


async def test_similarities_embed_the_text_once(store: KnowledgeStore) -> None:
    network = Network()
    vectors = ChunkVectors(store, embedder(network))
    network.texts = 0

    await vectors.similarities("my cat")

    assert network.texts == 1


async def test_vectors_kept_for_a_chunk_removed_are_not_loaded(
    indexed: KnowledgeStore,
) -> None:
    stamps = await indexed.stamps((await indexed.sources())[0].id)
    await indexed.remove_documents([stamps[PurePosixPath("cat.txt")].document])
    vectors = ChunkVectors(indexed, embedder(Network()))

    await vectors.load()

    assert len(vectors) == 2
    assert set((await vectors.similarities("pet")).nearest(3)) == {2, 3}
