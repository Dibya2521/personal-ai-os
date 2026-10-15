"""One-line files whose vectors and word scores can be worked out by hand.

The toy embedder (tests/retrieval/toy.py) gives a text the sum of its
words' rows: 1 on a meaning's axis (0 cat, pet, miso; 1 sister; 2 pune,
city; 3 database, server, host) and 0.1 on the common axis 4 for every
other word, an unknown one included. Without headers a chunk's text is its
line alone, so:

- cat.txt "my cat is miso" is (2, 0, 0, 0, 0.2), chunk 1.
- server.txt "the database server host kestrel" is (0, 0, 0, 3, 0.2), chunk 2.
- sister.txt "my sister lives in pune city" is (0, 1, 2, 0, 0.3), chunk 3.

"kestrel" is the one word only one file holds: BM25 gives it
ln((3 - 1 + 0.5) / (1 + 0.5)) times a length factor of exactly 1, since
server.txt's 5 words are the average of 4, 5 and 6.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from synthia.knowledge.chunk import Chunking
from synthia.knowledge.index import Indexer
from synthia.knowledge.meaning import ChunkVectors
from synthia.knowledge.retrieve import Found
from synthia.knowledge.service import Knowledge
from synthia.knowledge.store import KnowledgeStore
from synthia.retrieval.bm25 import K1
from tests.retrieval.toy import Network, embedder

HEADLESS = Chunking(32, header=False)
NOTES = {
    "cat.txt": "my cat is miso\n",
    "server.txt": "the database server host kestrel\n",
    "sister.txt": "my sister lives in pune city\n",
}
CAT = (2.0, 0.0, 0.0, 0.0, 0.2)
SERVER = (0.0, 0.0, 0.0, 3.0, 0.2)
SISTER = (0.0, 1.0, 2.0, 0.0, 0.3)
PET = (1.0, 0.0, 0.0, 0.0, 0.0)
KESTREL = (0.0, 0.0, 0.0, 0.0, 0.1)
PET_KESTREL = (1.0, 0.0, 0.0, 0.0, 0.1)
KESTREL_BM25 = math.log(2.5 / 1.5)
WORD_SCALE = math.log(2 * 3 + 1) * (K1 + 1)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / math.hypot(*a) / math.hypot(*b)


def planted(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode())
    return root


def citations(found: Found) -> list[str]:
    return [passage.citation() for passage in found.passages]


async def opened(db: Path, network: Network | None = None) -> Knowledge:
    return await Knowledge.open(db, embedder(network or Network()), chunking=HEADLESS)


@dataclass
class Wired:
    """A knowledge built by hand, so a test can see its store and its vectors."""

    knowledge: Knowledge
    store: KnowledgeStore
    vectors: ChunkVectors


def wired(store: KnowledgeStore) -> Wired:
    model = embedder(Network())
    vectors = ChunkVectors(store, model)
    indexer = Indexer(store, HEADLESS, model, model)
    return Wired(Knowledge(store, indexer, vectors), store, vectors)
