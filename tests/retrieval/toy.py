"""A toy embedding model whose vectors can be worked out by hand."""

from dataclasses import replace

import numpy as np

from synthia.retrieval.embed import Feeds, TextEmbedder
from synthia.retrieval.wordpiece import WordPiece
from tests.models.fakes import EMBEDDER
from tests.retrieval.test_wordpiece import SPECIALS

# Words that mean the same thing share a direction; every other word adds a
# little of one common direction, as filler words do to a real embedding.
MEANINGS = {
    "cat": 0,
    "pet": 0,
    "miso": 0,
    "sister": 1,
    "sibling": 1,
    "pune": 2,
    "city": 2,
    "host": 3,
    "database": 3,
    "server": 3,
}
FILLER = (
    "my",
    "what",
    "is",
    "the",
    "of",
    "name",
    "called",
    "lives",
    "in",
    "who",
    "owns",
    "kestrel",
    "osprey",
    "07",
    "02",
    "-",
    "?",
    "which",
    "a",
    "for",
)
VOCABULARY = {t: i for i, t in enumerate((*SPECIALS, *MEANINGS, *FILLER))}
COMMON = 4
TABLE = np.zeros((len(VOCABULARY), 5), np.float32)
for word, meaning in MEANINGS.items():
    TABLE[VOCABULARY[word], meaning] = 1.0
for word in FILLER:
    TABLE[VOCABULARY[word], COMMON] = 0.1
TABLE[VOCABULARY["[UNK]"], COMMON] = 0.1


class Network:
    """The first row of each text sums its words' rows, as [CLS] would."""

    def __init__(self) -> None:
        self.texts = 0

    def __call__(self, feeds: Feeds) -> np.ndarray:
        ids, mask = feeds["input_ids"], feeds["attention_mask"]
        self.texts += len(ids)
        hidden = TABLE[ids]
        hidden[:, 0] = (TABLE[ids] * mask[..., None]).sum(axis=1)
        return hidden


def embedder(network: Network) -> TextEmbedder:
    spec = replace(EMBEDDER, dimensions=TABLE.shape[1], max_tokens=32)
    return TextEmbedder(spec, WordPiece(VOCABULARY), network, token_types=False)
