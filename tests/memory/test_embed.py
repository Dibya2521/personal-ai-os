import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

from synthia.kernel.config import default_home
from synthia.memory.embed import BATCH, SEGMENT_IDS, EmbedError, Feeds, TextEmbedder
from synthia.memory.wordpiece import WordPiece
from synthia.models.catalogue import Embedder, Pooling, find
from synthia.models.install import BYTES_PER_GB, Installer
from tests.memory.test_wordpiece import SPECIALS, made_up_text
from tests.models.fakes import EMBEDDER

WORDS = ("my", "cat", "is", "called", "miso", "budget", "meeting", "friday", "?")
VOCABULARY = {t: i for i, t in enumerate((*SPECIALS, *WORDS))}
TABLE = (
    np.random.default_rng(0).standard_normal((len(VOCABULARY), 4)).astype(np.float32)
)
TEXTS = [
    " ".join(random.Random(n).choices(WORDS, k=n % 7 + 1)) for n in range(2 * BATCH + 3)
]


class Network:
    """Each token's row is its own; the first sums the text, as [CLS] would."""

    def __init__(self) -> None:
        self.fed: list[Feeds] = []

    def __call__(self, feeds: Feeds) -> np.ndarray:
        self.fed.append(feeds)
        ids, mask = feeds["input_ids"], feeds["attention_mask"]
        hidden = TABLE[ids]
        hidden[:, 0] = (TABLE[ids] * mask[..., None]).sum(axis=1)
        return hidden


def expected(text: str, pooling: Pooling) -> np.ndarray:
    """The vector worked out for one text alone: no batch, no padding."""
    rows = TABLE[WordPiece(VOCABULARY).ids(text, EMBEDDER.max_tokens)]
    rows[0] = rows.sum(axis=0)
    pooled = rows[0] if pooling is Pooling.CLS else rows.mean(axis=0)
    return pooled / np.linalg.norm(pooled)


def embedder(
    network: Network, pooling: Pooling = Pooling.CLS, *, token_types: bool = True
) -> TextEmbedder:
    spec = replace(EMBEDDER, pooling=pooling)
    return TextEmbedder(spec, WordPiece(VOCABULARY), network, token_types=token_types)


def installed(name: str) -> tuple[Embedder, Path]:
    """The real model in this machine's SYNTHIA home; skipped where it is absent."""
    spec = find(name)
    assert isinstance(spec, Embedder)
    models = Installer(default_home(), BYTES_PER_GB)
    if not models.installed(spec):
        pytest.skip(f"{name} is not installed (synthia models install {name})")
    return spec, models.path_of(spec)


@pytest.mark.parametrize("pooling", list(Pooling))
def test_each_vector_is_its_text_alone_whatever_the_batch(pooling: Pooling) -> None:
    vectors = embedder(Network(), pooling).vectors(TEXTS)

    assert vectors.shape == (len(TEXTS), 4)
    for text, vector in zip(TEXTS, vectors, strict=True):
        np.testing.assert_allclose(vector, expected(text, pooling), atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)


def test_texts_run_in_batches_shortest_first() -> None:
    network = Network()
    embedding = embedder(network)
    network.fed.clear()

    embedding.vectors(TEXTS)

    widths = [feeds["input_ids"].shape[1] for feeds in network.fed]
    assert [len(feeds["input_ids"]) for feeds in network.fed] == [BATCH, BATCH, 3]
    assert widths == sorted(widths)
    assert all(SEGMENT_IDS in feeds for feeds in network.fed)


def test_token_types_are_fed_only_to_a_network_that_takes_them() -> None:
    network = Network()
    embedder(network, token_types=False).vectors(TEXTS[:2])
    assert all(SEGMENT_IDS not in feeds for feeds in network.fed)


def test_no_texts_give_no_vectors() -> None:
    assert embedder(Network()).vectors([]).shape == (0, 4)


async def test_embedding_off_the_event_loop_gives_the_same_vectors() -> None:
    embedding = embedder(Network())
    np.testing.assert_array_equal(
        await embedding.embed(TEXTS), embedding.vectors(TEXTS)
    )


def test_a_network_of_the_wrong_width_is_refused() -> None:
    spec = replace(EMBEDDER, dimensions=5)
    with pytest.raises(EmbedError, match="gives 4 numbers, not 5"):
        TextEmbedder(spec, WordPiece(VOCABULARY), Network(), token_types=True)


def test_a_model_that_is_not_installed_cannot_load(tmp_path: Path) -> None:
    with pytest.raises(EmbedError, match="tiny-embedder"):
        TextEmbedder.load(tmp_path, EMBEDDER)


@pytest.mark.parametrize("name", ["bge-small-en-v1.5", "all-minilm-l6-v2-int8"])
def test_an_installed_model_puts_a_question_nearest_its_answer(name: str) -> None:
    spec, path = installed(name)
    embedding = TextEmbedder.load(path, spec)
    facts = embedding.vectors(
        [
            "My cat is called Miso and she hates the vacuum cleaner.",
            "The quarterly budget review is on Friday at ten.",
        ]
    )
    questions = embedding.vectors(
        ["What is the name of my cat?", "When is the budget meeting?"]
    )

    similarity = questions @ facts.T
    assert similarity.argmax(axis=1).tolist() == [0, 1]


@pytest.mark.parametrize("name", ["bge-small-en-v1.5", "all-minilm-l6-v2-int8"])
def test_an_installed_tokenizer_matches_the_tokenizers_library(name: str) -> None:
    spec, path = installed(name)
    ours = WordPiece.from_file(path / spec.tokenizer.name)
    theirs = Tokenizer.from_file(str(path / spec.tokenizer.name))
    # MiniLM's file pads every text to 128; ours never pads. No text here
    # reaches either model's token limit, so truncation is not in play.
    theirs.no_padding()
    rng = random.Random(4)
    texts = [made_up_text(rng) for _ in range(2000)]

    assert [ours.ids(t, spec.max_tokens) for t in texts] == [
        theirs.encode(t).ids for t in texts
    ]
