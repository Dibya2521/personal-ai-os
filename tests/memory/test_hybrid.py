import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from synthia.memory.embed import Feeds, TextEmbedder
from synthia.memory.hybrid import CANDIDATES, Recall, Recalled, fused
from synthia.memory.store import MemoryStore, Turn, WordScores
from synthia.memory.wordpiece import WordPiece
from tests.memory.test_embed import installed
from tests.memory.test_wordpiece import SPECIALS
from tests.models.fakes import EMBEDDER

MONDAY = datetime(2026, 9, 28, 9, 0, tzinfo=UTC)
DAY = timedelta(days=1)
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


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory.db")


async def remember(
    store: MemoryStore, *questions: str, at: datetime = MONDAY
) -> list[int]:
    chat = await store.begin_conversation("SYNTHIA", at)
    return [
        await store.add_turn(chat, Turn(q, "Noted.", at, "SYNTHIA", "local", "m"))
        for q in questions
    ]


async def recall_over(store: MemoryStore, network: Network, **options: float) -> Recall:
    recall = Recall(store, embedder(network), **options)
    await recall.backfill()
    return recall


def asked(found: list[Recalled]) -> list[str]:
    return [r.turn.question for r in found]


def test_fused_adds_the_scaled_word_score_to_the_similarity() -> None:
    words = WordScores({2: 3.0, 3: 1.0}, turns=10)

    scores = fused({1: 0.5, 2: 0.1}, words, 0.25)

    # The scale for 10 turns is ln(21) * 2.2 = 6.697949362991531.
    assert scores == pytest.approx(
        {1: 0.5, 2: 0.21197457002944922, 3: 0.037324856676483074}, rel=1e-12
    )


async def test_meaning_finds_a_turn_that_shares_no_word_with_the_question(
    store: MemoryStore,
) -> None:
    await remember(store, "my sister lives in pune", "my cat is called miso")
    recall = await recall_over(store, Network())

    found = await recall.find("what is the name of the pet", limit=2)

    assert asked(found) == ["my cat is called miso", "my sister lives in pune"]


async def test_words_tell_apart_turns_whose_meaning_is_the_same(
    store: MemoryStore,
) -> None:
    await remember(store, "the host kestrel - 07", "the host osprey - 02")
    network = Network()
    meaning_only = await recall_over(store, network, word_weight=0.0)
    both = Recall(store, embedder(network))
    await both.load()

    blurred = await meaning_only.find("who owns kestrel - 07 ?", limit=2)
    told_apart = await both.find("who owns kestrel - 07 ?", limit=2)

    assert asked(blurred)[0] == "the host osprey - 02"
    assert asked(told_apart)[0] == "the host kestrel - 07"


async def test_only_turns_inside_the_dates_come_back(store: MemoryStore) -> None:
    await remember(store, "my cat is called miso")
    await remember(store, "the pet cat again", at=MONDAY + 3 * DAY)
    recall = await recall_over(store, Network())

    found = await recall.find(
        "cat", since=MONDAY + DAY, until=MONDAY + 4 * DAY, limit=5
    )

    assert asked(found) == ["the pet cat again"]


async def test_a_question_with_no_words_gets_the_newest_turns(
    store: MemoryStore,
) -> None:
    await remember(store, "older")
    await remember(store, "newer", at=MONDAY + DAY)
    recall = await recall_over(store, Network())

    found = await recall.find(" ?! ", limit=5)

    assert [(r.turn.question, r.score) for r in found] == [
        ("newer", 0.0),
        ("older", 0.0),
    ]
    assert await recall.find(" ?! ", limit=5, at_least=0.5) == []


async def test_only_turns_scoring_at_least_the_floor_come_back(
    store: MemoryStore,
) -> None:
    await remember(store, "my sister lives in pune", "my cat is called miso")
    recall = await recall_over(store, Network())

    everything = await recall.find("what is the name of the pet", limit=5)
    close = await recall.find("what is the name of the pet", limit=5, at_least=0.5)

    assert len(everything) == 2
    assert asked(close) == ["my cat is called miso"]
    assert close[0].score >= 0.5


async def test_kept_vectors_are_loaded_not_embedded_again(store: MemoryStore) -> None:
    await remember(store, "my cat is called miso", "my sister lives in pune")
    first = Network()
    assert await (await recall_over(store, first)).backfill() == 0
    assert first.texts == 2 + 1  # two turns, and the width check at start
    again = Network()
    recall = Recall(store, embedder(again))
    await recall.load()

    found = await recall.find("pet", limit=1)

    assert asked(found) == ["my cat is called miso"]
    assert again.texts == 1 + 1  # the width check and the question


async def test_a_forgotten_turn_is_not_found_by_meaning_or_kept_again(
    store: MemoryStore,
) -> None:
    cat, _ = await remember(store, "my cat is called miso", "my sister lives in pune")
    recall = await recall_over(store, Network())
    waiting = await store.unembedded("other model", 1)

    await store.forget_turn(cat)
    recall.forget(cat)
    await recall.keep(*waiting)

    assert asked(await recall.find("pet", limit=5)) == ["my sister lives in pune"]
    assert await store.vectors(EMBEDDER.id) != []
    assert cat not in {v.turn for v in await store.vectors(EMBEDDER.id)}


async def test_without_an_embedding_model_turns_are_found_by_words(
    store: MemoryStore,
) -> None:
    await remember(store, "my cat is called miso", "my sister lives in pune")
    recall = Recall(store)
    await recall.load()
    await recall.keep(*await store.unembedded(EMBEDDER.id, 5))
    recall.forget(1)

    assert await recall.backfill() == 0
    assert await store.vectors(EMBEDDER.id) == []
    assert asked(await recall.find("pune", limit=5)) == ["my sister lives in pune"]
    assert asked(await recall.find("pet", limit=5)) == []


async def test_more_turns_than_candidates_still_rank_the_closest_first(
    store: MemoryStore,
) -> None:
    await remember(store, *(["my sister lives in pune"] * CANDIDATES), "miso the cat")
    recall = await recall_over(store, Network())

    assert asked(await recall.find("pet", limit=1)) == ["miso the cat"]


async def test_the_recall_set_is_found_better_by_both_than_by_meaning(
    tmp_path: Path,
) -> None:
    spec, directory = installed("all-minilm-l6-v2-int8")
    data = json.loads(
        (Path(__file__).parents[2] / "benchmarks/data/recall_set.json").read_text(
            encoding="utf-8"
        )
    )
    store = MemoryStore(tmp_path / "memory.db")
    pairs = [*data["pairs"], *data["named"]]
    ids = await remember(
        store, *(p["said"] for p in pairs), *data["other"], *data["near"]
    )
    real = TextEmbedder.load(directory, spec)
    meaning = Recall(store, real, word_weight=0.0)
    await meaning.backfill()
    both = Recall(store, real)
    await both.load()

    async def found(recall: Recall) -> tuple[int, int]:
        tops = [await recall.find(p["asked"], limit=5) for p in pairs]
        first = sum(top[0].turn.id == ids[n] for n, top in enumerate(tops))
        five = sum(ids[n] in [r.turn.id for r in top] for n, top in enumerate(tops))
        return first, five

    assert (await found(meaning), await found(both)) == ((38, 57), (39, 57))
