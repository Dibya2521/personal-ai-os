from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from synthia.kernel.config import default_home
from synthia.models.catalogue import RERANKERS, Reranker
from synthia.models.install import BYTES_PER_GB, Installer
from synthia.retrieval.embed import Feeds
from synthia.retrieval.rerank import CrossEncoder, RerankError
from synthia.retrieval.wordpiece import WordPiece
from tests.retrieval.toy import VOCABULARY, Judge, reranker


def test_scores_come_back_in_the_order_the_passages_were_given() -> None:
    judge = Judge("kestrel")
    passages = ["kestrel kestrel kestrel", "the cat", "kestrel", "kestrel kestrel"]

    assert reranker(judge).scores("which kestrel", passages) == [3.0, 0.0, 1.0, 2.0]


def test_pairs_run_in_batches_of_the_catalogued_size_shortest_first() -> None:
    judge = Judge("kestrel")
    encoder = reranker(judge, batch=2)
    judge.batches.clear()

    encoder.scores("q", ["a a a a", "a", "a a", "a a a"])

    assert judge.batches == [(2, 6), (2, 8)]


def test_the_question_is_type_0_and_the_passage_type_1() -> None:
    seen: list[Feeds] = []

    def network(feeds: Feeds) -> np.ndarray:
        seen.append(feeds)
        return np.zeros((len(feeds["input_ids"]), 1), np.float32)

    spec = replace(RERANKERS[0], max_tokens=32, batch=1)
    CrossEncoder(spec, WordPiece(VOCABULARY), network).scores("cat", ["pet city"])

    assert seen[-1]["token_type_ids"].tolist() == [[0, 0, 0, 1, 1, 1]]
    assert seen[-1]["attention_mask"].tolist() == [[1] * 6]


def test_a_network_giving_more_than_one_number_per_pair_is_refused() -> None:
    def network(feeds: Feeds) -> np.ndarray:
        return np.zeros((len(feeds["input_ids"]), 2), np.float32)

    spec = replace(RERANKERS[0], max_tokens=32)
    with pytest.raises(RerankError, match=r"gives \(1, 2\) numbers per pair"):
        CrossEncoder(spec, WordPiece(VOCABULARY), network)


def test_loading_from_a_folder_without_the_files_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RerankError, match="cannot load the reranker ms-marco"):
        CrossEncoder.load(tmp_path, RERANKERS[0])


async def test_rank_scores_off_the_event_loop() -> None:
    assert await reranker(Judge("cat")).rank("q", ["cat", "dog"]) == [1.0, 0.0]


@pytest.mark.parametrize("spec", RERANKERS, ids=lambda s: s.id)
def test_the_installed_reranker_puts_the_answer_first(spec: Reranker) -> None:
    models = Installer(default_home(), BYTES_PER_GB)
    if not models.installed(spec):
        pytest.skip(f"{spec.id} is not installed (synthia models install {spec.id})")
    encoder = CrossEncoder.load(models.path_of(spec), spec)

    berlin, museum = encoder.scores(
        "How many people live in Berlin?",
        [
            "Berlin has a population of 3,520,031 registered inhabitants.",
            "New York City is famous for the Metropolitan Museum of Art.",
        ],
    )

    assert berlin > 0 > museum


def test_the_full_precision_reranker_scores_the_same_in_a_batch_as_alone() -> None:
    spec = RERANKERS[1]
    models = Installer(default_home(), BYTES_PER_GB)
    if not models.installed(spec):
        pytest.skip(f"{spec.id} is not installed (synthia models install {spec.id})")
    encoder = CrossEncoder.load(models.path_of(spec), spec)
    words = [
        "the",
        "daily",
        "budget",
        "is",
        "claimed",
        "before",
        "a",
        "remote",
        "request",
        "is",
        "sent",
    ]
    passages = [" ".join(words[: 1 + i % len(words)] * (1 + i)) for i in range(20)]

    together = encoder.scores("where is the budget claimed?", passages)
    alone = [encoder.scores("where is the budget claimed?", [p])[0] for p in passages]

    assert spec.batch > 1
    assert together == pytest.approx(alone, abs=1e-5)
