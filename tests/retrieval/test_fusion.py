import pytest

from synthia.retrieval.fusion import WordScores, fused


def test_fused_adds_the_scaled_word_score_to_the_similarity() -> None:
    words = WordScores({2: 3.0, 3: 1.0}, indexed=10)

    scores = fused({1: 0.5, 2: 0.1}, words, 0.25)

    # The scale for 10 indexed texts is ln(21) * 2.2 = 6.697949362991531.
    assert scores == pytest.approx(
        {1: 0.5, 2: 0.21197457002944922, 3: 0.037324856676483074}, rel=1e-12
    )
