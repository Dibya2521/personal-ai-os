import wave
from pathlib import Path

import numpy as np
import pytest
from stt_candidates import edits, number_words, read_clip, words
from voice_record import INT16_MAX, save


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Remind me at 4:30.", "remind me at four thirty"),
        ("What is 27 times 48?", "what is twenty seven times forty eight"),
        ("Search the knowledge-based design", "search the knowledge based design"),
        ("What's the weather, SYNTHIA?", "what's the weather synthia"),
        ("'quoted' words", "quoted words"),
        ("the year 2026", "the year 2026"),
    ],
)
def test_words_compares_spoken_forms(text: str, expected: str) -> None:
    assert words(text) == expected.split()


@pytest.mark.parametrize(
    ("n", "spoken"),
    [
        (0, "zero"),
        (19, "nineteen"),
        (40, "forty"),
        (99, "ninety nine"),
        (305, "three hundred five"),
    ],
)
def test_number_words(n: int, spoken: str) -> None:
    assert number_words(n) == spoken.split()


def test_edits_counts_substitutions_insertions_and_deletions() -> None:
    assert edits(["a", "b", "c"], ["a", "b", "c"]) == 0
    assert edits(["a", "b", "c"], ["a", "x", "c"]) == 1
    assert edits(["a", "b", "c"], ["a", "c"]) == 1
    assert edits(["a", "b"], ["a", "b", "c", "d"]) == 2
    assert edits([], ["a"]) == 1


def test_a_saved_clip_reads_back_within_one_step_of_16_bits(tmp_path: Path) -> None:
    samples = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 2.0], np.float32)
    path = tmp_path / "clip.wav"
    save(path, samples)

    with wave.open(str(path), "rb") as clip:
        assert (clip.getnchannels(), clip.getsampwidth(), clip.getframerate()) == (
            1,
            2,
            16_000,
        )
    back = read_clip(path)
    np.testing.assert_allclose(back, np.clip(samples, -1, 1), atol=1 / INT16_MAX)
