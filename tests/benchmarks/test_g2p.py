import pytest
from g2p import (
    DEV,
    FIT,
    HELD_OUT,
    WindowRules,
    edits,
    fold,
    score,
    split,
    unstressed,
)

from synthia.voice.guess import AlignedWord, Alignment
from synthia.voice.phonemes import Pronunciation, Pronunciations

HELD_OUT_WORDS = ("abandon", "accent", "air", "angle", "april", "arm")
DEV_WORDS = ("absent", "also", "among", "ball")
FIT_WORDS = ("about", "above", "across", "act", "add", "after", "again", "age")


def test_fold_splits_words_the_same_way_every_run() -> None:
    assert {fold(word) for word in HELD_OUT_WORDS} == {HELD_OUT}
    assert {fold(word) for word in DEV_WORDS} == {DEV}
    assert {fold(word) for word in FIT_WORDS} == {FIT}


def test_split_keeps_words_with_the_same_letters_in_one_fold() -> None:
    dictionary = (
        "aim's EY1 M Z\naims EY1 M Z\nit's IH1 T S\nits IH1 T S\nabout AH0 B AW1 T\n"
    )

    folds = split(Pronunciations.parse(dictionary))

    letters = [word for words in folds.values() for word, _ in words]
    assert sorted(letters) == ["about", "aims", "its"]
    assert ("aims", ("EY1", "M", "Z")) in folds[fold("aims")]


@pytest.fixture
def rules() -> WindowRules:
    words = (
        AlignedWord("cat", (("K",), ("AE1",), ("T",))),
        AlignedWord("cot", (("K",), ("AA1",), ("T",))),
        AlignedWord("cut", (("K",), ("AH1",), ("T",))),
        AlignedWord("cell", (("S",), ("EH1",), ("L",), ())),
        AlignedWord("cent", (("S",), ("EH1",), ("N",), ("T",))),
    )
    return WindowRules.fit(Alignment(words, 0, ()))


@pytest.mark.parametrize(
    ("word", "phones"),
    [
        ("cat", ("K", "AE1", "T")),
        ("cut", ("K", "AH1", "T")),
        ("cell", ("S", "EH1", "L")),
        ("cent", ("S", "EH1", "N", "T")),
        ("cen", ("S", "EH1", "N")),
        ("tac", ("T", "AE1", "K")),
    ],
)
def test_window_rules_take_the_widest_window_seen(
    rules: WindowRules, word: str, phones: Pronunciation
) -> None:
    assert rules.guess(word) == phones


def test_window_rules_break_a_tie_for_the_shorter_output(rules: WindowRules) -> None:
    assert rules.guess("l") == ()


def test_window_rules_keep_only_windows_that_change_the_answer(
    rules: WindowRules,
) -> None:
    assert len(rules.rules) == 10


def test_edits_counts_substitutions_insertions_and_deletions() -> None:
    assert edits(("K", "AE1", "T"), ("K", "AE1", "T")) == 0
    assert edits(("K", "AE1", "T"), ("K", "AH1", "T", "S")) == 2
    assert edits((), ("K",)) == 1


def test_unstressed_drops_only_stress_digits() -> None:
    assert unstressed(("HH", "AH0", "L", "OW1", "ER2")) == ("HH", "AH", "L", "OW", "ER")


def test_score_compares_each_guess_with_the_dictionary() -> None:
    guesses = {"ab": ("AH1", "B"), "cd": ("K", "D")}

    result = score(guesses.__getitem__, [("ab", ("AH0", "B")), ("cd", ("K", "D"))])

    assert result.phone_error_rate == pytest.approx(0.25)
    assert result.unstressed_error_rate == 0
    assert result.word_accuracy == pytest.approx(0.5)
    assert result.ms_per_word >= 0
