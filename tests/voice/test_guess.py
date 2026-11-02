import json
import math
from collections.abc import Callable
from itertools import chain, pairwise, product
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from synthia.kernel.config import default_home
from synthia.models.catalogue import CMUDICT
from synthia.models.install import BYTES_PER_GB, Installer
from synthia.voice.guess import (
    FORMAT_VERSION,
    ORDER,
    Alignment,
    GraphoneModel,
    GuessError,
    align,
    cache_path,
    load_or_train,
    training_words,
)
from synthia.voice.phonemes import (
    PHONEMES,
    Phonemizer,
    Pronunciation,
    Pronunciations,
    spell,
    to_misaki,
)

SMALL = """\
box B AA1 K S
cat K AE1 T
cell S EH1 L
cent S EH1 N T
chop CH AA1 P
cop K AA1 P
cot K AA1 T
cut K AH1 T
fox F AA1 K S
go G OW1
graph G R AE1 F
hat HH AE1 T
it's IH1 T S
its IH1 T S
no N OW1
on AA1 N
phone F OW1 N
photo F OW1 T OW2
pin P IH1 N
shop SH AA1 P
so S OW1
thin TH IH1 N
tone T OW1 N
x EH1 K S
x-ray EH1 K S R EY1
"""


@pytest.fixture(scope="module")
def small() -> Pronunciations:
    return Pronunciations.parse(SMALL)


@pytest.fixture(scope="module")
def alignment(small: Pronunciations) -> Alignment:
    return align(training_words(small))


@pytest.fixture(scope="module")
def model(alignment: Alignment) -> GraphoneModel:
    return GraphoneModel.fit(alignment, order=3)


def test_training_words_keeps_plain_words_without_their_apostrophe(
    small: Pronunciations,
) -> None:
    pairs = training_words(small)
    words = dict(pairs)

    assert words["its"] == ("IH1", "T", "S")
    assert words["x"] == ("EH1", "K", "S")
    assert "x-ray" not in words
    assert "xray" not in words
    assert [letters for letters, _ in pairs].count("its") == 1
    assert len(pairs) == len(small) - 2


def test_align_gives_each_letter_the_phones_it_stands_for(
    alignment: Alignment,
) -> None:
    aligned = dict(alignment.words)

    assert aligned["phone"] == (("F",), (), ("OW1",), ("N",), ())
    assert aligned["box"] == (("B",), ("AA1",), ("K", "S"))
    assert aligned["graph"] == (("G",), ("R",), ("AE1",), ("F",), ())


def test_align_leaves_out_a_word_needing_three_phones_for_one_letter(
    small: Pronunciations, alignment: Alignment
) -> None:
    assert alignment.unaligned == 1
    assert "x" not in dict(alignment.words)
    assert len(alignment.words) == len(training_words(small)) - 1


def test_align_stops_at_the_first_pass_that_changes_the_fit_under_a_thousandth(
    alignment: Alignment,
) -> None:
    changes = [
        abs(now - before) / abs(before)
        for before, now in pairwise(alignment.log_likelihoods)
    ]

    assert 1 < len(alignment.log_likelihoods) <= 10
    assert changes[-1] <= 1e-3
    assert all(change > 1e-3 for change in changes[:-1])


def test_align_of_no_words_aligns_nothing() -> None:
    nothing = align([])

    assert nothing.words == ()
    assert nothing.unaligned == 0


@pytest.mark.parametrize(
    ("order", "empty", "message"),
    [
        (0, False, r"order of at least 1"),
        (3, True, r"order of at least 1 and a word"),
        (40, False, r"too many for n-grams of order=40"),
    ],
)
def test_fit_refuses_an_order_it_cannot_number_or_no_words(
    alignment: Alignment, order: int, *, empty: bool, message: str
) -> None:
    words = Alignment((), 0, ()) if empty else alignment

    with pytest.raises(ValueError, match=message):
        GraphoneModel.fit(words, order=order)


@pytest.mark.parametrize(
    ("word", "phones"),
    [
        ("phone", ("F", "OW1", "N")),
        ("box", ("B", "AA1", "K", "S")),
        ("cell", ("S", "EH1", "L")),
        ("graph", ("G", "R", "AE1", "F")),
        ("chop", ("CH", "AA1", "P")),
    ],
)
def test_guess_says_the_words_it_learned_from(
    model: GraphoneModel, word: str, phones: Pronunciation
) -> None:
    assert model.guess(word) == phones


def test_guess_reads_only_the_lower_case_letters(model: GraphoneModel) -> None:
    assert model.guess("Pho-ne!") == model.guess("phone")
    assert model.guess("IT'S") == ("IH1", "T", "S")


def test_guess_spells_a_word_with_a_letter_it_never_saw(model: GraphoneModel) -> None:
    assert model.guess("zap") == spell("zap")


def test_guess_refuses_a_word_without_letters(model: GraphoneModel) -> None:
    with pytest.raises(ValueError, match=r"no letter to spell in '123'"):
        model.guess("123")


def test_guess_refuses_a_beam_that_keeps_nothing(model: GraphoneModel) -> None:
    with pytest.raises(ValueError, match=r"beam=0"):
        model.guess("cat", beam=0)


def test_a_wider_beam_finds_what_a_greedy_search_misses(
    alignment: Alignment,
) -> None:
    bigrams = GraphoneModel.fit(alignment, order=2)

    assert bigrams.guess("cell", beam=1)[0] == "K"
    assert bigrams.guess("cell", beam=2) == ("S", "EH1", "L")


type Graphones = tuple[tuple[str, Pronunciation], ...]
EDGE = ("", ())


def reference_log_probability(
    alignment: Alignment, order: int
) -> Callable[[Graphones], float]:
    """Witten-Bell interpolation written from its formula with plain counters."""
    counts: dict[tuple[object, ...], dict[object, int]] = {}
    for word in alignment.words:
        padded = [EDGE] * (order - 1) + list(
            zip(word.letters, word.outputs, strict=True)
        )
        for i, graphone in enumerate([*padded[order - 1 :], EDGE], start=order - 1):
            for length in range(order):
                history = tuple(padded[i - length : i])
                after = counts.setdefault(history, {})
                after[graphone] = after.get(graphone, 0) + 1

    def probability(history: tuple[object, ...], graphone: object) -> float:
        after = counts.get(history, {})
        total = sum(after.values())
        if not history:
            return after.get(graphone, 0) / total
        shorter = probability(history[1:], graphone)
        if not after:
            return shorter
        types = len(after)
        return (after.get(graphone, 0) + types * shorter) / (total + types)

    def log_probability(graphones: Graphones) -> float:
        padded = [EDGE] * (order - 1) + list(graphones)
        return sum(
            math.log(probability(tuple(padded[i - order + 1 : i]), graphone))
            for i, graphone in enumerate([*graphones, EDGE], start=order - 1)
        )

    return log_probability


@pytest.mark.parametrize("order", [2, 3, 4, 5])
@pytest.mark.parametrize(
    "word",
    [
        "box",
        "phot",
        "shin",
        "cox",
        "tot",
        "chat",
        "thop",
        "ap",
        "he",
        "et",
        "ex",
        "photon",
        "chops",
        "thinx",
        "graphone",
    ],
)
def test_guess_finds_the_likeliest_reading_of_an_exact_n_gram_model(
    alignment: Alignment, order: int, word: str
) -> None:
    model = GraphoneModel.fit(alignment, order=order)
    log_probability = reference_log_probability(alignment, order)
    choices = [
        sorted(
            {
                (letter, output)
                for aligned in alignment.words
                for letter, output in zip(aligned.letters, aligned.outputs, strict=True)
                if letter == wanted
            }
        )
        for wanted in word
    ]
    readings = [
        (
            log_probability(graphones),
            tuple(chain.from_iterable(p for _, p in graphones)),
        )
        for graphones in product(*choices)
    ]
    best = max(score for score, _ in readings)
    guessed = max(
        score for score, phones in readings if phones == model.guess(word, beam=1000)
    )

    assert guessed == pytest.approx(best)


def test_the_phonemizer_says_an_unknown_word_through_the_guess(
    model: GraphoneModel,
) -> None:
    phonemizer = Phonemizer(Pronunciations.parse("hello HH AH0 L OW1\n"), model.guess)

    assert phonemizer.phonemes("hello chops") == (
        f"həlˈO {to_misaki(model.guess('chops'))}"  # noqa: RUF001
    )


def test_save_then_load_gives_the_same_model(
    model: GraphoneModel, tmp_path: Path
) -> None:
    path = tmp_path / "model.npz"
    model.save(path)

    loaded = GraphoneModel.load(path)

    assert loaded.order == model.order
    assert loaded.ngrams == model.ngrams
    for word in ("phone", "box", "cell", "tox", "photon", "shot"):
        assert loaded.guess(word) == model.guess(word)
    assert list(tmp_path.iterdir()) == [path]


type Arrays = dict[str, NDArray[np.generic]]


def _rewrite(path: Path, change: Callable[[Arrays], object]) -> None:
    with np.load(path) as saved:
        arrays: Arrays = dict(saved)
    change(arrays)
    np.savez(path, allow_pickle=False, **arrays)


def _header(arrays: Arrays, key: str, value: object) -> None:
    header = json.loads(str(arrays["header"]))
    header[key] = value
    arrays["header"] = np.array(json.dumps(header))


def _missing(path: Path) -> None:
    path.unlink()


def _garbage(path: Path) -> None:
    path.write_bytes(b"not a model")


def _truncated(path: Path) -> None:
    path.write_bytes(path.read_bytes()[:-100])


def _without_a_level(path: Path) -> None:
    _rewrite(path, lambda arrays: arrays.pop("keys_2"))


def _another_version(path: Path) -> None:
    _rewrite(path, lambda arrays: _header(arrays, "version", FORMAT_VERSION + 1))


def _a_phone_misaki_lacks(path: Path) -> None:
    _rewrite(path, lambda arrays: _header(arrays, "graphones", [["a", ["AH"]]]))


def _a_letter_outside_a_to_z(path: Path) -> None:
    _rewrite(path, lambda arrays: _header(arrays, "graphones", [["é", ["EY1"]]]))


def _float_keys(path: Path) -> None:
    def change(arrays: Arrays) -> None:
        arrays["keys_1"] = arrays["keys_1"].astype(np.float64)

    _rewrite(path, change)


def _unsorted_histories(path: Path) -> None:
    def change(arrays: Arrays) -> None:
        arrays["histories_2"] = arrays["histories_2"][::-1].copy()

    _rewrite(path, change)


def _an_order_too_high_to_number(path: Path) -> None:
    def change(arrays: Arrays) -> None:
        for length in range(3, 64):
            for name in ("keys", "log_probabilities", "histories", "backoffs"):
                arrays[f"{name}_{length}"] = arrays[f"{name}_2"]
        _header(arrays, "order", 64)

    _rewrite(path, change)


def _a_nan_unigram(path: Path) -> None:
    def change(arrays: Arrays) -> None:
        arrays["unigram"] = np.full_like(arrays["unigram"], np.nan)

    _rewrite(path, change)


def _a_positive_log_probability(path: Path) -> None:
    def change(arrays: Arrays) -> None:
        arrays["log_probabilities_1"] = np.abs(arrays["log_probabilities_1"])

    _rewrite(path, change)


def _a_key_past_its_level(path: Path) -> None:
    def change(arrays: Arrays) -> None:
        keys = arrays["keys_1"].copy()
        keys[-1] = np.iinfo(np.int64).max
        arrays["keys_1"] = keys

    _rewrite(path, change)


@pytest.mark.parametrize(
    "damage",
    [
        _missing,
        _garbage,
        _truncated,
        _without_a_level,
        _another_version,
        _a_phone_misaki_lacks,
        _a_letter_outside_a_to_z,
        _float_keys,
        _unsorted_histories,
        _an_order_too_high_to_number,
        _a_nan_unigram,
        _a_positive_log_probability,
        _a_key_past_its_level,
    ],
)
def test_load_refuses_a_missing_damaged_or_foreign_file(
    model: GraphoneModel, tmp_path: Path, damage: Callable[[Path], None]
) -> None:
    path = tmp_path / "model.npz"
    model.save(path)
    damage(path)

    with pytest.raises(GuessError, match=r"model\.npz"):
        GraphoneModel.load(path)


def test_cache_path_names_the_dictionary_and_the_format(tmp_path: Path) -> None:
    name = cache_path(tmp_path, CMUDICT).name

    assert CMUDICT.dictionary.sha256 in name
    assert name.endswith(f"-{FORMAT_VERSION}.npz")


def test_load_or_train_trains_and_saves_once(
    small: Pronunciations, tmp_path: Path
) -> None:
    trained = load_or_train(tmp_path, CMUDICT, small)
    loaded = load_or_train(tmp_path, CMUDICT, Pronunciations({}))

    assert trained.order == loaded.order == ORDER
    assert loaded.guess("phone") == trained.guess("phone") == ("F", "OW1", "N")
    assert list(tmp_path.iterdir()) == [cache_path(tmp_path, CMUDICT)]


def test_load_or_train_retrains_a_damaged_cache(
    small: Pronunciations, tmp_path: Path
) -> None:
    path = cache_path(tmp_path, CMUDICT)
    path.write_bytes(b"damaged")

    model = load_or_train(tmp_path, CMUDICT, small)

    assert model.guess("box") == ("B", "AA1", "K", "S")
    assert GraphoneModel.load(path).order == ORDER


def test_load_or_train_retrains_a_cache_of_another_order(
    alignment: Alignment, small: Pronunciations, tmp_path: Path
) -> None:
    path = cache_path(tmp_path, CMUDICT)
    GraphoneModel.fit(alignment, order=2).save(path)

    assert load_or_train(tmp_path, CMUDICT, small).order == ORDER
    assert GraphoneModel.load(path).order == ORDER


@pytest.fixture(scope="module")
def shipped() -> GraphoneModel:
    installer = Installer(default_home(), BYTES_PER_GB)
    path = cache_path(installer.path_of(CMUDICT), CMUDICT)
    if not path.is_file():
        pytest.skip(
            "no trained letter-to-sound model (uv run python benchmarks/g2p.py)"
        )
    return GraphoneModel.load(path)


def test_the_shipped_model_says_synthia(shipped: GraphoneModel) -> None:
    assert shipped.order == ORDER
    assert shipped.guess("synthia") == ("S", "IH1", "N", "TH", "IY0", "AH0")


@pytest.mark.parametrize("word", ["synthia", "kokoro", "qwen", "ollama", "zyxw"])
def test_every_shipped_guess_converts_to_phonemes(
    shipped: GraphoneModel, word: str
) -> None:
    assert set(to_misaki(shipped.guess(word))) <= PHONEMES
