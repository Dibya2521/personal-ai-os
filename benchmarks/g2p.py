"""Measure two letter-to-sound models learned from the CMU Pronouncing Dictionary.

    uv run python benchmarks/g2p.py

Both models start from one alignment of letters to phones, learned by
``synthia.voice.guess`` on the training words. Model A, window rules, gives
each letter the output it most often had among training letters with the same
widest window of letters around them. Model B is the graphone n-gram that
``synthia.voice.guess`` ships. A keyed hash of each word splits the plain words
for good, whatever the file order or Python's hash seed: a tenth is held out
to score, a tenth chooses the n-gram order and the beam, and the rest is
learned from.

The phone error rate is the edit distance from the guess to the dictionary's
first pronunciation, over the phones of that pronunciation, stress digits
included; then the same without them; word accuracy counts exact matches.
Seconds are wall time on the machine as it is, so the CPU load is printed.

Last, the shipped model is trained on every plain word and saved next to the
dictionary, where SYNTHIA loads it from.
"""

from __future__ import annotations

import functools
import hashlib
import string
import sys
import time
import tracemalloc
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import psutil

from synthia.kernel.config import load_settings
from synthia.models.catalogue import CMUDICT
from synthia.models.install import BYTES_PER_GB, Installer
from synthia.voice.guess import (
    BEAM,
    ORDER,
    Alignment,
    GraphoneModel,
    align,
    cache_path,
    train,
    training_words,
)
from synthia.voice.phonemes import Pronunciation, Pronunciations

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

HELD_OUT: Final = 0
DEV: Final = 1
FIT: Final = 2
_FOLDS: Final = 10
_SPLIT_KEY: Final = b"synthia-g2p"
_SPLIT_DIGEST_BYTES: Final = 8
WINDOW: Final = 4
_EDGE: Final = "#"
ORDERS: Final = range(3, 7)
BEAMS: Final = (8, 16, 32)
EXAMPLES: Final = 10
NAME: Final = "synthia"
CPU_SAMPLE_S: Final = 1.0
_STRESS: Final = "012"
_MS: Final = 1000
_MB: Final = 1024**2

type Pair = tuple[str, Pronunciation]


def fold(word: str) -> int:
    """Return ``word``'s fold: ``HELD_OUT``, ``DEV`` or ``FIT``."""
    digest = hashlib.blake2b(
        word.encode(), key=_SPLIT_KEY, digest_size=_SPLIT_DIGEST_BYTES
    ).digest()
    return min(int.from_bytes(digest) % _FOLDS, FIT)


def _padded(letters: str, width: int) -> str:
    return f"{_EDGE * width}{letters}{_EDGE * width}"


def _majorities(
    counts: Mapping[tuple[str, Pronunciation], int],
) -> dict[str, Pronunciation]:
    """Return each window's commonest output; ties go to the shorter, then the first."""
    majority: dict[str, Pronunciation] = {}
    ranked = sorted(
        counts.items(), key=lambda item: (-item[1], len(item[0][1]), item[0][1])
    )
    for (window, output), _ in ranked:
        majority.setdefault(window, output)
    return majority


@dataclass(frozen=True, slots=True)
class WindowRules:
    """Model A: each letter's commonest output in the widest window seen around it.

    A window is the letter with ``k`` letters each side, ``#`` past the word's
    edges, for ``k`` from ``width`` down to 0. Only windows whose output
    differs from the next narrower window's are kept, since the narrower one
    gives the same answer.
    """

    width: int
    rules: Mapping[str, Pronunciation]

    @classmethod
    def fit(cls, alignment: Alignment, width: int = WINDOW) -> WindowRules:
        """Learn the rules of ``alignment``'s words."""
        rules: dict[str, Pronunciation] = {}
        narrower: dict[str, Pronunciation] = {}
        for k in range(width + 1):
            counts: Counter[tuple[str, Pronunciation]] = Counter()
            for word in alignment.words:
                padded = _padded(word.letters, width)
                for centre, output in enumerate(word.outputs, start=width):
                    counts[padded[centre - k : centre + k + 1], output] += 1
            majority = _majorities(counts)
            rules |= {
                window: output
                for window, output in majority.items()
                if k == 0 or narrower[window[1:-1]] != output
            }
            narrower = majority
        return cls(width, rules)

    def guess(self, word: str) -> Pronunciation:
        """Return ``word``'s phones, read as lower-case letters a to z."""
        letters = "".join(c for c in word.casefold() if c in string.ascii_lowercase)
        padded = _padded(letters, self.width)
        phones: list[str] = []
        for centre in range(self.width, self.width + len(letters)):
            for k in range(self.width, -1, -1):
                output = self.rules.get(padded[centre - k : centre + k + 1])
                if output is not None:
                    phones += output
                    break
        return tuple(phones)


def edits(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Return the Levenshtein distance between two phone sequences."""
    row = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        previous, row[0] = row[0], i
        for j, hyp in enumerate(hypothesis, 1):
            previous, row[j] = (
                row[j],
                min(row[j] + 1, row[j - 1] + 1, previous + (ref != hyp)),
            )
    return row[-1]


def unstressed(phones: Pronunciation) -> Pronunciation:
    """Return ``phones`` without stress digits."""
    return tuple(phone.rstrip(_STRESS) for phone in phones)


@dataclass(frozen=True, slots=True)
class Score:
    """How a model's guesses compare with the dictionary's first pronunciations."""

    phone_error_rate: float
    unstressed_error_rate: float
    word_accuracy: float
    ms_per_word: float


def score(guess: Callable[[str], Pronunciation], words: Sequence[Pair]) -> Score:
    """Guess every word of ``words`` and compare each with its pronunciation."""
    errors = unstressed_errors = phones = right = 0
    took = 0.0
    for letters, wanted in words:
        began = time.perf_counter()
        got = guess(letters)
        took += time.perf_counter() - began
        errors += edits(wanted, got)
        unstressed_errors += edits(unstressed(wanted), unstressed(got))
        phones += len(wanted)
        right += got == wanted
    return Score(
        errors / phones,
        unstressed_errors / phones,
        right / len(words),
        took * _MS / len(words),
    )


def _row(name: str, result: Score) -> str:
    return (
        f"{name:28} {result.phone_error_rate:7.2%} {result.unstressed_error_rate:9.2%} "
        f"{result.word_accuracy:8.2%} {result.ms_per_word:8.2f}"
    )


def _header() -> str:
    return f"{'model':28} {'PER':>7} {'PER-ns':>9} {'word':>8} {'ms/word':>8}"


def split(pronunciations: Pronunciations) -> dict[int, list[Pair]]:
    """Return the plain words of each fold, folded by the letters a model sees.

    Folding by the dictionary's key would put "aim's" and "aims", the same
    letters once the apostrophe is dropped, in different folds.
    """
    folds: dict[int, list[Pair]] = {HELD_OUT: [], DEV: [], FIT: []}
    for letters, phones in training_words(pronunciations):
        folds[fold(letters)].append((letters, phones))
    return folds


def _fit_orders(alignment: Alignment) -> dict[int, GraphoneModel]:
    models: dict[int, GraphoneModel] = {}
    for order in ORDERS:
        began = time.perf_counter()
        models[order] = GraphoneModel.fit(alignment, order=order)
        took = time.perf_counter() - began
        print(f"  n={order}: {models[order].ngrams:,} n-grams, fit {took:.1f} s")
    return models


def _choose(
    models: Mapping[int, GraphoneModel], dev: Sequence[Pair]
) -> tuple[int, int]:
    """Return the order, then the beam, with the lowest dev error; ties go smaller."""
    print(f"dev ({len(dev):,} words), each order at beam {BEAM}")
    print(_header())
    by_order: dict[int, Score] = {}
    for order, model in models.items():
        by_order[order] = score(functools.partial(model.guess, beam=BEAM), dev)
        print(_row(f"graphone n={order} beam={BEAM}", by_order[order]), flush=True)
    order = min(by_order, key=lambda n: (by_order[n].phone_error_rate, n))
    by_beam = {BEAM: by_order[order]}
    for beam in BEAMS:
        if beam not in by_beam:
            by_beam[beam] = score(
                functools.partial(models[order].guess, beam=beam), dev
            )
            print(_row(f"graphone n={order} beam={beam}", by_beam[beam]), flush=True)
    beam = min(by_beam, key=lambda b: (by_beam[b].phone_error_rate, b))
    return order, beam


def _examples(
    rules: WindowRules, model: GraphoneModel, beam: int, held_out: Sequence[Pair]
) -> None:
    print(f"\n{'word':16} {'CMUdict':28} {'window rules':28} graphone n-gram")
    picks = [*held_out[:: len(held_out) // EXAMPLES][:EXAMPLES], (NAME, ())]
    for letters, wanted in picks:
        print(
            f"{letters:16} {' '.join(wanted) or '-':28} "
            f"{' '.join(rules.guess(letters)):28} "
            f"{' '.join(model.guess(letters, beam=beam))}"
        )


def _ship(installer: Installer, pronunciations: Pronunciations) -> None:
    path = cache_path(installer.path_of(CMUDICT), CMUDICT)
    tracemalloc.start()
    began = time.perf_counter()
    model = train(pronunciations)
    model.save(path)
    took = time.perf_counter() - began
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    began = time.perf_counter()
    loaded = GraphoneModel.load(path)
    load_s = time.perf_counter() - began
    tracemalloc.start()
    GraphoneModel.load(path)
    load_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    print(
        f"\nshipped n={model.order} beam={BEAM}: trained on every plain word in "
        f"{took:.0f} s with tracemalloc on, peak {peak / _MB:.0f} MB; "
        f"{model.ngrams:,} n-grams"
    )
    print(
        f"cache {path.name}: {path.stat().st_size / _MB:.1f} MB, loads in "
        f"{load_s:.2f} s, peak {load_peak / _MB:.0f} MB while loading"
    )
    print(f"{NAME}: {' '.join(loaded.guess(NAME))}")


def _align(words: Sequence[Pair]) -> Alignment:
    began = time.perf_counter()
    alignment = align(words)
    print(
        f"alignment: {time.perf_counter() - began:.0f} s, "
        f"{len(alignment.log_likelihoods)} Viterbi passes, "
        f"{len(alignment.words):,} aligned, {alignment.unaligned} not",
        flush=True,
    )
    for number, value in enumerate(alignment.log_likelihoods, start=1):
        print(f"  pass {number}: log likelihood {value:,.0f}")
    for letters, outputs in alignment.words:
        if letters in {"phone", "box"}:
            print(f"  {letters}: {outputs}")
    return alignment


def _fit_rules(alignment: Alignment) -> WindowRules:
    began = time.perf_counter()
    rules = WindowRules.fit(alignment)
    took = time.perf_counter() - began
    print(f"window rules: {len(rules.rules):,} rules, fit {took:.1f} s", flush=True)
    return rules


def main() -> int:
    """Print both models' scores, then train and save the shipped model."""
    installer = Installer(load_settings().home, BYTES_PER_GB)
    if not installer.installed(CMUDICT):
        print(
            "cmudict is not installed: synthia models install cmudict", file=sys.stderr
        )
        return 1
    pronunciations = Pronunciations.load(installer.path_of(CMUDICT), CMUDICT)
    print(f"CPU {psutil.cpu_percent(interval=CPU_SAMPLE_S):.0f} % busy at the start")
    folds = split(pronunciations)
    print(
        f"plain words: fit {len(folds[FIT]):,}, dev {len(folds[DEV]):,}, "
        f"held out {len(folds[HELD_OUT]):,}"
    )
    alignment = _align(folds[FIT])
    rules = _fit_rules(alignment)
    models = _fit_orders(alignment)
    order, beam = _choose(models, folds[DEV])
    held_out = folds[HELD_OUT]
    print(f"\nheld out ({len(held_out):,} words)")
    print(_header())
    print(_row(f"window rules k={WINDOW}", score(rules.guess, held_out)), flush=True)
    graphone = functools.partial(models[order].guess, beam=beam)
    print(
        _row(f"graphone n={order} beam={beam}", score(graphone, held_out)), flush=True
    )
    _examples(rules, models[order], beam, held_out)
    if (order, beam) != (ORDER, BEAM):
        print(
            f"\nchosen n={order} beam={beam} differ from shipped n={ORDER} beam={BEAM}"
        )
    _ship(installer, pronunciations)
    print(f"CPU {psutil.cpu_percent():.0f} % busy on average over the run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
