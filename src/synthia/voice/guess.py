"""Guess how a word the pronouncing dictionary lacks is said, from its letters.

The guess is learned from the dictionary itself, in two steps. First each
letter of a known word is aligned with the phones it stands for: nothing (the
``h`` of "phone"), one phone, or two (the ``x`` of "box" is ``K S``). A letter
with its phones is a graphone, so a known word becomes a sequence of
graphones. Then an n-gram model over graphones, smoothed by Witten-Bell
interpolation, gives the probability of each graphone after the few before
it. To guess a new word, a beam search reads its letters left to right,
keeping the likeliest graphone sequences, and their phones are the guess.

Training takes minutes in pure Python, so the model is saved next to the
dictionary and loaded from there.
"""

from __future__ import annotations

import contextlib
import math
import re
import string
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from synthia.kernel.errors import SynthiaError
from synthia.voice.phonemes import Pronunciation, spell, to_misaki

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from numpy.typing import NDArray

    from synthia.models.catalogue import Lexicon
    from synthia.voice.phonemes import Pronunciations

ORDER: Final = 5
"""Graphones in each n-gram: the one guessed and the ones before it."""

BEAM: Final = 32
"""Graphone sequences the search keeps after each letter."""

FORMAT_VERSION: Final = 1
"""Version of the saved model's layout, part of the cache file's name."""

_TRAINING_WORD: Final = re.compile(r"[a-z]+(?:'[a-z]+)?")
_APOSTROPHE: Final = "'"
_LETTERS: Final = frozenset(string.ascii_lowercase)
_MOST_PHONES: Final = 2
_LENGTH_WEIGHTS: Final = (0.1, 1.0, 0.05)
_UNSEEN: Final = -20.0
_SETTLED: Final = 1e-3
_MOST_ITERATIONS: Final = 10
_BOUNDARY: Final = 0
_LARGEST_KEY: Final = int(np.iinfo(np.int64).max)
_CACHE_STEM: Final = "g2p"

type _Pair = tuple[str, Pronunciation]
type _Scores = Mapping[str, Mapping[Pronunciation, float]]


class GuessError(SynthiaError):
    """A saved letter-to-sound model that cannot be loaded."""


class Graphone(NamedTuple):
    """A letter and the phones it stands for in one word."""

    letter: str
    phones: Pronunciation


class AlignedWord(NamedTuple):
    """A word's letters and the phones each letter stands for."""

    letters: str
    outputs: tuple[Pronunciation, ...]


@dataclass(frozen=True, slots=True)
class Alignment:
    """The aligned words, how many could not be aligned, and how fitting settled.

    ``log_likelihoods`` holds one total per Viterbi pass, so its length is the
    number of passes run.
    """

    words: tuple[AlignedWord, ...]
    unaligned: int
    log_likelihoods: tuple[float, ...]


class _Level(NamedTuple):
    """The n-grams after histories of one length.

    ``keys`` are ``history * size + graphone``, sorted, each with its log
    probability; ``histories`` are sorted, each with the log weight that scales
    the next shorter history's probabilities for graphones not seen after it.
    """

    keys: NDArray[np.int64]
    log_probabilities: NDArray[np.float64]
    histories: NDArray[np.int64]
    backoffs: NDArray[np.float64]


class _Header(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int
    order: int = Field(ge=1)
    graphones: tuple[Graphone, ...]


def training_words(pronunciations: Pronunciations) -> list[_Pair]:
    """Return each plain word's letters with its first pronunciation, each pair once.

    A plain word is letters with at most one apostrophe inside, which is
    dropped from the letters, so "it's" and "its" give one pair when they
    sound the same.
    """
    return list(
        dict.fromkeys(
            (word.replace(_APOSTROPHE, ""), pronunciations.variants(word)[0])
            for word in pronunciations
            if _TRAINING_WORD.fullmatch(word)
        )
    )


def align(words: Iterable[_Pair]) -> Alignment:
    """Align every letter of ``words`` with the phones it stands for.

    The first estimate counts every possible alignment, each letter weighted
    0.1, 1 or 0.05 as it takes 0, 1 or 2 phones. Viterbi passes then re-estimate from
    the single best alignment of each word until the total log likelihood
    changes by under 0.1 %, or for at most 10 passes. A word needing more than
    two phones for some letter cannot be aligned and is left out.
    """
    pairs = list(words)
    scores = _estimate(_expected_counts(pairs))
    log_likelihoods: list[float] = []
    while True:
        aligned, log_likelihood = _viterbi_all(pairs, scores)
        log_likelihoods.append(log_likelihood)
        if _settled(log_likelihoods):
            return Alignment(
                tuple(aligned), len(pairs) - len(aligned), tuple(log_likelihoods)
            )
        scores = _estimate(
            Counter(
                pair
                for word in aligned
                for pair in zip(word.letters, word.outputs, strict=True)
            )
        )


def _settled(log_likelihoods: Sequence[float]) -> bool:
    return len(log_likelihoods) >= _MOST_ITERATIONS or (
        len(log_likelihoods) > 1
        and abs(log_likelihoods[-1] - log_likelihoods[-2])
        <= _SETTLED * abs(log_likelihoods[-2])
    )


def _ways(letters: int, phones: int) -> list[list[float]]:
    """Return the weight of all ways ``a`` letters write ``b`` phones, by a and b."""
    ways = [[0.0] * (phones + 1) for _ in range(letters + 1)]
    ways[0][0] = 1.0
    for a in range(1, letters + 1):
        for b in range(phones + 1):
            ways[a][b] = sum(
                weight * ways[a - 1][b - took]
                for took, weight in enumerate(_LENGTH_WEIGHTS[: b + 1])
            )
    return ways


def _expected_counts(pairs: Sequence[_Pair]) -> dict[_Pair, float]:
    longest = max((len(letters) for letters, _ in pairs), default=0)
    ways = _ways(longest, _MOST_PHONES * longest)
    counts: defaultdict[_Pair, float] = defaultdict(float)
    for letters, phones in pairs:
        count, size = len(letters), len(phones)
        if size > _MOST_PHONES * count:
            continue
        total = ways[count][size]
        for i, letter in enumerate(letters):
            rest = ways[count - i - 1]
            for j in range(min(size, _MOST_PHONES * i) + 1):
                for took, weight in enumerate(_LENGTH_WEIGHTS[: size - j + 1]):
                    share = ways[i][j] * weight * rest[size - j - took] / total
                    if share > 0:
                        counts[letter, phones[j : j + took]] += share
    return counts


def _estimate(counts: Mapping[_Pair, float]) -> _Scores:
    totals: defaultdict[str, float] = defaultdict(float)
    for (letter, _), count in counts.items():
        totals[letter] += count
    scores: defaultdict[str, dict[Pronunciation, float]] = defaultdict(dict)
    for (letter, output), count in counts.items():
        scores[letter][output] = math.log(count / totals[letter])
    return scores


def _viterbi_all(
    pairs: Iterable[_Pair], scores: _Scores
) -> tuple[list[AlignedWord], float]:
    aligned: list[AlignedWord] = []
    total = 0.0
    for letters, phones in pairs:
        found = _viterbi(letters, phones, scores)
        if found is not None:
            aligned.append(found[0])
            total += found[1]
    return aligned, total


def _viterbi(
    letters: str, phones: Pronunciation, scores: _Scores
) -> tuple[AlignedWord, float] | None:
    count, size = len(letters), len(phones)
    if size > _MOST_PHONES * count:
        return None
    best = [0.0, *[-math.inf] * size]
    steps: list[list[int]] = []
    for i, letter in enumerate(letters):
        reachable = range(
            max(0, size - _MOST_PHONES * (count - i - 1)),
            min(size, _MOST_PHONES * (i + 1)) + 1,
        )
        best, step = _advance(best, phones, reachable, scores.get(letter, {}))
        steps.append(step)
    outputs: list[Pronunciation] = []
    end = size
    for step in reversed(steps):
        took = step[end]
        outputs.append(phones[end - took : end])
        end -= took
    return AlignedWord(letters, tuple(reversed(outputs))), best[size]


def _advance(
    best: Sequence[float],
    phones: Pronunciation,
    reachable: range,
    table: Mapping[Pronunciation, float],
) -> tuple[list[float], list[int]]:
    """Extend every best path by one letter, to each reachable phone count."""
    row = [-math.inf] * len(best)
    step = [0] * len(best)
    for j in reachable:
        for took in range(min(j, _MOST_PHONES) + 1):
            score = best[j - took] + table.get(phones[j - took : j], _UNSEEN)
            if score > row[j]:
                row[j], step[j] = score, took
    return row, step


class GraphoneModel:
    """An n-gram model over graphones, which guesses a word's phones.

    Graphone ``n`` of the model is ``graphones[n - 1]``; number 0 marks the
    edges of a word, before its first letter and after its last.
    """

    def __init__(
        self,
        graphones: Sequence[Graphone],
        unigram: NDArray[np.float64],
        levels: Sequence[_Level],
    ) -> None:
        self._graphones = tuple(graphones)
        self._unigram = unigram
        self._levels = tuple(levels)
        self._size = len(graphones) + 1
        self._states = self._size ** len(levels)
        by_letter: defaultdict[str, list[int]] = defaultdict(list)
        for number, graphone in enumerate(graphones, start=1):
            by_letter[graphone.letter].append(number)
        self._options = {
            letter: np.array(numbers, np.int64) for letter, numbers in by_letter.items()
        }

    @classmethod
    def fit(cls, alignment: Alignment, *, order: int = ORDER) -> GraphoneModel:
        """Count the graphone n-grams of ``alignment`` and smooth them.

        Raises:
            ValueError: If ``order`` is under 1, no word is aligned, or there
                are too many graphones to number n-grams of this order.
        """
        if order < 1 or not alignment.words:
            message = f"need an order of at least 1 and a word: {order=}"
            raise ValueError(message)
        graphones = sorted(
            {
                Graphone(letter, output)
                for word in alignment.words
                for letter, output in zip(word.letters, word.outputs, strict=True)
            }
        )
        size = len(graphones) + 1
        if size**order > _LARGEST_KEY:
            message = f"{size} graphones are too many for n-grams of {order=}"
            raise ValueError(message)
        sequence, predicted = _sequence(alignment, graphones, order)
        targets = sequence[predicted]
        unigram = np.log(np.bincount(targets, minlength=size) / len(targets))
        levels: list[_Level] = []
        histories = np.zeros_like(sequence)
        for length in range(1, order):
            before = np.concatenate([np.zeros(length, np.int64), sequence[:-length]])
            histories = histories + before * size ** (length - 1)
            levels.append(_level(histories[predicted], targets, size, unigram, levels))
        return cls(graphones, unigram, levels)

    @property
    def order(self) -> int:
        """Graphones in each n-gram."""
        return len(self._levels) + 1

    @property
    def ngrams(self) -> int:
        """N-grams the model holds, of every length."""
        return len(self._unigram) + sum(len(level.keys) for level in self._levels)

    def guess(self, word: str, *, beam: int = BEAM) -> Pronunciation:
        """Return the likeliest phones of ``word``, read as lower-case letters a to z.

        Other characters are dropped. A word with a letter the model never
        saw is spelled.

        Raises:
            ValueError: If ``beam`` is under 1 or ``word`` has no letter.
        """
        if beam < 1:
            message = f"the beam must keep at least one sequence: {beam=}"
            raise ValueError(message)
        letters = [letter for letter in word.casefold() if letter in _LETTERS]
        if not letters or not self._options.keys() >= set(letters):
            return spell(word)
        histories = np.zeros(1, np.int64)
        scores = np.zeros(1)
        trail: list[tuple[NDArray[np.intp], NDArray[np.int64]]] = []
        for letter in letters:
            histories, scores, step = self._extend(
                histories, scores, self._options[letter], beam
            )
            trail.append(step)
        ending = np.full_like(histories, _BOUNDARY)
        best = int(np.argmax(scores + self._log_probabilities(histories, ending)))
        outputs: list[Pronunciation] = []
        for parents, chosen in reversed(trail):
            outputs.append(self._graphones[int(chosen[best]) - 1].phones)
            best = int(parents[best])
        return tuple(chain.from_iterable(reversed(outputs)))

    def _extend(
        self,
        histories: NDArray[np.int64],
        scores: NDArray[np.float64],
        options: NDArray[np.int64],
        beam: int,
    ) -> tuple[
        NDArray[np.int64],
        NDArray[np.float64],
        tuple[NDArray[np.intp], NDArray[np.int64]],
    ]:
        """Extend every kept sequence by each graphone of the next letter.

        Of sequences ending in the same history only the likeliest can win,
        so only it is kept; then the ``beam`` likeliest are.
        """
        parents = np.repeat(np.arange(len(histories)), len(options))
        chosen = np.tile(options, len(histories))
        before = histories[parents]
        totals = scores[parents] + self._log_probabilities(before, chosen)
        after = (before * self._size + chosen) % self._states
        ranked = np.argsort(-totals, kind="stable")
        _, first = np.unique(after[ranked], return_index=True)
        kept = ranked[np.sort(first)][:beam]
        return after[kept], totals[kept], (parents[kept], chosen[kept])

    def _log_probabilities(
        self, histories: NDArray[np.int64], graphones: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        return _log_probabilities(
            self._unigram, self._levels, self._size, histories, graphones
        )

    def save(self, path: Path) -> None:
        """Write the model to ``path`` in one step, so a reader never sees half."""
        header = _Header(
            version=FORMAT_VERSION, order=self.order, graphones=self._graphones
        )
        arrays: dict[str, NDArray[np.generic]] = {
            "header": np.array(header.model_dump_json()),
            "unigram": self._unigram,
        }
        for length, level in enumerate(self._levels, start=1):
            arrays |= {
                f"{name}_{length}": array
                for name, array in zip(_Level._fields, level, strict=True)
            }
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=path.name, suffix=".part", delete_on_close=False
        ) as part:
            np.savez(part, allow_pickle=False, **arrays)
            part.close()
            Path(part.name).replace(path)

    @classmethod
    def load(cls, path: Path) -> GraphoneModel:
        """Read a model written by :meth:`save`.

        Raises:
            GuessError: If the file is missing, damaged or of another format.
        """
        try:
            # np.load leaks the file it opened when it is not a zip archive.
            with path.open("rb") as file, np.load(file, allow_pickle=False) as saved:
                header = _Header.model_validate_json(str(saved["header"]))
                unigram: NDArray[np.float64] = saved["unigram"]
                levels = [
                    _Level(*(saved[f"{name}_{length}"] for name in _Level._fields))
                    for length in range(1, header.order)
                ]
        except (OSError, EOFError, KeyError, ValueError, zipfile.BadZipFile) as error:
            message = f"cannot read the letter-to-sound model {path}: {error}"
            raise GuessError(message) from error
        if not _valid(header, unigram, levels):
            message = (
                f"{path} is not a letter-to-sound model of format {FORMAT_VERSION}"
            )
            raise GuessError(message)
        return cls(header.graphones, unigram, levels)


def _sequence(
    alignment: Alignment, graphones: Sequence[Graphone], order: int
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """Return every word's graphone numbers, each word after ``order - 1`` edges.

    The mask marks the numbers to predict: each graphone and the edge that
    ends its word.
    """
    number = {graphone: n for n, graphone in enumerate(graphones, start=1)}
    edges = [_BOUNDARY] * (order - 1)
    sequence: list[int] = []
    predicted: list[bool] = []
    for word in alignment.words:
        sequence += edges
        sequence += [
            number[Graphone(letter, output)]
            for letter, output in zip(word.letters, word.outputs, strict=True)
        ]
        sequence.append(_BOUNDARY)
        predicted += [False] * len(edges) + [True] * (len(word.letters) + 1)
    return np.array(sequence, np.int64), np.array(predicted, np.bool_)


def _level(
    histories: NDArray[np.int64],
    targets: NDArray[np.int64],
    size: int,
    unigram: NDArray[np.float64],
    shorter: Sequence[_Level],
) -> _Level:
    """Return the Witten-Bell n-grams after ``histories``, over the ``shorter`` ones.

    After a history seen N times followed by T different graphones, a
    graphone seen c times has probability (c + T p) / (N + T), where p is its
    probability after the history one graphone shorter.
    """
    keys, counts = np.unique(histories * size + targets, return_counts=True)
    seen, starts, types = np.unique(keys // size, return_index=True, return_counts=True)
    totals = np.add.reduceat(counts, starts)
    shorter_p = np.exp(
        _log_probabilities(unigram, shorter, size, keys // size, keys % size)
    )
    each_type = np.repeat(types, types)
    each_total = np.repeat(totals, types)
    return _Level(
        keys,
        np.log(counts + each_type * shorter_p) - np.log(each_total + each_type),
        seen,
        np.log(types) - np.log(totals + types),
    )


def _log_probabilities(
    unigram: NDArray[np.float64],
    levels: Sequence[_Level],
    size: int,
    histories: NDArray[np.int64],
    graphones: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Return the log probability of each graphone after its history.

    A history holds its graphones as digits in base ``size``, the latest
    lowest, so its last ``length`` graphones are it modulo ``size**length``.
    """
    result = unigram[graphones]
    for length, level in enumerate(levels, start=1):
        history = histories % size**length
        found, at = _search(level.keys, history * size + graphones)
        seen, where = _search(level.histories, history)
        weight = np.where(seen, level.backoffs[where], 0.0)
        result = np.where(found, level.log_probabilities[at], result + weight)
    return result


def _search(
    table: NDArray[np.int64], values: NDArray[np.int64]
) -> tuple[NDArray[np.bool_], NDArray[np.intp]]:
    at = np.minimum(np.searchsorted(table, values), len(table) - 1)
    return table[at] == values, at


def _valid(
    header: _Header, unigram: NDArray[np.float64], levels: Sequence[_Level]
) -> bool:
    size = len(header.graphones) + 1
    return (
        header.version == FORMAT_VERSION
        and size**header.order <= _LARGEST_KEY
        and all(_valid_graphone(graphone) for graphone in header.graphones)
        and unigram.dtype == np.float64
        and unigram.shape == (size,)
        and _valid_log_probabilities(unigram)
        and all(
            _valid_level(level, size, length)
            for length, level in enumerate(levels, start=1)
        )
    )


def _valid_log_probabilities(values: NDArray[np.float64]) -> bool:
    return bool(np.all(np.isfinite(values)) and np.all(values <= 0))


def _valid_graphone(graphone: Graphone) -> bool:
    try:
        to_misaki(graphone.phones)
    except ValueError:
        return False
    return graphone.letter in _LETTERS and len(graphone.phones) <= _MOST_PHONES


def _valid_level(level: _Level, size: int, length: int) -> bool:
    """Check one level; keys and histories are sorted, so the ends bound them all."""
    return (
        all(array.ndim == 1 for array in level)
        and level.keys.dtype == level.histories.dtype == np.int64
        and level.log_probabilities.dtype == level.backoffs.dtype == np.float64
        and len(level.keys) == len(level.log_probabilities) > 0
        and len(level.histories) == len(level.backoffs) > 0
        and bool(np.all(np.diff(level.keys) > 0))
        and bool(np.all(np.diff(level.histories) > 0))
        and level.keys[0] >= 0
        and level.keys[-1] < size ** (length + 1)
        and level.histories[0] >= 0
        and level.histories[-1] < size**length
        and _valid_log_probabilities(level.log_probabilities)
        and bool(np.all(np.isfinite(level.backoffs)))
    )


def train(pronunciations: Pronunciations, *, order: int = ORDER) -> GraphoneModel:
    """Learn a model from every plain word of ``pronunciations``; takes minutes.

    Raises:
        ValueError: If no word can be learned from.
    """
    return GraphoneModel.fit(align(training_words(pronunciations)), order=order)


def cache_path(directory: Path, spec: Lexicon) -> Path:
    """Return where the model learned from ``spec``'s dictionary is saved."""
    return directory / f"{_CACHE_STEM}-{spec.dictionary.sha256}-{FORMAT_VERSION}.npz"


def load_or_train(
    directory: Path, spec: Lexicon, pronunciations: Pronunciations
) -> GraphoneModel:
    """Load the saved model of ``spec``'s dictionary, training and saving it if needed.

    A saved model that is missing, damaged or of another order is replaced.

    Raises:
        ValueError: If no word of ``pronunciations`` can be learned from.
        OSError: If the model cannot be saved.
    """
    path = cache_path(directory, spec)
    with contextlib.suppress(GuessError):
        model = GraphoneModel.load(path)
        if model.order == ORDER:
            return model
    model = train(pronunciations)
    model.save(path)
    return model
