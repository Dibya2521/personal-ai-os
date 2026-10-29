"""Text to the misaki phonemes Kokoro reads, from the CMU Pronouncing Dictionary.

Kokoro-82M was trained on misaki phonemes: one character per sound, the
diphthongs written as capital letters (``A`` as in "day", ``I`` as in "my",
``O``, ``W``, ``Y``) and a stress mark immediately before the stressed vowel.
Pronunciations come from the CMU Pronouncing Dictionary, which writes each word
in ARPAbet with a stress digit on every vowel (0 none, 1 primary, 2 secondary);
this module maps them phone by phone. Text is first normalised into words to
look up, letters to spell and the punctuation Kokoro reads as pauses. A word
the dictionary does not know goes to a ``Guess``, which by default spells it.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass
from itertools import chain, product
from typing import TYPE_CHECKING, Final, NamedTuple

from synthia.kernel.errors import SynthiaError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping
    from pathlib import Path

    from synthia.models.catalogue import Lexicon

type Pronunciation = tuple[str, ...]
"""ARPAbet phones, each vowel with its stress digit: ``("HH", "AH0", "L", "OW1")``."""

type Guess = Callable[[str], Pronunciation]
"""Pronounce a casefolded word the dictionary does not know."""

_STRESS: Final[Mapping[str, str]] = {
    "0": "",
    "1": "\N{MODIFIER LETTER VERTICAL LINE}",
    "2": "\N{MODIFIER LETTER LOW VERTICAL LINE}",
}
_SCHWA: Final = "\N{LATIN SMALL LETTER SCHWA}"
_TURNED_R: Final = "\N{LATIN SMALL LETTER TURNED R}"
_TURNED_V: Final = "\N{LATIN SMALL LETTER TURNED V}"
_OPEN_ER: Final = f"\N{LATIN SMALL LETTER REVERSED OPEN E}{_TURNED_R}"
_VOWELS: Final[Mapping[str, str]] = {
    "AA": "\N{LATIN SMALL LETTER ALPHA}",
    "AE": "\N{LATIN SMALL LETTER AE}",
    "AO": "\N{LATIN SMALL LETTER OPEN O}",
    "AW": "W",
    "AY": "I",
    "EH": "\N{LATIN SMALL LETTER OPEN E}",
    "EY": "A",
    "IH": "\N{LATIN LETTER SMALL CAPITAL I}",
    "IY": "i",
    "OW": "O",
    "OY": "Y",
    "UH": "\N{LATIN SMALL LETTER UPSILON}",
    "UW": "u",
}
# AH and ER change their sound, not only their stress, when stressed.
_VOWELS_BY_STRESS: Final[Mapping[str, str]] = {
    "AH0": _SCHWA,
    "AH1": _TURNED_V,
    "AH2": _TURNED_V,
    "ER0": f"{_SCHWA}{_TURNED_R}",
    "ER1": _OPEN_ER,
    "ER2": _OPEN_ER,
}
_CONSONANTS: Final[Mapping[str, str]] = {
    "B": "b",
    "CH": "\N{LATIN SMALL LETTER TESH DIGRAPH}",
    "D": "d",
    "DH": "\N{LATIN SMALL LETTER ETH}",
    "F": "f",
    "G": "\N{LATIN SMALL LETTER SCRIPT G}",
    "HH": "h",
    "JH": "\N{LATIN SMALL LETTER DEZH DIGRAPH}",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "\N{LATIN SMALL LETTER ENG}",
    "P": "p",
    "R": _TURNED_R,
    "S": "s",
    "SH": "\N{LATIN SMALL LETTER ESH}",
    "T": "t",
    "TH": "\N{GREEK SMALL LETTER THETA}",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "\N{LATIN SMALL LETTER EZH}",
}
_MISAKI: Final[Mapping[str, str]] = {
    **_CONSONANTS,
    **{
        f"{vowel}{digit}": f"{mark}{symbol}"
        for (vowel, symbol), (digit, mark) in product(_VOWELS.items(), _STRESS.items())
    },
    **{
        phone: f"{_STRESS[phone[-1]]}{symbol}"
        for phone, symbol in _VOWELS_BY_STRESS.items()
    },
}

PHONEMES: Final[frozenset[str]] = frozenset("".join(_MISAKI.values()))
"""Every character ``to_misaki`` can write, the stress marks included."""

_ELLIPSIS: Final = "\N{HORIZONTAL ELLIPSIS}"
PUNCTUATION: Final[frozenset[str]] = frozenset(f";:,.!?{_ELLIPSIS}")
"""The punctuation the normaliser keeps, each a pause Kokoro reads."""

_LETTER_NAMES: Final[Mapping[str, Pronunciation]] = {
    "a": ("EY1",),
    "b": ("B", "IY1"),
    "c": ("S", "IY1"),
    "d": ("D", "IY1"),
    "e": ("IY1",),
    "f": ("EH1", "F"),
    "g": ("JH", "IY1"),
    "h": ("EY1", "CH"),
    "i": ("AY1",),
    "j": ("JH", "EY1"),
    "k": ("K", "EY1"),
    "l": ("EH1", "L"),
    "m": ("EH1", "M"),
    "n": ("EH1", "N"),
    "o": ("OW1",),
    "p": ("P", "IY1"),
    "q": ("K", "Y", "UW1"),
    "r": ("AA1", "R"),
    "s": ("EH1", "S"),
    "t": ("T", "IY1"),
    "u": ("Y", "UW1"),
    "v": ("V", "IY1"),
    "w": ("D", "AH1", "B", "AH0", "L", "Y", "UW0"),
    "x": ("EH1", "K", "S"),
    "y": ("W", "AY1"),
    "z": ("Z", "IY1"),
}

_POSSESSIVE: Final = "'s"
_POSSESSIVE_ENDINGS: Final[Mapping[str, Pronunciation]] = {
    **dict.fromkeys(("P", "T", "K", "F", "TH"), ("S",)),
    **dict.fromkeys(("S", "Z", "SH", "ZH", "CH", "JH"), ("IH0", "Z")),
}
_VOICED_ENDING: Final[Pronunciation] = ("Z",)
_SPELLED_CAPITALS: Final = range(2, 4)
_COMMENT: Final = "#"
_VARIANT: Final = "("

_ONES: Final = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
)
_TEENS: Final = (
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS: Final = (
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)
_SCALES: Final = ("thousand", "million", "billion")
_HUNDRED_WORD: Final = "hundred"
_ORDINALS: Final[Mapping[str, str]] = {
    "one": "first",
    "two": "second",
    "three": "third",
    "five": "fifth",
    "eight": "eighth",
    "nine": "ninth",
    "twelve": "twelfth",
    **{tens: f"{tens.removesuffix('y')}ieth" for tens in _TENS},
}
_ORDINAL_SUFFIX: Final = "th"
_MINUS: Final = "minus"
_POINT: Final = "point"
_OH: Final = "oh"
_OCLOCK: Final = "o'clock"
_PERCENT: Final = "percent"
_AND: Final = "and"
_MONTHS: Final = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

_TEN: Final = 10
_TWENTY: Final = 20
_HUNDRED: Final = 100
_THOUSAND: Final = 1000
_LARGEST: Final = _THOUSAND ** (len(_SCALES) + 1) - 1
_LARGEST_DIGITS: Final = len(str(_LARGEST))
_ZERO_DIGIT: Final = "0"
_YEARS: Final = range(1100, 2100)
_YEAR_DIGITS: Final = 4
_PLAIN_YEARS: Final = range(2000, 2010)
_MONTH_NUMBERS: Final = range(1, len(_MONTHS) + 1)
_DAYS: Final = range(1, 32)
_HOURS: Final = range(24)
_MINUTES: Final = range(60)
_CENT_DIGITS: Final = 2
_ONE_DOLLAR: Final = "1"
_THOUSANDS_SEPARATOR: Final = ","

_HYPHEN: Final = "-"
_COMMA: Final = ","
_PERIOD: Final = "."
_REPLACEMENTS: Final = (
    ("\N{RIGHT SINGLE QUOTATION MARK}", "'"),
    ("\N{MINUS SIGN}", _HYPHEN),
    ("...", _ELLIPSIS),
    ("--", f" {_HYPHEN} "),
    ("\N{EN DASH}", f" {_HYPHEN} "),
    ("\N{EM DASH}", f" {_HYPHEN} "),
    ("&", f" {_AND} "),
)
_CORE: Final = re.compile(r"[A-Za-z0-9$-](?:.*[A-Za-z0-9%])?", re.DOTALL)
_RUN: Final = re.compile(r"(?P<digits>[0-9]+)|(?P<letters>[A-Za-z]+)|.", re.DOTALL)
_DOTTED_INITIALISM: Final = re.compile(r"[A-Za-z](?:\.[A-Za-z])+")
_AMOUNT: Final = (
    r"(?P<minus>-)?(?P<integer>[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)"
    r"(?:\.(?P<fraction>[0-9]+))?"
)


class PhonemeError(SynthiaError):
    """A pronouncing dictionary that cannot be loaded."""


class _Unit(NamedTuple):
    one: str
    many: str


_DOLLARS: Final = _Unit("dollar", "dollars")
_CENTS: Final = _Unit("cent", "cents")


class Pronunciations:
    """Every pronunciation of each word, keyed by the casefolded word."""

    def __init__(self, entries: Mapping[str, tuple[Pronunciation, ...]]) -> None:
        self._entries = entries

    @classmethod
    def parse(cls, text: str) -> Pronunciations:
        """Read a dictionary in the CMU format: ``word PH PH ...`` per line.

        A variant is written ``word(2)`` and kept after the word's first
        pronunciation, in file order. ``#`` starts a comment.

        Raises:
            ValueError: If a line has a word but no phones.
        """
        entries: dict[str, tuple[Pronunciation, ...]] = {}
        for number, line in enumerate(text.splitlines(), start=1):
            fields = line.partition(_COMMENT)[0].split()
            if not fields:
                continue
            word, *phones = fields
            if not phones:
                message = f"line {number}: {word!r} has no phones"
                raise ValueError(message)
            key = word.partition(_VARIANT)[0].casefold()
            entries[key] = (*entries.get(key, ()), tuple(map(sys.intern, phones)))
        return cls(entries)

    @classmethod
    def load(cls, directory: Path, spec: Lexicon) -> Pronunciations:
        """Load ``spec``'s installed dictionary from ``directory``.

        Raises:
            PhonemeError: If the file is missing, unreadable or not UTF-8.
            ValueError: If a line has a word but no phones.
        """
        try:
            text = (directory / spec.dictionary.name).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            message = f"cannot read the pronouncing dictionary {spec.id}: {error}"
            raise PhonemeError(message) from error
        return cls.parse(text)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def first(self, word: str) -> Pronunciation | None:
        """Return the first pronunciation of ``word``, ``None`` if it is unknown."""
        variants = self.variants(word)
        return variants[0] if variants else None

    def variants(self, word: str) -> tuple[Pronunciation, ...]:
        """Return every pronunciation of ``word`` in file order."""
        return self._entries.get(word.casefold(), ())


@dataclass(frozen=True, slots=True)
class Word:
    """A word to look up in the dictionary."""

    text: str


@dataclass(frozen=True, slots=True)
class Letters:
    """Letters to say one by one, by name."""

    text: str


@dataclass(frozen=True, slots=True)
class Mark:
    """A kept punctuation symbol."""

    symbol: str


type Token = Word | Letters | Mark
"""One piece of normalised text."""

type _Reader = Callable[[re.Match[str]], list[Token] | None]

_ABBREVIATIONS: Final[Mapping[str, tuple[Token, ...]]] = {
    "mr.": (Word("mister"),),
    "mrs.": (Word("missus"),),
    "ms.": (Word("ms"),),
    "dr.": (Word("doctor"),),
    "vs.": (Word("versus"),),
    "vs": (Word("versus"),),
    "etc.": (Word("et"), Word("cetera")),
    "e.g.": (Word("for"), Word("example")),
    "i.e.": (Word("that"), Word("is")),
    "a.m.": (Letters("am"),),
    "p.m.": (Letters("pm"),),
}


def to_misaki(pronunciation: Pronunciation) -> str:
    """Return ``pronunciation`` in misaki phonemes, each stress mark before its vowel.

    Raises:
        ValueError: If a phone is not ARPAbet, or is a vowel without a stress digit.
    """
    try:
        return "".join(_MISAKI[phone] for phone in pronunciation)
    except KeyError as error:
        message = f"not an ARPAbet phone with its stress: {error.args[0]!r}"
        raise ValueError(message) from error


def spell(word: str) -> Pronunciation:
    """Return ``word`` spelled out, letter name after letter name.

    Characters other than the letters a to z are skipped.

    Raises:
        ValueError: If ``word`` has no letter to spell.
    """
    phones = tuple(
        chain.from_iterable(_LETTER_NAMES.get(letter, ()) for letter in word.casefold())
    )
    if not phones:
        message = f"no letter to spell in {word!r}"
        raise ValueError(message)
    return phones


def _words(words: Iterable[str]) -> list[Token]:
    return [Word(word) for word in words]


def _marks(text: str) -> list[Token]:
    return [Mark(symbol) for symbol in text if symbol in PUNCTUATION]


def _cardinal(number: int) -> list[str]:
    if number < _TEN:
        return [_ONES[number]]
    if number < _TWENTY:
        return [_TEENS[number - _TEN]]
    if number < _HUNDRED:
        return [_TENS[(number - _TWENTY) // _TEN], *_cardinal_rest(number % _TEN)]
    if number < _THOUSAND:
        hundreds, rest = divmod(number, _HUNDRED)
        return [_ONES[hundreds], _HUNDRED_WORD, *_cardinal_rest(rest)]
    power = max(p for p in range(1, len(_SCALES) + 1) if number >= _THOUSAND**p)
    head, rest = divmod(number, _THOUSAND**power)
    return [*_cardinal(head), _SCALES[power - 1], *_cardinal_rest(rest)]


def _cardinal_rest(number: int) -> list[str]:
    return _cardinal(number) if number else []


def _ordinal(number: int) -> list[str]:
    *head, last = _cardinal(number)
    return [*head, _ORDINALS.get(last, f"{last}{_ORDINAL_SUFFIX}")]


def _digits(digits: str) -> list[str]:
    return [_ONES[int(digit)] for digit in digits]


def _integer(digits: str) -> list[str]:
    """Return ``digits`` as words, one by one when 0-first or too long."""
    if len(digits) > _LARGEST_DIGITS or (
        len(digits) > 1 and digits.startswith(_ZERO_DIGIT)
    ):
        return _digits(digits)
    return _cardinal(int(digits))


def _amount(minus: str | None, integer: str, fraction: str | None) -> list[str]:
    sign = [] if minus is None else [_MINUS]
    point = [] if fraction is None else [_POINT, *_digits(fraction)]
    return [*sign, *_integer(integer.replace(_THOUSANDS_SEPARATOR, "")), *point]


def _pair(number: int, *, zero: str) -> list[str]:
    """Return the minutes of a time or the last two digits of a year."""
    if number == 0:
        return [zero]
    if number < _TEN:
        return [_OH, *_cardinal(number)]
    return _cardinal(number)


def _year_words(digits: str) -> list[str]:
    year = int(digits)
    if year not in _YEARS:
        return _integer(digits)
    if year in _PLAIN_YEARS:
        return _cardinal(year)
    century, rest = divmod(year, _HUNDRED)
    return [*_cardinal(century), *_pair(rest, zero=_HUNDRED_WORD)]


def _read_date(found: re.Match[str]) -> list[Token] | None:
    month, day = int(found["month"]), int(found["day"])
    if month not in _MONTH_NUMBERS or day not in _DAYS:
        return None
    return [
        Word(_MONTHS[month - 1]),
        *_words(_ordinal(day)),
        Mark(_COMMA),
        *_words(_year_words(found["year"])),
    ]


def _read_money(found: re.Match[str]) -> list[Token]:
    minus = found["sign"] or found["minus"]
    integer, fraction = found["integer"], found["fraction"]
    if fraction is None or len(fraction) != _CENT_DIGITS:
        one = integer == _ONE_DOLLAR and fraction is None
        unit = _DOLLARS.one if one else _DOLLARS.many
        return _words([*_amount(minus, integer, fraction), unit])
    dollars = [
        *_amount(minus, integer, None),
        _DOLLARS.one if integer == _ONE_DOLLAR else _DOLLARS.many,
    ]
    cents = int(fraction)
    if cents == 0:
        return _words(dollars)
    return _words(
        [*dollars, *_cardinal(cents), _CENTS.one if cents == 1 else _CENTS.many]
    )


def _read_percent(found: re.Match[str]) -> list[Token]:
    return _words(
        [*_amount(found["minus"], found["integer"], found["fraction"]), _PERCENT]
    )


def _read_time(found: re.Match[str]) -> list[Token] | None:
    hour, minute = int(found["hour"]), int(found["minute"])
    if hour not in _HOURS or minute not in _MINUTES:
        return None
    return _words([*_cardinal(hour), *_pair(minute, zero=_OCLOCK)])


def _read_ordinal(found: re.Match[str]) -> list[Token] | None:
    digits = found["integer"]
    if len(digits) > _LARGEST_DIGITS:
        return None
    return _words(_ordinal(int(digits)))


def _read_year(found: re.Match[str]) -> list[Token]:
    return _words(_year_words(found[0]))


def _read_decade(found: re.Match[str]) -> list[Token]:
    digits = found["digits"]
    *head, last = (
        _year_words(digits) if len(digits) == _YEAR_DIGITS else _integer(digits)
    )
    plural = f"{last.removesuffix('y')}ies" if last.endswith("y") else f"{last}s"
    return _words([*head, plural])


def _read_initialism(found: re.Match[str]) -> list[Token]:
    return [Letters(found[0].replace(_PERIOD, ""))]


def _read_number(found: re.Match[str]) -> list[Token]:
    return _words(_amount(found["minus"], found["integer"], found["fraction"]))


def _read_word(found: re.Match[str]) -> list[Token]:
    return [Word(found[0])]


_READERS: Final[tuple[tuple[re.Pattern[str], _Reader], ...]] = (
    (
        re.compile(r"(?P<year>[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})"),
        _read_date,
    ),
    (re.compile(rf"(?P<sign>-)?\${_AMOUNT}"), _read_money),
    (re.compile(rf"{_AMOUNT}%"), _read_percent),
    (re.compile(r"(?P<hour>[0-9]{1,2}):(?P<minute>[0-9]{2})"), _read_time),
    (re.compile(r"(?P<integer>[0-9]+)(?:st|nd|rd|th)", re.IGNORECASE), _read_ordinal),
    (re.compile(r"[0-9]{4}"), _read_year),
    (re.compile(r"(?P<digits>[0-9]{2}|[0-9]{4})'?s"), _read_decade),
    (_DOTTED_INITIALISM, _read_initialism),
    (re.compile(_AMOUNT), _read_number),
    (re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*"), _read_word),
)
"""Whole-token forms, tried in order; a reader returns ``None`` to pass."""


def _mixed(core: str) -> list[Token]:
    """Return a token of letters, digits and symbols, read run by run."""
    tokens: list[Token] = []
    for run in _RUN.finditer(core):
        match run.lastgroup:
            case "digits":
                tokens += _words(_integer(run[0]))
            case "letters":
                tokens.append(Letters(run[0]) if len(run[0]) == 1 else Word(run[0]))
            case _:
                tokens += _marks(run[0])
    return tokens


def _core(core: str) -> list[Token]:
    for pattern, read in _READERS:
        found = pattern.fullmatch(core)
        if found is not None and (tokens := read(found)) is not None:
            return tokens
    if _HYPHEN in core:
        return list(chain.from_iterable(_core(part) for part in core.split(_HYPHEN)))
    return _mixed(core)


def _chunk(chunk: str) -> list[Token]:
    """Return the tokens of one whitespace-separated piece of text."""
    if chunk == _HYPHEN:
        return [Mark(_COMMA)]
    found = _CORE.search(chunk)
    if found is None:
        return _marks(chunk)
    lead, core, trail = chunk[: found.start()], found[0], chunk[found.end() :]
    with_period = (
        _ABBREVIATIONS.get(f"{core}{_PERIOD}".casefold())
        if trail.startswith(_PERIOD)
        else None
    )
    if with_period is not None:
        return [*_marks(lead), *with_period, *_marks(trail.removeprefix(_PERIOD))]
    if _DOTTED_INITIALISM.fullmatch(core):
        trail = trail.removeprefix(_PERIOD)
    bare = _ABBREVIATIONS.get(core.casefold())
    return [*_marks(lead), *(_core(core) if bare is None else bare), *_marks(trail)]


def normalise(text: str) -> list[Token]:
    """Return ``text`` as words to look up, letters to spell and kept punctuation.

    Numbers, years, times, dates, money and percentages become words; accents
    are dropped; quotes, brackets and other symbols are dropped.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    folded = "".join(c for c in decomposed if not unicodedata.combining(c))
    for old, new in _REPLACEMENTS:
        folded = folded.replace(old, new)
    return list(chain.from_iterable(_chunk(chunk) for chunk in folded.split()))


class Phonemizer:
    """Text to misaki phonemes, through the first pronunciation of each word.

    A word of two or three capitals is spelled, as an initialism; a longer one
    is looked up and spelled when missing. A word ending in 's the dictionary
    lacks is its stem with the ending the stem's last sound takes. Any other
    unknown word goes to ``guess``, casefolded.
    """

    def __init__(self, pronunciations: Pronunciations, guess: Guess = spell) -> None:
        self._pronunciations = pronunciations
        self._guess = guess

    def phonemes(self, text: str) -> str:
        """Return ``text`` as misaki phonemes: words one space apart, marks attached.

        Raises:
            ValueError: If the dictionary or the guess gives a phone misaki lacks.
        """
        pieces: list[str] = []
        for token in normalise(text):
            if isinstance(token, Mark):
                if pieces:
                    pieces.append(token.symbol)
                continue
            if pieces:
                pieces.append(" ")
            pieces.append(to_misaki(self._pronounce(token)))
        return "".join(pieces)

    def _pronounce(self, token: Word | Letters) -> Pronunciation:
        word = token.text
        if isinstance(token, Letters) or _initialism(word):
            return spell(word)
        known = self._pronunciations.first(word)
        if known is None:
            known = self._possessive(word)
        if known is not None:
            return known
        return spell(word) if word.isupper() else self._guess(word.casefold())

    def _possessive(self, word: str) -> Pronunciation | None:
        if not word.casefold().endswith(_POSSESSIVE):
            return None
        stem = word[: -len(_POSSESSIVE)]
        known = spell(stem) if _initialism(stem) else self._pronunciations.first(stem)
        if known is None:
            return None
        return (*known, *_POSSESSIVE_ENDINGS.get(known[-1], _VOICED_ENDING))


def _initialism(word: str) -> bool:
    return word.isalpha() and word.isupper() and len(word) in _SPELLED_CAPITALS
