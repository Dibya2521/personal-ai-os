"""The Porter stemmer, rule for rule as SQLite's FTS5 ``porter`` tokenizer runs it.

Martin Porter's 1980 algorithm strips English suffixes in five steps, so
"connected", "connecting" and "connection" all become "connect". FTS5 runs it
on each token's UTF-8 bytes, skips tokens shorter than 3 or longer than 64
bytes, and uses the algorithm's later rules ("bli" to "ble", "logi" to
"log"). This follows it byte for byte, so a word stems here exactly as it
does in the memory store's full-text index; any byte outside a to z counts as
a consonant, as it does there.

A suffix counts only when the word is longer than it. Each step tries its
suffixes in order and stops at the first one the word ends with, whether or
not that rule's condition then holds.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Final

MIN_STEMMED_BYTES: Final = 3
MAX_STEMMED_BYTES: Final = 64
_VOWELS: Final = frozenset(b"aeiou")
_Y: Final = ord("y")

_STEP2: Final = (
    (b"ational", b"ate"),
    (b"tional", b"tion"),
    (b"enci", b"ence"),
    (b"anci", b"ance"),
    (b"izer", b"ize"),
    (b"logi", b"log"),
    (b"bli", b"ble"),
    (b"alli", b"al"),
    (b"entli", b"ent"),
    (b"eli", b"e"),
    (b"ousli", b"ous"),
    (b"ization", b"ize"),
    (b"ation", b"ate"),
    (b"ator", b"ate"),
    (b"alism", b"al"),
    (b"iveness", b"ive"),
    (b"fulness", b"ful"),
    (b"ousness", b"ous"),
    (b"aliti", b"al"),
    (b"iviti", b"ive"),
    (b"biliti", b"ble"),
)
_STEP3: Final = (
    (b"ical", b"ic"),
    (b"ness", b""),
    (b"icate", b"ic"),
    (b"iciti", b"ic"),
    (b"ful", b""),
    (b"ative", b""),
    (b"alize", b"al"),
)
_STEP4: Final = (
    b"al",
    b"ance",
    b"ence",
    b"er",
    b"ic",
    b"able",
    b"ible",
    b"ant",
    b"ement",
    b"ment",
    b"ent",
    b"ion",
    b"ou",
    b"ism",
    b"ate",
    b"iti",
    b"ous",
    b"ive",
    b"ize",
)


def _vowel(byte: int, *, after_consonant: bool) -> bool:
    return byte in _VOWELS or (after_consonant and byte == _Y)


def _consonants(stem: bytes) -> list[bool]:
    marks: list[bool] = []
    consonant = False
    for byte in stem:
        consonant = not _vowel(byte, after_consonant=consonant)
        marks.append(consonant)
    return marks


def _ends(word: bytes, suffix: bytes) -> bool:
    return len(word) > len(suffix) and word.endswith(suffix)


def measure(stem: bytes) -> int:
    """Return Porter's m: how many vowel-then-consonant runs ``stem`` holds."""
    return sum(not a and b for a, b in pairwise(_consonants(stem)))


def has_vowel(stem: bytes) -> bool:
    """Return whether ``stem`` holds a vowel; a "y" counts unless it comes first."""
    return any(_vowel(byte, after_consonant=i > 0) for i, byte in enumerate(stem))


def ends_cvc(stem: bytes) -> bool:
    """Return Porter's *o: consonant, vowel, consonant, the last not w, x or y."""
    if not stem or stem[-1] in b"wxy":
        return False
    return _consonants(stem)[-3:] == [True, False, True]


def _step1a(word: bytes) -> bytes:
    if not word.endswith(b"s"):
        return word
    if word.endswith(b"es"):
        return word[:-2] if _ends(word, b"sses") or _ends(word, b"ies") else word[:-1]
    return word if word.endswith(b"ss") else word[:-1]


def _step1b(word: bytes) -> bytes:
    if _ends(word, b"eed"):
        return word[:-1] if measure(word[:-3]) > 0 else word
    for suffix in (b"ed", b"ing"):
        if _ends(word, suffix):
            stem = word[: -len(suffix)]
            return _step1b_tidy(stem) if has_vowel(stem) else word
    return word


def _step1b_tidy(stem: bytes) -> bytes:
    if any(_ends(stem, suffix) for suffix in (b"at", b"bl", b"iz")):
        return stem + b"e"
    last = stem[-1]
    doubled = len(stem) > 1 and last == stem[-2]
    if doubled and not _vowel(last, after_consonant=False) and last not in b"lsz":
        return stem[:-1]
    if measure(stem) == 1 and ends_cvc(stem):
        return stem + b"e"
    return stem


def _step1c(word: bytes) -> bytes:
    if word.endswith(b"y") and has_vowel(word[:-1]):
        return word[:-1] + b"i"
    return word


def _replace(word: bytes, rules: tuple[tuple[bytes, bytes], ...]) -> bytes:
    for suffix, replacement in rules:
        if _ends(word, suffix):
            stem = word[: -len(suffix)]
            return stem + replacement if measure(stem) > 0 else word
    return word


def _step4(word: bytes) -> bytes:
    for suffix in _STEP4:
        if _ends(word, suffix):
            stem = word[: -len(suffix)]
            keep = suffix == b"ion" and not stem.endswith((b"s", b"t"))
            return word if keep or measure(stem) <= 1 else stem
    return word


def _step5(word: bytes) -> bytes:
    if word.endswith(b"e"):
        stem = word[:-1]
        m = measure(stem)
        if m > 1 or (m == 1 and not ends_cvc(stem)):
            word = stem
    if _ends(word, b"ll") and measure(word[:-1]) > 1:
        word = word[:-1]
    return word


def stem(token: bytes) -> bytes:
    """Return the Porter stem of ``token``, lower-case UTF-8 bytes."""
    if not MIN_STEMMED_BYTES <= len(token) <= MAX_STEMMED_BYTES:
        return token
    word = _step1c(_step1b(_step1a(token)))
    word = _replace(_replace(word, _STEP2), _STEP3)
    return _step5(_step4(word))
