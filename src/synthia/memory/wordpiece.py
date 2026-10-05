"""BERT's WordPiece tokenizer: text to the token ids an embedding model reads.

It follows the rules a ``tokenizer.json`` of the BERT kind names, in the
order the Hugging Face tokenizers library applies them:

1. Clean: drop control and format characters, U+0000 and U+FFFD; every kind
   of whitespace becomes a space.
2. Put spaces around CJK ideographs, so each is a word of its own.
3. Strip accents (NFD, then drop non-spacing marks) and lower-case.
4. Split on whitespace, and make every punctuation character a word.
5. Split each word greedily into the longest pieces in the vocabulary, all
   but the first written with a ``##`` prefix; a word with no such split,
   or longer than 100 characters, becomes ``[UNK]``.
6. Wrap in ``[CLS]`` ... ``[SEP]``, cut to the model's limit.

One difference on purpose: text that spells a special token ("[SEP]") is
read as text, never as the token, so nothing typed can steer the encoder.
"""

from __future__ import annotations

import json
import string
import unicodedata
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

UNKNOWN: Final = "[UNK]"
START: Final = "[CLS]"
END: Final = "[SEP]"
PREFIX: Final = "##"
MAX_WORD_CHARS: Final = 100
_CJK: Final = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0xF900, 0xFAFF),
    (0x2F800, 0x2FA1F),
)
_KEEP_CONTROLS: Final = frozenset("\t\n\r")
_DROPPED: Final = frozenset(("Cc", "Cf", "Cn", "Co"))
_DROPPED_CHARS: Final = frozenset("\0\N{REPLACEMENT CHARACTER}")
_EXPECTED: Final = {
    ("normalizer", "type"): "BertNormalizer",
    ("normalizer", "clean_text"): True,
    ("normalizer", "handle_chinese_chars"): True,
    ("normalizer", "lowercase"): True,
    ("pre_tokenizer", "type"): "BertPreTokenizer",
    ("model", "type"): "WordPiece",
    ("model", "unk_token"): UNKNOWN,
    ("model", "continuing_subword_prefix"): PREFIX,
    ("model", "max_input_chars_per_word"): MAX_WORD_CHARS,
}


class TokenizerError(ValueError):
    """A tokenizer file this WordPiece does not reproduce."""


def _cjk(char: str) -> bool:
    point = ord(char)
    return any(low <= point <= high for low, high in _CJK)


def _punctuation(char: str) -> bool:
    return char in string.punctuation or unicodedata.category(char).startswith("P")


def normalized(text: str) -> str:
    """Return ``text`` cleaned, CJK spaced, accents stripped and lower-cased."""
    kept: list[str] = []
    for char in text:
        # Tab and newlines are controls too, but count as whitespace; other
        # controls go even when they are whitespace (U+0085).
        if char in _KEEP_CONTROLS:
            kept.append(" ")
        elif char in _DROPPED_CHARS or unicodedata.category(char) in _DROPPED:
            continue
        elif char.isspace():
            kept.append(" ")
        elif _cjk(char):
            kept.append(f" {char} ")
        else:
            kept.append(char)
    split = unicodedata.normalize("NFD", "".join(kept))
    # Character by character: str.lower() would apply the Greek final-sigma rule.
    return "".join(c.lower() for c in split if unicodedata.category(c) != "Mn")


def words(text: str) -> list[str]:
    """Return the words of normalised ``text``; each punctuation mark is one."""
    found: list[str] = []
    current: list[str] = []
    for char in text:
        if char.isspace() or _punctuation(char):
            if current:
                found.append("".join(current))
                current = []
            if not char.isspace():
                found.append(char)
        else:
            current.append(char)
    if current:
        found.append("".join(current))
    return found


class WordPiece:
    """A WordPiece vocabulary, and the rules that turn text into its ids."""

    def __init__(self, vocabulary: Mapping[str, int]) -> None:
        """Use ``vocabulary``, which must hold the four special tokens.

        Raises:
            TokenizerError: If a special token is missing.
        """
        missing = [t for t in (UNKNOWN, START, END) if t not in vocabulary]
        if missing:
            message = f"the vocabulary lacks {', '.join(missing)}"
            raise TokenizerError(message)
        self._vocabulary = vocabulary
        self._unknown = vocabulary[UNKNOWN]

    @classmethod
    def from_file(cls, path: Path) -> WordPiece:
        """Read a ``tokenizer.json`` that uses exactly BERT's rules.

        Raises:
            TokenizerError: If the file names other rules, or is not one.
            OSError: If it cannot be read.
        """
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
            for (part, key), value in _EXPECTED.items():
                if spec[part][key] != value:
                    message = f"{path}: {part}.{key} is {spec[part][key]!r}"
                    raise TokenizerError(message)
            accents = spec["normalizer"]["strip_accents"]
            vocabulary = spec["model"]["vocab"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            message = f"{path} is not a WordPiece tokenizer file"
            raise TokenizerError(message) from error
        if accents not in (None, True):
            message = f"{path}: accents are kept, which these rules do not do"
            raise TokenizerError(message)
        return cls(vocabulary)

    def pieces(self, word: str) -> list[int]:
        """Return the ids of ``word`` split into the longest known pieces."""
        if len(word) > MAX_WORD_CHARS:
            return [self._unknown]
        found: list[int] = []
        start = 0
        while start < len(word):
            for end in range(len(word), start, -1):
                piece = word[start:end] if start == 0 else PREFIX + word[start:end]
                id_ = self._vocabulary.get(piece)
                if id_ is not None:
                    found.append(id_)
                    start = end
                    break
            else:
                return [self._unknown]
        return found

    def ids(self, text: str, max_tokens: int) -> list[int]:
        """Return the ids of ``text`` inside the start and end tokens, cut to fit."""
        inner: list[int] = []
        room = max_tokens - 2
        for word in words(normalized(text)):
            inner.extend(self.pieces(word))
            if len(inner) >= room:
                break
        return [self._vocabulary[START], *inner[:room], self._vocabulary[END]]
