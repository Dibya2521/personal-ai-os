r"""A file's text as lines, what a reader says about it, and which reader reads it.

The contract between readers and the chunker is :class:`Outline`: line spans
and heading trails, never text. A reader is a pure function of the decoded
file; adding a format is one row in the table :func:`format_of` reads, and
every chunking then works on it unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Final

from synthia.knowledge.model import DocumentKind, Link

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import PurePosixPath

SNIFF_BYTES: Final = 8192
"""How far a NUL byte marks a file as binary, the sniff ``tools/files.py`` uses."""
BYTE_ORDER_MARK: Final = "\N{ZERO WIDTH NO-BREAK SPACE}"


class UnreadableError(ValueError):
    """A file that is not text a reader can take: binary, or not UTF-8."""


@dataclass(frozen=True, slots=True)
class Text:
    r"""A decoded file: one tuple of lines per page (one page unless a PDF).

    Lines are split at ``\n`` only, with a trailing ``\r`` dropped, so line
    numbers are an editor's. A UTF-8 byte order mark is not part of line 1.
    """

    pages: tuple[tuple[str, ...], ...]

    @classmethod
    def of_bytes(cls, data: bytes) -> Text:
        """Decode a text file strictly.

        Raises:
            UnreadableError: If the first 8,192 bytes hold a NUL, or the bytes
                are not UTF-8.
        """
        nul = data.find(b"\0", 0, SNIFF_BYTES)
        if nul >= 0:
            message = f"a NUL byte at offset {nul}: not a text file"
            raise UnreadableError(message)
        try:
            decoded = data.decode("utf-8")
        except UnicodeDecodeError as error:
            message = f"not UTF-8: {error.reason} at byte {error.start}"
            raise UnreadableError(message) from error
        lines = decoded.removeprefix(BYTE_ORDER_MARK).split("\n")
        return cls((tuple(line.removesuffix("\r") for line in lines),))

    def lines(self, page: int | None = None) -> tuple[str, ...]:
        """Return the lines of ``page`` (1-based), or of the only page when None.

        Raises:
            ValueError: If ``page`` is None and there are several pages.
            IndexError: If there is no such page.
        """
        if page is None:
            if len(self.pages) != 1:
                message = f"name a page: the text has {len(self.pages)}"
                raise ValueError(message)
            return self.pages[0]
        if not 1 <= page <= len(self.pages):
            message = f"no page {page} in {len(self.pages)}"
            raise IndexError(message)
        return self.pages[page - 1]

    def span(self, page: int | None, start: int, end: int) -> str:
        r"""Return lines ``start..end`` of ``page``, inclusive, joined by ``\n``.

        The one place a chunk's body is made; nothing else builds body text.

        Raises:
            IndexError: If the lines are not ``1 <= start <= end <= len``.
        """
        lines = self.lines(page)
        if not 1 <= start <= end <= len(lines):
            message = f"lines {start}..{end} are not within 1..{len(lines)}"
            raise IndexError(message)
        return "\n".join(lines[start - 1 : end])


@dataclass(frozen=True, slots=True)
class Block:
    """A run of lines under one heading trail, in one page.

    Blocks of one :class:`Outline` are in order, never overlap, never cross a
    page, and together hold every non-blank line. ``unit`` groups consecutive
    blocks the chunker keeps together when they fit as a whole (a Python class
    and its methods); a block alone in its unit is its own.
    """

    page: int | None
    start: int
    end: int
    trail: tuple[str, ...]
    unit: int


@dataclass(frozen=True, slots=True)
class FoundLink:
    """A link and the line it is on, so it joins the chunk holding that line."""

    link: Link
    page: int | None
    line: int


@dataclass(frozen=True, slots=True)
class Outline:
    """What a reader found in a document: title, blocks and links."""

    title: str
    blocks: tuple[Block, ...]
    links: tuple[FoundLink, ...] = ()
    note: str = ""


type Reader = Callable[[Text, PurePosixPath], Outline]
"""Reads a decoded file, and its path for the title."""
type Decoder = Callable[[bytes], Text]


@dataclass(frozen=True, slots=True)
class Format:
    """How files of one kind are decoded and read."""

    kind: DocumentKind
    decode: Decoder
    read: Reader


@cache
def _formats() -> Mapping[str, Format]:
    # The readers import this module, so the table is built on first use.
    from synthia.knowledge.markdown import read_markdown  # noqa: PLC0415

    markdown = Format(DocumentKind.MARKDOWN, Text.of_bytes, read_markdown)
    return {".md": markdown, ".markdown": markdown}


def format_of(path: PurePosixPath) -> Format | None:
    """Return the format that reads ``path``, or None to skip the file.

    A suffix with no format is skipped unread, so a vault's images and
    attachments are never opened.
    """
    return _formats().get(path.suffix.casefold())
