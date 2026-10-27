"""PDF files as text: the lines pypdf extracts, one page of lines per PDF page."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from synthia.knowledge.text import NEWLINE, Block, Outline, Text, UnreadableError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import PurePosixPath

    from pypdf import PdfReader


def pdf_text(data: bytes) -> Text:
    """Decode a PDF into the text of each of its pages.

    Raises:
        UnreadableError: If the file is encrypted, even when an empty password
            opens it, or pypdf cannot parse it.
    """
    # pypdf takes longer to import than the indexer, so only a PDF pays for it.
    from pypdf import PdfReader  # noqa: PLC0415
    from pypdf.errors import PdfReadError  # noqa: PLC0415

    try:
        return Text(_pages(PdfReader(io.BytesIO(data))))
    except PdfReadError as error:
        message = f"not a PDF pypdf can read: {error}"
        raise UnreadableError(message) from error


def _pages(reader: PdfReader) -> tuple[tuple[str, ...], ...]:
    if reader.is_encrypted:
        message = "encrypted: not read"
        raise UnreadableError(message)
    return tuple(tuple(NEWLINE.split(page.extract_text())) for page in reader.pages)


def read_pdf(text: Text, path: PurePosixPath) -> Outline:
    """Return each page holding text as one block, under the file name.

    Pages are numbered from 1 even in a one-page PDF, so every citation names
    its page. The note names the pages with no text: pypdf reads none from a
    scanned page, which is an image.
    """
    written: list[int] = []
    blank: list[int] = []
    for number in range(1, len(text.pages) + 1):
        holds_text = any(line.strip() for line in text.lines(number))
        (written if holds_text else blank).append(number)
    blocks = tuple(
        Block(page, 1, len(text.lines(page)), (), unit)
        for unit, page in enumerate(written)
    )
    return Outline(path.name, blocks, note=_blank_note(blank))


def _blank_note(pages: Sequence[int]) -> str:
    if not pages:
        return ""
    if len(pages) == 1:
        return f"page {pages[0]} holds no text (scanned or images only)"
    numbers = ", ".join(map(str, pages))
    return f"pages {numbers} hold no text (scanned or images only)"
