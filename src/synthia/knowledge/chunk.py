"""An outline cut into chunks that fit the embedder.

Readers know formats; this module alone knows tokens. Every chunking the
eval scores is a value of :class:`Chunking`, applied to any reader's outline,
so a new format gets every chunking for free.

Each line is counted once with the embedder's own tokenizer, and a span
costs the sum of its lines plus its header. That sum is the count of the
text the embedder reads, because the tokenizer splits on whitespace and a
line break is whitespace; packing is then arithmetic on prefix sums.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from itertools import accumulate, groupby
from typing import TYPE_CHECKING, Final, Protocol

from synthia.knowledge.model import Draft, Link

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    from synthia.knowledge.text import Block, Outline, Text
    from synthia.models.catalogue import Embedder

READER_VERSION: Final = 1
"""Bumped by any change to a reader or to this module that moves chunk
boundaries. It is part of every recipe, so the next index run reads every
document again instead of keeping chunks cut the old way."""
HEADER_SEPARATOR: Final = " > "
WINDOW_LINES: Final = 40
FRAME_TOKENS: Final = 2
"""The start and end tokens the embedder wraps every text in."""
HEADER_SHARE: Final = 4
"""A header takes at most a quarter of the budget, so a chunk is mostly text:
a longer one drops its middle headings, then the title, then goes."""


class TokenCounter(Protocol):
    """Counts the tokens of a text, without the start and end tokens."""

    def count(self, text: str) -> int:
        """Return how many pieces ``text`` splits into, uncut."""
        ...


class Splitter(Enum):
    """Where chunk boundaries come from."""

    STRUCTURE = auto()
    """Units, then blocks, then blank lines, then lines."""
    LINES = auto()
    """Windows of whole lines, ignoring structure: the default."""


@dataclass(frozen=True, slots=True)
class Chunking:
    """How documents are cut: one value the index records and the eval varies.

    Raises:
        ValueError: If ``max_tokens`` leaves no room inside the start and end
            tokens, or ``window_lines`` is not positive.
    """

    max_tokens: int
    splitter: Splitter = Splitter.STRUCTURE
    header: bool = True
    window_lines: int = WINDOW_LINES

    def __post_init__(self) -> None:
        if self.max_tokens <= FRAME_TOKENS:
            message = f"max_tokens={self.max_tokens} leaves no room for text"
            raise ValueError(message)
        if self.window_lines < 1:
            message = f"window_lines={self.window_lines} is not positive"
            raise ValueError(message)

    @classmethod
    def for_embedder(cls, spec: Embedder) -> Chunking:
        """Return the default chunking at ``spec``'s token limit.

        Windows of :data:`WINDOW_LINES` lines without a header, the best
        measured on 61 questions about this repository at commit ffd814e (49
        answerable, ``benchmarks/rag_eval.py``). With the fused score at word
        weight 0.2 they reach nDCG@10 0.461, against 0.390 for the earlier
        default of sections with a header at weight 0.1: higher in 98.1 % of
        2,000 paired bootstrap resamples over the questions, by 0.015 to 0.128
        (90 % interval). Over sections with a header at the same weight 0.2
        the win is not settled (76.6 % of resamples); windows are the simpler
        cut, so they stay the default until a better one is measured. A
        likely reason they win: sections are never packed with their small
        neighbours, so at a 256-token limit the same 261 files gave 3,906
        section chunks against 2,085 windows, about half the size.
        """
        return cls(spec.max_tokens, Splitter.LINES, header=False)

    @property
    def budget(self) -> int:
        """Tokens a chunk's header and body may hold together."""
        return self.max_tokens - FRAME_TOKENS

    def recipe(self) -> str:
        """Return a stable name for this chunking and :data:`READER_VERSION`.

        Stored on every document, as in ``structure+header/256/40/r1``.
        """
        header = "+header" if self.header else ""
        splitter = self.splitter.name.lower()
        return (
            f"{splitter}{header}/{self.max_tokens}/{self.window_lines}"
            f"/r{READER_VERSION}"
        )


@dataclass(frozen=True, slots=True)
class Drafted:
    """A document's drafts in line order."""

    drafts: tuple[Draft, ...]

    @property
    def cut(self) -> int:
        """How many drafts are a single line longer than the budget."""
        return sum(draft.cut for draft in self.drafts)


@dataclass(frozen=True, slots=True)
class _Page:
    number: int | None
    before: tuple[int, ...]
    """``before[n]`` is the tokens of lines 1..n."""
    blank: tuple[bool, ...]

    @classmethod
    def of(
        cls, number: int | None, lines: Sequence[str], counter: TokenCounter
    ) -> _Page:
        counts = (counter.count(line) for line in lines)
        blank = tuple(not line.strip() for line in lines)
        return cls(number, tuple(accumulate(counts, initial=0)), blank)

    @property
    def size(self) -> int:
        return len(self.blank)

    def tokens(self, start: int, end: int) -> int:
        return self.before[end] - self.before[start - 1]

    def is_blank(self, line: int) -> bool:
        return self.blank[line - 1]

    def trimmed(self, start: int, end: int) -> tuple[int, int] | None:
        while start <= end and self.is_blank(start):
            start += 1
        while end >= start and self.is_blank(end):
            end -= 1
        return (start, end) if start <= end else None

    def paragraphs(self, start: int, end: int) -> Iterator[tuple[int, int]]:
        first: int | None = None
        for line in range(start, end + 1):
            if not self.is_blank(line):
                first = line if first is None else first
            elif first is not None:
                yield first, line - 1
                first = None
        if first is not None:
            yield first, end


@dataclass(frozen=True, slots=True)
class _Frame:
    """A page and the header every span cut from it under one trail carries."""

    page: _Page
    header: str
    room: int

    def fits(self, start: int, end: int) -> bool:
        return self.page.tokens(start, end) <= self.room


@dataclass(frozen=True, slots=True)
class _Span:
    frame: _Frame
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _Frames:
    title: str
    pages: Mapping[int | None, _Page]
    counter: TokenCounter
    chunking: Chunking

    def of(self, page: int | None, trail: tuple[str, ...]) -> _Frame:
        header = self._header(trail) if self.chunking.header else ""
        room = self.chunking.budget - self.counter.count(header)
        return _Frame(self.pages[page], header, room)

    def _header(self, trail: tuple[str, ...]) -> str:
        """Return the fullest header within a share of the budget, else ""."""
        limit = self.chunking.budget // HEADER_SHARE
        nearest = trail[-1:]
        for parts in ((self.title, *trail), (self.title, *nearest), nearest):
            header = _joined(parts)
            if self.counter.count(header) <= limit:
                return header
        return ""


def _joined(parts: Iterable[str]) -> str:
    # A note's H1 is usually its title too; it is said once.
    return HEADER_SEPARATOR.join(part for part, _ in groupby(p for p in parts if p))


def _atoms(frame: _Frame, start: int, end: int) -> Iterator[tuple[int, int]]:
    """Yield the paragraphs that fit, and each line of those that do not."""
    for first, last in frame.page.paragraphs(start, end):
        if frame.fits(first, last):
            yield first, last
        else:
            yield from ((line, line) for line in range(first, last + 1))


def _packed(frame: _Frame, atoms: Iterable[tuple[int, int]]) -> Iterator[_Span]:
    """Join consecutive atoms greedily while the joined span fits."""
    span: _Span | None = None
    for first, last in atoms:
        if span is not None and frame.fits(span.start, last):
            span = _Span(frame, span.start, last)
            continue
        if span is not None:
            yield span
        span = _Span(frame, first, last)
    if span is not None:
        yield span


def _unit(frames: _Frames, blocks: Sequence[Block]) -> Iterator[_Span]:
    first, last = blocks[0], blocks[-1]
    if len(blocks) > 1 and first.page == last.page:
        frame = frames.of(first.page, first.trail)
        whole = frame.page.trimmed(first.start, last.end)
        if whole is not None and frame.fits(*whole):
            yield _Span(frame, *whole)
            return
    for block in blocks:
        frame = frames.of(block.page, block.trail)
        yield from _packed(frame, _atoms(frame, block.start, block.end))


def _by_structure(frames: _Frames, outline: Outline) -> Iterator[_Span]:
    for _, unit in groupby(outline.blocks, key=lambda block: block.unit):
        yield from _unit(frames, tuple(unit))


def _windows(frame: _Frame, width: int) -> Iterator[_Span]:
    page = frame.page
    line = 1
    while line <= page.size:
        window = page.trimmed(line, min(line + width - 1, page.size))
        if window is None:
            line += width
            continue
        start, end = window
        while end > start and (page.is_blank(end) or not frame.fits(start, end)):
            end -= 1
        yield _Span(frame, start, end)
        line = end + 1


def _by_lines(frames: _Frames, _outline: Outline) -> Iterator[_Span]:
    for number in frames.pages:
        yield from _windows(frames.of(number, ()), frames.chunking.window_lines)


_SPLITTERS: Final[Mapping[Splitter, Callable[[_Frames, Outline], Iterator[_Span]]]] = {
    Splitter.STRUCTURE: _by_structure,
    Splitter.LINES: _by_lines,
}


def _links_by_draft(spans: Sequence[_Span], outline: Outline) -> dict[int, list[Link]]:
    owner: dict[tuple[int | None, int], int] = {}
    for ordinal, span in enumerate(spans):
        for line in range(span.start, span.end + 1):
            owner[span.frame.page.number, line] = ordinal
    links: dict[int, list[Link]] = defaultdict(list)
    for found in outline.links:
        ordinal = owner.get((found.page, found.line))
        if ordinal is not None:
            links[ordinal].append(found.link)
    return links


def chunked(
    text: Text, outline: Outline, counter: TokenCounter, chunking: Chunking
) -> Drafted:
    """Cut ``outline`` of ``text`` into drafts that fit ``chunking``.

    Each draft's body is ``text.span(page, start_line, end_line)``, with no
    blank line at either end; drafts are in line order, never overlap, never
    cross a page, and together hold every non-blank line once. A draft's
    header and body fit the budget together, except a single line that does
    not fit alone: it is a draft of its own, marked ``cut``. Each link joins
    the draft holding its line.

    Structure keeps a unit of blocks whole when it fits, else each block whole
    when it fits, else packs the block's paragraphs greedily, splitting a
    paragraph that does not fit into its lines. Lines takes windows of
    ``window_lines`` lines, each shrunk from its end until it fits.
    """
    # Blocks hold every non-blank line, so the pages they name are all the
    # pages with text.
    pages = {
        number: _Page.of(number, text.lines(number), counter)
        for number in dict.fromkeys(block.page for block in outline.blocks)
    }
    frames = _Frames(outline.title, pages, counter, chunking)
    spans = tuple(_SPLITTERS[chunking.splitter](frames, outline))
    links = _links_by_draft(spans, outline)
    return Drafted(
        tuple(
            Draft(
                ordinal,
                span.frame.page.number,
                span.start,
                span.end,
                span.frame.header,
                text.span(span.frame.page.number, span.start, span.end),
                tuple(links[ordinal]),
                cut=not span.frame.fits(span.start, span.end),
            )
            for ordinal, span in enumerate(spans)
        )
    )
