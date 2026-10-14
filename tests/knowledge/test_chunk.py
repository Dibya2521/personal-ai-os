import shutil
import subprocess
from pathlib import Path, PurePosixPath

import pytest
from hypothesis import given
from hypothesis import strategies as st

from synthia.knowledge import chunk
from synthia.knowledge.chunk import (
    HEADER_SEPARATOR,
    Chunking,
    Drafted,
    Splitter,
    TokenCounter,
    chunked,
)
from synthia.knowledge.markdown import read_markdown
from synthia.knowledge.model import Chunk, ChunkId, DocumentId, Draft, Link, LinkKind
from synthia.knowledge.text import Block, FoundLink, Outline, Text
from synthia.retrieval.wordpiece import WordPiece, normalized, words
from tests.models.fakes import EMBEDDER

ROOT = Path(__file__).parents[2]
SPECIALS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")


class Words:
    """Counts whitespace-separated words: simple enough to check by eye."""

    def count(self, text: str) -> int:
        return len(text.split())


WORDS = Words()


def outline(
    *blocks: Block, title: str = "T", links: tuple[FoundLink, ...] = ()
) -> Outline:
    return Outline(title, blocks, links)


def block(
    start: int, end: int, *trail: str, unit: int = 0, page: int | None = None
) -> Block:
    return Block(page, start, end, trail, unit)


def cut(
    lines: tuple[str, ...],
    shape: Outline,
    chunking: Chunking,
    counter: TokenCounter = WORDS,
) -> list[tuple[int, int, str, bool]]:
    drafted = chunked(Text((lines,)), shape, counter, chunking)
    return [(d.start_line, d.end_line, d.header, d.cut) for d in drafted.drafts]


# 18 words of text; with the 3-word header "T > A" the unit costs 21.
CLASS = (
    "class A has six words here",
    "def f has six words here",
    "def g has six words here",
)
CLASS_SHAPE = outline(block(1, 1, "A"), block(2, 2, "A", "f"), block(3, 3, "A", "g"))


def test_a_unit_that_fits_is_one_draft_under_its_first_blocks_header() -> None:
    assert cut(CLASS, CLASS_SHAPE, Chunking(23)) == [(1, 3, "T > A", False)]


def test_a_unit_that_does_not_fit_is_cut_at_its_blocks() -> None:
    assert cut(CLASS, CLASS_SHAPE, Chunking(22)) == [
        (1, 1, "T > A", False),
        (2, 2, "T > A > f", False),
        (3, 3, "T > A > g", False),
    ]


def test_a_unit_never_joins_blocks_on_two_pages() -> None:
    text = Text((("a",), ("b",)))
    shape = outline(block(1, 1, page=1), block(1, 1, page=2))

    drafted = chunked(text, shape, WORDS, Chunking(100, header=False))

    assert [(d.page, d.body) for d in drafted.drafts] == [(1, "a"), (2, "b")]


def test_a_block_over_budget_packs_its_paragraphs_greedily() -> None:
    lines = ("a b", "c", "", "d e f", "", "g h")

    assert cut(lines, outline(block(1, 6)), Chunking(8, header=False)) == [
        (1, 4, "", False),
        (6, 6, "", False),
    ]


def test_a_paragraph_over_budget_packs_its_lines_and_a_long_line_is_cut() -> None:
    lines = ("1 2", "3 4", "5 6 7", "8 9 10 11 12 13 14", "15")

    drafted = chunked(
        Text((lines,)), outline(block(1, 5)), WORDS, Chunking(8, header=False)
    )

    assert [(d.start_line, d.end_line, d.cut) for d in drafted.drafts] == [
        (1, 2, False),
        (3, 3, False),
        (4, 4, True),
        (5, 5, False),
    ]
    assert drafted.cut == 1


def test_blank_lines_are_trimmed_and_a_blank_block_is_no_draft() -> None:
    lines = ("", "a", "", "b", "", "", "  ")
    shape = outline(block(1, 5), block(6, 7, unit=1))

    drafted = chunked(Text((lines,)), shape, WORDS, Chunking(100, header=False))

    assert [(d.start_line, d.end_line, d.body) for d in drafted.drafts] == [
        (2, 4, "a\n\nb")
    ]


def test_a_unit_of_blank_blocks_is_no_draft() -> None:
    shape = outline(block(1, 1), block(2, 2))

    assert cut(("", ""), shape, Chunking(100)) == []


def test_the_header_counts_against_the_budget() -> None:
    lines = ("a b c d e f g h", "i j k l m n o p")
    shape = outline(block(1, 2, "Part", "Sec"), title="Note")

    assert cut(lines, shape, Chunking(22)) == [
        (1, 1, "Note > Part > Sec", False),
        (2, 2, "Note > Part > Sec", False),
    ]
    assert cut(lines, shape, Chunking(22, header=False)) == [(1, 2, "", False)]


def test_an_empty_heading_leaves_no_empty_part_in_the_header() -> None:
    shape = outline(block(1, 1, "", "Sec"), title="Note")

    assert cut(("a",), shape, Chunking(100)) == [(1, 1, "Note > Sec", False)]


@pytest.mark.parametrize(
    ("title", "trail", "header"),
    [
        ("Note", ("Part", "Sec"), "Note > Part > Sec"),
        ("Note", ("Part", "Sub", "Sec"), "Note > Sec"),
        ("A long note title", ("Part", "Sec"), "Sec"),
        ("A long note title", ("a much much longer heading here",), ""),
    ],
)
def test_a_header_over_a_quarter_of_the_budget_is_shortened(
    title: str, trail: tuple[str, ...], header: str
) -> None:
    shape = outline(block(1, 1, *trail), title=title)

    assert cut(("a",), shape, Chunking(22)) == [(1, 1, header, False)]


def test_a_header_says_a_heading_once_when_the_title_repeats_it() -> None:
    shape = outline(block(1, 1, "Note", "Sec"), title="Note")

    assert cut(("a",), shape, Chunking(100)) == [
        (1, 1, f"Note{HEADER_SEPARATOR}Sec", False)
    ]


def test_links_join_the_draft_holding_their_line() -> None:
    first, second, lost = (Link(LinkKind.WIKILINK, k, k) for k in ("x", "y", "z"))
    shape = outline(
        block(1, 1),
        block(2, 5, unit=1),
        links=(
            FoundLink(second, None, 3),
            FoundLink(first, None, 1),
            FoundLink(lost, None, 5),
        ),
    )

    drafted = chunked(
        Text((("a", "b", "", "c", ""),)), shape, WORDS, Chunking(100, header=False)
    )

    assert [(d.end_line, d.links) for d in drafted.drafts] == [
        (1, (first,)),
        (4, (second,)),
    ]


def test_lines_splitter_takes_windows_and_shrinks_them_to_fit() -> None:
    lines = ("a", "b", "", "", "", "c", "d e f g h", "i")
    shape = outline(block(1, 3, "ignored"), block(4, 8, "too", unit=1))
    chunking = Chunking(6, Splitter.LINES, window_lines=3)

    assert cut(lines, shape, chunking) == [
        (1, 2, "T", False),
        (6, 6, "T", False),
        (7, 7, "T", True),
        (8, 8, "T", False),
    ]


def test_drafts_are_numbered_in_order_with_their_bodies() -> None:
    drafted = chunked(
        Text((("a", "", "b c"),)),
        outline(block(1, 3)),
        WORDS,
        Chunking(4, header=False),
    )

    assert drafted == Drafted(
        (
            Draft(0, None, 1, 1, "", "a", ()),
            Draft(1, None, 3, 3, "", "b c", ()),
        )
    )


def test_recipe_names_every_field_and_the_reader_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variants = [
        Chunking(256),
        Chunking(512),
        Chunking(256, Splitter.LINES),
        Chunking(256, header=False),
        Chunking(256, window_lines=20),
    ]

    assert [v.recipe() for v in variants] == [
        "structure+header/256/40/r1",
        "structure+header/512/40/r1",
        "lines+header/256/40/r1",
        "structure/256/40/r1",
        "structure+header/256/20/r1",
    ]
    monkeypatch.setattr(chunk, "READER_VERSION", 2)
    assert Chunking(256).recipe() == "structure+header/256/40/r2"


def test_chunking_for_an_embedder_uses_its_token_limit() -> None:
    chunking = Chunking.for_embedder(EMBEDDER)

    assert chunking == Chunking(8)
    assert chunking.budget == 6


@pytest.mark.parametrize(
    ("max_tokens", "window_lines", "message"),
    [(2, 40, "max_tokens=2 leaves no room"), (3, 0, "window_lines=0 is not positive")],
)
def test_chunking_refuses_sizes_with_no_room(
    max_tokens: int, window_lines: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Chunking(max_tokens, window_lines=window_lines)


def tracked_markdown() -> list[PurePosixPath]:
    git = shutil.which("git")
    if git is None or not (ROOT / ".git").exists():
        pytest.skip("the repository's Markdown is listed by git")
    listed = subprocess.run(  # noqa: S603 - a resolved git with fixed arguments
        [git, "ls-files", "*.md"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return [PurePosixPath(line) for line in listed.stdout.splitlines()]


@pytest.fixture(scope="module")
def corpus() -> list[tuple[Text, Outline]]:
    documents: list[tuple[Text, Outline]] = []
    for path in tracked_markdown():
        text = Text.of_bytes((ROOT / path).read_bytes())
        documents.append((text, read_markdown(text, path)))
    return documents


@pytest.fixture(scope="module")
def wordpiece(corpus: list[tuple[Text, Outline]]) -> WordPiece:
    """Pieces as the WordPiece tests make them; long words lose their whole form."""
    pieces: set[str] = set()
    for text, _ in corpus:
        for word in words(normalized("\n".join(text.lines()))):
            pieces.update((word[:3], *word, *(f"##{c}" for c in word[1:])))
            pieces.update(f"##{word[i:]}" for i in range(1, len(word)))
            if len(word) <= 6:
                pieces.add(word)
    return WordPiece({t: i for i, t in enumerate((*SPECIALS, *sorted(pieces)))})


def assert_whole(
    text: Text, shape: Outline, drafted: Drafted, counter: TokenCounter, budget: int
) -> None:
    """Check the chunker's promises for a one-page text."""
    covered: list[int] = []
    for ordinal, draft in enumerate(drafted.drafts):
        assert draft.ordinal == ordinal
        assert draft.page is None
        assert draft.body == text.span(None, draft.start_line, draft.end_line)
        lines = draft.body.split("\n")
        assert lines[0].strip()
        assert lines[-1].strip()
        cost = counter.count(draft.header) + sum(counter.count(line) for line in lines)
        assert (cost > budget) is draft.cut
        assert not draft.cut or draft.start_line == draft.end_line
        covered.extend(range(draft.start_line, draft.end_line + 1))
    assert covered == sorted(set(covered))
    lines = text.lines()
    assert {n for n in covered if lines[n - 1].strip()} == {
        n for n, line in enumerate(lines, start=1) if line.strip()
    }
    assert sum(len(d.links) for d in drafted.drafts) == len(shape.links)


@pytest.mark.parametrize("max_tokens", [42, 502], ids=["budget 40", "budget 500"])
@pytest.mark.parametrize("splitter", list(Splitter), ids=lambda s: s.name.lower())
def test_every_tracked_markdown_file_cuts_whole_with_words(
    corpus: list[tuple[Text, Outline]], max_tokens: int, splitter: Splitter
) -> None:
    chunking = Chunking(max_tokens, splitter)
    for text, shape in corpus:
        drafted = chunked(text, shape, WORDS, chunking)
        assert_whole(text, shape, drafted, WORDS, chunking.budget)


@pytest.mark.parametrize("max_tokens", [42, 502], ids=["budget 40", "budget 500"])
def test_every_tracked_markdown_file_cuts_whole_with_wordpiece(
    corpus: list[tuple[Text, Outline]], wordpiece: WordPiece, max_tokens: int
) -> None:
    chunking = Chunking(max_tokens)
    assert len(corpus) > 10
    for text, shape in corpus:
        drafted = chunked(text, shape, wordpiece, chunking)
        assert_whole(text, shape, drafted, wordpiece, chunking.budget)
        for d in drafted.drafts:
            indexed = Chunk(
                ChunkId(0),
                DocumentId(0),
                d.ordinal,
                d.page,
                d.start_line,
                d.end_line,
                d.header,
                d.body,
            ).indexed_text()
            assert wordpiece.count(indexed) == wordpiece.count(d.header) + sum(
                wordpiece.count(line) for line in d.body.split("\n")
            )


LINE_SHAPES = (
    "",
    "   ",
    "# Head",
    "## Sub [[x]]",
    "```",
    "~~~",
    "---",
    "title: Made up",
    "related: [[r]]",
    "one two three",
    "a b c d e f g h i j k l m n o p",
    "see [x](../y.md) and `[[no]]`",
)


@given(
    st.lists(st.sampled_from(LINE_SHAPES), min_size=1, max_size=60),
    st.integers(min_value=3, max_value=30),
    st.sampled_from(list(Splitter)),
    st.booleans(),
)
def test_any_markdown_cuts_whole(
    lines: list[str], max_tokens: int, splitter: Splitter, header: bool
) -> None:
    text = Text((tuple(lines),))
    shape = read_markdown(text, PurePosixPath("n/note.md"))
    chunking = Chunking(max_tokens, splitter, header=header, window_lines=7)

    assert_whole(
        text, shape, chunked(text, shape, WORDS, chunking), WORDS, chunking.budget
    )
