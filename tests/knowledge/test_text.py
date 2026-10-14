from pathlib import PurePosixPath

import pytest

from synthia.knowledge.model import DocumentKind
from synthia.knowledge.text import SNIFF_BYTES, Text, UnreadableError, format_of


def test_lines_end_at_crlf_cr_or_lf_and_nothing_else() -> None:
    text = Text.of_bytes(b"one\r\ntwo\rthree\x0cstill three\n\r\n")

    assert text.pages == (("one", "two", "three\x0cstill three", "", ""),)


def test_a_byte_order_mark_is_not_part_of_line_one() -> None:
    assert Text.of_bytes(b"\xef\xbb\xbf# Title\nbody").lines() == ("# Title", "body")


def test_a_nul_byte_near_the_start_marks_a_binary_file() -> None:
    with pytest.raises(UnreadableError, match="a NUL byte at offset 3"):
        Text.of_bytes(b"abc\0def")


def test_a_nul_byte_past_the_sniff_is_text() -> None:
    data = b"a" * SNIFF_BYTES + b"\0"

    assert Text.of_bytes(data).lines() == ("a" * SNIFF_BYTES + "\0",)


def test_bytes_that_are_not_utf8_are_refused() -> None:
    with pytest.raises(
        UnreadableError, match="not UTF-8: invalid start byte at byte 2"
    ):
        Text.of_bytes(b"ok\xff")


def test_span_joins_inclusive_lines_of_a_page() -> None:
    text = Text((("a", "b", "c"), ("d", "e")))

    assert text.span(1, 2, 3) == "b\nc"
    assert text.span(2, 1, 1) == "d"
    assert Text((("x", "y"),)).span(None, 1, 2) == "x\ny"


@pytest.mark.parametrize(
    ("start", "end"), [(0, 1), (2, 1), (1, 4)], ids=["zero", "reversed", "past"]
)
def test_span_refuses_lines_outside_the_page(start: int, end: int) -> None:
    with pytest.raises(IndexError, match=rf"lines {start}\.\.{end} are not within"):
        Text((("a", "b", "c"),)).span(None, start, end)


def test_lines_of_a_paged_text_need_a_page_that_exists() -> None:
    text = Text((("a",), ("b",)))

    assert text.lines(2) == ("b",)
    with pytest.raises(ValueError, match="name a page: the text has 2"):
        text.lines()
    with pytest.raises(IndexError, match="no page 3 in 2"):
        text.lines(3)


@pytest.mark.parametrize("name", ["a.md", "b.markdown", "C.MD"])
def test_markdown_files_are_read_as_markdown(name: str) -> None:
    found = format_of(PurePosixPath("notes", name))

    assert found is not None
    assert found.kind is DocumentKind.MARKDOWN
    outline = found.read(found.decode(b"# Fusion\nbody"), PurePosixPath(name))
    assert outline.title == "Fusion"


@pytest.mark.parametrize(
    ("name", "kind", "title"),
    [
        ("pkg/mod.py", DocumentKind.PYTHON, "pkg.mod"),
        ("pkg/mod.PYI", DocumentKind.PYTHON, "mod.PYI"),
        ("notes/todo.txt", DocumentKind.TEXT, "todo.txt"),
        ("pyproject.toml", DocumentKind.TEXT, "pyproject.toml"),
        ("web/app.tsx", DocumentKind.TEXT, "app.tsx"),
    ],
)
def test_python_and_plain_text_files_have_their_readers(
    name: str, kind: DocumentKind, title: str
) -> None:
    path = PurePosixPath(name)
    found = format_of(path)

    assert found is not None
    assert found.kind is kind
    assert found.read(found.decode(b"x = 1\n"), path).title == title


@pytest.mark.parametrize("name", ["photo.png", "Makefile", "notes.md.bak"])
def test_files_with_no_format_are_skipped(name: str) -> None:
    assert format_of(PurePosixPath(name)) is None
