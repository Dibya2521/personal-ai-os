from pathlib import Path, PurePosixPath

import pytest

from synthia.knowledge.chunk import Chunking, chunked
from synthia.knowledge.model import DocumentKind
from synthia.knowledge.retrieve import Side, Variant
from synthia.knowledge.text import Format, Outline, Text, UnreadableError, format_of
from tests.knowledge.checks import WORDS
from tests.knowledge.corpus import citations, opened
from tests.knowledge.pdfs import encrypted, pdf_of
from tests.knowledge.test_index import indexed

PATH = PurePosixPath("papers/report.pdf")
SCANNED = "(scanned or images only)"


def pdf_format() -> Format:
    form = format_of(PATH)
    assert form is not None
    return form


def read(data: bytes) -> tuple[Text, Outline]:
    form = pdf_format()
    text = form.decode(data)
    return text, form.read(text, PATH)


def drafts(data: bytes) -> list[tuple[int | None, int, int, str]]:
    text, outline = read(data)
    drafted = chunked(text, outline, WORDS, Chunking(40, header=False))
    return [(d.page, d.start_line, d.end_line, d.body) for d in drafted.drafts]


def test_a_pdf_reads_as_its_lines_page_by_page() -> None:
    text, outline = read(pdf_of([["alpha one", "alpha (two)"], ["beta"]]))

    assert pdf_format().kind is DocumentKind.PDF
    assert text.pages == (("alpha one", "alpha (two)"), ("beta",))
    assert outline.title == "report.pdf"
    assert outline.note == ""


def test_a_section_broken_by_a_page_keeps_each_page_s_lines_on_it() -> None:
    data = pdf_of([["Methods", "we measured it"], ["and then again", "twice"]])

    assert drafts(data) == [
        (1, 1, 2, "Methods\nwe measured it"),
        (2, 1, 2, "and then again\ntwice"),
    ]


def test_a_one_page_pdf_still_numbers_its_page() -> None:
    assert drafts(pdf_of([["only page"]])) == [(1, 1, 1, "only page")]


@pytest.mark.parametrize(
    ("pages", "note"),
    [
        ([["a"], [], ["c"], [" "]], f"pages 2, 4 hold no text {SCANNED}"),
        ([["a"], []], f"page 2 holds no text {SCANNED}"),
        ([list[str]()], f"page 1 holds no text {SCANNED}"),
    ],
    ids=["two", "one", "all"],
)
def test_a_page_with_no_text_is_noted_and_cut_into_nothing(
    pages: list[list[str]], note: str
) -> None:
    written = [n for n, lines in enumerate(pages, start=1) if "".join(lines).strip()]

    assert read(pdf_of(pages))[1].note == note
    assert [page for page, *_ in drafts(pdf_of(pages))] == written


@pytest.mark.parametrize("password", ["", "secret"], ids=["empty", "set"])
def test_an_encrypted_pdf_is_not_read_even_when_no_password_opens_it(
    password: str,
) -> None:
    data = encrypted(pdf_of([["hidden"]]), password)

    with pytest.raises(UnreadableError, match=r"^encrypted: not read$"):
        pdf_format().decode(data)


@pytest.mark.parametrize(
    "data",
    [b"not a pdf at all", b"", pdf_of([["cut short"]])[:200]],
    ids=["garbage", "empty", "truncated"],
)
def test_bytes_pypdf_cannot_parse_are_unreadable(data: bytes) -> None:
    with pytest.raises(UnreadableError, match=r"^not a PDF pypdf can read: \w"):
        pdf_format().decode(data)


async def test_an_indexed_pdf_is_found_by_its_words_and_cited_by_page(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "report.pdf").write_bytes(
        pdf_of([["summary of findings"], ["the kestrel nests", "on the roof"]])
    )
    (root / "memo.pdf").write_bytes(pdf_of([["a heron waits"]]))
    (root / "scan.pdf").write_bytes(pdf_of([[]]))
    db = tmp_path / "knowledge.db"
    knowledge = await opened(db)
    await knowledge.add(root)
    await knowledge.index()

    words = Variant(Side.WORDS)
    kestrel = await knowledge.search("kestrel", variant=words)
    heron = await knowledge.search("heron", variant=words)

    assert citations(kestrel) == ["notes/report.pdf:p.2:1-2"]
    assert citations(heron) == ["notes/memo.pdf:p.1:1"]
    assert indexed(db)["scan.pdf"] == (f"page 1 holds no text {SCANNED}", ())
