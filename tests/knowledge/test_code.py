from itertools import pairwise
from pathlib import PurePosixPath

import pytest

from synthia.knowledge.chunk import Chunking, chunked
from synthia.knowledge.code import read_python
from synthia.knowledge.model import Link, LinkKind, module_name
from synthia.knowledge.text import Block, Outline, Text, read_lines
from tests.knowledge.checks import ROOT, WORDS, assert_whole, tracked

MODULE = PurePosixPath("m.py")


def read(*lines: str, path: PurePosixPath = MODULE) -> Outline:
    return read_python(Text((lines,)), path)


def imports(*lines: str, path: PurePosixPath) -> list[tuple[int, str, str]]:
    return [(f.line, f.link.target, f.link.key) for f in read(*lines, path=path).links]


def drafts(lines: tuple[str, ...], max_tokens: int) -> list[tuple[int, int, str]]:
    text = Text((lines,))
    drafted = chunked(text, read_python(text, MODULE), WORDS, Chunking(max_tokens))
    return [(d.start_line, d.end_line, d.header) for d in drafted.drafts]


@pytest.mark.parametrize(
    ("path", "name"),
    [
        ("src/synthia/memory/store.py", "synthia.memory.store"),
        ("pkg/__init__.py", "pkg"),
        ("src.py", "src"),
        ("src/src/x.py", "src.x"),
        ("src/__init__.py", None),
        ("__init__.py", None),
        ("foo.bar.py", None),
        ("pkg/stub.pyi", None),
        ("notes.md", None),
    ],
)
def test_module_name_is_how_python_would_import_the_file(
    path: str, name: str | None
) -> None:
    assert module_name(PurePosixPath(path)) == name


def test_a_decorated_function_keeps_its_decorators_and_the_comments_above() -> None:
    outline = read(
        "import os",
        "",
        "# Why f exists.",
        "# More of why.",
        "@cache",
        "@other(",
        "    1,",
        ")",
        "def f():",
        "    return 1",
    )

    assert outline == Outline(
        "m",
        (Block(None, 1, 2, (), 0), Block(None, 3, 10, ("def f",), 1)),
        outline.links,
    )


def test_a_comment_a_blank_line_above_a_def_stays_with_the_module() -> None:
    assert read("# loose", "", "def f():", "    pass").blocks == (
        Block(None, 1, 2, (), 0),
        Block(None, 3, 4, ("def f",), 1),
    )


def test_the_docstring_definitions_and_main_guard_are_blocks_in_order() -> None:
    outline = read(
        '"""Module doc."""',
        "",
        "import sys",
        "",
        "",
        "async def fetch():",
        "    return 0",
        "",
        "",
        'if __name__ == "__main__":',
        "    sys.exit(fetch())",
    )

    assert outline.blocks == (
        Block(None, 1, 5, (), 0),
        Block(None, 6, 7, ("async def fetch",), 1),
        Block(None, 8, 11, (), 2),
    )


def test_nested_definitions_stay_inside_their_parent() -> None:
    outline = read(
        "def outer():",
        "    def inner():",
        "        return 1",
        "",
        "    class Local:",
        "        pass",
        "    return inner",
    )

    assert outline.blocks == (Block(None, 1, 7, ("def outer",), 0),)


CLASS = (
    "# The store.",
    "@dataclass",
    "class A:",
    '    """Doc of A, long enough to not fit."""',
    "",
    "    x = 1",
    "",
    "    # About f.",
    "    @property",
    "    def f(self):",
    "        return 1",
    "",
    "    async def g(self):",
    "        return 2",
    "    y = 3",
)


def test_a_class_is_one_unit_of_its_head_and_each_method() -> None:
    assert read(*CLASS).blocks == (
        Block(None, 1, 7, ("class A",), 0),
        Block(None, 8, 11, ("class A", "def f"), 0),
        Block(None, 13, 14, ("class A", "async def g"), 0),
        Block(None, 15, 15, ("class A",), 0),
    )


def test_a_class_that_fits_is_one_draft() -> None:
    assert drafts(CLASS, 502) == [(1, 15, "m > class A")]


def test_a_class_that_does_not_fit_is_cut_per_method_with_its_trail() -> None:
    assert drafts(CLASS, 34) == [
        (1, 6, "m > class A"),
        (8, 11, "m > class A > def f"),
        (13, 14, "m > class A > async def g"),
        (15, 15, "m > class A"),
    ]


def test_a_class_without_methods_is_one_block_and_its_header_is_its_own() -> None:
    assert read("class A(", "    Base,", "):", "    x = 1", "def f(): ...").blocks == (
        Block(None, 1, 4, ("class A",), 0),
        Block(None, 5, 5, ("def f",), 1),
    )


def test_a_method_right_under_a_multiline_class_line_leaves_it_to_the_class() -> None:
    assert read("class A(", "    Base,", "):", "    def f(self): ...").blocks == (
        Block(None, 1, 3, ("class A",), 0),
        Block(None, 4, 4, ("class A", "def f"), 0),
    )


def test_a_string_ending_in_a_comment_like_line_stays_with_its_statement() -> None:
    assert read('x = """', '# text"""', "@cache", "def f(): ...").blocks == (
        Block(None, 1, 2, (), 0),
        Block(None, 3, 4, ("def f",), 1),
    )


def test_a_file_with_old_mac_line_endings_parses_on_the_same_lines() -> None:
    text = Text.of_bytes(b"x = 1\rdef f():\r    pass\r")

    outline = read_python(text, MODULE)

    assert outline.note == ""
    assert outline.blocks == (
        Block(None, 1, 1, (), 0),
        Block(None, 2, 3, ("def f",), 1),
    )


def test_a_syntax_warning_in_the_file_is_not_a_failure() -> None:
    outline = read('x = "\\d"', "def f(): ...")

    assert outline.note == ""
    assert outline.blocks[-1] == Block(None, 2, 2, ("def f",), 1)


def test_imports_of_every_form_link_on_their_first_line() -> None:
    found = imports(
        "import a.b, c as d",
        "from e.f import (",
        "    g,",
        "    h as i,",
        ")",
        "from x import *",
        "def f():",
        "    import inner",
        path=PurePosixPath("pkg/sub/mod.py"),
    )

    assert found == [
        (1, "a.b", "a.b"),
        (1, "c", "c"),
        (2, "e.f", "e.f"),
        (2, "e.f.g", "e.f.g"),
        (2, "e.f.h", "e.f.h"),
        (6, "x", "x"),
        (8, "inner", "inner"),
    ]


def test_relative_imports_resolve_against_the_modules_package() -> None:
    found = imports(
        "from . import j",
        "from .k import l",
        "from ..m import n",
        "from ... import o",
        path=PurePosixPath("src/pkg/sub/mod.py"),
    )

    assert found == [
        (1, ".", "pkg.sub"),
        (1, ".j", "pkg.sub.j"),
        (2, ".k", "pkg.sub.k"),
        (2, ".k.l", "pkg.sub.k.l"),
        (3, "..m", "pkg.m"),
        (3, "..m.n", "pkg.m.n"),
    ]


def test_a_package_init_resolves_relative_imports_against_itself() -> None:
    found = imports(
        "from . import j", "from .. import up", path=PurePosixPath("pkg/__init__.py")
    )

    assert found == [(1, ".", "pkg"), (1, ".j", "pkg.j")]


def test_a_top_level_module_drops_its_relative_imports() -> None:
    assert imports("from . import j", "import k", path=PurePosixPath("top.py")) == [
        (2, "k", "k")
    ]


def test_a_bare_standard_library_name_is_no_link() -> None:
    found = imports(
        "import types, os.path, pydantic",
        "from typing import TYPE_CHECKING",
        "from profile.models import User",
        "from . import types",
        path=PurePosixPath("pkg/mod.py"),
    )

    assert found == [
        (1, "os.path", "os.path"),
        (1, "pydantic", "pydantic"),
        (2, "typing.TYPE_CHECKING", "typing.TYPE_CHECKING"),
        (3, "profile.models", "profile.models"),
        (3, "profile.models.User", "profile.models.User"),
        (4, ".", "pkg"),
        (4, ".types", "pkg.types"),
    ]


@pytest.mark.parametrize(
    "source",
    [
        ("def f(): pass; x = 1", "y = 2"),
        ("\tclass A:", "\t\tdef f(self): ...", "x = 1\x0cy = 2"),
        ("# only", "# comments"),
        (
            "# above",
            "@overload",
            "def f(x: int) -> int: ...",
            "@overload",
            "def f(): ...",
        ),
        ("@decorator(", "    arg,", ")", "class A:", '    """Doc."""'),
        ("class A:", "    class B: ...", "    x = 1", "    def f(self): ...", "y = 2"),
        ("if __name__ == '__main__':", "    main()"),
        ("",),
    ],
    ids=[
        "semicolons",
        "tabs and form feed",
        "comments",
        "overloads",
        "decorator",
        "nested class",
        "main",
        "empty",
    ],
)
def test_odd_python_shapes_cut_whole(source: tuple[str, ...]) -> None:
    text = Text((source,))
    outline = read_python(text, MODULE)
    drafted = chunked(text, outline, WORDS, Chunking(12))

    assert_whole(text, outline, drafted, WORDS, Chunking(12).budget)


def test_import_links_are_import_kind() -> None:
    assert read("import a").links[0].link == Link(LinkKind.IMPORT, "a", "a")


@pytest.mark.parametrize(
    "line",
    [
        "def f(:",
        "x = 1\0",
        "x = '\ud800'",
        "x = 1" + " + 1" * 100_000,
        "x = " + "not " * 100_000 + "1",
    ],
    ids=["syntax", "nul", "surrogate", "recursion", "parser stack"],
)
def test_a_file_that_does_not_parse_is_read_by_lines_and_says_so(line: str) -> None:
    outline = read(line, "", "more")

    assert outline.title == "m.py"
    assert outline.blocks == (Block(None, 1, 3, (), 0),)
    assert outline.links == ()
    assert outline.note.startswith("did not parse: ")
    assert outline.note.endswith("; read by lines")


def test_plain_text_is_one_block_under_the_file_name() -> None:
    text = Text((("a b", "", "c d"),))

    assert read_lines(text, PurePosixPath("notes/todo.txt")) == Outline(
        "todo.txt", (Block(None, 1, 3, (), 0),)
    )


def test_paged_plain_text_is_one_block_per_page_with_lines() -> None:
    text = Text((("a", "b"), (), ("c",)))

    assert read_lines(text, PurePosixPath("x.txt")).blocks == (
        Block(1, 1, 2, (), 0),
        Block(3, 1, 1, (), 1),
    )


def test_plain_text_packs_paragraphs_then_splits_one_that_does_not_fit() -> None:
    text = Text((("a b", "", "c d", "", "e f g h", "i j k l"),))
    outline = read_lines(text, PurePosixPath("t.txt"))

    drafted = chunked(text, outline, WORDS, Chunking(10))

    assert [(d.start_line, d.end_line, d.header) for d in drafted.drafts] == [
        (1, 3, "t.txt"),
        (5, 5, "t.txt"),
        (6, 6, "t.txt"),
    ]


@pytest.fixture(scope="module")
def corpus() -> list[tuple[Text, Outline]]:
    documents: list[tuple[Text, Outline]] = []
    for path in tracked("*.py"):
        text = Text.of_bytes((ROOT / path).read_bytes())
        documents.append((text, read_python(text, path)))
    return documents


def test_every_tracked_python_file_parses_into_ordered_blocks(
    corpus: list[tuple[Text, Outline]],
) -> None:
    assert len(corpus) > 100
    for text, shape in corpus:
        assert shape.note == ""
        ends = [(block.start, block.end) for block in shape.blocks]
        assert all(start <= end for start, end in ends)
        assert all(a[1] < b[0] for a, b in pairwise(ends))
        assert all(end <= len(text.lines()) for _, end in ends)


@pytest.mark.parametrize("max_tokens", [42, 502], ids=["budget 40", "budget 500"])
def test_every_tracked_python_file_cuts_whole_with_words(
    corpus: list[tuple[Text, Outline]], max_tokens: int
) -> None:
    chunking = Chunking(max_tokens)
    for text, shape in corpus:
        drafted = chunked(text, shape, WORDS, chunking)
        assert_whole(text, shape, drafted, WORDS, chunking.budget)
