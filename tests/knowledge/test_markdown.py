from pathlib import PurePosixPath

import pytest

from synthia.knowledge.markdown import read_markdown
from synthia.knowledge.model import Link, LinkKind
from synthia.knowledge.text import Block, FoundLink, Outline, Text

NOTE = PurePosixPath("notes/a.md")


def read(*lines: str, path: PurePosixPath = NOTE) -> Outline:
    return read_markdown(Text((lines,)), path)


def links(*lines: str, path: PurePosixPath = NOTE) -> list[Link]:
    return [found.link for found in read(*lines, path=path).links]


def wiki(target: str, key: str, anchor: str | None = None) -> Link:
    return Link(LinkKind.WIKILINK, target, key, anchor)


def test_a_comment_in_a_backtick_fence_is_not_a_heading() -> None:
    outline = read("# Title", "```py", "# not a heading", "```", "## Real", "text")

    assert outline == Outline(
        "Title",
        (
            Block(None, 1, 4, ("Title",), 0),
            Block(None, 5, 6, ("Title", "Real"), 1),
        ),
    )


def test_a_tilde_fence_holding_backticks_stays_one_fence() -> None:
    outline = read("~~~", "```", "# no", "```", "~~~", "# Yes", "x")

    assert outline == Outline(
        "Yes", (Block(None, 1, 5, (), 0), Block(None, 6, 7, ("Yes",), 1))
    )


def test_a_fence_closes_only_on_one_at_least_as_long() -> None:
    outline = read("````", "```", "# no", "````", "# Yes", "x")

    assert outline.blocks == (
        Block(None, 1, 4, (), 0),
        Block(None, 5, 6, ("Yes",), 1),
    )


def test_an_unclosed_fence_runs_to_the_end() -> None:
    outline = read("# A", "```", "# B", "[[x]]")

    assert outline == Outline("A", (Block(None, 1, 4, ("A",), 0),))


def test_a_backtick_line_with_a_backtick_after_it_is_inline_code() -> None:
    outline = read("```not`a fence", "# Heading", "x")

    assert outline.blocks == (
        Block(None, 1, 1, (), 0),
        Block(None, 2, 3, ("Heading",), 1),
    )


def test_frontmatter_gives_the_title_and_related_links() -> None:
    outline = read(
        "---",
        'title: "Fusion notes"',
        "related: [[A]], [[b|B]]",
        "tags: [x]",
        "patterns:",
        '  - "[[Folder/C.md#Part]]"',
        "up: [[D]]",
        "---",
        "# Heading",
        "See [[E]].",
    )

    assert outline == Outline(
        "Fusion notes",
        (Block(None, 1, 8, (), 0), Block(None, 9, 10, ("Heading",), 1)),
        (
            FoundLink(Link(LinkKind.RELATED, "A", "a"), None, 3),
            FoundLink(Link(LinkKind.RELATED, "b", "b"), None, 3),
            FoundLink(
                Link(LinkKind.RELATED, "Folder/C.md", "folder/c", "Part"), None, 6
            ),
            FoundLink(wiki("D", "d"), None, 7),
            FoundLink(wiki("E", "e"), None, 10),
        ),
    )


def test_frontmatter_never_closed_is_text() -> None:
    outline = read("---", "title: X", "# H", "x")

    assert outline == Outline(
        "H", (Block(None, 1, 2, (), 0), Block(None, 3, 4, ("H",), 1))
    )


def test_a_document_with_no_heading_is_one_block_titled_by_its_stem() -> None:
    outline = read("just text", "", path=PurePosixPath("notes/plain.md"))

    assert outline == Outline("plain", (Block(None, 1, 2, (), 0),))


def test_a_bare_heading_joins_the_next_and_a_bare_last_one_the_one_before() -> None:
    outline = read("intro", "# First", "", "# Second", "body", "## Last", "## End")

    assert outline == Outline(
        "First",
        (Block(None, 1, 1, (), 0), Block(None, 2, 7, ("Second",), 1)),
    )


def test_a_bare_last_heading_after_frontmatter_alone_stays_a_block() -> None:
    outline = read("---", "title: T", "---", "# Only")

    assert outline == Outline(
        "T", (Block(None, 1, 3, (), 0), Block(None, 4, 4, ("Only",), 1))
    )


def test_a_rule_on_the_first_line_is_not_frontmatter_without_a_key() -> None:
    outline = read("---", "# Head", "text", "---", "# Real", "body")

    assert outline == Outline(
        "Head",
        (
            Block(None, 1, 1, (), 0),
            Block(None, 2, 4, ("Head",), 1),
            Block(None, 5, 6, ("Real",), 2),
        ),
    )


def test_frontmatter_may_close_with_three_dots() -> None:
    outline = read("---", "title: T", "...", "# H", "x")

    assert outline == Outline(
        "T", (Block(None, 1, 3, (), 0), Block(None, 4, 5, ("H",), 1))
    )


def test_a_file_with_no_lines_has_no_blocks() -> None:
    outline = read_markdown(Text(((),)), PurePosixPath("empty.md"))

    assert outline == Outline("empty", ())


def test_trails_hold_the_enclosing_headings() -> None:
    outline = read("# A", "x", "## B", "y", "### C", "z", "## D", "w")

    assert [block.trail for block in outline.blocks] == [
        ("A",),
        ("A", "B"),
        ("A", "B", "C"),
        ("A", "D"),
    ]


@pytest.mark.parametrize(
    ("line", "title"),
    [
        ("# Closing ##", "Closing"),
        ("# C#", "C#"),
        ("   # Indented", "Indented"),
        ("## Second level", "a"),
        ("#tag", "a"),
        ("####### seven", "a"),
    ],
    ids=["closing marks", "hash in text", "indented", "h2", "tag", "too deep"],
)
def test_the_title_is_an_h1_text_or_the_stem(line: str, title: str) -> None:
    assert read(line).title == title


def test_setext_headings_are_text() -> None:
    assert read("Title", "=====", "text").blocks == (Block(None, 1, 3, (), 0),)


def test_crlf_lines_and_a_byte_order_mark_read_as_plain_lines() -> None:
    text = Text.of_bytes(b"\xef\xbb\xbf---\r\ntitle: T\r\n---\r\n# A\r\n[[x]]\r\n")

    assert read_markdown(text, NOTE) == Outline(
        "T",
        (Block(None, 1, 3, (), 0), Block(None, 4, 6, ("A",), 1)),
        (FoundLink(wiki("x", "x"), None, 5),),
    )


def test_wikilinks_name_the_casefolded_stem() -> None:
    assert links("[[a|b]] [[a#h]] ![[a]] [[Fusion.MD|f]] [[ ]]") == [
        wiki("a", "a"),
        wiki("a", "a", "h"),
        wiki("a", "a"),
        wiki("Fusion.MD", "fusion"),
    ]


def test_a_wikilink_naming_folders_is_keyed_by_its_path_from_the_root() -> None:
    assert links(
        "[[Notes/Fusion.MD|f]] [[/Notes/./Fusion]] [[Files/Data.csv]] [[Notes/]]"
    ) == [
        wiki("Notes/Fusion.MD", "notes/fusion"),
        wiki("/Notes/./Fusion", "notes/fusion"),
        wiki("Files/Data.csv", "files/data.csv"),
        wiki("Notes/", "notes"),
    ]


def test_a_wikilink_to_another_note_suffix_or_an_attachment() -> None:
    assert links(
        "[[Note.markdown]] [[file.pdf]] ![[photo.png]] ![[Note]] ![[Other.md]]"
    ) == [
        wiki("Note.markdown", "note"),
        wiki("file.pdf", "file.pdf"),
        wiki("Note", "note"),
        wiki("Other.md", "other"),
    ]


def test_a_table_cells_escaped_alias_pipe_is_a_pipe() -> None:
    assert links(r"| [[a\|b]] | [[c\|d]] |") == [wiki("a", "a"), wiki("c", "c")]


def test_a_link_target_may_hold_parentheses_and_an_image_is_not_a_link() -> None:
    assert links("[x](a_(1).md) ![alt](p.png) [y](b.md)") == [
        Link(LinkKind.MARKDOWN, "a_(1).md", "notes/a_(1).md"),
        Link(LinkKind.MARKDOWN, "b.md", "notes/b.md"),
    ]


def test_markdown_links_resolve_from_the_files_folder() -> None:
    assert links(
        "[x](../b.md) [y](https://e.com) [z](#top) [w](../../up.md)"
        " [v](<sub dir/c%20d.md#Sec>) [u](/root.md 'Root') [t](./same.md?raw=1)"
    ) == [
        Link(LinkKind.MARKDOWN, "../b.md", "b.md"),
        Link(LinkKind.MARKDOWN, "sub dir/c%20d.md#Sec", "notes/sub dir/c d.md", "Sec"),
        Link(LinkKind.MARKDOWN, "/root.md", "root.md"),
        Link(LinkKind.MARKDOWN, "./same.md?raw=1", "notes/same.md"),
    ]


def test_a_markdown_link_above_the_root_is_skipped() -> None:
    root = PurePosixPath("a.md")

    assert links("[x](../../up.md) [y](../up.md) [z](./)", path=root) == []
    assert links("[x](b/../c.md)", path=root) == [
        Link(LinkKind.MARKDOWN, "b/../c.md", "c.md")
    ]


def test_links_in_fences_and_inline_code_are_ignored() -> None:
    assert links(
        "`[[a]]` ``[[b]] ` [[c]]`` [[d]]", "```", "[[e]] [f](f.md)", "```", "[[g]]"
    ) == [wiki("d", "d"), wiki("g", "g")]
