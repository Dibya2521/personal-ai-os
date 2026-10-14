"""Markdown read into heading sections, with frontmatter and every link.

One pass over the lines, as a small state machine (:class:`_State`):
frontmatter, then prose and fenced blocks. A ``#`` line is a heading only
outside a fence; a fence closes only on the same character, at least as long
as it opened, so a tilde fence holding backticks stays one fence. Setext
headings (a line of ``===`` under text) are not read as headings.

Frontmatter is not parsed as YAML. Each ``key:`` line at column 0 sets the
current key, and every wikilink on a line under ``related`` or ``patterns``
is a ``RELATED`` link, which covers inline and block lists alike. ``title:``
gives the title. Any other shape stays text, never an error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Final
from urllib.parse import unquote

from synthia.knowledge.model import Link, LinkKind
from synthia.knowledge.text import Block, FoundLink, Outline

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from synthia.knowledge.text import Text

LINK: Final = re.compile(
    r"(?P<wiki>(?P<embed>!)?\[\[(?P<target>[^\]|#\n]+)(?:#(?P<anchor>[^\]|\n]+))?"
    r"(?:\|[^\]\n]*)?\]\])"
    r"|(?P<image>!)?\[[^\]\n]*\]\((?P<href><[^>\n]*>|(?:[^()\s]|\([^()\s]*\))+)"
    r"""(?:[ \t]+(?:"[^"\n]*"|'[^'\n]*'))?[ \t]*\)"""
)
NOTE_SUFFIXES: Final = (".md", ".markdown")
CODE_SPAN: Final = re.compile(r"(?<!`)(?P<ticks>`+)(?!`).+?(?<!`)(?P=ticks)(?!`)")
HEADING: Final = re.compile(
    r"^ {0,3}(?P<marks>#{1,6})(?:[ \t]+(?P<text>.*?))?(?:[ \t]+#+)?[ \t]*$"
)
FENCE_OPEN: Final = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
FENCE_CLOSE: Final = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})[ \t]*$")
FRONTMATTER_KEY: Final = re.compile(
    r"^(?P<key>[A-Za-z_][\w .-]*?)[ \t]*:(?:[ \t]+(?P<value>.*?))?[ \t]*$"
)
FRONTMATTER_FENCE: Final = "---"
FRONTMATTER_CLOSERS: Final = frozenset((FRONTMATTER_FENCE, "..."))
SCHEME: Final = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
RELATED_KEYS: Final = frozenset(("related", "patterns"))
QUOTES: Final = "\"'"


class _State(Enum):
    FRONTMATTER = auto()
    PROSE = auto()
    FENCED = auto()


@dataclass(slots=True)
class _Walk:
    """What one pass over the lines has seen so far."""

    path: PurePosixPath
    frontmatter_end: int
    state: _State = _State.PROSE
    fence: str = ""
    key: str = ""
    title: str = ""
    h1: str = ""
    open_headings: list[tuple[int, str]] = field(default_factory=list[tuple[int, str]])
    headings: list[tuple[int, tuple[str, ...]]] = field(
        default_factory=list[tuple[int, tuple[str, ...]]]
    )
    links: list[FoundLink] = field(default_factory=list[FoundLink])

    def take(self, number: int, line: str) -> None:
        if self.state is _State.FENCED:
            self._fenced(line)
        elif self.state is _State.FRONTMATTER:
            self._frontmatter(number, line)
        else:
            self._prose(number, line)

    def _frontmatter(self, number: int, line: str) -> None:
        if number == self.frontmatter_end:
            self.state = _State.PROSE
            return
        key = FRONTMATTER_KEY.match(line)
        if key is not None:
            self.key = key["key"].casefold()
            if self.key == "title" and key["value"]:
                self.title = _unquoted(key["value"])
        related = self.key in RELATED_KEYS
        self._links(number, line, LinkKind.RELATED if related else LinkKind.WIKILINK)

    def _fenced(self, line: str) -> None:
        closing = FENCE_CLOSE.match(line)
        if closing is None:
            return
        fence = closing["fence"]
        if fence[0] == self.fence[0] and len(fence) >= len(self.fence):
            self.state = _State.PROSE

    def _prose(self, number: int, line: str) -> None:
        opening = FENCE_OPEN.match(line)
        # A backtick in a backtick fence's info string makes the line inline code.
        if opening is not None and not (
            opening["fence"][0] == "`" and "`" in opening["info"]
        ):
            self.state = _State.FENCED
            self.fence = opening["fence"]
            return
        heading = HEADING.match(line)
        if heading is not None:
            self._heading(number, len(heading["marks"]), heading["text"] or "")
        self._links(number, CODE_SPAN.sub(" ", line), LinkKind.WIKILINK)

    def _heading(self, number: int, level: int, text: str) -> None:
        while self.open_headings and self.open_headings[-1][0] >= level:
            self.open_headings.pop()
        self.open_headings.append((level, text))
        self.headings.append((number, tuple(t for _, t in self.open_headings)))
        if level == 1 and not self.h1:
            self.h1 = text

    def _links(self, number: int, line: str, wiki: LinkKind) -> None:
        # A table cell escapes a wikilink's alias pipe as \|.
        for match in LINK.finditer(line.replace(r"\|", "|")):
            if match["wiki"]:
                link = wikilink(match, wiki)
            elif match["image"]:
                continue
            else:
                link = markdown_link(match["href"], self.path)
            if link is not None:
                self.links.append(FoundLink(link, None, number))


def _unquoted(value: str) -> str:
    if len(value) > 1 and value[0] == value[-1] and value[0] in QUOTES:
        return value[1:-1]
    return value


def _frontmatter_end(lines: Sequence[str]) -> int:
    """Return the line closing the frontmatter, or 0 when there is none.

    A ``---`` first line opens it only when a ``key:`` line follows before the
    closing ``---`` or ``...``; otherwise the first line is a horizontal rule.
    """
    if not lines or lines[0].rstrip() != FRONTMATTER_FENCE:
        return 0
    for number in range(2, len(lines) + 1):
        if lines[number - 1].rstrip() in FRONTMATTER_CLOSERS:
            keyed = any(FRONTMATTER_KEY.match(line) for line in lines[1 : number - 1])
            return number if keyed else 0
    return 0


def _sections(
    size: int, after: int, headings: Sequence[tuple[int, tuple[str, ...]]]
) -> Iterator[tuple[int, int, tuple[str, ...]]]:
    """Yield the text before the first heading, then each heading's section."""
    starts = [*(line for line, _ in headings), size + 1]
    if starts[0] > after + 1:
        yield after + 1, starts[0] - 1, ()
    for (line, trail), following in zip(headings, starts[1:], strict=True):
        yield line, following - 1, trail


def _blocks(lines: Sequence[str], walk: _Walk) -> tuple[Block, ...]:
    blocks: list[Block] = []
    if walk.frontmatter_end:
        blocks.append(Block(None, 1, walk.frontmatter_end, (), 0))
    sections = list(_sections(len(lines), walk.frontmatter_end, walk.headings))
    heading_lines = {line for line, _ in walk.headings}
    start: int | None = None
    for index, (first, last, trail) in enumerate(sections):
        bare = first in heading_lines and not any(
            line.strip() for line in lines[first:last]
        )
        if bare and index < len(sections) - 1:
            start = first if start is None else start
            continue
        if bare and blocks and blocks[-1].start > walk.frontmatter_end:
            blocks[-1] = replace(blocks[-1], end=last)
            continue
        begin = first if start is None else start
        blocks.append(Block(None, begin, last, trail, len(blocks)))
        start = None
    return tuple(blocks)


def read_markdown(text: Text, path: PurePosixPath) -> Outline:
    """Return ``text``'s sections as blocks, its title, and its links.

    Title: frontmatter ``title``, else the first H1, else the file stem.
    Blocks: the frontmatter first, with an empty trail; then the text before
    the first heading, also with an empty trail; then one block per heading
    section, from the heading line to the line before the next heading of any
    level, with trail = the enclosing heading texts. A section holding only
    its heading line joins the next (the last one joins the one before), so
    no chunk is a bare heading. Each block is its own unit. Links inside
    fences and inline code are not links, nor are images.
    """
    lines = text.lines()
    end = _frontmatter_end(lines)
    walk = _Walk(path, end, _State.FRONTMATTER if end else _State.PROSE)
    for number, line in enumerate(lines, start=1):
        walk.take(number, line)
    title = walk.title or walk.h1 or path.stem
    return Outline(title, _blocks(lines, walk), tuple(walk.links))


def wikilink(match: re.Match[str], kind: LinkKind) -> Link | None:
    """Return the link a wikilink match names, or None when it names no note.

    ``key`` is the casefolded last path component of the target, without a
    ``.md`` or ``.markdown`` suffix, so "Notes/Fusion.md" and "fusion" match
    the same note; ``anchor`` is the heading after ``#``; the alias is display
    text only. An embedded attachment (``![[photo.png]]``) is not a link.
    """
    target = match["target"].strip()
    name = PurePosixPath(target).name.casefold()
    key = next((name.removesuffix(s) for s in NOTE_SUFFIXES if name.endswith(s)), name)
    attachment = key == name and PurePosixPath(name).suffix != ""
    if not key or (match["embed"] and attachment):
        return None
    anchor = match["anchor"]
    return Link(kind, target, key, anchor.strip() if anchor else None)


def markdown_link(href: str, path: PurePosixPath) -> Link | None:
    """Return the link ``[text](href)`` names, or None for one leaving the source.

    A URL with a scheme, a bare ``#anchor`` and a path that climbs above the
    source root are not document links. ``key`` is the target resolved
    against ``path``'s folder (or the root, for a target starting with
    ``/``), percent-decoded, as a posix path from the root.
    """
    target = href[1:-1] if href.startswith("<") else href
    if SCHEME.match(target):
        return None
    location, _, anchor = target.partition("#")
    location = location.partition("?")[0]
    key = _resolved(location, path) if location else None
    if key is None:
        return None
    return Link(LinkKind.MARKDOWN, target, key, unquote(anchor) or None)


def _resolved(location: str, path: PurePosixPath) -> str | None:
    parts = [] if location.startswith("/") else list(path.parent.parts)
    for written in location.split("/"):
        part = unquote(written)
        if part == "..":
            if not parts:
                return None
            parts.pop()
        elif part not in {"", "."}:
            parts.append(part)
    return "/".join(parts) or None
