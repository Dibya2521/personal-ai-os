"""Python read with the standard library's ``ast``.

Python: each top-level function and class is a block, starting at its first
decorator or at the run of comment lines directly above it. The module
docstring and the statements between definitions are blocks with an empty
trail, so every non-blank line is in one. A class is one unit of several
blocks: its head (decorators, class line, docstring, body up to the first
method), then each method, then any statements between or after methods. The
chunker keeps it whole when it fits and splits it per method when it does
not. A trail names definitions as written: ``("class Store", "async def get")``.

Imports anywhere in the file become ``IMPORT`` links on their statement's
first line. ``import a.b`` links ``a.b``; ``from a.b import c, d`` links
``a.b``, ``a.b.c`` and ``a.b.d``, since only the index knows whether ``c`` is
a module, and a name that is not stays unresolved. A relative import resolves
against the file's package; one climbing above the top is dropped. A bare
standard library name (``import types``) is no link: the index matches a bare
name as a dotted suffix, so it would reach any of the person's modules sharing
it. A dotted one stays: it reaches only a module ending in the same parts.

A file that does not parse is read as plain text and says so in its note.
"""

from __future__ import annotations

import ast
import sys
import warnings
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

from synthia.knowledge.model import Link, LinkKind, module_name
from synthia.knowledge.text import Block, FoundLink, Outline, read_lines

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import PurePosixPath

    from synthia.knowledge.text import Text

type _Definition = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
type _Kinds = tuple[type[_Definition], ...]

KEYWORDS: Final[Mapping[type[ast.stmt], str]] = {
    ast.FunctionDef: "def",
    ast.AsyncFunctionDef: "async def",
    ast.ClassDef: "class",
}
_FUNCTIONS: Final[_Kinds] = (ast.FunctionDef, ast.AsyncFunctionDef)
_DEFINITIONS: Final[_Kinds] = (*_FUNCTIONS, ast.ClassDef)
_LEADS: Final = ("#", "@")
"""A line starting with one of these, right above a definition, belongs to it."""


@dataclass(frozen=True, slots=True)
class _Piece:
    """Lines ``first..last``: one definition, or the text between two (None)."""

    first: int
    last: int
    node: _Definition | None


@dataclass(frozen=True, slots=True)
class _Lines:
    lines: Sequence[str]

    def has_text(self, first: int, last: int) -> bool:
        return any(line.strip() for line in self.lines[first - 1 : last])

    def head(self, node: _Definition, floor: int) -> int:
        """Return ``node``'s first line, above ``floor``: decorators and comments."""
        first = min((node.lineno, *(d.lineno for d in node.decorator_list)))
        while first - 1 > floor and self.lines[first - 2].lstrip().startswith(_LEADS):
            first -= 1
        return first

    def pieces(
        self, body: Sequence[ast.stmt], floor: int, last: int, kinds: _Kinds
    ) -> Iterator[_Piece]:
        """Yield ``body``'s ``kinds`` and the text around them, in ``floor + 1..last``.

        ``floor`` is the line above, which a definition's head never reaches.
        A run of text with no non-blank line is no piece.
        """
        cursor = floor + 1
        for node in body:
            if isinstance(node, kinds):
                first = self.head(node, floor)
                if self.has_text(cursor, first - 1):
                    yield _Piece(cursor, first - 1, None)
                yield _Piece(first, _end(node), node)
                cursor = _end(node) + 1
            floor = _end(node)
        if self.has_text(cursor, last):
            yield _Piece(cursor, last, None)


def _end(node: ast.stmt) -> int:
    return node.end_lineno or node.lineno


def _trail(node: _Definition | None) -> tuple[str, ...]:
    return () if node is None else (f"{KEYWORDS[type(node)]} {node.name}",)


def _class_blocks(
    lines: _Lines, first: int, node: ast.ClassDef, unit: int
) -> Iterator[Block]:
    trail = _trail(node)
    inner = list(lines.pieces(node.body, node.lineno, _end(node), _FUNCTIONS))
    head_end = next((p.first - 1 for p in inner if p.node is not None), _end(node))
    yield Block(None, first, head_end, trail, unit)
    for piece in inner:
        if piece.first > head_end:
            yield Block(
                None, piece.first, piece.last, (*trail, *_trail(piece.node)), unit
            )


def _blocks(lines: _Lines, tree: ast.Module) -> tuple[Block, ...]:
    blocks: list[Block] = []
    top = lines.pieces(tree.body, 0, len(lines.lines), _DEFINITIONS)
    for unit, piece in enumerate(top):
        if isinstance(piece.node, ast.ClassDef):
            blocks.extend(_class_blocks(lines, piece.first, piece.node, unit))
        else:
            blocks.append(
                Block(None, piece.first, piece.last, _trail(piece.node), unit)
            )
    return tuple(blocks)


def _absolute(module: str | None, level: int, package: str | None) -> str | None:
    """Return what ``from <level dots><module>`` names inside ``package``.

    None when it climbs above the top package, or the file has none.
    """
    if level == 0:
        return module
    parts = [] if package is None else package.split(".")
    if level > len(parts):
        return None
    return ".".join([*parts[: len(parts) - level + 1], *filter(None, [module])])


def _imported(node: ast.Import | ast.ImportFrom, package: str | None) -> Iterator[Link]:
    if isinstance(node, ast.Import):
        links = [Link(LinkKind.IMPORT, a.name, a.name) for a in node.names]
    else:
        absolute = _absolute(node.module, node.level, package)
        if absolute is None:
            return
        dots = "." * node.level
        links = [Link(LinkKind.IMPORT, dots + (node.module or ""), absolute)]
        links.extend(
            Link(
                LinkKind.IMPORT,
                dots + ".".join(filter(None, [node.module, alias.name])),
                f"{absolute}.{alias.name}",
            )
            for alias in node.names
            if alias.name != "*"
        )
    yield from (link for link in links if link.key not in sys.stdlib_module_names)


def _links(tree: ast.Module, package: str | None) -> tuple[FoundLink, ...]:
    found: list[FoundLink] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            found.extend(
                FoundLink(link, None, node.lineno) for link in _imported(node, package)
            )
    return tuple(sorted(found, key=lambda link: link.line))


def _parsed(lines: Sequence[str], path: PurePosixPath) -> ast.Module:
    # A SyntaxWarning is about the file's code, and under -W error it would
    # fail the parse of a valid file.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ast.parse("\n".join(lines), filename=path.name)


def read_python(text: Text, path: PurePosixPath) -> Outline:
    """Return the module's definitions as blocks and its imports as links.

    Title: the module name (:func:`~synthia.knowledge.model.module_name`),
    else the file name. A file that does not parse is read by
    :func:`read_lines`, with the parser's message in the outline's note.
    """
    lines = text.lines()
    try:
        tree = _parsed(lines, path)
    # Too deep a nesting surfaces as RecursionError or, from the parser's own
    # stack, MemoryError; neither is the indexer's to die of.
    except (SyntaxError, ValueError, RecursionError, MemoryError) as error:
        note = f"did not parse: {error}; read by lines"
        return replace(read_lines(text, path), note=note)
    package = module_name(path.with_name("__init__.py"))
    title = module_name(path) or path.name
    return Outline(title, _blocks(_Lines(lines), tree), _links(tree, package))
