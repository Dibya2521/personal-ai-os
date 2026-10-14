r"""What knowledge is made of: sources, documents, chunks and the links between them.

Every type here is a frozen record. Line numbers are 1-based and inclusive,
counted the way an editor counts them: a line ends at ``\n`` only. A PDF
counts lines within each page.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, NewType

if TYPE_CHECKING:
    from pathlib import Path, PurePosixPath

SourceId = NewType("SourceId", int)
DocumentId = NewType("DocumentId", int)
ChunkId = NewType("ChunkId", int)


class SourceKind(StrEnum):
    """How a source was found: a plain folder, or an Obsidian-style vault.

    Informational: links resolve by the same rule in both. Stored as text,
    hence ``StrEnum``.
    """

    FILES = "files"
    VAULT = "vault"


class DocumentKind(StrEnum):
    """Which reader a document went through."""

    MARKDOWN = "markdown"
    PYTHON = "python"
    TEXT = "text"
    PDF = "pdf"


class LinkKind(StrEnum):
    """Where a link was found, which also decides how its target is matched.

    ``WIKILINK`` and ``RELATED`` match a document's casefolded file stem,
    ``MARKDOWN`` its path relative to the source root, ``IMPORT`` its dotted
    module name, whole or as a dotted suffix. Only documents of the link's
    own source are matched.
    """

    WIKILINK = "wikilink"
    RELATED = "related"
    MARKDOWN = "markdown"
    IMPORT = "import"


@dataclass(frozen=True, slots=True)
class Source:
    """A folder the person added, read-only to SYNTHIA.

    ``root`` is the resolved real path. ``name`` is unique among sources and
    starts every citation from this source. ``excludes`` are posix paths
    relative to ``root``: the folders the person keeps out.
    """

    id: SourceId
    name: str
    root: Path
    kind: SourceKind
    private: bool
    excludes: tuple[PurePosixPath, ...]


@dataclass(frozen=True, slots=True)
class Document:
    """One file of a source, as last indexed.

    ``recipe`` names how it was chunked; a document whose recipe differs from
    the index's is read again even when its bytes did not change. ``note``
    says what went wrong in reading, "" when nothing did.
    """

    id: DocumentId
    source: SourceId
    path: PurePosixPath
    kind: DocumentKind
    mtime_ns: int
    size: int
    sha256: str
    recipe: str
    title: str
    note: str = ""


@dataclass(frozen=True, slots=True)
class Chunk:
    """A run of one document's lines, the unit that is found and cited.

    ``header`` is the contextual path ("ADR 0012 Memory > Decision > Fusion"),
    or "" when there is none. Embedding and the word index see ``header`` and
    ``body`` together; a citation names ``body``'s lines only. ``page`` is set
    for a PDF and None otherwise. ``cut`` marks a single line too long for the
    embedder, which reads only its start.
    """

    id: ChunkId
    document: DocumentId
    ordinal: int
    page: int | None
    start_line: int
    end_line: int
    header: str
    body: str
    cut: bool = False

    def indexed_text(self) -> str:
        """Return what the embedder and the word index read: header, then body."""
        return f"{self.header}\n{self.body}" if self.header else self.body


@dataclass(frozen=True, slots=True)
class Link:
    """A link written in a chunk, kept whether or not its target exists.

    ``target`` is as written ("Fusion notes", "../adr/0012.md",
    "synthia.retrieval.fusion"); ``key`` is its normalised form for matching
    (see :class:`LinkKind`); ``anchor`` is the heading after ``#``, if any.
    Which document it reaches is never stored but derived when read, so a
    note renamed or added later resolves without a pass over the links.
    """

    kind: LinkKind
    target: str
    key: str
    anchor: str | None = None


@dataclass(frozen=True, slots=True)
class Draft:
    """A chunk as the chunker made it, before the store gives it an id."""

    ordinal: int
    page: int | None
    start_line: int
    end_line: int
    header: str
    body: str
    links: tuple[Link, ...]
    cut: bool = False
