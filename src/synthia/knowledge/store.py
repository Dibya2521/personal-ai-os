"""knowledge.db: sources, documents, chunks, their words, vectors and links.

The file is rebuildable from the sources, so it is its own file beside
memory.db: a rebuild or a corrupt index never touches what cannot be rebuilt.
It follows :class:`synthia.memory.store.MemoryStore`'s rules: WAL set once, a
connection per call, ``BEGIN IMMEDIATE`` for writes, every public method run
off the event loop, FTS5 external content kept by triggers.

Three choices carry the module:

- Chunks are immutable. A changed document is replaced whole in one
  transaction, so a crash leaves the old, consistent document, which the next
  run sees as changed. Ids are ``AUTOINCREMENT``, never reused, so a vector
  held in memory can never be mistaken for a new chunk's.
- Which document a link reaches is derived, never stored: the view
  ``resolved_links`` matches by kind each time it is read. Adding or renaming
  a note re-resolves every link to it without a pass.
- Cascades carry every removal: a source takes its documents, a document its
  chunks, a chunk its vector, its links and (by trigger) its words.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final

from synthia.knowledge.model import (
    Chunk,
    ChunkId,
    Document,
    DocumentId,
    DocumentKind,
    Source,
    SourceId,
    SourceKind,
    module_name,
)
from synthia.retrieval.fts import FTS_TOKENIZER, match_expression
from synthia.retrieval.fusion import WordScores

if TYPE_CHECKING:
    from collections.abc import Collection, Generator, Iterable, Sequence

    from synthia.knowledge.model import Draft

KNOWLEDGE_FILE: Final = Path("knowledge.db")
SCHEMA_VERSION: Final = 1
BUSY_TIMEOUT_S: Final = 5.0

_SCHEMA: Final = (
    """
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    root TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL CHECK (kind IN ('files', 'vault')),
    private INTEGER NOT NULL CHECK (private IN (0, 1)),
    excludes TEXT NOT NULL
)
""",
    """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('markdown', 'python', 'text', 'pdf')),
    mtime_ns INTEGER NOT NULL,
    size INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    recipe TEXT NOT NULL,
    title TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    stem_key TEXT NOT NULL,
    module_key TEXT,
    UNIQUE (source, path)
)
""",
    "CREATE INDEX IF NOT EXISTS documents_by_stem ON documents(source, stem_key)",
    """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    page INTEGER,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    header TEXT NOT NULL,
    body TEXT NOT NULL,
    cut INTEGER NOT NULL CHECK (cut IN (0, 1)),
    UNIQUE (document, ordinal),
    CHECK (start_line >= 1 AND end_line >= start_line)
)
""",
    f"""
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_text USING fts5(
    header, body,
    content='chunks', content_rowid='id', tokenize='{FTS_TOKENIZER}'
)
""",
    """
CREATE TRIGGER IF NOT EXISTS chunk_indexed AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_text(rowid, header, body)
    VALUES (new.id, new.header, new.body);
END
""",
    """
CREATE TRIGGER IF NOT EXISTS chunk_unindexed AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_text(chunks_text, rowid, header, body)
    VALUES ('delete', old.id, old.header, old.body);
END
""",
    """
CREATE TABLE IF NOT EXISTS chunk_vectors (
    chunk INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    vector BLOB NOT NULL
)
""",
    """
CREATE TABLE IF NOT EXISTS links (
    chunk INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('wikilink', 'related', 'markdown', 'import')),
    target TEXT NOT NULL,
    key TEXT NOT NULL,
    anchor TEXT,
    PRIMARY KEY (chunk, position)
)
""",
    # A note beats another file of its stem (as in Obsidian), an exact module a
    # dotted suffix (substr: "_" is a LIKE wildcard); a document never links to
    # itself. Subqueries, not a window: SQLite does not push a WHERE on chunk
    # through a window function.
    """
CREATE VIEW IF NOT EXISTS resolved_links AS
SELECT l.chunk, l.position, l.kind, l.anchor, d.id AS document, d.title
FROM links l
JOIN chunks c ON c.id = l.chunk
JOIN documents here ON here.id = c.document
JOIN documents d ON d.id = CASE
    WHEN l.kind IN ('wikilink', 'related') THEN (
        SELECT t.id FROM documents t
        WHERE t.source = here.source AND t.stem_key = l.key
        ORDER BY t.kind != 'markdown', length(t.path), t.path
        LIMIT 1)
    WHEN l.kind = 'markdown' THEN (
        SELECT t.id FROM documents t
        WHERE t.source = here.source AND t.path = l.key)
    ELSE coalesce((
        SELECT t.id FROM documents t
        WHERE t.source = here.source AND t.module_key = l.key
        ORDER BY length(t.path), t.path
        LIMIT 1), (
        SELECT t.id FROM documents t
        WHERE t.source = here.source
            AND substr(t.module_key, -length(l.key) - 1) = '.' || l.key
        ORDER BY length(t.path), t.path
        LIMIT 1))
END
WHERE d.id != here.id
""",
)
_CHUNK_COLUMNS: Final = (
    "c.id, c.document, c.ordinal, c.page, c.start_line, c.end_line,"
    " c.header, c.body, c.cut"
)
_DOCUMENT_COLUMNS: Final = (
    "d.id, d.source, d.path, d.kind, d.mtime_ns, d.size, d.sha256, d.recipe,"
    " d.title, d.note"
)
_SOURCE_COLUMNS: Final = "id, name, root, kind, private, excludes"

type _ChunkRow = tuple[int, int, int, int | None, int, int, str, str, int]
type _DocumentRow = tuple[int, int, str, str, int, int, str, str, str, str]
type _SourceRow = tuple[int, str, str, str, int, str]


@dataclass(frozen=True, slots=True)
class Stamp:
    """What decides whether a document must be read again."""

    document: DocumentId
    mtime_ns: int
    size: int
    sha256: str
    recipe: str


@dataclass(frozen=True, slots=True)
class Reading:
    """A document as read: everything :meth:`KnowledgeStore.replace_document` writes."""

    source: SourceId
    path: PurePosixPath
    kind: DocumentKind
    mtime_ns: int
    size: int
    sha256: str
    recipe: str
    title: str
    note: str
    drafts: tuple[Draft, ...]


@dataclass(frozen=True, slots=True)
class Replaced:
    """What a replace did: the document, the chunk ids removed, how many were added."""

    document: DocumentId
    removed: tuple[ChunkId, ...]
    added: int


@dataclass(frozen=True, slots=True)
class Row:
    """A chunk with its document and source, as a passage is built from it."""

    chunk: Chunk
    document: Document
    source_name: str
    private: bool


@dataclass(frozen=True, slots=True)
class LinkTarget:
    """A document a chunk's link reaches, and the heading it names, if any."""

    document: DocumentId
    title: str
    anchor: str | None


def _stem_key(path: PurePosixPath) -> str:
    return path.stem.casefold()


def _ids(ids: Iterable[int]) -> str:
    return json.dumps(list(ids))


def _source(row: _SourceRow) -> Source:
    id_, name, root, kind, private, excludes = row
    return Source(
        SourceId(id_),
        name,
        Path(root),
        SourceKind(kind),
        bool(private),
        tuple(PurePosixPath(p) for p in json.loads(excludes)),
    )


def _document(row: _DocumentRow) -> Document:
    id_, source, path, kind, mtime_ns, size, sha256, recipe, title, note = row
    return Document(
        DocumentId(id_),
        SourceId(source),
        PurePosixPath(path),
        DocumentKind(kind),
        mtime_ns,
        size,
        sha256,
        recipe,
        title,
        note,
    )


def _chunk(row: _ChunkRow) -> Chunk:
    id_, document, ordinal, page, start_line, end_line, header, body, cut = row
    return Chunk(
        ChunkId(id_),
        DocumentId(document),
        ordinal,
        page,
        start_line,
        end_line,
        header,
        body,
        bool(cut),
    )


class KnowledgeStore:
    """The knowledge index in one SQLite file.

    Writes are made by one actor at a time; reads may run beside a write,
    which WAL allows.
    """

    def __init__(self, path: Path) -> None:
        """Open ``path``, creating the file and its tables if needed.

        Raises:
            sqlite3.DatabaseError: If the file is not a database, or was made
                by a newer SYNTHIA (its ``user_version`` is above
                :data:`SCHEMA_VERSION`).
        """
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._opened() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
        with self._transaction() as connection:
            (version,) = connection.execute("PRAGMA user_version").fetchone()
            if version > SCHEMA_VERSION:
                message = f"{path} has schema {version}, newer than {SCHEMA_VERSION}"
                raise sqlite3.DatabaseError(message)
            for statement in _SCHEMA:
                connection.execute(statement)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    async def put_source(
        self,
        root: Path,
        name: str,
        kind: SourceKind,
        *,
        private: bool,
        excludes: Sequence[PurePosixPath],
    ) -> Source:
        """Add the source at ``root``, or update its kind and flags if already added.

        Idempotent: adding the same root twice leaves one source, under the
        name it was first added with.

        Raises:
            sqlite3.IntegrityError: If another root already has ``name``.
        """
        return await asyncio.to_thread(
            self._put_source, root, name, kind, private=private, excludes=excludes
        )

    async def remove_source(self, source: SourceId) -> tuple[ChunkId, ...]:
        """Remove ``source`` and everything under it; return the chunk ids removed."""
        return await asyncio.to_thread(self._remove_source, source)

    async def sources(self) -> list[Source]:
        """Return every source, in the order added."""
        return await asyncio.to_thread(self._sources)

    async def stamps(self, source: SourceId) -> dict[PurePosixPath, Stamp]:
        """Return each document of ``source`` by path, in one query."""
        return await asyncio.to_thread(self._stamps, source)

    async def touch(self, document: DocumentId, mtime_ns: int, size: int) -> None:
        """Record a new stat for a document whose bytes did not change."""
        await asyncio.to_thread(self._touch, document, mtime_ns, size)

    async def replace_document(self, reading: Reading) -> Replaced:
        """Make ``reading`` the document at its path, in one transaction.

        Idempotent: when the stored sha256 and recipe equal ``reading``'s,
        nothing is written, ``removed`` is empty and ``added`` is 0. A
        document replaced keeps its id; its chunks get new ones.
        """
        return await asyncio.to_thread(self._replace, reading)

    async def remove_documents(
        self, documents: Collection[DocumentId]
    ) -> tuple[ChunkId, ...]:
        """Remove ``documents``; return the chunk ids removed."""
        return await asyncio.to_thread(self._remove_documents, documents)

    async def unembedded(self, model: str, limit: int) -> list[tuple[ChunkId, str]]:
        """Return up to ``limit`` chunks with no vector from ``model``, oldest first.

        Each comes with its indexed text, what the embedder reads.
        """
        return await asyncio.to_thread(self._unembedded, model, limit)

    async def put_vectors(
        self, model: str, vectors: Sequence[tuple[ChunkId, bytes]]
    ) -> list[ChunkId]:
        """Keep each chunk's vector from ``model``, replacing any before it.

        Return the chunks whose vector was kept: a chunk removed meanwhile is
        skipped.
        """
        return await asyncio.to_thread(self._put_vectors, model, vectors)

    async def vectors(self, model: str) -> list[tuple[ChunkId, bytes]]:
        """Return every vector kept from ``model``, by chunk id."""
        return await asyncio.to_thread(self._vectors, model)

    async def word_scores(
        self, query: str, *, limit: int, within: Collection[ChunkId] | None = None
    ) -> WordScores:
        """Return the ``limit`` chunks best for ``query``'s words, by BM25.

        With ``within``, only those chunks are scored. ``indexed`` is the
        number of chunks, which the fused score's scale needs.
        """
        return await asyncio.to_thread(self._word_scores, query, limit, within)

    async def rows(self, chunks: Sequence[ChunkId]) -> list[Row]:
        """Return the chunks still present with their document and source, in order."""
        return await asyncio.to_thread(self._rows, chunks)

    async def linked(
        self, chunks: Collection[ChunkId]
    ) -> dict[ChunkId, tuple[LinkTarget, ...]]:
        """Return, per chunk, each document its links resolve to, in link order.

        A chunk none of whose links resolve is left out.
        """
        return await asyncio.to_thread(self._linked, chunks)

    async def optimize(self) -> None:
        """Merge the full-text index's segments; refresh the planner's statistics."""
        await asyncio.to_thread(self._optimize)

    @contextmanager
    def _opened(self) -> Generator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self._path, timeout=BUSY_TIMEOUT_S, isolation_level=None
        )
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection]:
        with self._opened() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            connection.execute("COMMIT")

    def _put_source(
        self,
        root: Path,
        name: str,
        kind: SourceKind,
        *,
        private: bool,
        excludes: Sequence[PurePosixPath],
    ) -> Source:
        with self._opened() as connection:
            row = connection.execute(
                "INSERT INTO sources (name, root, kind, private, excludes)"  # noqa: S608 - constant columns
                " VALUES (?, ?, ?, ?, ?) ON CONFLICT (root) DO UPDATE"
                " SET kind = excluded.kind, private = excluded.private,"
                f" excludes = excluded.excludes RETURNING {_SOURCE_COLUMNS}",
                (
                    name,
                    str(root),
                    kind.value,
                    int(private),
                    json.dumps([str(p) for p in excludes]),
                ),
            ).fetchone()
        return _source(row)

    def _remove_source(self, source: SourceId) -> tuple[ChunkId, ...]:
        with self._transaction() as connection:
            removed = tuple(
                ChunkId(chunk)
                for (chunk,) in connection.execute(
                    "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document"
                    " WHERE d.source = ? ORDER BY c.id",
                    (source,),
                )
            )
            connection.execute("DELETE FROM sources WHERE id = ?", (source,))
        return removed

    def _sources(self) -> list[Source]:
        with self._opened() as connection:
            rows = connection.execute(
                f"SELECT {_SOURCE_COLUMNS} FROM sources ORDER BY id"  # noqa: S608 - constant columns
            ).fetchall()
        return [_source(row) for row in rows]

    def _stamps(self, source: SourceId) -> dict[PurePosixPath, Stamp]:
        with self._opened() as connection:
            rows = connection.execute(
                "SELECT path, id, mtime_ns, size, sha256, recipe FROM documents"
                " WHERE source = ?",
                (source,),
            ).fetchall()
        return {
            PurePosixPath(path): Stamp(DocumentId(id_), mtime_ns, size, sha, recipe)
            for path, id_, mtime_ns, size, sha, recipe in rows
        }

    def _touch(self, document: DocumentId, mtime_ns: int, size: int) -> None:
        with self._opened() as connection:
            connection.execute(
                "UPDATE documents SET mtime_ns = ?, size = ? WHERE id = ?",
                (mtime_ns, size, document),
            )

    def _replace(self, reading: Reading) -> Replaced:
        with self._transaction() as connection:
            old = connection.execute(
                "SELECT id, sha256, recipe FROM documents"
                " WHERE source = ? AND path = ?",
                (reading.source, str(reading.path)),
            ).fetchone()
            if old is not None and old[1:] == (reading.sha256, reading.recipe):
                return Replaced(DocumentId(old[0]), (), 0)
            previous = None if old is None else old[0]
            removed = tuple(
                ChunkId(chunk)
                for (chunk,) in connection.execute(
                    "SELECT id FROM chunks WHERE document = ? ORDER BY id", (previous,)
                )
            )
            connection.execute("DELETE FROM chunks WHERE document = ?", (previous,))
            document = self._upsert_document(connection, reading)
            for draft in reading.drafts:
                self._insert_chunk(connection, document, draft)
        return Replaced(document, removed, len(reading.drafts))

    @staticmethod
    def _upsert_document(
        connection: sqlite3.Connection, reading: Reading
    ) -> DocumentId:
        (document,) = connection.execute(
            "INSERT INTO documents (source, path, kind, mtime_ns, size, sha256,"
            " recipe, title, note, stem_key, module_key)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (source, path) DO UPDATE SET kind = excluded.kind,"
            " mtime_ns = excluded.mtime_ns, size = excluded.size,"
            " sha256 = excluded.sha256, recipe = excluded.recipe,"
            " title = excluded.title, note = excluded.note RETURNING id",
            (
                reading.source,
                str(reading.path),
                reading.kind.value,
                reading.mtime_ns,
                reading.size,
                reading.sha256,
                reading.recipe,
                reading.title,
                reading.note,
                _stem_key(reading.path),
                module_name(reading.path),
            ),
        ).fetchone()
        return DocumentId(document)

    @staticmethod
    def _insert_chunk(
        connection: sqlite3.Connection, document: DocumentId, draft: Draft
    ) -> None:
        cursor = connection.execute(
            "INSERT INTO chunks (document, ordinal, page, start_line, end_line,"
            " header, body, cut) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                document,
                draft.ordinal,
                draft.page,
                draft.start_line,
                draft.end_line,
                draft.header,
                draft.body,
                int(draft.cut),
            ),
        )
        connection.executemany(
            "INSERT INTO links (chunk, position, kind, target, key, anchor)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    cursor.lastrowid,
                    position,
                    link.kind.value,
                    link.target,
                    link.key,
                    link.anchor,
                )
                for position, link in enumerate(draft.links)
            ],
        )

    def _remove_documents(
        self, documents: Collection[DocumentId]
    ) -> tuple[ChunkId, ...]:
        wanted = _ids(documents)
        with self._transaction() as connection:
            removed = tuple(
                ChunkId(chunk)
                for (chunk,) in connection.execute(
                    "SELECT id FROM chunks WHERE document IN"
                    " (SELECT value FROM json_each(?)) ORDER BY id",
                    (wanted,),
                )
            )
            connection.execute(
                "DELETE FROM documents WHERE id IN (SELECT value FROM json_each(?))",
                (wanted,),
            )
        return removed

    def _unembedded(self, model: str, limit: int) -> list[tuple[ChunkId, str]]:
        with self._opened() as connection:
            rows = connection.execute(
                f"SELECT {_CHUNK_COLUMNS} FROM chunks c"  # noqa: S608 - constant columns
                " LEFT JOIN chunk_vectors v ON v.chunk = c.id AND v.model = ?"
                " WHERE v.chunk IS NULL ORDER BY c.id LIMIT ?",
                (model, limit),
            ).fetchall()
        chunks = [_chunk(row) for row in rows]
        return [(chunk.id, chunk.indexed_text()) for chunk in chunks]

    def _put_vectors(
        self, model: str, vectors: Sequence[tuple[ChunkId, bytes]]
    ) -> list[ChunkId]:
        kept: list[ChunkId] = []
        with self._transaction() as connection:
            for chunk, vector in vectors:
                cursor = connection.execute(
                    "INSERT OR REPLACE INTO chunk_vectors (chunk, model, vector)"
                    " SELECT ?, ?, ? WHERE EXISTS (SELECT 1 FROM chunks WHERE id = ?)",
                    (chunk, model, vector, chunk),
                )
                if cursor.rowcount:
                    kept.append(chunk)
        return kept

    def _vectors(self, model: str) -> list[tuple[ChunkId, bytes]]:
        with self._opened() as connection:
            rows = connection.execute(
                "SELECT chunk, vector FROM chunk_vectors"
                " WHERE model = ? ORDER BY chunk",
                (model,),
            ).fetchall()
        return [(ChunkId(chunk), vector) for chunk, vector in rows]

    def _word_scores(
        self, query: str, limit: int, within: Collection[ChunkId] | None
    ) -> WordScores:
        expression = match_expression(query)
        with self._opened() as connection:
            (count,) = connection.execute("SELECT count(*) FROM chunks").fetchone()
            if not expression:
                return WordScores({}, count)
            rows = connection.execute(
                "SELECT rowid, bm25(chunks_text) FROM chunks_text"
                " WHERE chunks_text MATCH :expression AND (:within IS NULL"
                " OR rowid IN (SELECT value FROM json_each(:within)))"
                " ORDER BY bm25(chunks_text), rowid LIMIT :limit",
                {
                    "expression": expression,
                    "within": None if within is None else _ids(within),
                    "limit": limit,
                },
            ).fetchall()
        # FTS5 negates BM25 so that ascending order is best first.
        return WordScores({chunk: -score for chunk, score in rows}, count)

    def _rows(self, chunks: Sequence[ChunkId]) -> list[Row]:
        with self._opened() as connection:
            rows = connection.execute(
                f"SELECT {_CHUNK_COLUMNS}, {_DOCUMENT_COLUMNS}, s.name, s.private"  # noqa: S608 - constant columns
                " FROM chunks c JOIN documents d ON d.id = c.document"
                " JOIN sources s ON s.id = d.source"
                " WHERE c.id IN (SELECT value FROM json_each(?))",
                (_ids(chunks),),
            ).fetchall()
        found = {
            row[0]: Row(_chunk(row[:9]), _document(row[9:19]), row[19], bool(row[20]))
            for row in rows
        }
        return [found[chunk] for chunk in chunks if chunk in found]

    def _linked(
        self, chunks: Collection[ChunkId]
    ) -> dict[ChunkId, tuple[LinkTarget, ...]]:
        with self._opened() as connection:
            rows = connection.execute(
                "SELECT chunk, document, title, anchor FROM resolved_links"
                " WHERE chunk IN (SELECT value FROM json_each(?))"
                " ORDER BY chunk, position",
                (_ids(chunks),),
            ).fetchall()
        targets: defaultdict[ChunkId, list[LinkTarget]] = defaultdict(list)
        for chunk, document, title, anchor in rows:
            targets[ChunkId(chunk)].append(
                LinkTarget(DocumentId(document), title, anchor)
            )
        return {chunk: tuple(found) for chunk, found in targets.items()}

    def _optimize(self) -> None:
        with self._opened() as connection:
            connection.execute(
                "INSERT INTO chunks_text (chunks_text) VALUES ('optimize')"
            )
            connection.execute("PRAGMA optimize")
