"""Knowledge: the person's folders, indexed and searchable, behind one object.

It holds the store, the chunks' vectors and the indexer, and one lock under
which every write runs: adding a source, removing one, and an index pass. A
``remove`` that arrives mid-pass therefore waits for the pass, instead of
deleting a source the pass is still writing. Searches take no lock: SQLite's
WAL lets them read beside a writer, and the held vectors change only on the
event loop, between a search's awaits.
"""

from __future__ import annotations

import asyncio
import itertools
from contextlib import asynccontextmanager
from pathlib import PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Final, Self

from synthia.knowledge.chunk import Chunking
from synthia.knowledge.index import Indexer, Unsplit
from synthia.knowledge.meaning import ChunkVectors
from synthia.knowledge.model import SourceKind
from synthia.knowledge.retrieve import DEFAULT_VARIANT, Ranker
from synthia.knowledge.store import KnowledgeStore

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterable, Sequence
    from pathlib import Path

    from synthia.knowledge.index import IndexReport, Pause
    from synthia.knowledge.model import ChunkId, Source, SourceId
    from synthia.knowledge.retrieve import Found, Passage, Variant
    from synthia.knowledge.store import LinkTarget
    from synthia.retrieval.embed import TextEmbedder

SEARCH_LIMIT: Final = 6
VAULT_MARKER: Final = ".obsidian"
"""The settings folder Obsidian keeps in a vault's root."""
UNNAMED: Final = "source"
"""The name of a source whose root has none, such as a drive's root."""
FIRST_SUFFIX: Final = 2
"""A second folder of a name is ``name-2``, then ``name-3``."""
_PARENT: Final = ".."


class KnowledgeError(ValueError):
    """A request knowledge refuses: a folder it cannot add, or a bad exclude."""


def _exclude(given: PurePosixPath) -> PurePosixPath:
    r"""Return ``given`` as the walker compares it, with ``/`` between names.

    A backslash separates names, as on Windows, so ``diary\`` keeps the
    folder ``diary`` out instead of matching nothing.

    Raises:
        KnowledgeError: If ``given`` is absolute, climbs out of the root with
            ``..``, or is the root itself.
    """
    windows = PureWindowsPath(str(given))
    exclude = PurePosixPath(windows.as_posix())
    if exclude.is_absolute() or windows.anchor:
        message = f"the exclude {given} is absolute: name a folder inside the root"
        raise KnowledgeError(message)
    if _PARENT in exclude.parts:
        message = f"the exclude {given} climbs out of the root with {_PARENT}"
        raise KnowledgeError(message)
    if not exclude.parts:
        message = f"the exclude {given} is the whole root"
        raise KnowledgeError(message)
    return exclude


def _folder(root: Path) -> tuple[Path, SourceKind]:
    """Return ``root`` resolved and its kind; blocking."""
    try:
        resolved = root.resolve(strict=True)
    except OSError as error:
        message = f"{root} cannot be added: {error}"
        raise KnowledgeError(message) from error
    if not resolved.is_dir():
        message = f"{root} is not a folder"
        raise KnowledgeError(message)
    vault = (resolved / VAULT_MARKER).is_dir()
    return resolved, SourceKind.VAULT if vault else SourceKind.FILES


def _check_apart(root: Path, sources: Iterable[Source]) -> None:
    for source in sources:
        if source.root in root.parents or root in source.root.parents:
            message = f"{root} overlaps the source {source.name} at {source.root}"
            raise KnowledgeError(message)


def _free_name(root: Path, sources: Iterable[Source]) -> str:
    """Return the name a new source at ``root`` is added under."""
    taken = {source.name for source in sources}
    base = root.name or UNNAMED
    numbered = (f"{base}-{n}" for n in itertools.count(FIRST_SUFFIX))
    return next(name for name in itertools.chain([base], numbered) if name not in taken)


class Knowledge:
    """The person's folders, indexed and searchable."""

    def __init__(
        self, store: KnowledgeStore, indexer: Indexer, vectors: ChunkVectors
    ) -> None:
        """Serve ``store``, kept in step by ``indexer``, searched through ``vectors``.

        :meth:`open` builds all three over one store.
        """
        self._store = store
        self._indexer = indexer
        self._vectors = vectors
        self._ranker = Ranker(store, vectors)
        self._writing = asyncio.Lock()
        self._queued = 0
        self._steps = Unsplit()

    @classmethod
    async def open(
        cls, path: Path, embedder: TextEmbedder, *, chunking: Chunking | None = None
    ) -> Self:
        """Open the index at ``path`` and hold the vectors it keeps from ``embedder``.

        ``embedder`` also counts the chunker's tokens. ``chunking`` defaults
        to :meth:`Chunking.for_embedder`; a document cut another way is read
        again on the next pass.

        Raises:
            sqlite3.DatabaseError: If the file is not the index, or a newer one.
            OSError: If its folder cannot be created.
        """
        store = await asyncio.to_thread(KnowledgeStore, path)
        vectors = ChunkVectors(store, embedder)
        await vectors.load()
        if chunking is None:
            chunking = Chunking.for_embedder(embedder.spec)
        return cls(store, Indexer(store, chunking, embedder, embedder), vectors)

    @property
    def write_waiting(self) -> bool:
        """Whether an add, a remove or a pass waits for the one running."""
        return self._queued > 0

    @asynccontextmanager
    async def _write(self) -> AsyncGenerator[None]:
        self._queued += 1
        try:
            await self._writing.acquire()
        finally:
            self._queued -= 1
        try:
            yield
        finally:
            self._writing.release()

    async def add(
        self,
        root: Path,
        *,
        private: bool | None = None,
        excludes: Sequence[PurePosixPath] | None = None,
    ) -> Source:
        """Add the folder ``root`` as a source, or update its flags if added before.

        Its kind is vault when ``root/.obsidian`` is a folder. Its name is the
        folder's, then ``-2``, ``-3`` while that is taken. ``private`` and
        ``excludes`` left None keep what the source has; a new source is then
        not private and excludes nothing. An exclude is stored with ``/``
        between names, a backslash read as one too. Nothing is indexed until
        :meth:`index`.

        Raises:
            KnowledgeError: If ``root`` is not an existing folder, is inside
                or holds another source's root, or an exclude is absolute,
                climbs out with ``..`` or is the root itself.
        """
        wanted = None if excludes is None else [_exclude(e) for e in excludes]
        async with self._write():
            resolved, kind = await asyncio.to_thread(_folder, root)
            sources = await self._store.sources()
            _check_apart(resolved, sources)
            earlier = next((s for s in sources if s.root == resolved), None)
            if earlier is None:
                name, was_private, were_excluded = (
                    _free_name(resolved, sources),
                    False,
                    (),
                )
            else:
                name, was_private, were_excluded = (
                    earlier.name,
                    earlier.private,
                    earlier.excludes,
                )
            return await self._store.put_source(
                resolved,
                name,
                kind,
                private=was_private if private is None else private,
                excludes=were_excluded if wanted is None else wanted,
            )

    async def remove(self, root: Path) -> bool:
        """Remove the source at ``root`` and everything indexed from it; False if none.

        ``root`` may be gone from the disk. Cancelled once the store has begun
        the removal, it finishes it, vectors forgotten, before the caller sees
        the cancellation.
        """
        async with self._write():
            sources = await self._store.sources()
            resolved = await asyncio.to_thread(root.resolve)
            for source in sources:
                if source.root == resolved:
                    await self._steps.run(self._removed(source.id))
                    return True
            return False

    async def _removed(self, source: SourceId) -> None:
        self._vectors.forget(await self._store.remove_source(source))

    async def sources(self) -> list[Source]:
        """Return every source, in the order added."""
        return await self._store.sources()

    async def index(self, pause: Pause | None = None) -> IndexReport:
        """Bring every source in step with its files, and embed what is new.

        ``pause`` is awaited before each file and each embedding round; what
        it raises ends the pass there, keeping what was written.
        """
        async with self._write():
            return await self._indexer.run(
                pause=pause,
                on_removed=self._vectors.forget,
                on_embedded=self._vectors.hold,
            )

    async def search(
        self,
        query: str,
        *,
        limit: int = SEARCH_LIMIT,
        variant: Variant | None = None,
        private: bool = True,
    ) -> Found:
        """Return up to ``limit`` passages for ``query``, best first.

        ``variant`` None is the default ranking. With ``private`` False,
        passages of private sources are left out and counted.

        Raises:
            ValueError: If ``limit`` is below 1.
        """
        if limit < 1:
            message = f"limit={limit} is below 1"
            raise ValueError(message)
        chosen = DEFAULT_VARIANT if variant is None else variant
        return await self._ranker.find(
            query, limit=limit, variant=chosen, private=private
        )

    async def links(
        self, passages: Iterable[Passage]
    ) -> dict[ChunkId, tuple[LinkTarget, ...]]:
        """Return, per passage's chunk, the documents its links reach, in link order.

        A link reaches only documents of its own source, so a public passage
        never names a private document. A chunk none of whose links reach a
        document is left out.
        """
        return await self._store.linked({passage.chunk.id for passage in passages})
