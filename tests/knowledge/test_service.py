import asyncio
import shutil
import threading
from collections.abc import Collection, Coroutine, Generator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import override

import pytest

from synthia.knowledge.index import IndexReport
from synthia.knowledge.model import ChunkId, DocumentId, Source, SourceId, SourceKind
from synthia.knowledge.retrieve import Found, Side, Variant
from synthia.knowledge.service import Knowledge, KnowledgeError
from synthia.knowledge.store import KnowledgeStore, Reading, Replaced
from tests.knowledge.corpus import (
    HEADLESS,
    NOTES,
    Wired,
    citations,
    opened,
    planted,
    wired,
)
from tests.retrieval.toy import Network, embedder
from tests.timing import HANG_TIMEOUT_S

MEANING = Variant(Side.MEANING)
WORDS = Variant(Side.WORDS)


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "home" / "knowledge.db"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return planted(tmp_path / "notes", NOTES)


@pytest.fixture
async def knowledge(db: Path) -> Knowledge:
    return await opened(db)


async def test_open_chunks_at_the_embedders_limit_by_default(
    db: Path, root: Path
) -> None:
    knowledge = await Knowledge.open(db, embedder(Network()))
    await knowledge.add(root)
    await knowledge.index()

    found = await knowledge.search("miso", variant=WORDS)

    (passage,) = found.passages
    assert passage.document.recipe == "structure+header/32/40/r1/s2"
    assert passage.chunk.header == "cat.txt"


async def test_open_holds_the_vectors_the_index_kept(db: Path, root: Path) -> None:
    first = await opened(db)
    await first.add(root)
    await first.index()
    network = Network()
    model = embedder(network)
    network.texts = 0

    second = await Knowledge.open(db, model, chunking=HEADLESS)
    found = await second.search("pet", limit=1, variant=MEANING)

    assert citations(found) == ["notes/cat.txt:1"]
    assert network.texts == 1


async def test_index_holds_new_vectors_and_forgets_replaced_ones(
    db: Path, root: Path
) -> None:
    shelf = wired(KnowledgeStore(db))
    await shelf.knowledge.add(root)
    await shelf.knowledge.index()
    (root / "cat.txt").write_bytes(b"my sibling is miso\n")

    report = await shelf.knowledge.index()
    found = await shelf.knowledge.search("sibling", limit=1, variant=MEANING)

    assert replace(report, seconds=0.0) == IndexReport(
        changed=1, unchanged=2, chunks=1, embedded=1
    )
    assert len(shelf.vectors) == 3
    (passage,) = found.passages
    assert (passage.chunk.id, passage.chunk.body) == (4, "my sibling is miso")
    assert await shelf.knowledge.search("cat", variant=WORDS) == Found((), 0)


async def test_remove_forgets_every_vector_of_the_source(db: Path, root: Path) -> None:
    shelf = wired(KnowledgeStore(db))
    await shelf.knowledge.add(root)
    await shelf.knowledge.index()

    removed = await shelf.knowledge.remove(root)

    assert removed
    assert len(shelf.vectors) == 0
    assert await shelf.knowledge.sources() == []
    assert await shelf.knowledge.search("pet", variant=MEANING) == Found((), 0)


async def test_remove_finds_a_root_gone_from_the_disk(
    knowledge: Knowledge, root: Path, tmp_path: Path
) -> None:
    other = planted(tmp_path / "other", {"x.txt": "my pet\n"})
    await knowledge.add(other)
    await knowledge.add(root)
    await knowledge.index()
    shutil.rmtree(root)

    removed = await knowledge.remove(root)

    assert removed
    assert [s.name for s in await knowledge.sources()] == ["other"]


async def test_remove_of_a_root_never_added_is_false(
    knowledge: Knowledge, root: Path, tmp_path: Path
) -> None:
    await knowledge.add(root)

    removed = await knowledge.remove(tmp_path / "elsewhere")

    assert not removed
    assert [s.name for s in await knowledge.sources()] == ["notes"]


async def test_add_names_a_source_after_its_folder_and_does_not_index(
    knowledge: Knowledge, root: Path
) -> None:
    source = await knowledge.add(root)

    assert (source.name, source.root, source.kind) == (
        "notes",
        root.resolve(),
        SourceKind.FILES,
    )
    assert (source.private, source.excludes) == (False, ())
    assert await knowledge.sources() == [source]
    assert await knowledge.search("miso") == Found((), 0)


@pytest.mark.parametrize(
    ("folder", "kind"), [(True, SourceKind.VAULT), (False, SourceKind.FILES)]
)
async def test_add_finds_a_vault_by_its_obsidian_folder(
    knowledge: Knowledge, root: Path, folder: bool, kind: SourceKind
) -> None:
    marker = root / ".obsidian"
    if folder:
        marker.mkdir()
    else:
        marker.write_bytes(b"{}")

    source = await knowledge.add(root)

    assert source.kind is kind


async def test_add_numbers_a_name_already_taken(
    knowledge: Knowledge, tmp_path: Path
) -> None:
    roots = [
        tmp_path / "a" / "notes",
        tmp_path / "b" / "notes-2",
        tmp_path / "c" / "notes",
    ]
    names: list[str] = []
    for each in roots:
        each.mkdir(parents=True)
        names.append((await knowledge.add(each)).name)

    assert names == ["notes", "notes-2", "notes-3"]


async def test_add_names_a_root_with_no_name_source(
    knowledge: Knowledge, tmp_path: Path
) -> None:
    drive = Path(tmp_path.anchor)

    source = await knowledge.add(drive)

    assert source.name == "source"


async def test_add_again_updates_the_flags_and_keeps_the_name(
    knowledge: Knowledge, root: Path
) -> None:
    first = await knowledge.add(root)

    again = await knowledge.add(
        root / ".." / "notes", private=True, excludes=[PurePosixPath("drafts")]
    )

    assert again == Source(
        first.id,
        "notes",
        first.root,
        SourceKind.FILES,
        private=True,
        excludes=(PurePosixPath("drafts"),),
    )
    assert await knowledge.sources() == [again]


async def test_add_again_without_flags_keeps_them(
    knowledge: Knowledge, root: Path
) -> None:
    first = await knowledge.add(root, private=True, excludes=[PurePosixPath("drafts")])

    again = await knowledge.add(root)

    assert again == first
    assert (again.private, again.excludes) == (True, (PurePosixPath("drafts"),))


async def test_add_again_changes_only_the_flag_it_names(
    knowledge: Knowledge, root: Path
) -> None:
    await knowledge.add(root, private=True, excludes=[PurePosixPath("drafts")])

    public = await knowledge.add(root, private=False)
    cleared = await knowledge.add(root, excludes=())

    assert (public.private, public.excludes) == (False, (PurePosixPath("drafts"),))
    assert (cleared.private, cleared.excludes) == (False, ())


async def test_add_keeps_an_exclude_inside_the_root(
    knowledge: Knowledge, root: Path
) -> None:
    source = await knowledge.add(
        root, excludes=[PurePosixPath("./drafts"), PurePosixPath("a/b")]
    )

    assert source.excludes == (PurePosixPath("drafts"), PurePosixPath("a/b"))


@pytest.mark.parametrize(("given", "folder"), [("diary\\", "diary"), ("a\\b", "a/b")])
async def test_add_reads_a_backslash_in_an_exclude_as_a_separator(
    knowledge: Knowledge, root: Path, given: str, folder: str
) -> None:
    planted(root / folder, {"pet.txt": "my diary says osprey\n"})

    source = await knowledge.add(root, excludes=[PurePosixPath(given)])
    report = await knowledge.index()

    assert source.excludes == (PurePosixPath(folder),)
    assert replace(report, seconds=0.0) == IndexReport(added=3, chunks=3, embedded=3)
    assert await knowledge.search("osprey", variant=WORDS) == Found((), 0)


async def test_add_refuses_what_is_not_a_folder(
    knowledge: Knowledge, root: Path
) -> None:
    with pytest.raises(KnowledgeError, match=r"cat\.txt is not a folder$"):
        await knowledge.add(root / "cat.txt")
    with pytest.raises(KnowledgeError, match=r"missing cannot be added: "):
        await knowledge.add(root / "missing")
    assert await knowledge.sources() == []


async def test_add_refuses_a_root_inside_or_holding_another(
    knowledge: Knowledge, root: Path, tmp_path: Path
) -> None:
    await knowledge.add(root)
    (root / "sub").mkdir()

    with pytest.raises(KnowledgeError, match=r"overlaps the source notes at "):
        await knowledge.add(root / "sub")
    with pytest.raises(KnowledgeError, match=r"overlaps the source notes at "):
        await knowledge.add(tmp_path)
    assert [s.name for s in await knowledge.sources()] == ["notes"]


@pytest.mark.parametrize(
    ("exclude", "reason"),
    [
        ("/abs", "is absolute"),
        ("//server/share", "is absolute"),
        ("C:/abs", "is absolute"),
        ("C:abs", "is absolute"),
        ("\\\\server\\share\\x", "is absolute"),
        ("\\abs", "is absolute"),
        ("..", "climbs out"),
        ("a/../b", "climbs out"),
        ("a\\..\\b", "climbs out"),
        ("..\\outside", "climbs out"),
        (".", "is the whole root"),
        ("", "is the whole root"),
    ],
)
async def test_add_refuses_an_exclude_outside_the_root(
    knowledge: Knowledge, root: Path, exclude: str, reason: str
) -> None:
    with pytest.raises(KnowledgeError, match=rf"^the exclude .* {reason}"):
        await knowledge.add(root, excludes=[PurePosixPath(exclude)])
    assert await knowledge.sources() == []


class Watched(KnowledgeStore):
    """Records, at each read of the sources, whether an index pass is running."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.indexing: asyncio.Task[IndexReport] | None = None
        self.read_while_indexing: list[bool] = []

    async def sources(self) -> list[Source]:
        running = self.indexing is not None and not self.indexing.done()
        self.read_while_indexing.append(running)
        return await super().sources()


async def test_remove_waits_for_a_running_pass(db: Path, root: Path) -> None:
    store = Watched(db)
    shelf = wired(store)
    await shelf.knowledge.add(root)
    store.read_while_indexing.clear()
    paused, release = asyncio.Event(), asyncio.Event()

    async def pause() -> None:
        paused.set()
        await release.wait()

    store.indexing = asyncio.create_task(shelf.knowledge.index(pause))
    await asyncio.wait_for(paused.wait(), HANG_TIMEOUT_S)
    removal = asyncio.create_task(shelf.knowledge.remove(root))
    release.set()
    report = await asyncio.wait_for(store.indexing, HANG_TIMEOUT_S)
    removed = await asyncio.wait_for(removal, HANG_TIMEOUT_S)

    # The pass read the sources as it began; the removal read them only after.
    assert store.read_while_indexing == [True, False]
    assert replace(report, seconds=0.0) == IndexReport(added=3, chunks=3, embedded=3)
    assert removed
    assert len(shelf.vectors) == 0
    assert await shelf.knowledge.sources() == []


class Gated(KnowledgeStore):
    """Holds the next write named :attr:`gate` in its thread until released."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.gate = ""
        self.entered = threading.Event()
        self.release = threading.Event()
        self.left = threading.Event()

    @contextmanager
    def _gated(self, write: str) -> Generator[None]:
        if write != self.gate:
            yield
            return
        self.gate = ""
        self.entered.set()
        self.release.wait(HANG_TIMEOUT_S)
        try:
            yield
        finally:
            self.left.set()

    @override
    def _replace(self, reading: Reading) -> Replaced:
        with self._gated("replace_document"):
            return super()._replace(reading)

    @override
    def _remove_documents(
        self, documents: Collection[DocumentId]
    ) -> tuple[ChunkId, ...]:
        with self._gated("remove_documents"):
            return super()._remove_documents(documents)

    @override
    def _put_vectors(
        self, model: str, vectors: Sequence[tuple[ChunkId, bytes]]
    ) -> list[ChunkId]:
        with self._gated("put_vectors"):
            return super()._put_vectors(model, vectors)

    @override
    def _remove_source(self, source: SourceId) -> tuple[ChunkId, ...]:
        with self._gated("remove_source"):
            return super()._remove_source(source)


async def cancelled_inside(
    store: Gated, write: str, call: Coroutine[object, object, object]
) -> None:
    """Cancel ``call`` while ``write`` runs in its thread, then let the write end."""
    store.gate = write
    task = asyncio.create_task(call)
    try:
        assert await asyncio.to_thread(store.entered.wait, HANG_TIMEOUT_S)
        task.cancel()
    finally:
        store.release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, HANG_TIMEOUT_S)
    assert await asyncio.to_thread(store.left.wait, HANG_TIMEOUT_S)


async def held_and_kept(shelf: Wired) -> tuple[set[int], set[int]]:
    """Return the chunk ids held for search by meaning, and those the store keeps."""
    near = await shelf.vectors.similarities("pet")
    kept = await shelf.store.vectors(embedder(Network()).spec.id)
    return set(near.nearest(len(shelf.vectors))), {chunk for chunk, _ in kept}


async def test_a_pass_cancelled_while_keeping_vectors_still_holds_them(
    db: Path, root: Path
) -> None:
    store = Gated(db)
    shelf = wired(store)
    await shelf.knowledge.add(root)

    await cancelled_inside(store, "put_vectors", shelf.knowledge.index())

    assert await held_and_kept(shelf) == ({1, 2, 3}, {1, 2, 3})
    found = await shelf.knowledge.search("pet", limit=1, variant=MEANING)
    assert citations(found) == ["notes/cat.txt:1"]


async def test_a_pass_cancelled_while_replacing_a_file_forgets_its_old_chunks(
    db: Path, root: Path
) -> None:
    store = Gated(db)
    shelf = wired(store)
    await shelf.knowledge.add(root)
    await shelf.knowledge.index()
    (root / "cat.txt").write_bytes(b"my sibling is miso\n")

    await cancelled_inside(store, "replace_document", shelf.knowledge.index())

    assert await held_and_kept(shelf) == ({2, 3}, {2, 3})
    await shelf.knowledge.index()
    assert await held_and_kept(shelf) == ({2, 3, 4}, {2, 3, 4})
    found = await shelf.knowledge.search("sibling", limit=1, variant=MEANING)
    assert citations(found) == ["notes/cat.txt:1"]


async def test_a_pass_cancelled_while_removing_a_file_forgets_its_chunks(
    db: Path, root: Path
) -> None:
    store = Gated(db)
    shelf = wired(store)
    await shelf.knowledge.add(root)
    await shelf.knowledge.index()
    (root / "sister.txt").unlink()

    await cancelled_inside(store, "remove_documents", shelf.knowledge.index())

    assert await held_and_kept(shelf) == ({1, 2}, {1, 2})
    found = await shelf.knowledge.search("pet", variant=MEANING)
    assert citations(found) == ["notes/cat.txt:1", "notes/server.txt:1"]


async def test_a_removal_cancelled_in_the_store_still_forgets_the_vectors(
    db: Path, root: Path
) -> None:
    store = Gated(db)
    shelf = wired(store)
    await shelf.knowledge.add(root)
    await shelf.knowledge.index()

    await cancelled_inside(store, "remove_source", shelf.knowledge.remove(root))

    assert await held_and_kept(shelf) == (set(), set())
    assert await shelf.knowledge.sources() == []
    assert await shelf.knowledge.search("pet", variant=MEANING) == Found((), 0)
