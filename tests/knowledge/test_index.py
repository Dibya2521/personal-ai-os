import asyncio
import os
import sqlite3
import sys
import threading
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath

import numpy as np
import pytest
from numpy.typing import NDArray

from synthia.knowledge.chunk import Chunking
from synthia.knowledge.index import MAX_FILE_BYTES, Indexer, IndexReport, Unsplit
from synthia.knowledge.model import ChunkId, Source, SourceKind
from synthia.knowledge.store import KnowledgeStore, Reading, Replaced
from synthia.knowledge.text import Format, Outline, Text, format_of
from tests.knowledge.checks import WORDS
from tests.retrieval.toy import Network, embedder
from tests.timing import HANG_TIMEOUT_S

CHUNKING = Chunking(40)
HELD_BACK = "held back: looks like it holds a secret"
TREE = {
    "a.md": "# Alpha\n\nfirst note\n",
    "pkg/b.py": "def b():\n    return 1\n",
    "c.txt": "plain words\n",
}
INDEXED = {
    "a.md": ("", ("# Alpha\n\nfirst note",)),
    "c.txt": ("", ("plain words",)),
    "pkg/b.py": ("", ("def b():\n    return 1",)),
}


@dataclass
class Spy:
    """The real formats, recording each file decoded and the thread that read it."""

    fails: str = ""
    undecodable: str = ""
    decoded: list[str] = field(default_factory=list[str])
    threads: list[str] = field(default_factory=list[str])

    def __call__(self, path: PurePosixPath) -> Format | None:
        form = format_of(path)
        if form is None:
            return None

        def decode(data: bytes) -> Text:
            self.decoded.append(path.as_posix())
            if path.as_posix() == self.undecodable:
                message = "the decoder broke"
                raise RuntimeError(message)
            return form.decode(data)

        def read(text: Text, path: PurePosixPath) -> Outline:
            self.threads.append(threading.current_thread().name)
            if path.as_posix() == self.fails:
                message = "boom"
                raise ValueError(message)
            return form.read(text, path)

        return replace(form, decode=decode, read=read)


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "home" / "knowledge.db"


@pytest.fixture
def store(db: Path) -> KnowledgeStore:
    return KnowledgeStore(db)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return planted(tmp_path / "notes", TREE)


def planted(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode())
    return root


async def added(
    store: KnowledgeStore, root: Path, *excludes: str, name: str = "notes"
) -> Source:
    return await store.put_source(
        root,
        name,
        SourceKind.FILES,
        private=False,
        excludes=[PurePosixPath(e) for e in excludes],
    )


def indexed(db: Path) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Each document by path: its note and its chunks' bodies in order."""
    with closing(sqlite3.connect(db)) as connection:
        rows = connection.execute(
            "SELECT d.path, d.note, c.body FROM documents d"
            " LEFT JOIN chunks c ON c.document = d.id ORDER BY d.path, c.ordinal"
        ).fetchall()
    found: dict[str, tuple[str, tuple[str, ...]]] = {}
    for path, note, body in rows:
        _, bodies = found.get(path, (note, ()))
        found[path] = (note, bodies if body is None else (*bodies, body))
    return found


def chunk_ids(db: Path) -> list[int]:
    with closing(sqlite3.connect(db)) as connection:
        return [i for (i,) in connection.execute("SELECT id FROM chunks ORDER BY id")]


def of(report: IndexReport) -> IndexReport:
    return replace(report, seconds=0.0)


def indexer(
    store: KnowledgeStore, spy: Spy | None = None, **options: object
) -> Indexer:
    chunking = options.get("chunking", CHUNKING)
    assert isinstance(chunking, Chunking)
    return Indexer(store, chunking, WORDS, formats=spy or format_of)


async def test_run_adds_every_file_a_format_reads(
    store: KnowledgeStore, root: Path, db: Path
) -> None:
    (root / "image.png").write_bytes(b"\x89PNG")
    await added(store, root)

    report = await indexer(store).run()

    assert of(report) == IndexReport(added=3, chunks=3)
    assert indexed(db) == INDEXED
    assert report.seconds > 0.0


async def test_run_with_nothing_changed_opens_no_file(
    store: KnowledgeStore, root: Path, db: Path
) -> None:
    await added(store, root)
    await indexer(store).run()
    before = chunk_ids(db)
    spy = Spy()

    report = await indexer(store, spy).run()

    assert of(report) == IndexReport(unchanged=3)
    assert spy.decoded == []
    assert chunk_ids(db) == before


async def test_run_records_a_new_stat_when_the_bytes_are_the_same(
    store: KnowledgeStore, root: Path, db: Path
) -> None:
    source = await added(store, root)
    await indexer(store).run()
    before = chunk_ids(db)
    os.utime(root / "a.md", ns=(1_000_000_000, 1_000_000_000))
    spy = Spy()

    report = await indexer(store, spy).run()

    assert of(report) == IndexReport(unchanged=3)
    assert spy.decoded == []
    assert chunk_ids(db) == before
    stamps = await store.stamps(source.id)
    assert stamps[PurePosixPath("a.md")].mtime_ns == 1_000_000_000
    assert of(await indexer(store, spy).run()) == IndexReport(unchanged=3)
    assert spy.decoded == []


async def test_run_replaces_an_edited_file_and_reports_its_old_chunks(
    store: KnowledgeStore, root: Path, db: Path
) -> None:
    await added(store, root)
    await indexer(store).run()
    old = chunk_ids(db)
    (root / "c.txt").write_bytes(b"other words\n\nand more\n")
    removed: list[ChunkId] = []

    report = await indexer(store).run(on_removed=removed.extend)

    assert of(report) == IndexReport(changed=1, unchanged=2, chunks=1)
    assert indexed(db)["c.txt"] == ("", ("other words\n\nand more",))
    assert removed == [old[1]]


async def test_a_new_recipe_reads_every_file_again(
    store: KnowledgeStore, root: Path, db: Path
) -> None:
    source = await added(store, root)
    await indexer(store).run()
    lines = Chunking(40, window_lines=2)

    report = await indexer(store, chunking=lines).run()

    assert of(report) == IndexReport(changed=3, chunks=3)
    stamps = await store.stamps(source.id)
    assert {s.recipe for s in stamps.values()} == {"structure+header/40/2/r1/s2"}
    assert indexed(db) == INDEXED


async def test_a_renamed_file_is_removed_and_added(
    store: KnowledgeStore, root: Path, db: Path
) -> None:
    await added(store, root)
    await indexer(store).run()
    (root / "a.md").rename(root / "d.md")

    report = await indexer(store).run()

    assert of(report) == IndexReport(added=1, unchanged=2, removed=1, chunks=1)
    assert sorted(indexed(db)) == ["c.txt", "d.md", "pkg/b.py"]


async def test_a_deleted_file_is_removed_with_its_chunks(
    store: KnowledgeStore, root: Path, db: Path
) -> None:
    await added(store, root)
    await indexer(store).run()
    first = chunk_ids(db)
    (root / "pkg" / "b.py").unlink()
    removed: list[ChunkId] = []

    report = await indexer(store).run(on_removed=removed.extend)

    assert of(report) == IndexReport(unchanged=2, removed=1)
    assert sorted(indexed(db)) == ["a.md", "c.txt"]
    assert removed == [first[2]]


async def test_excluded_and_dot_paths_are_left_out(
    store: KnowledgeStore, tmp_path: Path, db: Path
) -> None:
    root = planted(
        tmp_path / "notes",
        {
            "kept.md": "kept\n",
            "private/a.md": "out\n",
            "private/deep/b.md": "out\n",
            "drafts/skip.md": "out\n",
            "drafts/keep.md": "kept\n",
            ".obsidian/app.md": "out\n",
            ".hidden.md": "out\n",
            "privately.md": "kept\n",
            "PRIVATE/c.md": "out\n",
            "Drafts/Skip.md": "out\n",
        },
    )
    await added(store, root, "Private", "drafts/skip.md")

    report = await indexer(store).run()

    assert of(report) == IndexReport(added=3, chunks=3)
    assert sorted(indexed(db)) == ["drafts/keep.md", "kept.md", "privately.md"]


async def test_a_folder_excluded_later_loses_its_documents(
    store: KnowledgeStore, root: Path, db: Path
) -> None:
    await added(store, root)
    await indexer(store).run()
    await added(store, root, "pkg")

    report = await indexer(store).run()

    assert of(report) == IndexReport(unchanged=2, removed=1)
    assert sorted(indexed(db)) == ["a.md", "c.txt"]


async def test_a_symlink_is_passed_over(
    store: KnowledgeStore, root: Path, tmp_path: Path, db: Path
) -> None:
    outside = planted(tmp_path / "outside", {"o.md": "outside\n"})
    try:
        (root / "link.md").symlink_to(outside / "o.md")
        (root / "folder").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("creating symlinks needs a privilege this account lacks")
    await added(store, root)

    report = await indexer(store).run()

    assert of(report) == IndexReport(added=3, passed_over=2, chunks=3)
    assert sorted(indexed(db)) == ["a.md", "c.txt", "pkg/b.py"]


async def test_a_junction_is_passed_over(
    store: KnowledgeStore, root: Path, tmp_path: Path, db: Path
) -> None:
    if sys.platform != "win32":
        pytest.skip("a junction is a Windows folder link")
    import _winapi  # noqa: PLC0415 - Windows only

    outside = planted(tmp_path / "outside", {"o.md": "outside\n"})
    _winapi.CreateJunction(str(outside), str(root / "folder"))
    await added(store, root)

    report = await indexer(store).run()

    assert of(report) == IndexReport(added=3, passed_over=1, chunks=3)
    assert sorted(indexed(db)) == ["a.md", "c.txt", "pkg/b.py"]


async def test_a_name_that_is_not_utf8_is_passed_over(
    store: KnowledgeStore, root: Path, db: Path
) -> None:
    try:
        (root / "\udc80.md").write_bytes(b"lost\n")
    except (OSError, UnicodeEncodeError):
        pytest.skip("this file system cannot hold a name that is not UTF-8")
    await added(store, root)

    report = await indexer(store).run()

    assert of(report) == IndexReport(added=3, passed_over=1, chunks=3)
    assert indexed(db) == INDEXED


async def test_a_missing_root_is_unavailable_and_keeps_its_documents(
    store: KnowledgeStore, root: Path, db: Path, tmp_path: Path
) -> None:
    await added(store, root)
    elsewhere = planted(tmp_path / "other", {"x.md": "x\n"})
    await added(store, elsewhere, name="other")
    await indexer(store).run()
    root.rename(tmp_path / "unplugged")

    report = await indexer(store).run()

    assert of(report) == IndexReport(unchanged=1, unavailable=("notes",))
    assert sorted(indexed(db)) == ["a.md", "c.txt", "pkg/b.py", "x.md"]


async def test_a_root_that_lists_empty_keeps_its_documents(
    store: KnowledgeStore, root: Path, db: Path
) -> None:
    await added(store, root)
    await indexer(store).run()
    for path in sorted(root.rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()

    report = await indexer(store).run()

    assert of(report) == IndexReport(unavailable=("notes",))
    assert sorted(indexed(db)) == ["a.md", "c.txt", "pkg/b.py"]


async def test_two_passes_at_once_run_one_after_the_other(
    store: KnowledgeStore, root: Path
) -> None:
    await added(store, root)
    index = indexer(store)

    first, second = await asyncio.gather(index.run(), index.run())

    assert of(first) == IndexReport(added=3, chunks=3)
    assert of(second) == IndexReport(unchanged=3)


async def test_a_folder_that_cannot_be_listed_keeps_its_documents(
    store: KnowledgeStore, root: Path, db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await added(store, root)
    await indexer(store).run()
    denied = root / "pkg"
    scandir = os.scandir

    def refusing(path: Path) -> object:
        if Path(path) == denied:
            message = "denied"
            raise PermissionError(message)
        return scandir(path)

    monkeypatch.setattr(os, "scandir", refusing)
    (root / "c.txt").unlink()

    report = await indexer(store).run()

    assert of(report) == IndexReport(unchanged=1, removed=1)
    assert sorted(indexed(db)) == ["a.md", "pkg/b.py"]


async def test_a_file_that_is_not_text_is_noted_and_not_read_again(
    store: KnowledgeStore, root: Path, db: Path
) -> None:
    (root / "blob.txt").write_bytes(b"\x00\x01\x02")
    await added(store, root)

    first = await indexer(store).run()
    spy = Spy()
    second = await indexer(store, spy).run()

    assert of(first) == IndexReport(added=3, unreadable=1, chunks=3)
    assert indexed(db)["blob.txt"] == ("a NUL byte at offset 0: not a text file", ())
    assert of(second) == IndexReport(unchanged=4)
    assert spy.decoded == []


async def test_a_reader_that_raises_leaves_a_note(
    store: KnowledgeStore, root: Path, db: Path
) -> None:
    await added(store, root)

    report = await indexer(store, Spy(fails="a.md")).run()

    assert of(report) == IndexReport(added=2, unreadable=1, chunks=2)
    assert indexed(db)["a.md"] == ("the reader failed: ValueError: boom", ())


async def test_a_decoder_that_raises_leaves_a_note_and_the_pass_goes_on(
    store: KnowledgeStore, root: Path, db: Path
) -> None:
    await added(store, root)

    report = await indexer(store, Spy(undecodable="a.md")).run()

    assert of(report) == IndexReport(added=2, unreadable=1, chunks=2)
    assert indexed(db)["a.md"] == (
        "the reader failed: RuntimeError: the decoder broke",
        (),
    )


async def test_a_file_that_grows_but_stays_over_the_limit_is_noted_again(
    store: KnowledgeStore, tmp_path: Path, db: Path
) -> None:
    root = planted(tmp_path / "notes", {"huge.csv": "a" * (MAX_FILE_BYTES + 1)})
    huge = root / "huge.csv"
    await added(store, root)
    await indexer(store).run()
    with huge.open("ab") as more:
        more.write(b"b")

    grown = await indexer(store).run()
    after = await indexer(store).run()

    assert of(grown) == IndexReport(changed=0, unreadable=1)
    assert indexed(db)["huge.csv"] == ("not read: 2,097,154 bytes, over 2,097,152", ())
    assert of(after) == IndexReport(unchanged=1)


async def test_a_file_that_grows_over_the_limit_after_the_walk_is_not_kept(
    store: KnowledgeStore, root: Path, db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await added(store, root)
    read_bytes = Path.read_bytes

    def grown(path: Path) -> bytes:
        if path.name == "c.txt":
            return b"x" * (MAX_FILE_BYTES + 1)
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", grown)

    report = await indexer(store).run()

    assert of(report) == IndexReport(added=2, unreadable=1, chunks=2)
    assert indexed(db)["c.txt"] == ("not read: 2,097,153 bytes, over 2,097,152", ())


async def test_a_file_over_the_size_limit_is_noted_and_never_read(
    store: KnowledgeStore, tmp_path: Path, db: Path
) -> None:
    root = planted(tmp_path / "notes", {"small.txt": "fine\n"})
    (root / "huge.csv").write_bytes(b"a,b\n" * (MAX_FILE_BYTES // 4) + b"x")
    await added(store, root)

    first = await indexer(store).run()
    second = await indexer(store).run()

    assert of(first) == IndexReport(added=1, unreadable=1, chunks=1)
    assert of(second) == IndexReport(unchanged=2)
    assert indexed(db)["huge.csv"] == ("not read: 2,097,153 bytes, over 2,097,152", ())


async def test_a_file_that_looks_like_it_holds_a_secret_is_held_back(
    store: KnowledgeStore, tmp_path: Path, db: Path
) -> None:
    value = "Zq81" * 5
    root = planted(
        tmp_path / "notes",
        {
            "secrets.yaml": "nothing here\n",
            "AWS_Credentials.toml": "nothing here\n",
            "old_passwords.md": "nothing here\n",
            "id_rsa.txt": "nothing here\n",
            "key.md": f"intro\n\n-----BEGIN {'RSA'} PRIVATE KEY-----\nabc\n",
            "config.py": f'API_KEY = "{value}"\n',
            "app.yaml": f"github_token: '{value}'\n",
            "run.ini": f"password={value}\n",
            "call.json": f'{{"client_secret": "{value}"}}\n',
            "code.py": "token = get_token()\npassword: str\n",
            "names.py": "token = settings.github_access_token\nmax_tokens = 512\n",
            "example.toml": (
                'api_key = "${API_KEY_FROM_ENV}"\nsecret = "<put the secret here>"\n'
            ),
            "blank.yaml": 'token: "xxxxxxxxxxxxxxxxxx"\npasswd: changeme-changeme-1\n',
            "prose.md": "The token = a long run of words that is not a secret.\n",
            "aws.txt": "AWS_ACCESS_KEY_ID=AKIA" + "ABCDEFGHIJ234567\n",
            "db.md": "Connect to postgres://admin:" + "hunter2@db.local/app\n",
            "gh.md": "use ghp_" + "a1B2" * 9 + " here\n",
            "llm.md": "the key sk-" + "x9Y8" * 6 + " works\n",
            "names.toml": (
                "tokenizer = build_tokenizer_v2_fast\ntoken_budget=1000000000000000\n"
            ),
        },
    )
    await added(store, root)

    report = await indexer(store).run()

    assert of(report) == IndexReport(added=7, held_back=12, chunks=7)
    held = {path for path, (note, _) in indexed(db).items() if note == HELD_BACK}
    assert held == {
        "AWS_Credentials.toml",
        "app.yaml",
        "aws.txt",
        "call.json",
        "config.py",
        "db.md",
        "gh.md",
        "id_rsa.txt",
        "key.md",
        "llm.md",
        "run.ini",
        "secrets.yaml",
    }
    assert indexed(db)["old_passwords.md"] == ("", ("nothing here",))
    assert indexed(db)["code.py"] == ("", ("token = get_token()\npassword: str",))


class FailsOnce(KnowledgeStore):
    """Fails the second document it is asked to write, once."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.writes = 0

    async def replace_document(self, reading: Reading) -> Replaced:
        self.writes += 1
        if self.writes == 2:
            message = "the disk is full"
            raise OSError(message)
        return await super().replace_document(reading)


async def test_a_pass_that_crashed_converges_on_the_next(
    root: Path, tmp_path: Path
) -> None:
    crashing = FailsOnce(tmp_path / "crashed.db")
    await added(crashing, root)
    with pytest.raises(OSError, match=r"^the disk is full$"):
        await indexer(crashing).run()

    report = await indexer(crashing).run()

    assert of(report) == IndexReport(added=2, unchanged=1, chunks=2)
    assert indexed(tmp_path / "crashed.db") == INDEXED


async def test_a_file_gone_before_it_is_read_is_kept_until_the_next_pass(
    store: KnowledgeStore, root: Path, db: Path
) -> None:
    await added(store, root)
    await indexer(store).run()
    (root / "c.txt").write_bytes(b"edited words\n")
    files = iter(["a.md", "c.txt"])

    async def deleting() -> None:
        if next(files, None) == "c.txt":
            (root / "c.txt").unlink()

    first = await indexer(store).run(pause=deleting)
    second = await indexer(store).run()

    assert of(first) == IndexReport(unchanged=2, passed_over=1)
    assert of(second) == IndexReport(unchanged=2, removed=1)
    assert sorted(indexed(db)) == ["a.md", "pkg/b.py"]


@dataclass
class Embedded:
    pairs: list[tuple[ChunkId, NDArray[np.float32]]] = field(
        default_factory=list[tuple[ChunkId, NDArray[np.float32]]]
    )

    def __call__(self, pairs: Sequence[tuple[ChunkId, NDArray[np.float32]]]) -> None:
        self.pairs.extend(pairs)


async def test_embedding_fills_every_chunk_once(
    store: KnowledgeStore, root: Path, db: Path
) -> None:
    await added(store, root)
    network = Network()
    model = embedder(network)
    run = Indexer(store, CHUNKING, WORDS, model).run
    network.texts = 0
    embedded = Embedded()
    pauses: list[None] = []

    async def pause() -> None:
        pauses.append(None)

    first = await run(pause=pause, on_embedded=embedded)
    second = await run(on_embedded=embedded)

    assert of(first) == IndexReport(added=3, chunks=3, embedded=3)
    assert of(second) == IndexReport(unchanged=3)
    assert len(pauses) == 4
    assert network.texts == 3
    kept = await store.vectors(model.spec.id)
    assert [chunk for chunk, _ in kept] == chunk_ids(db)
    assert [chunk for chunk, _ in embedded.pairs] == chunk_ids(db)
    for (_, vector), (_, held) in zip(kept, embedded.pairs, strict=True):
        assert np.frombuffer(vector, np.float32).tolist() == held.tolist()


async def test_files_are_read_on_one_thread_off_the_event_loop(
    store: KnowledgeStore, root: Path
) -> None:
    source = await added(store, root)
    spy = Spy()

    report = await indexer(store, spy).run(source)

    assert of(report) == IndexReport(added=3, chunks=3)
    assert spy.threads == ["synthia-reader_0"] * 3
    assert threading.current_thread().name == "MainThread"


@dataclass
class Step:
    """A step that waits for the test, then records what it saw."""

    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    finished: asyncio.Event = field(default_factory=asyncio.Event)
    error: Exception | None = None
    caller_done: list[bool] = field(default_factory=list[bool])

    async def __call__(self, caller: Callable[[], asyncio.Task[int]]) -> int:
        self.started.set()
        await self.release.wait()
        self.caller_done.append(caller().done())
        self.finished.set()
        if self.error is not None:
            raise self.error
        return 1


def started(steps: Unsplit, step: Step) -> asyncio.Task[int]:
    task: asyncio.Task[int] = asyncio.create_task(steps.run(step(lambda: task)))
    return task


async def test_unsplit_returns_what_the_step_returns() -> None:
    step = Step()
    step.release.set()

    assert await started(Unsplit(), step) == 1


async def test_unsplit_finishes_the_step_before_its_cancelled_caller_returns() -> None:
    step = Step()
    caller = started(Unsplit(), step)
    await asyncio.wait_for(step.started.wait(), HANG_TIMEOUT_S)

    caller.cancel()
    step.release.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(caller, HANG_TIMEOUT_S)
    assert step.caller_done == [False]


async def test_unsplit_logs_a_step_that_fails_after_its_caller_was_cancelled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    step = Step(error=OSError("the disk is full"))
    caller = started(Unsplit(), step)
    await asyncio.wait_for(step.started.wait(), HANG_TIMEOUT_S)

    caller.cancel()
    step.release.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(caller, HANG_TIMEOUT_S)
    (record,) = caplog.records
    assert record.getMessage() == "a write whose caller was cancelled failed"
    assert record.exc_info is not None
    assert record.exc_info[1] is step.error


async def test_unsplit_cancelled_twice_lets_the_step_run_on() -> None:
    step = Step()
    caller = started(Unsplit(), step)
    await asyncio.wait_for(step.started.wait(), HANG_TIMEOUT_S)

    caller.cancel()
    # Let the first cancellation reach the caller before the second.
    await asyncio.sleep(0)
    caller.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(caller, HANG_TIMEOUT_S)
    assert not step.finished.is_set()
    step.release.set()
    await asyncio.wait_for(step.finished.wait(), HANG_TIMEOUT_S)
    assert step.caller_done == [True]
