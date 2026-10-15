"""Keep a knowledge store in step with its sources' files.

One pass, per source: walk the folder, compare each file with what the store
holds, read what changed, remove what is gone; then embed every chunk that
has no vector. Each decision comes from the store's state and the files'
state, never from a queue in memory, so a pass is idempotent and a crash
anywhere converges on the next pass:

- A file whose ``(mtime_ns, size)`` and recipe equal its stamp is not opened.
- A file whose stat changed is hashed; the same sha256 and recipe only record
  the new stat.
- Anything else is read and replaced in one transaction. A file that is not
  text, that a reader fails on, or that looks like it holds a secret is
  stored with no chunks and a note, so it is not read again until it changes.
- Nothing is removed on missing evidence: a source whose root cannot be
  listed is reported unavailable, a folder that cannot be listed keeps its
  documents, and only a file absent from a folder that was listed goes.
- Embedding runs last, from the chunks the store says have no vector, so
  chunks written before a crash are embedded by the next pass.
- A cancelled pass stops after the store write in progress and its report
  to ``on_removed`` or ``on_embedded``, never between the two.
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import partial
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final

from synthia.knowledge.chunk import chunked
from synthia.knowledge.model import DocumentKind
from synthia.knowledge.store import Reading
from synthia.knowledge.text import UnreadableError, format_of

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine, Iterable, Sequence

    import numpy as np
    from numpy.typing import NDArray

    from synthia.knowledge.chunk import Chunking, TokenCounter
    from synthia.knowledge.model import ChunkId, DocumentId, Draft, Source, SourceId
    from synthia.knowledge.store import KnowledgeStore, Stamp
    from synthia.knowledge.text import Format, Text
    from synthia.retrieval.embed import TextEmbedder

logger = logging.getLogger(__name__)

EMBED_BATCH: Final = 64
"""Chunks taken from the store per embedding round; the embedder's own spec
decides how many of them share a network run."""
HELD_BACK: Final = "held back: looks like it holds a secret"
MAX_FILE_BYTES: Final = 2 * 1024**2
"""A larger file is noted and never read: about four times the largest note
of a 1,281-note vault (541 KB), and eight times the largest file here."""

SECRET_RULES: Final = 2
"""Part of every document's recipe, so a change to the rules below re-reads
every file instead of leaving what the old rules let in or held back."""
SECRET_NAMES: Final = (
    *("*secret*", "*credential*", "*password*"),
    *("*.pem", "*.key", "id_rsa*", "*.pfx", "*.p12"),
)
"""Names, matched without case, of plain-text and configuration files kept
out unread; a note or a module so named is judged by its text instead."""
_KNOWN_FORMATS: Final = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|\bAKIA[0-9A-Z]{16}\b"
    r"|\bgh[pousr]_[A-Za-z0-9]{36,}"
    r"|\bsk-[A-Za-z0-9_-]{20,}"
    r"|[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@"
)
_ASSIGNED: Final = re.compile(
    r"""(?ix)
    (?:api_?key|secret|token|passw(?:or)?d)[\w-]*["']?[ \t]*[:=][ \t]*
    (?: "(?P<double>[^"\s]{16,})"
      | '(?P<single>[^'\s]{16,})'
      | (?P<bare>[^\s"'=<>()\[\]{},;\#]{16,})(?=[\s,;\#]|$) )
    """
)
_PLACEHOLDER_STARTS: Final = ("<", "${", "{{", "%(")

# Readers wrap ast.parse in warnings.catch_warnings, which swaps process-wide
# state, so every read in the process runs on this one thread.
_READER: Final = ThreadPoolExecutor(max_workers=1, thread_name_prefix="synthia-reader")

type Pause = Callable[[], Awaitable[None]]
"""Awaited before each file and each embedding round, so a waiting turn runs first."""
type Formats = Callable[[PurePosixPath], Format | None]
type OnRemoved = Callable[[Sequence[ChunkId]], None]
type OnEmbedded = Callable[[Sequence[tuple[ChunkId, NDArray[np.float32]]]], None]


@dataclass(frozen=True, slots=True)
class IndexReport:
    """What one pass did.

    ``unreadable`` files (not text, or a reader failed) and ``held_back``
    ones (they look like they hold a secret) are stored with no chunks and a
    note, and not read again until they change. ``passed_over`` counts links,
    names that are not UTF-8 and files that could not be opened this time;
    none is stored, and the last are tried again on the next pass. ``chunks``
    and ``cut`` count the chunks written. ``unavailable`` names the sources
    whose root could not be listed; their documents were kept.
    """

    added: int = 0
    changed: int = 0
    unchanged: int = 0
    removed: int = 0
    unreadable: int = 0
    held_back: int = 0
    passed_over: int = 0
    chunks: int = 0
    cut: int = 0
    embedded: int = 0
    unavailable: tuple[str, ...] = ()
    seconds: float = 0.0


def _secret_name(name: str) -> bool:
    folded = name.casefold()
    return any(fnmatch.fnmatchcase(folded, pattern) for pattern in SECRET_NAMES)


def _placeholder(value: str) -> bool:
    folded = value.casefold()
    return (
        folded.startswith(_PLACEHOLDER_STARTS)
        or "changeme" in folded
        or not folded.strip("x*.-_")
    )


def _secret_value(match: re.Match[str]) -> bool:
    bare = match.group("bare")
    # Unquoted, code is a name (``settings.api_token``, ``build_v2_fast``) or a
    # number; a secret mixes letters and digits outside a name.
    if bare is not None and (
        bare.isdigit()
        or not any(c.isdigit() for c in bare)
        or "_" in bare
        or "." in bare
    ):
        return False
    value = bare or match.group("double") or match.group("single")
    return not _placeholder(value)


def _holds_secret(text: Text) -> bool:
    """Return whether a line holds something that looks like a secret.

    Caught: a private key block; an AWS access key id, a GitHub token, an
    ``sk-`` key; a URL with a password in it; and ``api_key``, ``secret``,
    ``token`` or ``password`` (and names starting so) set by ``=`` or ``:``
    to 16 or more characters that are not a placeholder, quoted, or unquoted
    mixing letters and digits. It knows names and these formats, not every
    secret: a bare JWT or a key under another name is missed. It errs on
    holding back: a long quoted value under such a name is held even when it
    is not a secret.
    """
    return any(
        _KNOWN_FORMATS.search(line) is not None
        or any(_secret_value(m) for m in _ASSIGNED.finditer(line))
        for page in text.pages
        for line in page
    )


@dataclass(frozen=True, slots=True)
class _File:
    path: PurePosixPath
    real: Path
    mtime_ns: int
    size: int
    format: Format


@dataclass(frozen=True, slots=True)
class _Walked:
    files: tuple[_File, ...]
    unlisted: tuple[PurePosixPath, ...]
    """Folders, or entries of unknown kind, whose documents must be kept."""
    passed_over: int


def _within(path: PurePosixPath, folders: Iterable[PurePosixPath]) -> bool:
    return any(path == folder or folder in path.parents for folder in folders)


def _excluded(path: PurePosixPath, excludes: Iterable[PurePosixPath]) -> bool:
    # Without case, as Windows and macOS name files: a folder kept out for
    # privacy stays out however its name is typed.
    def folded(p: PurePosixPath) -> PurePosixPath:
        return PurePosixPath(str(p).casefold())

    return _within(folded(path), (folded(e) for e in excludes))


def _listed(folder: Path) -> list[os.DirEntry[str]]:
    with os.scandir(folder) as entries:
        return sorted(entries, key=lambda entry: entry.name)


@dataclass(slots=True)
class _Walk:
    source: Source
    formats: Formats
    files: list[_File] = field(default_factory=list[_File])
    unlisted: list[PurePosixPath] = field(default_factory=list[PurePosixPath])
    passed_over: int = 0

    def visit(
        self, folder: PurePosixPath, entry: os.DirEntry[str]
    ) -> tuple[PurePosixPath, list[os.DirEntry[str]]] | None:
        """Take in one entry; return it with its entries if it is a folder to walk."""
        if entry.name.startswith("."):
            return None
        try:
            entry.name.encode()
        except UnicodeEncodeError:
            self.passed_over += 1
            return None
        path = folder / entry.name
        if _excluded(path, self.source.excludes):
            return None
        try:
            return self._inspected(path, entry)
        except OSError:
            self.unlisted.append(path)
            return None

    def _inspected(
        self, path: PurePosixPath, entry: os.DirEntry[str]
    ) -> tuple[PurePosixPath, list[os.DirEntry[str]]] | None:
        if entry.is_symlink() or entry.is_junction():
            self.passed_over += 1
            return None
        if entry.is_dir(follow_symlinks=False):
            return path, _listed(Path(entry.path))
        form = self.formats(path)
        if form is not None and entry.is_file(follow_symlinks=False):
            stat = entry.stat(follow_symlinks=False)
            real = Path(entry.path)
            self.files.append(_File(path, real, stat.st_mtime_ns, stat.st_size, form))
        return None


def _walk(source: Source, formats: Formats) -> _Walked:
    """Return the files of ``source`` a format reads; blocking.

    Raises:
        OSError: If the root cannot be listed.
    """
    walk = _Walk(source, formats)
    folders = [(PurePosixPath(), _listed(source.root))]
    while folders:
        folder, entries = folders.pop()
        for entry in entries:
            inner = walk.visit(folder, entry)
            if inner is not None:
                folders.append(inner)
    files = tuple(sorted(walk.files, key=lambda file: file.path))
    return _Walked(files, tuple(walk.unlisted), walk.passed_over)


def _vanished(stamps: dict[PurePosixPath, Stamp], walked: _Walked) -> list[DocumentId]:
    seen = {file.path for file in walked.files}
    return [
        stamp.document
        for path, stamp in stamps.items()
        if path not in seen and not _within(path, walked.unlisted)
    ]


class _Kind(Enum):
    TEXT = auto()
    UNREADABLE = auto()
    HELD_BACK = auto()


@dataclass(frozen=True, slots=True)
class _Content:
    kind: _Kind
    title: str
    note: str
    drafts: tuple[Draft, ...] = ()
    cut: int = 0


@dataclass(frozen=True, slots=True)
class _Unopened:
    """The file could not be opened: it is kept as stored and tried next pass."""


@dataclass(frozen=True, slots=True)
class _Same:
    document: DocumentId


@dataclass(frozen=True, slots=True)
class _Read:
    reading: Reading
    content: _Content


type _Outcome = _Unopened | _Same | _Read


def _failed(name: str, error: Exception) -> _Content:
    # One file a reader cannot take must not stop the pass, nor the embedding
    # after it; the note keeps it from being read until it changes.
    note = f"the reader failed: {type(error).__name__}: {error}"
    return _Content(_Kind.UNREADABLE, name, note)


async def _no_pause() -> None:
    return


def _ignored(_: Sequence[object]) -> None:
    return


async def _in_reader[R](work: Callable[[], R]) -> R:
    return await asyncio.get_running_loop().run_in_executor(_READER, work)


def _log_failure(step: asyncio.Task[object]) -> None:
    error = None if step.cancelled() else step.exception()
    if error is not None:
        logger.error("a write whose caller was cancelled failed", exc_info=error)


class Unsplit:
    """Runs a store write and the report of what it changed as one step.

    A store write runs in a thread, which goes on after the task awaiting it
    is cancelled: the write commits, but the code after its ``await`` never
    runs, so whatever follows the store (vectors held in memory) misses the
    change. Here the write and its report run in a task the cancellation
    does not reach. The cancelled caller waits for that task, then sees the
    cancellation; cancelled again while waiting, it stops waiting and the
    task runs on to its end. A failure no caller is left to see is logged.
    """

    def __init__(self) -> None:
        """Run no step yet."""
        self._running: set[asyncio.Task[object]] = set()

    async def run[R](self, step: Coroutine[object, object, R]) -> R:
        """Run ``step`` to its end, even if the caller is cancelled meanwhile.

        Raises:
            asyncio.CancelledError: If the caller was cancelled; ``step`` has
                finished, or runs on if the wait for it was cancelled too.
        """
        task = asyncio.create_task(step)
        # The event loop keeps only a weak reference to a task.
        self._running.add(task)
        task.add_done_callback(self._running.discard)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            task.add_done_callback(_log_failure)
            await asyncio.wait([task])
            raise


@dataclass(slots=True)
class _Pass:
    """One pass's hooks and counts."""

    pause: Pause
    on_removed: OnRemoved
    on_embedded: OnEmbedded
    added: int = 0
    changed: int = 0
    unchanged: int = 0
    removed: int = 0
    unreadable: int = 0
    held_back: int = 0
    passed_over: int = 0
    chunks: int = 0
    cut: int = 0
    embedded: int = 0
    unavailable: list[str] = field(default_factory=list[str])

    def written(self, content: _Content, *, new: bool) -> None:
        if content.kind is _Kind.UNREADABLE:
            self.unreadable += 1
        elif content.kind is _Kind.HELD_BACK:
            self.held_back += 1
        else:
            self.added += new
            self.changed += not new
            self.chunks += len(content.drafts)
            self.cut += content.cut

    @property
    def wrote(self) -> bool:
        return (
            self.added + self.changed + self.removed + self.unreadable + self.held_back
            > 0
        )

    def report(self, seconds: float) -> IndexReport:
        return IndexReport(
            added=self.added,
            changed=self.changed,
            unchanged=self.unchanged,
            removed=self.removed,
            unreadable=self.unreadable,
            held_back=self.held_back,
            passed_over=self.passed_over,
            chunks=self.chunks,
            cut=self.cut,
            embedded=self.embedded,
            unavailable=tuple(self.unavailable),
            seconds=seconds,
        )


class Indexer:
    """Brings a knowledge store in step with its sources. One pass at a time."""

    def __init__(
        self,
        store: KnowledgeStore,
        chunking: Chunking,
        counter: TokenCounter,
        embedder: TextEmbedder | None = None,
        *,
        formats: Formats = format_of,
    ) -> None:
        """Index into ``store``; without ``embedder``, chunks stay unembedded.

        ``counter`` counts tokens for ``chunking``; ``formats`` says which
        reader takes a path, None to leave it out.
        """
        self._store = store
        self._chunking = chunking
        self._recipe = f"{chunking.recipe()}/s{SECRET_RULES}"
        self._counter = counter
        self._embedder = embedder
        self._formats = formats
        self._one_pass = asyncio.Lock()
        self._steps = Unsplit()

    async def run(
        self,
        source: Source | None = None,
        *,
        pause: Pause | None = None,
        on_removed: OnRemoved | None = None,
        on_embedded: OnEmbedded | None = None,
    ) -> IndexReport:
        """Bring ``source``, or every source when None, in step with its files.

        A second call waits for the first to finish. ``on_removed`` gets the
        ids of chunks removed, ``on_embedded`` each chunk embedded with its
        vector, so vectors held elsewhere follow. Each store write and its
        hand-over run as one :class:`Unsplit` step, so a cancelled pass still
        hands over what its last write changed. A crash still loses ids
        written but not yet handed over, so such an owner loads its vectors
        from the store when it starts.
        """
        async with self._one_pass:
            started = time.perf_counter()
            run = _Pass(
                pause or _no_pause, on_removed or _ignored, on_embedded or _ignored
            )
            for each in await self._store.sources() if source is None else [source]:
                await self._source(run, each)
            if run.wrote:
                await self._store.optimize()
            await self._embed(run)
            return run.report(time.perf_counter() - started)

    async def _source(self, run: _Pass, source: Source) -> None:
        try:
            walked = await _in_reader(partial(_walk, source, self._formats))
        except OSError as error:
            logger.warning("source %s is unavailable: %s", source.name, error)
            run.unavailable.append(source.name)
            return
        stamps = await self._store.stamps(source.id)
        if not walked.files and not walked.unlisted and stamps:
            # An unmounted share or an unsynced folder lists as empty: never
            # take that as every file deleted.
            logger.warning("source %s lists empty; its documents are kept", source.name)
            run.unavailable.append(source.name)
            return
        run.passed_over += walked.passed_over
        for file in walked.files:
            await run.pause()
            await self._file(run, source.id, file, stamps.get(file.path))
        gone = _vanished(stamps, walked)
        if gone:
            await self._steps.run(self._removed(run, gone))
            run.removed += len(gone)

    async def _removed(self, run: _Pass, gone: Sequence[DocumentId]) -> None:
        run.on_removed(await self._store.remove_documents(gone))

    async def _file(
        self, run: _Pass, source: SourceId, file: _File, stamp: Stamp | None
    ) -> None:
        if stamp is not None and (stamp.mtime_ns, stamp.size, stamp.recipe) == (
            file.mtime_ns,
            file.size,
            self._recipe,
        ):
            run.unchanged += 1
            return
        outcome = await _in_reader(partial(self._outcome, source, file, stamp))
        match outcome:
            case _Unopened():
                run.passed_over += 1
            case _Same(document):
                await self._store.touch(document, file.mtime_ns, file.size)
                run.unchanged += 1
            case _:
                await self._steps.run(self._replaced(run, outcome.reading))
                run.written(outcome.content, new=stamp is None)

    async def _replaced(self, run: _Pass, reading: Reading) -> None:
        replaced = await self._store.replace_document(reading)
        if replaced.removed:
            run.on_removed(replaced.removed)

    def _outcome(self, source: SourceId, file: _File, stamp: Stamp | None) -> _Outcome:
        if file.size > MAX_FILE_BYTES:
            return self._too_large(source, file, file.size)
        try:
            data = file.real.read_bytes()
        except OSError as error:
            logger.warning("cannot open %s: %s", file.real, error)
            return _Unopened()
        if len(data) > MAX_FILE_BYTES:
            return self._too_large(source, file, len(data))
        sha256 = hashlib.sha256(data).hexdigest()
        if stamp is not None and (stamp.sha256, stamp.recipe) == (sha256, self._recipe):
            return _Same(stamp.document)
        return self._read(source, file, sha256, self._content(file, data))

    def _too_large(self, source: SourceId, file: _File, size: int) -> _Read:
        # The hash names the stat, so the note is written again whenever the
        # file changes and stays over the limit.
        note = f"not read: {size:,} bytes, over {MAX_FILE_BYTES:,}"
        content = _Content(_Kind.UNREADABLE, "", note)
        return self._read(source, file, f"not read:{size}:{file.mtime_ns}", content)

    def _read(
        self, source: SourceId, file: _File, sha256: str, content: _Content
    ) -> _Read:
        reading = Reading(
            source,
            file.path,
            file.format.kind,
            file.mtime_ns,
            file.size,
            sha256,
            self._recipe,
            content.title or file.path.name,
            content.note,
            content.drafts,
        )
        return _Read(reading, content)

    def _content(self, file: _File, data: bytes) -> _Content:
        name = file.path.name
        if file.format.kind is DocumentKind.TEXT and _secret_name(name):
            return _Content(_Kind.HELD_BACK, name, HELD_BACK)
        try:
            text = file.format.decode(data)
        except UnreadableError as error:
            return _Content(_Kind.UNREADABLE, name, str(error))
        except Exception as error:
            logger.exception("could not decode %s", file.real)
            return _failed(name, error)
        if _holds_secret(text):
            return _Content(_Kind.HELD_BACK, name, HELD_BACK)
        try:
            outline = file.format.read(text, file.path)
            drafted = chunked(text, outline, self._counter, self._chunking)
        except Exception as error:
            logger.exception("the reader failed on %s", file.real)
            return _failed(name, error)
        return _Content(
            _Kind.TEXT, outline.title, outline.note, drafted.drafts, drafted.cut
        )

    async def _embed(self, run: _Pass) -> None:
        if self._embedder is None:
            return
        model = self._embedder.spec.id
        while waiting := await self._store.unembedded(model, EMBED_BATCH):
            await run.pause()
            vectors = await self._embedder.embed([text for _, text in waiting])
            pairs = [
                (chunk, vector)
                for (chunk, _), vector in zip(waiting, vectors, strict=True)
            ]
            run.embedded += await self._steps.run(self._kept(run, model, pairs))

    async def _kept(
        self,
        run: _Pass,
        model: str,
        pairs: Sequence[tuple[ChunkId, NDArray[np.float32]]],
    ) -> int:
        """Keep ``pairs``' vectors, hand over those kept, and return how many."""
        kept = set(
            await self._store.put_vectors(
                model, [(chunk, vector.tobytes()) for chunk, vector in pairs]
            )
        )
        held = [(chunk, vector) for chunk, vector in pairs if chunk in kept]
        run.on_embedded(held)
        return len(held)
