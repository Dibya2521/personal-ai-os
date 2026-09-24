"""Install and remove catalogued runtimes and models inside a disk budget.

Everything lives under ``SYNTHIA_HOME``: ``models/<id>/`` holds a model's files,
``runtimes/<id>/`` an unpacked llama.cpp build, and ``downloads/<id>/`` a
runtime's archives until they are unpacked. The budget counts all three, part
files included, because they all take disk.

The budget is checked twice for a runtime: before downloading, against the
bytes still to fetch, and before unpacking, against the exact unpacked size
read from the archives themselves.
"""

from __future__ import annotations

import asyncio
import shutil
from typing import TYPE_CHECKING, Final

from synthia.kernel.errors import SynthiaError
from synthia.models.archive import unpack, unpacked_size
from synthia.models.catalogue import Model, Runtime
from synthia.models.download import fetch, part_path

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import httpx

    from synthia.models.catalogue import Download

BYTES_PER_MB: Final = 1024**2
BYTES_PER_GB: Final = 1024**3
# Never fill the disk: other programs, and SQLite's journal, need room to write.
KEEP_FREE_BYTES: Final = 1 * BYTES_PER_GB
MODELS_DIR: Final = "models"
RUNTIMES_DIR: Final = "runtimes"
DOWNLOADS_DIR: Final = "downloads"
COMPLETE_MARKER: Final = ".complete"


class DiskBudgetError(SynthiaError):
    """An install would exceed the disk budget or leave the disk nearly full."""


def size_text(count: int) -> str:
    """Return ``count`` bytes for a person: megabytes below a gigabyte."""
    if count < BYTES_PER_GB:
        return f"{count / BYTES_PER_MB:.1f} MB"
    return f"{count / BYTES_PER_GB:.2f} GB"


def free_bytes(path: Path) -> int:
    """Return the free bytes on the disk that holds, or will hold, ``path``."""
    existing = next(p for p in (path, *path.parents) if p.exists())
    return shutil.disk_usage(existing).free


def _tree_bytes(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _discard(_: Download, __: int) -> None:
    return None


async def _fetch(
    client: httpx.AsyncClient,
    file: Download,
    directory: Path,
    on_progress: Callable[[Download, int], None],
) -> Path:
    return await fetch(client, file, directory, lambda count: on_progress(file, count))


class Installer:
    """Installs into ``home``, keeping everything installed within ``budget_bytes``."""

    def __init__(
        self,
        home: Path,
        budget_bytes: int,
        free_bytes: Callable[[Path], int] = free_bytes,
    ) -> None:
        self._home = home
        self._budget = budget_bytes
        self._free_bytes = free_bytes

    def path_of(self, item: Runtime | Model) -> Path:
        """Return where ``item`` is installed."""
        kind = MODELS_DIR if isinstance(item, Model) else RUNTIMES_DIR
        return self._home / kind / item.id

    def _downloads_of(self, runtime: Runtime) -> Path:
        return self._home / DOWNLOADS_DIR / runtime.id

    def installed(self, item: Runtime | Model) -> bool:
        """Return whether ``item`` is completely installed."""
        path = self.path_of(item)
        if isinstance(item, Runtime):
            return (path / COMPLETE_MARKER).is_file()
        return all(
            (path / f.name).is_file() and (path / f.name).stat().st_size == f.size
            for f in item.files
        )

    def used_bytes(self) -> int:
        """Return the bytes everything installed or half-downloaded takes."""
        return sum(
            _tree_bytes(self._home / d)
            for d in (MODELS_DIR, RUNTIMES_DIR, DOWNLOADS_DIR)
        )

    def to_download(self, item: Runtime | Model) -> int:
        """Return the bytes still to fetch for ``item``; part files count as fetched."""
        if self.installed(item):
            return 0
        if isinstance(item, Model):
            files, directory = item.files, self.path_of(item)
        else:
            files, directory = item.archives, self._downloads_of(item)
        return sum(self._missing(f, directory) for f in files)

    @staticmethod
    def _missing(file: Download, directory: Path) -> int:
        if (directory / file.name).is_file():
            return 0
        part = part_path(file, directory)
        return file.size - (part.stat().st_size if part.is_file() else 0)

    def _check_room(self, name: str, needed: int) -> None:
        used = self.used_bytes()
        if used + needed > self._budget:
            message = (
                f"installing {name} needs {size_text(needed)} more, and "
                f"{size_text(used)} of the {size_text(self._budget)} disk budget is "
                "used; raise SYNTHIA_DISK_BUDGET_GB or remove something first"
            )
            raise DiskBudgetError(message)
        free = self._free_bytes(self._home)
        if free - needed < KEEP_FREE_BYTES:
            message = (
                f"installing {name} needs {size_text(needed)} and would leave "
                f"under {size_text(KEEP_FREE_BYTES)} free on the disk "
                f"({size_text(free)} free now)"
            )
            raise DiskBudgetError(message)

    async def install(
        self,
        client: httpx.AsyncClient,
        item: Runtime | Model,
        on_progress: Callable[[Download, int], None] = _discard,
    ) -> Path:
        """Install ``item`` and return where it is; already installed is a no-op.

        Raises:
            DiskBudgetError: Before any byte is written, if it would not fit.
            DownloadError: If a file cannot be fetched whole; run again to resume.
            ArchiveError: If a runtime archive cannot be unpacked safely.
        """
        path = self.path_of(item)
        if await asyncio.to_thread(self.installed, item):
            return path
        needed = await asyncio.to_thread(self.to_download, item)
        await asyncio.to_thread(self._check_room, item.id, needed)
        if isinstance(item, Model):
            for file in item.files:
                await _fetch(client, file, path, on_progress)
            return path
        downloads = self._downloads_of(item)
        archives = [
            await _fetch(client, a, downloads, on_progress) for a in item.archives
        ]
        await asyncio.to_thread(self._unpack, item, archives)
        return path

    def _unpack(self, runtime: Runtime, archives: list[Path]) -> None:
        self._check_room(runtime.id, sum(unpacked_size(a) for a in archives))
        target = self.path_of(runtime)
        unpack(archives, target)
        (target / COMPLETE_MARKER).write_text(f"{runtime.build}\n", encoding="utf-8")
        shutil.rmtree(self._downloads_of(runtime))

    def remove(self, item: Runtime | Model) -> int:
        """Delete ``item`` and its half-downloaded files; return the bytes freed."""
        paths = [self.path_of(item)]
        if isinstance(item, Runtime):
            paths.append(self._downloads_of(item))
        freed = sum(_tree_bytes(p) for p in paths)
        for path in paths:
            if path.exists():
                shutil.rmtree(path)
        return freed
