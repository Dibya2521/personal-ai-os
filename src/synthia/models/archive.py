"""Unpack llama.cpp archives into one directory, and nothing outside it.

The archives are verified against the catalogue before they get here, so the
path checks are defence in depth. They still refuse rather than repair: a member
aimed outside the directory means the archive is not what it claims to be.
Everything lands in a staging directory first and is renamed into place whole,
so a failed unpack never leaves a half-installed runtime.
"""

from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Final

from synthia.kernel.errors import SynthiaError

if TYPE_CHECKING:
    from collections.abc import Sequence

STAGING_SUFFIX: Final = ".unpacking"


class ArchiveError(SynthiaError):
    """An archive is unreadable, of an unknown kind, or reaches outside its folder."""


def _is_zip(archive: Path) -> bool:
    return archive.name.endswith(".zip")


def _is_tar(archive: Path) -> bool:
    return archive.name.endswith(".tar.gz")


def unpacked_size(archive: Path) -> int:
    """Return the bytes ``archive`` holds once unpacked.

    Raises:
        ArchiveError: If it is not a readable zip or gzipped tar.
    """
    try:
        if _is_zip(archive):
            with zipfile.ZipFile(archive) as zipped:
                return sum(info.file_size for info in zipped.infolist())
        if _is_tar(archive):
            with tarfile.open(archive, "r:gz") as tarred:
                return sum(member.size for member in tarred if member.isfile())
    except (zipfile.BadZipFile, tarfile.TarError, EOFError) as error:
        message = f"{archive.name} is not a readable archive: {error}"
        raise ArchiveError(message) from None
    message = f"{archive.name}: unknown archive type"
    raise ArchiveError(message)


def unpack(archives: Sequence[Path], target: Path) -> None:
    """Unpack every archive into ``target``, replacing what was there.

    Raises:
        ArchiveError: If an archive is unreadable, of an unknown kind, or has a
            member that would land outside ``target``. ``target`` is untouched.
    """
    staging = target.with_name(f"{target.name}{STAGING_SUFFIX}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        for archive in archives:
            _extract(archive, staging)
    except BaseException:
        shutil.rmtree(staging)
        raise
    if target.exists():
        shutil.rmtree(target)
    staging.replace(target)


def _extract(archive: Path, into: Path) -> None:
    try:
        if _is_zip(archive):
            _unzip(archive, into)
        elif _is_tar(archive):
            with tarfile.open(archive, "r:gz") as tarred:
                root = into.resolve()
                for name in tarred.getnames():
                    _check_name(archive, name, root)
                tarred.extractall(root, filter="data")
        else:
            message = f"{archive.name}: unknown archive type"
            raise ArchiveError(message)
    except tarfile.FilterError as error:
        message = f"{archive.name}: {error}; refusing to unpack it"
        raise ArchiveError(message) from None
    except (zipfile.BadZipFile, tarfile.TarError, EOFError) as error:
        message = f"{archive.name} is not a readable archive: {error}"
        raise ArchiveError(message) from None


def _check_name(archive: Path, name: str, root: Path) -> None:
    """Refuse a member whose name leaves ``root``, rather than rewriting the name."""
    windows = PureWindowsPath(name)
    unsafe = (
        PurePosixPath(name).is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in windows.parts
        or not (root / name).resolve().is_relative_to(root)
    )
    if unsafe:
        message = (
            f"{archive.name}: {name!r} would land outside the install "
            "directory; refusing to unpack it"
        )
        raise ArchiveError(message)


def _unzip(archive: Path, into: Path) -> None:
    root = into.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for name in zipped.namelist():
            _check_name(archive, name, root)
        zipped.extractall(root)  # noqa: S202 - every name was checked above
