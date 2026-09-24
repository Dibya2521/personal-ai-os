"""Fetch one catalogued file, resumably, and keep it only if it is exact.

Bytes arrive in ``<name>.part`` and are hashed as they are written. The file is
renamed to ``<name>`` only after its size and SHA-256 match the catalogue, and
the rename is atomic, so a file under its final name has always been verified.
A lost connection keeps the part file, and the next run asks the server for the
rest with an HTTP ``Range`` request instead of starting a 2.7 GB file again.

Disk work runs in a worker thread: hashing a 2.7 GB part file on the event loop
would stall everything else SYNTHIA is doing for seconds.
"""

from __future__ import annotations

import asyncio
import hashlib
from http import HTTPStatus
from typing import TYPE_CHECKING, BinaryIO, Final, Protocol

import httpx

from synthia.kernel.errors import SynthiaError

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from synthia.models.catalogue import Download

HASH_CHUNK_BYTES: Final = 1024 * 1024
PART_SUFFIX: Final = ".part"


class DownloadError(SynthiaError):
    """A file could not be fetched whole. Running again resumes where it stopped."""


class ChecksumError(DownloadError):
    """A file arrived whole but is not the file the catalogue describes."""


class _Digest(Protocol):
    def update(self, data: bytes, /) -> None: ...
    def hexdigest(self) -> str: ...


def _discard(_: int) -> None:
    return None


def part_path(download: Download, directory: Path) -> Path:
    """Return where ``download`` is written while it arrives."""
    return directory / f"{download.name}{PART_SUFFIX}"


def _start(download: Download, directory: Path) -> tuple[int, _Digest] | None:
    """Return the bytes already fetched and their digest, or ``None`` if installed."""
    final = directory / download.name
    if final.is_file() and final.stat().st_size == download.size:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    part = part_path(download, directory)
    have = part.stat().st_size if part.is_file() else 0
    digest = hashlib.sha256()
    if have > download.size:
        part.unlink()
        return 0, digest
    if have:
        with part.open("rb") as file:
            while chunk := file.read(HASH_CHUNK_BYTES):
                digest.update(chunk)
    return have, digest


def _finish(download: Download, directory: Path, digest: _Digest) -> Path:
    part = part_path(download, directory)
    if digest.hexdigest() != download.sha256:
        part.unlink()
        message = (
            f"{download.name} does not match its recorded SHA-256 and was deleted; "
            "run again to fetch it afresh"
        )
        raise ChecksumError(message)
    return part.replace(directory / download.name)


async def fetch(
    client: httpx.AsyncClient,
    download: Download,
    directory: Path,
    on_progress: Callable[[int], None] = _discard,
) -> Path:
    """Download ``download`` into ``directory`` and return its verified path.

    ``on_progress`` receives each number of bytes as it lands, starting with
    whatever an earlier run already fetched.

    Raises:
        DownloadError: If the server refuses or resumes at the wrong place, the
            connection is lost, or more bytes arrive than expected.
        ChecksumError: If the finished file has the wrong SHA-256; it is deleted.
    """
    start = await asyncio.to_thread(_start, download, directory)
    if start is None:
        on_progress(download.size)
        return directory / download.name
    have, digest = start
    on_progress(have)
    if have < download.size:
        part = part_path(download, directory)
        digest = await _receive(client, download, part, start, on_progress)
    return await asyncio.to_thread(_finish, download, directory, digest)


def _append(file: BinaryIO, digest: _Digest, chunk: bytes) -> None:
    file.write(chunk)
    digest.update(chunk)


async def _copy(
    response: httpx.Response,
    part: Path,
    start: tuple[int, _Digest],
    limit: int,
    on_progress: Callable[[int], None],
) -> int:
    """Append the body to ``part``; a count above ``limit`` means it stopped early."""
    have, digest = start
    file = await asyncio.to_thread(part.open, "ab" if have else "wb")
    try:
        async for chunk in response.aiter_bytes():
            have += len(chunk)
            if have > limit:
                break
            await asyncio.to_thread(_append, file, digest, chunk)
            on_progress(len(chunk))
    finally:
        await asyncio.to_thread(file.close)
    return have


async def _receive(
    client: httpx.AsyncClient,
    download: Download,
    part: Path,
    progress: tuple[int, _Digest],
    on_progress: Callable[[int], None],
) -> _Digest:
    have, digest = progress
    # A range counts encoded bytes; unencoded, the bytes on the wire are the file's.
    headers = {"Accept-Encoding": "identity"}
    if have:
        headers["Range"] = f"bytes={have}-"
    try:
        async with client.stream(
            "GET", download.url, headers=headers, follow_redirects=True
        ) as response:
            if not await asyncio.to_thread(_resumes, response, download, part, have):
                have, digest = 0, hashlib.sha256()
            have = await _copy(
                response, part, (have, digest), download.size, on_progress
            )
    except httpx.TransportError as error:
        message = (
            f"{download.name}: the connection was lost ({type(error).__name__}); "
            "run again to resume"
        )
        raise DownloadError(message) from None
    if have > download.size:
        await asyncio.to_thread(part.unlink)
        message = f"{download.name}: the server sent more than {download.size:,} bytes"
        raise DownloadError(message)
    if have < download.size:
        message = (
            f"{download.name}: the download stopped at {have:,} of "
            f"{download.size:,} bytes; run again to resume"
        )
        raise DownloadError(message)
    return digest


def _resumes(
    response: httpx.Response, download: Download, part: Path, have: int
) -> bool:
    """Return whether ``response`` continues the part file rather than restarting it."""
    if have and response.status_code == HTTPStatus.PARTIAL_CONTENT:
        if response.headers.get("content-range", "").startswith(f"bytes {have}-"):
            return True
        part.unlink()
        message = f"{download.name}: the server resumed at the wrong offset"
        raise DownloadError(message)
    if response.status_code == HTTPStatus.OK:
        return False
    if have and response.status_code == HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE:
        # The file on the server is not the one the part came from.
        part.unlink()
    message = f"{download.name}: the server answered HTTP {response.status_code}"
    raise DownloadError(message)
