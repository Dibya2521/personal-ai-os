import hashlib
import random
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import httpx
import pytest

from synthia.models.catalogue import Download
from synthia.models.download import (
    HASH_CHUNK_BYTES,
    ChecksumError,
    DownloadError,
    fetch,
    part_path,
)

DATA = random.Random(7).randbytes(3 * HASH_CHUNK_BYTES + 123)
URL = "https://files.test/weights.gguf"
FILE = Download("weights.gguf", URL, len(DATA), hashlib.sha256(DATA).hexdigest())


class Dropping(httpx.AsyncByteStream):
    """Send ``body`` up to ``cut`` bytes, then lose the connection."""

    def __init__(self, body: bytes, cut: int) -> None:
        self.body = body
        self.cut = cut

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.body[: self.cut]
        message = "connection reset"
        raise httpx.ReadError(message)


class Server:
    """Serve ``body`` at ``URL``, and record what was asked."""

    def __init__(
        self,
        body: bytes = DATA,
        *,
        ranges: bool = True,
        cut: int | None = None,
        status: int = 200,
    ) -> None:
        self.body = body
        self.ranges = ranges
        self.cut = cut
        self.status = status
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx.Response(self.status)
        wanted = request.headers.get("range")
        if wanted and self.ranges:
            start = int(wanted.removeprefix("bytes=").removesuffix("-"))
            rest = self.body[start:]
            headers = {
                "content-range": f"bytes {start}-{len(self.body) - 1}/{len(self.body)}"
            }
            return httpx.Response(206, content=rest, headers=headers)
        if self.cut is not None:
            cut, self.cut = self.cut, None
            return httpx.Response(200, stream=Dropping(self.body, cut))
        return httpx.Response(200, content=self.body)


async def run(
    server: Callable[[httpx.Request], httpx.Response],
    directory: Path,
    file: Download = FILE,
) -> tuple[Path, list[int]]:
    seen: list[int] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        path = await fetch(client, file, directory, seen.append)
    return path, seen


async def test_a_file_is_fetched_verified_and_renamed(tmp_path: Path) -> None:
    path, seen = await run(Server(), tmp_path)

    assert path == tmp_path / FILE.name
    assert path.read_bytes() == DATA
    assert not part_path(FILE, tmp_path).exists()
    assert sum(seen) == len(DATA)


async def test_an_installed_file_is_not_fetched_again(tmp_path: Path) -> None:
    (tmp_path / FILE.name).write_bytes(DATA)
    server = Server()

    _, seen = await run(server, tmp_path)

    assert server.requests == []
    assert seen == [len(DATA)]


async def test_a_lost_connection_keeps_the_part_and_the_next_run_resumes(
    tmp_path: Path,
) -> None:
    server = Server(cut=HASH_CHUNK_BYTES + 5)

    with pytest.raises(
        DownloadError, match=r"connection was lost.*run again to resume"
    ):
        await run(server, tmp_path)
    assert part_path(FILE, tmp_path).stat().st_size == HASH_CHUNK_BYTES + 5

    path, seen = await run(server, tmp_path)

    assert server.requests[-1].headers["range"] == f"bytes={HASH_CHUNK_BYTES + 5}-"
    assert path.read_bytes() == DATA
    assert seen[0] == HASH_CHUNK_BYTES + 5
    assert sum(seen) == len(DATA)


async def test_a_server_that_ignores_the_range_restarts_the_file(
    tmp_path: Path,
) -> None:
    part_path(FILE, tmp_path).write_bytes(DATA[:1000])

    path, _ = await run(Server(ranges=False), tmp_path)

    assert path.read_bytes() == DATA


async def test_a_resume_at_the_wrong_offset_discards_the_part(tmp_path: Path) -> None:
    part_path(FILE, tmp_path).write_bytes(DATA[:1000])

    def wrong(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            206, content=DATA[500:], headers={"content-range": "bytes 500-9/10"}
        )

    with pytest.raises(DownloadError, match="resumed at the wrong offset"):
        await run(wrong, tmp_path)
    assert not part_path(FILE, tmp_path).exists()


async def test_a_file_with_the_wrong_digest_is_deleted(tmp_path: Path) -> None:
    tampered = bytes([DATA[0] ^ 1]) + DATA[1:]

    with pytest.raises(ChecksumError, match="does not match its recorded SHA-256"):
        await run(Server(tampered), tmp_path)
    assert list(tmp_path.iterdir()) == []


async def test_more_bytes_than_recorded_is_refused_and_discarded(
    tmp_path: Path,
) -> None:
    with pytest.raises(DownloadError, match="sent more than"):
        await run(Server(DATA + b"extra"), tmp_path)
    assert list(tmp_path.iterdir()) == []


async def test_a_short_body_is_kept_to_resume(tmp_path: Path) -> None:
    with pytest.raises(DownloadError, match="stopped at 10 of"):
        await run(Server(DATA[:10]), tmp_path)
    assert part_path(FILE, tmp_path).stat().st_size == 10


@pytest.mark.parametrize("status", [404, 403, 500])
async def test_an_http_error_names_the_status_and_writes_nothing(
    tmp_path: Path, status: int
) -> None:
    with pytest.raises(DownloadError, match=f"answered HTTP {status}"):
        await run(Server(status=status), tmp_path / "models")
    assert list((tmp_path / "models").iterdir()) == []


async def test_a_part_longer_than_the_file_is_discarded(tmp_path: Path) -> None:
    part_path(FILE, tmp_path).write_bytes(DATA + b"stale")
    server = Server()

    path, _ = await run(server, tmp_path)

    assert "range" not in server.requests[0].headers
    assert path.read_bytes() == DATA


async def test_a_complete_part_is_verified_without_a_request(tmp_path: Path) -> None:
    part_path(FILE, tmp_path).write_bytes(DATA)
    server = Server()

    path, _ = await run(server, tmp_path)

    assert server.requests == []
    assert path.read_bytes() == DATA


async def test_a_complete_but_corrupt_part_is_deleted(tmp_path: Path) -> None:
    part_path(FILE, tmp_path).write_bytes(bytes(len(DATA)))

    with pytest.raises(ChecksumError):
        await run(Server(), tmp_path)
    assert list(tmp_path.iterdir()) == []


async def test_redirects_are_followed_and_keep_the_range(tmp_path: Path) -> None:
    part_path(FILE, tmp_path).write_bytes(DATA[:77])
    cdn = Server()

    def origin(request: httpx.Request) -> httpx.Response:
        if request.url.host == "files.test":
            return httpx.Response(302, headers={"location": "https://cdn.test/blob"})
        return cdn(request)

    path, _ = await run(origin, tmp_path)

    assert cdn.requests[0].headers["range"] == "bytes=77-"
    assert path.read_bytes() == DATA


async def test_progress_is_optional(tmp_path: Path) -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(Server())) as client:
        path = await fetch(client, FILE, tmp_path)

    assert path.read_bytes() == DATA


async def test_the_bytes_are_asked_for_unencoded(tmp_path: Path) -> None:
    server = Server()

    await run(server, tmp_path)

    assert server.requests[0].headers["accept-encoding"] == "identity"


async def test_a_refused_range_discards_the_part_so_the_next_run_starts_over(
    tmp_path: Path,
) -> None:
    part_path(FILE, tmp_path).write_bytes(DATA[:1000])

    with pytest.raises(DownloadError, match="answered HTTP 416"):
        await run(lambda _: httpx.Response(416), tmp_path)
    assert not part_path(FILE, tmp_path).exists()


async def test_a_server_error_while_resuming_keeps_the_part(tmp_path: Path) -> None:
    part_path(FILE, tmp_path).write_bytes(DATA[:1000])

    with pytest.raises(DownloadError, match="answered HTTP 503"):
        await run(lambda _: httpx.Response(503), tmp_path)
    assert part_path(FILE, tmp_path).stat().st_size == 1000
