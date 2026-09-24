import hashlib
import io
import zipfile
from pathlib import Path

import httpx
import pytest

from synthia.models.catalogue import Backend, Download, Model, Runtime, Target
from synthia.models.download import part_path
from synthia.models.install import (
    BYTES_PER_GB,
    COMPLETE_MARKER,
    DiskBudgetError,
    Installer,
    free_bytes,
    gigabytes,
)

PLENTY = 100 * BYTES_PER_GB


def zipped(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def described(name: str, data: bytes) -> Download:
    return Download(
        name, f"https://files.test/{name}", len(data), hashlib.sha256(data).hexdigest()
    )


WEIGHTS = b"w" * 5000
PROJECTOR = b"p" * 700
SERVER_ZIP = zipped({"llama-server.exe": b"MZ" * 300})
CUDART_ZIP = zipped({"cudart64_12.dll": b"cu" * 200})
# Deflate shrinks a megabyte of zeros to about a kilobyte: cheap to fetch, big unpacked.
BOMB_ZIP = zipped({"huge.bin": bytes(1024 * 1024)})
FILES = {
    "m.gguf": WEIGHTS,
    "mmproj.gguf": PROJECTOR,
    "cpu.zip": SERVER_ZIP,
    "cudart.zip": CUDART_ZIP,
    "bomb.zip": BOMB_ZIP,
}
MODEL = Model(
    "tiny",
    "r",
    "abc",
    "mit",
    described("m.gguf", WEIGHTS),
    described("mmproj.gguf", PROJECTOR),
)
CPU = Runtime(Target.WINDOWS_X64, Backend.CPU, (described("cpu.zip", SERVER_ZIP),))
CUDA = Runtime(
    Target.WINDOWS_X64,
    Backend.CUDA,
    (described("cpu.zip", SERVER_ZIP), described("cudart.zip", CUDART_ZIP)),
)
BOMB = Runtime(Target.LINUX_X64, Backend.CPU, (described("bomb.zip", BOMB_ZIP),))


class Files:
    def __init__(self) -> None:
        self.requests: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        name = request.url.path.removeprefix("/")
        self.requests.append(name)
        return httpx.Response(200, content=FILES[name])


def installer(
    home: Path, files: Files, budget: int = PLENTY, free: int = PLENTY
) -> Installer:
    client = httpx.AsyncClient(transport=httpx.MockTransport(files))
    return Installer(home, budget, client, lambda _: free)


async def test_a_model_installs_every_file_and_reports_progress(
    tmp_path: Path,
) -> None:
    files = Files()
    seen: dict[str, int] = {}
    models = installer(tmp_path, files)

    path = await models.install(
        MODEL, lambda f, n: seen.__setitem__(f.name, seen.get(f.name, 0) + n)
    )

    assert path == tmp_path / "models" / "tiny"
    assert (path / "m.gguf").read_bytes() == WEIGHTS
    assert models.installed(MODEL)
    assert models.to_download(MODEL) == 0
    assert seen == {"m.gguf": len(WEIGHTS), "mmproj.gguf": len(PROJECTOR)}


async def test_a_runtime_is_unpacked_marked_and_its_archives_removed(
    tmp_path: Path,
) -> None:
    models = installer(tmp_path, Files())

    path = await models.install(CUDA)

    assert sorted(p.name for p in path.iterdir()) == [
        COMPLETE_MARKER,
        "cudart64_12.dll",
        "llama-server.exe",
    ]
    assert (path / COMPLETE_MARKER).read_text(encoding="utf-8") == "b11130\n"
    assert not (tmp_path / "downloads" / CUDA.id).exists()
    assert models.installed(CUDA)
    assert not models.installed(CPU)


async def test_installing_again_fetches_nothing(tmp_path: Path) -> None:
    files = Files()
    models = installer(tmp_path, files)
    await models.install(CPU)
    await models.install(MODEL)
    fetched = len(files.requests)

    await models.install(CPU)
    await models.install(MODEL)

    assert len(files.requests) == fetched == 3


async def test_over_the_budget_is_refused_before_any_byte(tmp_path: Path) -> None:
    files = Files()
    models = installer(tmp_path, files, budget=len(WEIGHTS))

    with pytest.raises(DiskBudgetError, match="raise SYNTHIA_DISK_BUDGET_GB"):
        await models.install(MODEL)
    assert files.requests == []
    assert models.used_bytes() == 0


async def test_a_nearly_full_disk_is_refused(tmp_path: Path) -> None:
    models = installer(tmp_path, Files(), free=BYTES_PER_GB + len(WEIGHTS))

    with pytest.raises(DiskBudgetError, match=r"under 1\.00 GB free"):
        await models.install(MODEL)


async def test_an_archive_too_big_once_unpacked_is_kept_but_not_unpacked(
    tmp_path: Path,
) -> None:
    models = installer(tmp_path, Files(), budget=512 * 1024)

    with pytest.raises(DiskBudgetError, match=r"installing llama\.cpp-b11130-linux"):
        await models.install(BOMB)
    assert not models.path_of(BOMB).exists()
    assert (tmp_path / "downloads" / BOMB.id / "bomb.zip").is_file()
    assert models.to_download(BOMB) == 0


async def test_a_part_file_counts_as_fetched_and_as_used(tmp_path: Path) -> None:
    directory = tmp_path / "models" / "tiny"
    directory.mkdir(parents=True)
    part_path(MODEL.weights, directory).write_bytes(WEIGHTS[:1000])
    models = installer(tmp_path, Files())

    assert models.to_download(MODEL) == MODEL.size - 1000
    assert models.used_bytes() == 1000
    assert not models.installed(MODEL)


async def test_remove_deletes_everything_and_reports_the_bytes(
    tmp_path: Path,
) -> None:
    models = installer(tmp_path, Files())
    await models.install(MODEL)
    await models.install(CPU)
    (tmp_path / "downloads" / CPU.id).mkdir(parents=True)
    (tmp_path / "downloads" / CPU.id / "cpu.zip.part").write_bytes(b"12345")

    assert models.remove(MODEL) == MODEL.size
    left = models.used_bytes()

    assert left > 5
    assert models.remove(CPU) == left
    assert models.used_bytes() == 0
    assert models.remove(MODEL) == 0


def test_free_bytes_measures_the_nearest_existing_folder(tmp_path: Path) -> None:
    assert free_bytes(tmp_path / "not" / "yet") == free_bytes(tmp_path) > 0


def test_gigabytes_are_binary_like_the_doctor() -> None:
    assert gigabytes(3 * BYTES_PER_GB // 2) == "1.50 GB"
