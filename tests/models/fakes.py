"""Small runtimes and models, and a server that has their files."""

import hashlib
import io
import zipfile

import httpx

from synthia.models.catalogue import Backend, Download, Model, Runtime, Target


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
