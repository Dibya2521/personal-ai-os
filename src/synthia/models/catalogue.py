"""Every file SYNTHIA can install, pinned to an exact size and SHA-256.

A download is trusted only if it matches the digest recorded here, so the digest
must come from somewhere other than the download itself: the GitHub release API
for llama.cpp, the Hugging Face tree API for the model. Model files are fetched
from a pinned revision, because a branch such as ``main`` can move under a
recorded hash.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

LLAMA_CPP_BUILD: Final = "b11130"
_RELEASES: Final = (
    f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_BUILD}"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


class Target(StrEnum):
    """An operating system and processor a llama.cpp build is made for."""

    WINDOWS_X64 = "windows-x64"
    LINUX_X64 = "linux-x64"
    MACOS_ARM64 = "macos-arm64"
    MACOS_X64 = "macos-x64"


class Backend(StrEnum):
    """What a llama.cpp build runs the model on."""

    CPU = "cpu"
    VULKAN = "vulkan"
    CUDA = "cuda"
    METAL = "metal"


@dataclass(frozen=True, slots=True)
class Download:
    """One file to fetch, and what it must be once fetched.

    Raises:
        ValueError: If the name is not a plain file name, the size is not
            positive or the digest is not 64 lower-case hex digits.
    """

    name: str
    url: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name or "\\" in self.name or ".." in self.name:
            message = f"not a plain file name: {self.name!r}"
            raise ValueError(message)
        if self.size <= 0:
            message = f"{self.name}: size must be positive"
            raise ValueError(message)
        if not _SHA256.fullmatch(self.sha256):
            message = f"{self.name}: not a SHA-256 digest: {self.sha256!r}"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class Runtime:
    """A llama.cpp build: its archive, plus the CUDA runtime a CUDA build needs."""

    target: Target
    backend: Backend
    archives: tuple[Download, ...]
    build: str = LLAMA_CPP_BUILD

    @property
    def id(self) -> str:
        """Name used on the command line and as the install directory."""
        return f"llama.cpp-{self.build}-{self.target}-{self.backend}"

    @property
    def size(self) -> int:
        """Bytes to download."""
        return sum(archive.size for archive in self.archives)


@dataclass(frozen=True, slots=True)
class Model:
    """A model's weights, and the projector that lets it see images, if any."""

    id: str
    repo: str
    revision: str
    licence: str
    weights: Download
    projector: Download | None = None

    @property
    def files(self) -> tuple[Download, ...]:
        """Every file of the model, weights first."""
        return (
            (self.weights,)
            if self.projector is None
            else (self.weights, self.projector)
        )

    @property
    def size(self) -> int:
        """Bytes to download."""
        return sum(file.size for file in self.files)

    @property
    def vision(self) -> bool:
        """Whether the model can take images."""
        return self.projector is not None


def _release(name: str, size: int, sha256: str) -> Download:
    return Download(name, f"{_RELEASES}/{name}", size, sha256)


def _hugging_face(
    repo: str, revision: str, name: str, size: int, sha256: str
) -> Download:
    return Download(
        name, f"https://huggingface.co/{repo}/resolve/{revision}/{name}", size, sha256
    )


# From GET https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b11130.
RUNTIMES: Final = (
    Runtime(
        Target.WINDOWS_X64,
        Backend.CPU,
        (
            _release(
                "llama-b11130-bin-win-cpu-x64.zip",
                18_558_075,
                # pragma: allowlist nextline secret
                "147b02e233de5136bc7f80b1a16ce7040a675bc91c908bfeb0c2abfec08ba716",
            ),
        ),
    ),
    Runtime(
        Target.WINDOWS_X64,
        Backend.VULKAN,
        (
            _release(
                "llama-b11130-bin-win-vulkan-x64.zip",
                32_123_579,
                # pragma: allowlist nextline secret
                "dd7eead919e618e2b4112819a98369691852caf23b243e5d8d8eefe72bed5322",
            ),
        ),
    ),
    Runtime(
        Target.WINDOWS_X64,
        Backend.CUDA,
        (
            _release(
                "llama-b11130-bin-win-cuda-12.4-x64.zip",
                253_783_111,
                # pragma: allowlist nextline secret
                "f91cdc739d60be1111622ce1db0e3d3b225ddbbaa550b3a017c2f4aace36aaa3",
            ),
            _release(
                "cudart-llama-bin-win-cuda-12.4-x64.zip",
                391_443_627,
                # pragma: allowlist nextline secret
                "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
            ),
        ),
    ),
    Runtime(
        Target.LINUX_X64,
        Backend.CPU,
        (
            _release(
                "llama-b11130-bin-ubuntu-x64.tar.gz",
                16_996_245,
                # pragma: allowlist nextline secret
                "85b47d65f8b0cb31ba9f83b66389405a7f814f3ac858728d235294270afb2b7a",
            ),
        ),
    ),
    Runtime(
        Target.LINUX_X64,
        Backend.VULKAN,
        (
            _release(
                "llama-b11130-bin-ubuntu-vulkan-x64.tar.gz",
                30_600_381,
                # pragma: allowlist nextline secret
                "26e264ebe831519538622f03dc852cbe7c3985e58a7af1f7840d8ec2f9f4bacc",
            ),
        ),
    ),
    Runtime(
        Target.LINUX_X64,
        Backend.CUDA,
        (
            _release(
                "llama-b11130-bin-ubuntu-cuda-12.8-x64.tar.gz",
                168_876_350,
                # pragma: allowlist nextline secret
                "4fbafbc7880f059775799248b0c7fc111bea48e1fb62897053ed5250fe5f464c",
            ),
            _release(
                "cudart-llama-b11130-bin-ubuntu-cuda-12.8-x64.tar.gz",
                594_373_751,
                # pragma: allowlist nextline secret
                "edad9908a6bff31322d47bcd4fed5d983b60b8478f3eb1c8264a4f7e974ccae6",
            ),
        ),
    ),
    Runtime(
        Target.MACOS_ARM64,
        Backend.METAL,
        (
            _release(
                "llama-b11130-bin-macos-arm64.tar.gz",
                11_200_079,
                # pragma: allowlist nextline secret
                "eee14b0d2c3915b1c45a4577886f4ef4f9f616914a40b22d836f1c4ef351530a",
            ),
        ),
    ),
    Runtime(
        Target.MACOS_X64,
        Backend.CPU,
        (
            _release(
                "llama-b11130-bin-macos-x64.tar.gz",
                11_236_436,
                # pragma: allowlist nextline secret
                "872b176393d769241cc995fafc6db33e969f20d8fc54900a2a11e39dfd475e97",
            ),
        ),
    ),
)

_QWEN_REPO: Final = "unsloth/Qwen3.5-4B-GGUF"
_QWEN_REVISION: Final = (
    # pragma: allowlist nextline secret
    "e87f176479d0855a907a41277aca2f8ee7a09523"
)

# From GET https://huggingface.co/api/models/unsloth/Qwen3.5-4B-GGUF and its tree.
MODELS: Final = (
    Model(
        "qwen3.5-4b",
        _QWEN_REPO,
        _QWEN_REVISION,
        "apache-2.0",
        _hugging_face(
            _QWEN_REPO,
            _QWEN_REVISION,
            "Qwen3.5-4B-Q4_K_M.gguf",
            2_740_937_888,
            # pragma: allowlist nextline secret
            "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4",
        ),
        _hugging_face(
            _QWEN_REPO,
            _QWEN_REVISION,
            "mmproj-F16.gguf",
            672_423_616,
            # pragma: allowlist nextline secret
            "cd88edcf8d031894960bb0c9c5b9b7e1fea6ebee02b9f7ce925a00d12891f864",
        ),
    ),
)
DEFAULT_MODEL: Final = "qwen3.5-4b"

_SYSTEMS: Final = {"windows": "windows", "linux": "linux", "darwin": "macos"}
_MACHINES: Final = {
    "amd64": "x64",
    "x86_64": "x64",
    "arm64": "arm64",
    "aarch64": "arm64",
}


def target_of(system: str, machine: str) -> Target | None:
    """Return the build target for ``platform.system()`` and ``platform.machine()``.

    ``None`` means no build is catalogued for this machine, so SYNTHIA runs
    remote only.
    """
    os_name = _SYSTEMS.get(system.lower())
    cpu = _MACHINES.get(machine.lower())
    if os_name is None or cpu is None:
        return None
    try:
        return Target(f"{os_name}-{cpu}")
    except ValueError:
        return None


def recommended(target: Target, *, nvidia: bool) -> tuple[Runtime, ...]:
    """Return the builds to install on ``target``: the fastest likely, then the floor.

    The CPU build is always included as the floor to fall back to. On an Apple
    Silicon Mac the one build runs on Metal or the CPU.
    """
    builds = {r.backend: r for r in RUNTIMES if r.target == target}
    accelerator = Backend.CUDA if nvidia else Backend.VULKAN
    order = (Backend.METAL, accelerator, Backend.CPU)
    return tuple(builds[b] for b in order if b in builds)


def find(name: str) -> Runtime | Model:
    """Return the runtime or model called ``name``.

    Raises:
        KeyError: If nothing in the catalogue has that name.
    """
    for item in (*RUNTIMES, *MODELS):
        if item.id == name:
            return item
    raise KeyError(name)
