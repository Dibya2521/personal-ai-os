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


class Pooling(StrEnum):
    """How an embedding model's per-token outputs become one vector."""

    CLS = "cls"
    MEAN = "mean"


@dataclass(frozen=True, slots=True)
class Embedder:
    """A sentence embedding model: an ONNX file and its WordPiece tokenizer.

    ``batch`` is how many texts may run through the network together. A model
    quantised to int8 as it runs takes each layer's scale from the whole
    batch, padding included, so its vector for a text changes with the texts
    beside it; such a model runs one text at a time.
    """

    id: str
    repo: str
    revision: str
    licence: str
    model: Download
    tokenizer: Download
    pooling: Pooling
    dimensions: int
    max_tokens: int
    batch: int

    @property
    def files(self) -> tuple[Download, ...]:
        """Every file of the model, the network first."""
        return (self.model, self.tokenizer)

    @property
    def size(self) -> int:
        """Bytes to download."""
        return sum(file.size for file in self.files)


type Item = Runtime | Model | Embedder
"""Anything ``synthia models`` can install."""


def _release(name: str, size: int, sha256: str) -> Download:
    return Download(name, f"{_RELEASES}/{name}", size, sha256)


def _hugging_face(  # noqa: PLR0913
    repo: str,
    revision: str,
    name: str,
    size: int,
    sha256: str,
    *,
    path: str | None = None,
) -> Download:
    """Return a file of ``repo`` at ``revision``, saved as ``name``.

    ``path`` is where it sits in the repository, when not at the top as ``name``.
    """
    where = path or name
    return Download(
        name, f"https://huggingface.co/{repo}/resolve/{revision}/{where}", size, sha256
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

_BGE_REPO: Final = "BAAI/bge-small-en-v1.5"
_BGE_REVISION: Final = (
    # pragma: allowlist nextline secret
    "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
)
_MINILM_REPO: Final = "sentence-transformers/all-MiniLM-L6-v2"
_MINILM_REVISION: Final = (
    # pragma: allowlist nextline secret
    "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)

# From GET https://huggingface.co/api/models/<repo>/tree/<revision>. The ONNX
# files are in Git LFS, whose id is their SHA-256; tokenizer.json is a plain git
# file, so its SHA-256 was taken from a copy whose git blob id matched the tree.
EMBEDDERS: Final = (
    Embedder(
        "bge-small-en-v1.5",
        _BGE_REPO,
        _BGE_REVISION,
        "mit",
        _hugging_face(
            _BGE_REPO,
            _BGE_REVISION,
            "model.onnx",
            133_093_490,
            # pragma: allowlist nextline secret
            "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35",
            path="onnx/model.onnx",
        ),
        _hugging_face(
            _BGE_REPO,
            _BGE_REVISION,
            "tokenizer.json",
            711_396,
            # pragma: allowlist nextline secret
            "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
        ),
        Pooling.CLS,
        dimensions=384,
        max_tokens=512,
        batch=16,
    ),
    Embedder(
        "all-minilm-l6-v2-int8",
        _MINILM_REPO,
        _MINILM_REVISION,
        "apache-2.0",
        _hugging_face(
            _MINILM_REPO,
            _MINILM_REVISION,
            "model.onnx",
            23_026_053,
            # pragma: allowlist nextline secret
            "4278337fd0ff3c68bfb6291042cad8ab363e1d9fbc43dcb499fe91c871902474",
            path="onnx/model_qint8_avx512.onnx",
        ),
        _hugging_face(
            _MINILM_REPO,
            _MINILM_REVISION,
            "tokenizer.json",
            466_247,
            # pragma: allowlist nextline secret
            "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
        ),
        Pooling.MEAN,
        dimensions=384,
        # sentence-transformers trains and runs it with at most 256 tokens.
        max_tokens=256,
        # Batched, a text's vector moved by up to 0.028 with its neighbours.
        batch=1,
    ),
)
# Measured on benchmarks/data/recall_set.json: bge-small's recall at 1 (0.625),
# more at 5 (0.925 against 0.875), about half its time to embed a turn even one
# text at a time against bge's sixteen, a sixth of its size.
DEFAULT_EMBEDDER: Final = EMBEDDERS[1]
_SYSTEMS: Final = {"windows": "windows", "linux": "linux", "darwin": "macos"}
_MACHINES: Final = {
    "amd64": "x64",
    "x86_64": "x64",
    "arm64": "arm64",
    "aarch64": "arm64",
}


def target_of(system: str, machine: str) -> Target | None:
    """Return the build target for ``platform.system()`` and ``platform.machine()``.

    ``None`` means no build is catalogued for this machine, so SYNTHIA cannot
    answer offline there.
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


def find(name: str) -> Item:
    """Return the runtime, model or embedding model called ``name``.

    Raises:
        KeyError: If nothing in the catalogue has that name.
    """
    for item in (*RUNTIMES, *MODELS, *EMBEDDERS):
        if item.id == name:
            return item
    raise KeyError(name)
