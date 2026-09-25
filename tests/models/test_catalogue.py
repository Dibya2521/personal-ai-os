import pytest

from synthia.kernel.config import DEFAULT_LOCAL_MODEL
from synthia.models.catalogue import (
    LLAMA_CPP_BUILD,
    MODELS,
    RUNTIMES,
    Backend,
    Download,
    Model,
    Runtime,
    Target,
    find,
    recommended,
    target_of,
)

DIGEST = "0" * 64
ASSET_TARGET = {
    Target.WINDOWS_X64: "-bin-win-",
    Target.LINUX_X64: "-bin-ubuntu-",
    Target.MACOS_ARM64: "-bin-macos-arm64",
    Target.MACOS_X64: "-bin-macos-x64",
}


def test_every_name_in_the_catalogue_is_unique() -> None:
    ids = [item.id for item in (*RUNTIMES, *MODELS)]
    files = [d.name for r in RUNTIMES for d in r.archives] + [
        f.name for m in MODELS for f in m.files
    ]

    assert len(ids) == len(set(ids))
    assert len(files) == len(set(files))


@pytest.mark.parametrize("runtime", RUNTIMES, ids=lambda r: r.id)
def test_each_runtime_archive_is_the_build_it_claims(runtime: Runtime) -> None:
    main = runtime.archives[0]

    assert main.name.startswith(f"llama-{LLAMA_CPP_BUILD}-")
    assert ASSET_TARGET[runtime.target] in main.name
    for archive in runtime.archives:
        assert archive.url.endswith(
            f"/releases/download/{LLAMA_CPP_BUILD}/{archive.name}"
        )
    if runtime.backend in {Backend.VULKAN, Backend.CUDA}:
        assert f"-{runtime.backend}-" in main.name
    if runtime.backend == Backend.CUDA:
        assert runtime.archives[1].name.startswith("cudart-")


def test_model_files_come_from_the_pinned_revision() -> None:
    for model in MODELS:
        for file in model.files:
            assert file.url == (
                f"https://huggingface.co/{model.repo}/resolve/{model.revision}/{file.name}"
            )


def test_this_laptops_download_is_the_recorded_total() -> None:
    builds = recommended(Target.WINDOWS_X64, nvidia=False)
    model = find(DEFAULT_LOCAL_MODEL)

    assert sum(b.size for b in builds) + model.size == 3_464_043_158


@pytest.mark.parametrize(
    ("system", "machine", "target"),
    [
        ("Windows", "AMD64", Target.WINDOWS_X64),
        ("Linux", "x86_64", Target.LINUX_X64),
        ("Darwin", "arm64", Target.MACOS_ARM64),
        ("Darwin", "x86_64", Target.MACOS_X64),
        ("Linux", "aarch64", None),
        ("Windows", "ARM64", None),
        ("FreeBSD", "amd64", None),
        ("Linux", "riscv64", None),
    ],
)
def test_target_of_maps_the_platform_names(
    system: str, machine: str, target: Target | None
) -> None:
    assert target_of(system, machine) == target


@pytest.mark.parametrize(
    ("target", "nvidia", "backends"),
    [
        (Target.WINDOWS_X64, False, (Backend.VULKAN, Backend.CPU)),
        (Target.WINDOWS_X64, True, (Backend.CUDA, Backend.CPU)),
        (Target.LINUX_X64, False, (Backend.VULKAN, Backend.CPU)),
        (Target.LINUX_X64, True, (Backend.CUDA, Backend.CPU)),
        (Target.MACOS_ARM64, False, (Backend.METAL,)),
        (Target.MACOS_X64, True, (Backend.CPU,)),
    ],
)
def test_recommended_puts_the_accelerator_first_and_keeps_the_cpu_floor(
    target: Target, nvidia: bool, backends: tuple[Backend, ...]
) -> None:
    builds = recommended(target, nvidia=nvidia)

    assert tuple(b.backend for b in builds) == backends
    assert all(b.target == target for b in builds)


def test_find_returns_runtimes_and_models_and_refuses_unknown_names() -> None:
    assert find(DEFAULT_LOCAL_MODEL).id == DEFAULT_LOCAL_MODEL
    assert find(RUNTIMES[0].id) is RUNTIMES[0]
    with pytest.raises(KeyError, match="gpt-5"):
        find("gpt-5")


def test_a_model_without_a_projector_cannot_see() -> None:
    weights = Download("w.gguf", "https://example.test/w.gguf", 10, DIGEST)
    model = Model("m", "r", "abc", "mit", weights)

    assert model.files == (weights,)
    assert model.size == 10
    assert not model.vision
    assert MODELS[0].vision


@pytest.mark.parametrize(
    ("name", "size", "digest", "problem"),
    [
        ("../evil", 1, DIGEST, "not a plain file name"),
        ("dir/file", 1, DIGEST, "not a plain file name"),
        ("dir\\file", 1, DIGEST, "not a plain file name"),
        ("", 1, DIGEST, "not a plain file name"),
        ("f", 0, DIGEST, "size must be positive"),
        ("f", 1, "A" * 64, "not a SHA-256 digest"),
        ("f", 1, "0" * 63, "not a SHA-256 digest"),
    ],
)
def test_a_download_refuses_what_could_not_be_safe(
    name: str, size: int, digest: str, problem: str
) -> None:
    with pytest.raises(ValueError, match=problem):
        Download(name, "https://example.test/f", size, digest)
