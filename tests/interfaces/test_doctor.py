from dataclasses import replace
from pathlib import Path

import pytest

from synthia.interfaces.doctor import (
    LOCAL_MODEL,
    Check,
    Facts,
    LocalFacts,
    Status,
    _nearest_existing,  # pyright: ignore[reportPrivateUsage]
    evaluate,
    gather_facts,
    overall,
)
from synthia.kernel.config import LocalBackend, Settings
from synthia.models.catalogue import Backend, Target
from synthia.models.install import BYTES_PER_GB, Installer

INSTALLED = LocalFacts(
    Target.LINUX_X64,
    model_known=True,
    model_installed=True,
    builds=(Backend.VULKAN, Backend.CPU),
)
HEALTHY = Facts(
    python=(3, 12, 9),
    system="Linux-6.12",
    physical_cores=8,
    logical_cores=16,
    memory_gb=32.0,
    cuda_driver=False,
    home_exists=True,
    disk_free_gb=120.0,
    local=INSTALLED,
)


def settings(
    tmp_path: Path,
    *,
    key: str | None = "sk-test",
    backend: LocalBackend = LocalBackend.AUTO,
) -> Settings:
    return Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        home=tmp_path,
        openrouter_api_key=key,  # pyright: ignore[reportArgumentType]
        local_backend=backend,
    )


def statuses(checks: list[Check]) -> dict[str, Status]:
    return {c.name: c.status for c in checks}


def test_a_healthy_machine_passes_everything(tmp_path: Path) -> None:
    checks = evaluate(settings(tmp_path), HEALTHY)

    assert overall(checks) is Status.OK
    assert [c.name for c in checks] == [
        "python",
        "system",
        "cpu",
        "memory",
        "accelerator",
        LOCAL_MODEL,
        "home",
        "disk",
        "openrouter",
    ]


@pytest.mark.parametrize(
    ("change", "name", "expected"),
    [
        ({"python": (3, 11, 4)}, "python", Status.FAIL),
        ({"memory_gb": 4.0}, "memory", Status.WARN),
        ({"disk_free_gb": 0.4}, "disk", Status.FAIL),
        ({"disk_free_gb": 5.0}, "disk", Status.WARN),
        ({"cuda_driver": True}, "accelerator", Status.OK),
        ({"home_exists": False}, "home", Status.OK),
    ],
)
def test_each_fact_moves_its_own_check(
    tmp_path: Path, change: dict[str, object], name: str, expected: Status
) -> None:
    checks = evaluate(settings(tmp_path), replace(HEALTHY, **change))  # pyright: ignore[reportArgumentType]

    assert statuses(checks)[name] is expected


@pytest.mark.parametrize(
    ("local", "backend", "status", "detail"),
    [
        (INSTALLED, LocalBackend.AUTO, Status.OK, "qwen3.5-4b on vulkan, then cpu"),
        (
            replace(INSTALLED, model_installed=False, builds=()),
            LocalBackend.AUTO,
            Status.WARN,
            (
                "qwen3.5-4b and a llama.cpp build not installed, remote only; "
                "see synthia models list"
            ),
        ),
        (
            replace(INSTALLED, builds=()),
            LocalBackend.CUDA,
            Status.WARN,
            "the cuda build not installed, remote only; see synthia models list",
        ),
        (
            replace(INSTALLED, target=None),
            LocalBackend.AUTO,
            Status.WARN,
            "no llama.cpp build suits this machine, remote only",
        ),
        (
            replace(INSTALLED, model_known=False),
            LocalBackend.AUTO,
            Status.FAIL,
            "SYNTHIA_LOCAL_MODEL=qwen3.5-4b is not a model in the catalogue",
        ),
    ],
    ids=["installed", "nothing", "chosen build missing", "no build here", "unknown"],
)
def test_the_local_model_check_says_what_runs_or_what_is_missing(
    tmp_path: Path,
    local: LocalFacts,
    backend: LocalBackend,
    status: Status,
    detail: str,
) -> None:
    checks = evaluate(
        settings(tmp_path, backend=backend), replace(HEALTHY, local=local)
    )

    check = next(c for c in checks if c.name == LOCAL_MODEL)
    assert (check.status, check.detail) == (status, detail)


def test_a_missing_key_is_a_warning_and_its_value_never_shows(tmp_path: Path) -> None:
    unset = evaluate(settings(tmp_path, key=None), HEALTHY)
    planted = "sk-or-v1-doctor-planted-key"  # pragma: allowlist secret
    report = evaluate(settings(tmp_path, key=planted), HEALTHY)

    assert overall(unset) is Status.WARN
    assert all(planted not in c.detail for c in report)


def test_unknown_core_counts_are_shown_as_unknown(tmp_path: Path) -> None:
    facts = replace(HEALTHY, physical_cores=None, logical_cores=None)

    cpu = next(c for c in evaluate(settings(tmp_path), facts) if c.name == "cpu")

    assert cpu.detail == "? cores, ? threads"


def test_overall_is_the_worst_status_and_ok_when_empty() -> None:
    assert overall([]) is Status.OK


def test_gather_facts_reads_this_machine(tmp_path: Path) -> None:
    home = tmp_path / "not" / "yet"
    facts = gather_facts(settings(home), Installer(home, BYTES_PER_GB))

    assert facts.python[0] == 3
    assert facts.memory_gb > 0
    assert facts.disk_free_gb > 0
    assert not facts.home_exists
    assert facts.local.model_known
    assert not facts.local.model_installed
    assert facts.local.builds == ()


def test_gather_facts_notes_an_unknown_local_model(tmp_path: Path) -> None:
    unknown = settings(tmp_path).model_copy(update={"local_model": "gpt-5"})

    facts = gather_facts(unknown, Installer(tmp_path, BYTES_PER_GB))

    assert not facts.local.model_known


def test_nearest_existing_walks_up_to_a_real_directory(tmp_path: Path) -> None:
    assert _nearest_existing(tmp_path / "a" / "b") == tmp_path
    assert _nearest_existing(tmp_path) == tmp_path


def test_nearest_existing_falls_back_to_the_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def nothing_exists(_: Path) -> bool:
        return False

    monkeypatch.setattr(Path, "exists", nothing_exists)

    assert _nearest_existing(Path("relative") / "path") == Path()
