"""``synthia doctor``: check the machine and the configuration.

Split in two so the verdicts can be tested without the hardware:
:func:`gather_facts` reads the system, and :func:`evaluate` is a pure function
from those facts and the settings to a list of checks.
"""

from __future__ import annotations

import platform
import shutil
import sys
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    from synthia.kernel.config import Settings

BYTES_PER_GB = 1024**3
MIN_PYTHON = (3, 12)
MIN_MEMORY_GB = 8.0
MIN_FREE_DISK_GB = 1.0


class Status(IntEnum):
    """Outcome of one check. The worst one is the command's exit code."""

    OK = 0
    WARN = 1
    FAIL = 2


@dataclass(frozen=True, slots=True)
class Check:
    """One line of the report."""

    name: str
    status: Status
    detail: str


@dataclass(frozen=True, slots=True)
class Facts:
    """What the machine looks like, read once."""

    python: tuple[int, int, int]
    system: str
    physical_cores: int | None
    logical_cores: int | None
    memory_gb: float
    cuda_driver: bool
    home_exists: bool
    disk_free_gb: float


def _nearest_existing(path: Path) -> Path:
    """Return ``path`` or its closest ancestor that exists, for measuring disk."""
    for candidate in (path, *path.parents):
        if candidate.exists():
            return candidate
    return Path(path.anchor or ".")


def gather_facts(home: Path) -> Facts:
    """Read the facts from this machine."""
    return Facts(
        python=(sys.version_info.major, sys.version_info.minor, sys.version_info.micro),
        system=platform.platform(terse=True),
        physical_cores=psutil.cpu_count(logical=False),
        logical_cores=psutil.cpu_count(logical=True),
        memory_gb=psutil.virtual_memory().total / BYTES_PER_GB,
        # nvidia-smi ships with the NVIDIA driver on both Windows and Linux.
        cuda_driver=shutil.which("nvidia-smi") is not None,
        home_exists=home.is_dir(),
        disk_free_gb=shutil.disk_usage(_nearest_existing(home)).free / BYTES_PER_GB,
    )


def evaluate(settings: Settings, facts: Facts) -> list[Check]:
    """Return the report for ``facts`` under ``settings``."""
    return [
        _python(facts),
        Check("system", Status.OK, facts.system),
        _cpu(facts),
        _memory(facts),
        _accelerator(facts),
        _home(settings, facts),
        _disk(settings, facts),
        _openrouter(settings),
    ]


def _python(facts: Facts) -> Check:
    version = ".".join(map(str, facts.python))
    if facts.python[:2] < MIN_PYTHON:
        return Check("python", Status.FAIL, f"{version}, 3.12 or newer is required")
    return Check("python", Status.OK, version)


def _cpu(facts: Facts) -> Check:
    cores = facts.physical_cores or "?"
    threads = facts.logical_cores or "?"
    return Check("cpu", Status.OK, f"{cores} cores, {threads} threads")


def _memory(facts: Facts) -> Check:
    detail = f"{facts.memory_gb:.1f} GB"
    if facts.memory_gb < MIN_MEMORY_GB:
        return Check("memory", Status.WARN, f"{detail}, local models need 8 GB or more")
    return Check("memory", Status.OK, detail)


def _accelerator(facts: Facts) -> Check:
    if facts.cuda_driver:
        return Check("accelerator", Status.OK, "NVIDIA driver found, CUDA available")
    return Check("accelerator", Status.OK, "no CUDA GPU, inference runs on the CPU")


def _home(settings: Settings, facts: Facts) -> Check:
    if facts.home_exists:
        return Check("home", Status.OK, str(settings.home))
    return Check("home", Status.OK, f"{settings.home} (created on first use)")


def _disk(settings: Settings, facts: Facts) -> Check:
    budget = settings.disk_budget_gb
    detail = f"{facts.disk_free_gb:.1f} GB free, budget {budget:g} GB"
    if facts.disk_free_gb < MIN_FREE_DISK_GB:
        return Check("disk", Status.FAIL, detail)
    if facts.disk_free_gb < budget:
        return Check("disk", Status.WARN, f"{detail}, less free space than the budget")
    return Check("disk", Status.OK, detail)


def _openrouter(settings: Settings) -> Check:
    if settings.openrouter_api_key is None:
        return Check(
            "openrouter",
            Status.WARN,
            "SYNTHIA_OPENROUTER_API_KEY not set, remote models disabled",
        )
    return Check("openrouter", Status.OK, "API key set")


def overall(checks: list[Check]) -> Status:
    """Return the worst status in ``checks``."""
    return max((check.status for check in checks), default=Status.OK)
