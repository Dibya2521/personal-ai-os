"""``synthia models``: list, install and remove local runtimes and models.

The functions here take the catalogue items and the installer as arguments, so
the command line only resolves names and wires the real ones in.
"""

from __future__ import annotations

import platform
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

from synthia.models.catalogue import (
    DEFAULT_MODEL,
    MODELS,
    RUNTIMES,
    Model,
    Runtime,
    Target,
    find,
    recommended,
    target_of,
)
from synthia.models.install import size_text

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import httpx
    from rich.console import Console

    from synthia.models.catalogue import Download
    from synthia.models.install import Installer

INSTALLED: Final = "installed"
PARTIAL: Final = "partial"
MISSING: Final = "-"


@dataclass(frozen=True, slots=True)
class Machine:
    """What decides which builds suit this computer."""

    target: Target | None
    nvidia: bool


def this_machine() -> Machine:
    """Read the build target and whether an NVIDIA driver is present."""
    return Machine(
        target_of(platform.system(), platform.machine()),
        # nvidia-smi ships with the NVIDIA driver on both Windows and Linux.
        shutil.which("nvidia-smi") is not None,
    )


def suggested(machine: Machine) -> tuple[Runtime | Model, ...]:
    """Return what ``synthia models install`` installs when given no names."""
    if machine.target is None:
        return ()
    return (*recommended(machine.target, nvidia=machine.nvidia), find(DEFAULT_MODEL))


def resolve(names: Sequence[str]) -> tuple[Runtime | Model, ...]:
    """Return the catalogue items called ``names``.

    Raises:
        KeyError: Naming every name the catalogue does not have.
    """
    unknown = [n for n in names if not _known(n)]
    if unknown:
        raise KeyError(", ".join(unknown))
    return tuple(find(n) for n in dict.fromkeys(names))


def _known(name: str) -> bool:
    return any(item.id == name for item in (*RUNTIMES, *MODELS))


def _state(installer: Installer, item: Runtime | Model) -> str:
    if installer.installed(item):
        return INSTALLED
    return PARTIAL if installer.to_download(item) < item.size else MISSING


def list_table(installer: Installer, machine: Machine) -> Table:
    """Return every catalogue item with its size and state; * marks a pick."""
    picks = {item.id for item in suggested(machine)}
    table = Table(box=None, pad_edge=False)
    for column in ("", "name", "size", "state"):
        table.add_column(column)
    for item in (*RUNTIMES, *MODELS):
        mark = "*" if item.id in picks else ""
        table.add_row(mark, item.id, size_text(item.size), _state(installer, item))
    return table


async def install(
    installer: Installer,
    client: httpx.AsyncClient,
    items: Sequence[Runtime | Model],
    console: Console,
    confirm: Callable[[str], bool],
) -> None:
    """Show the plan, ask, then install ``items`` with a progress bar.

    Raises:
        DiskBudgetError: If an item does not fit; earlier items stay installed.
        DownloadError: If a file cannot be fetched whole; run again to resume.
        ArchiveError: If a runtime archive cannot be unpacked safely.
    """
    wanted = [
        (item, installer.to_download(item))
        for item in items
        if not installer.installed(item)
    ]
    if not wanted:
        console.print("everything asked for is installed")
        return
    width = max(len(item.id) for item, _ in wanted)
    for item, left in wanted:
        console.print(f"  {item.id:<{width}}  {size_text(left)} to download")
    total = sum(left for _, left in wanted)
    if not confirm(f"download {size_text(total)} and install?"):
        console.print("nothing installed")
        return
    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        DownloadColumn(binary_units=True),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        for item, _ in wanted:
            task = progress.add_task(item.id, total=item.size)

            def advance(_: Download, count: int, task: TaskID = task) -> None:
                progress.advance(task, count)

            await installer.install(client, item, advance)


def remove(
    installer: Installer, items: Sequence[Runtime | Model], console: Console
) -> None:
    """Delete ``items`` and say how much disk each gave back."""
    for item in items:
        freed = installer.remove(item)
        if freed:
            console.print(f"removed {item.id}, {size_text(freed)} freed")
        else:
            console.print(f"{item.id} was not installed")
