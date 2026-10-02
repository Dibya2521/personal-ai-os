import asyncio
import os
import sys
from pathlib import Path

import pytest

from synthia.tools import process_tree
from synthia.tools.process_tree import ProcessTree
from tests.timing import HANG_TIMEOUT_S


async def test_input_goes_in_and_output_comes_back(tmp_path: Path) -> None:
    tree = await ProcessTree.start(
        [sys.executable, "-c", "print(input()[::-1])"],
        stdin=b"olleh\n",
        cwd=tmp_path,
        env=dict(os.environ),
        limit=100,
    )
    await asyncio.wait_for(tree.exited.wait(), HANG_TIMEOUT_S)
    await asyncio.wait_for(tree.end(), HANG_TIMEOUT_S)

    assert (bytes(tree.output), tree.exit_code) == (f"hello{os.linesep}".encode(), 0)
    assert not tree.overflowed.is_set()


async def test_ending_twice_is_safe(tmp_path: Path) -> None:
    tree = await ProcessTree.start(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=b"",
        cwd=tmp_path,
        env=dict(os.environ),
        limit=100,
    )
    await asyncio.wait_for(tree.end(), HANG_TIMEOUT_S)
    await asyncio.wait_for(tree.end(), HANG_TIMEOUT_S)

    assert tree.exit_code not in {None, 0}


class RefusedError(Exception):
    pass


async def test_a_process_that_could_not_be_contained_is_still_ended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(_self: object, _pid: int) -> None:
        raise RefusedError

    monkeypatch.setattr(process_tree._Containment, "hold", refuse)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    # start re-raises only after ending the process; a hang would time out instead.
    with pytest.raises(RefusedError):
        await asyncio.wait_for(
            ProcessTree.start(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                stdin=b"",
                cwd=tmp_path,
                env=dict(os.environ),
                limit=100,
            ),
            HANG_TIMEOUT_S,
        )


async def test_a_program_that_cannot_start_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        await ProcessTree.start(
            [str(tmp_path / "no-such-program")],
            stdin=b"",
            cwd=tmp_path,
            env=dict(os.environ),
            limit=100,
        )
