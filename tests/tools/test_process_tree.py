import asyncio
import os
import sys
from pathlib import Path

import pytest

from synthia.tools import process_tree
from synthia.tools.process_tree import ProcessTree
from tests.timing import HANG_TIMEOUT_S


async def started(
    argv: list[str], tmp_path: Path, **options: object
) -> tuple[ProcessTree, bytearray]:
    output = bytearray()
    tree = await ProcessTree.start(
        argv,
        cwd=tmp_path,
        env=dict(os.environ),
        on_output=output.extend,
        **options,  # type: ignore[arg-type]
    )
    return tree, output


async def test_input_sent_after_start_reaches_the_process(tmp_path: Path) -> None:
    tree, output = await started(
        [sys.executable, "-c", "print(input()[::-1])"], tmp_path
    )
    tree.send(b"olleh\n")
    tree.close_input()
    await asyncio.wait_for(tree.exited.wait(), HANG_TIMEOUT_S)
    await asyncio.wait_for(tree.end(), HANG_TIMEOUT_S)

    assert (bytes(output), tree.exit_code) == (f"hello{os.linesep}".encode(), 0)


async def test_the_input_stays_open_for_a_conversation(tmp_path: Path) -> None:
    code = "import sys\nfor line in sys.stdin:\n    print(line.strip().upper())"
    lines: asyncio.Queue[bytes] = asyncio.Queue()
    output = bytearray()

    def take(data: bytes) -> None:
        output.extend(data)
        lines.put_nowait(data)

    tree = await ProcessTree.start(
        [sys.executable, "-u", "-c", code],
        cwd=tmp_path,
        env=dict(os.environ),
        on_output=take,
    )
    try:
        for word in (b"one", b"two"):
            tree.send(word + b"\n")
            while word.upper() not in output:
                await asyncio.wait_for(lines.get(), HANG_TIMEOUT_S)
    finally:
        await asyncio.wait_for(tree.end(), HANG_TIMEOUT_S)

    assert bytes(output).split() == [b"ONE", b"TWO"]


async def test_errors_can_go_to_a_file_apart_from_the_output(tmp_path: Path) -> None:
    code = "import sys\nprint('out')\nprint('err', file=sys.stderr)"
    with (tmp_path / "errors.log").open("wb") as errors:
        tree, output = await started(
            [sys.executable, "-c", code], tmp_path, errors=errors
        )
        await asyncio.wait_for(tree.exited.wait(), HANG_TIMEOUT_S)
        await asyncio.wait_for(tree.end(), HANG_TIMEOUT_S)

    assert bytes(output).split() == [b"out"]
    assert (tmp_path / "errors.log").read_bytes().split() == [b"err"]


async def test_ending_twice_is_safe_and_input_after_it_is_ignored(
    tmp_path: Path,
) -> None:
    tree, _ = await started(
        [sys.executable, "-c", "import time; time.sleep(60)"], tmp_path
    )
    await asyncio.wait_for(tree.end(), HANG_TIMEOUT_S)
    await asyncio.wait_for(tree.end(), HANG_TIMEOUT_S)
    tree.send(b"too late\n")

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
            started([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path),
            HANG_TIMEOUT_S,
        )


async def test_a_program_that_cannot_start_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        await started([str(tmp_path / "no-such-program")], tmp_path)
