import json
import subprocess
import sys
from pathlib import Path

import pytest

from synthia.agent.policy import Decision, Policy
from synthia.agent.tools import FunctionTool, ToolError
from synthia.tools import local_tools
from synthia.tools.files import MAX_READ_CHARS, AllowedRoots, file_tools

WINDOWS = sys.platform == "win32"


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    allowed = tmp_path / "allowed"
    (allowed / "notes").mkdir(parents=True)
    (allowed / "notes" / "todo.txt").write_text("buy milk\n", encoding="utf-8")
    (allowed / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "allowed2").mkdir()
    (tmp_path / "allowed2" / "neighbour.txt").write_text("next door", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("outside", encoding="utf-8")
    return tmp_path


def tools(*roots: Path) -> tuple[FunctionTool, FunctionTool]:
    return file_tools(AllowedRoots.of(roots))


def args(**values: object) -> str:
    return json.dumps(values)


async def refused(tool: FunctionTool, arguments: str) -> str:
    with pytest.raises(ToolError) as raised:
        await tool.run(arguments)
    return str(raised.value)


async def test_a_relative_path_is_read_from_the_first_folder(tree: Path) -> None:
    read, _ = tools(tree / "allowed")

    assert await read.run(args(path="notes/todo.txt")) == "buy milk\n"


async def test_an_absolute_path_inside_any_folder_is_read(tree: Path) -> None:
    read, _ = tools(tree / "allowed", tree / "allowed2")

    assert await read.run(args(path=str(tree / "allowed2" / "neighbour.txt"))) == (
        "next door"
    )


@pytest.mark.parametrize(
    "path",
    [
        "../secret.txt",
        "notes/../../secret.txt",
        "../allowed2/neighbour.txt",
        "SECRET",
    ],
)
async def test_no_spelling_of_a_path_leaves_the_folder(tree: Path, path: str) -> None:
    read, _ = tools(tree / "allowed")
    target = str(tree / "secret.txt") if path == "SECRET" else path

    assert await refused(read, args(path=target)) == (
        f"{target} is outside the allowed folders"
    )


async def test_a_folder_whose_name_starts_like_an_allowed_one_is_outside(
    tree: Path,
) -> None:
    read, _ = tools(tree / "allowed")
    neighbour = str(tree / "allowed2" / "neighbour.txt")

    assert await refused(read, args(path=neighbour)) == (
        f"{neighbour} is outside the allowed folders"
    )


async def test_a_symlink_out_of_the_folder_is_refused(tree: Path) -> None:
    link = tree / "allowed" / "escape.txt"
    try:
        link.symlink_to(tree / "secret.txt")
    except OSError:
        pytest.skip("creating symlinks needs a privilege this account lacks")
    read, _ = tools(tree / "allowed")

    assert await refused(read, args(path="escape.txt")) == (
        "escape.txt is outside the allowed folders"
    )


@pytest.fixture
def junction(tree: Path) -> Path:
    door = tree / "allowed" / "door"
    subprocess.run(  # noqa: S603
        ["cmd", "/c", "mklink", "/J", str(door), str(tree)],  # noqa: S607
        check=True,
        capture_output=True,
    )
    return door


@pytest.mark.skipif(not WINDOWS, reason="junctions are a Windows feature")
async def test_a_junction_out_of_the_folder_is_refused(
    tree: Path, junction: Path
) -> None:
    assert junction.is_dir()
    read, _ = tools(tree / "allowed")

    assert await refused(read, args(path="door/secret.txt")) == (
        "door/secret.txt is outside the allowed folders"
    )


@pytest.mark.skipif(not WINDOWS, reason="Windows paths ignore case")
async def test_a_root_spelled_in_another_case_still_matches(tree: Path) -> None:
    read, _ = tools(Path(str(tree / "allowed").upper()))

    assert await read.run(args(path="b.txt")) == "b"


async def test_with_no_folders_allowed_nothing_is_read(tree: Path) -> None:
    read, listing = tools()

    for tool, arguments in (
        (read, args(path=str(tree / "secret.txt"))),
        (listing, args(directory=".")),
    ):
        assert await refused(tool, arguments) == (
            "no folders are allowed for reading; "
            "SYNTHIA_FILE_ROOTS names the ones that are"
        )


async def test_a_missing_file_and_a_folder_are_not_files(tree: Path) -> None:
    read, _ = tools(tree / "allowed")

    assert await refused(read, args(path="nope.txt")) == "nope.txt is not a file"
    assert await refused(read, args(path="notes")) == "notes is not a file"


async def test_a_binary_file_is_refused(tree: Path) -> None:
    (tree / "allowed" / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR")
    read, _ = tools(tree / "allowed")

    assert await refused(read, args(path="image.png")) == (
        "image.png is a binary file, not text"
    )


async def test_a_long_file_is_cut_and_says_so(tree: Path) -> None:
    (tree / "allowed" / "long.txt").write_text(
        "x" * (MAX_READ_CHARS + 5), encoding="utf-8"
    )
    read, _ = tools(tree / "allowed")

    text = await read.run(args(path="long.txt"))

    assert (
        text
        == "x" * MAX_READ_CHARS
        + f"\n[the first 20000 of {MAX_READ_CHARS + 5} characters]"
    )


async def test_bytes_that_are_not_utf8_are_replaced(tree: Path) -> None:
    (tree / "allowed" / "latin.txt").write_bytes(b"caf\xe9")
    read, _ = tools(tree / "allowed")

    assert await read.run(args(path="latin.txt")) == "caf\N{REPLACEMENT CHARACTER}"


async def test_a_listing_is_sorted_and_marks_folders(tree: Path) -> None:
    _, listing = tools(tree / "allowed")

    assert await listing.run(args(directory=".")) == "b.txt\nnotes/"


async def test_a_listing_stops_at_the_limit_and_says_so(tree: Path) -> None:
    for i in range(3):
        (tree / "allowed" / "notes" / f"{i}.txt").write_text("", encoding="utf-8")
    _, listing = tools(tree / "allowed")

    assert await listing.run(args(directory="notes", limit=2)) == (
        "0.txt\n1.txt\n[2 of 4 entries]"
    )


async def test_an_empty_folder_says_so(tree: Path) -> None:
    (tree / "allowed" / "empty").mkdir()
    _, listing = tools(tree / "allowed")

    assert await listing.run(args(directory="empty")) == "(empty)"


async def test_listing_outside_or_a_file_is_refused(tree: Path) -> None:
    _, listing = tools(tree / "allowed")

    assert await refused(listing, args(directory="..")) == (
        ".. is outside the allowed folders"
    )
    assert await refused(listing, args(directory="b.txt")) == "b.txt is not a folder"


@pytest.mark.parametrize("limit", [0, 501])
async def test_a_listing_limit_out_of_range_never_runs(tree: Path, limit: int) -> None:
    _, listing = tools(tree / "allowed")

    assert "limit" in await refused(listing, args(directory=".", limit=limit))


def test_a_home_relative_root_is_expanded(
    monkeypatch: pytest.MonkeyPatch, tree: Path
) -> None:
    monkeypatch.setenv("USERPROFILE" if WINDOWS else "HOME", str(tree))

    assert AllowedRoots.of([Path("~/allowed")]).roots == ((tree / "allowed").resolve(),)


def test_the_local_tools_are_four_reads_and_python_which_asks() -> None:
    box = local_tools()

    assert [
        (s.name, Policy().decide(t)) for s, t in zip(box.specs(), box, strict=True)
    ] == [
        ("current_time", Decision.ALLOW),
        ("calculate", Decision.ALLOW),
        ("read_file", Decision.ALLOW),
        ("list_files", Decision.ALLOW),
        ("run_python", Decision.ASK),
    ]
