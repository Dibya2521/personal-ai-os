import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from synthia.models.archive import (
    STAGING_SUFFIX,
    ArchiveError,
    unpack,
    unpacked_size,
)

SERVER = b"MZ llama-server"
LIBRARY = b"ggml " * 100


def make_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zipped:
        for name, data in members.items():
            zipped.writestr(name, data)
    return path


def add_file(tarred: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o755
    tarred.addfile(info, io.BytesIO(data))


def add_link(tarred: tarfile.TarFile, name: str, target: str) -> None:
    info = tarfile.TarInfo(name)
    info.type = tarfile.SYMTYPE
    info.linkname = target
    tarred.addfile(info)


def make_tar(path: Path, members: dict[str, bytes], links: dict[str, str]) -> Path:
    with tarfile.open(path, "w:gz") as tarred:
        for name, data in members.items():
            add_file(tarred, name, data)
        for name, target in links.items():
            add_link(tarred, name, target)
    return path


def test_a_zip_unpacks_with_its_folders(tmp_path: Path) -> None:
    archive = make_zip(
        tmp_path / "llama.zip", {"llama-server.exe": SERVER, "lib/ggml.dll": LIBRARY}
    )
    target = tmp_path / "runtime"

    unpack([archive], target)

    assert (target / "llama-server.exe").read_bytes() == SERVER
    assert (target / "lib" / "ggml.dll").read_bytes() == LIBRARY
    assert unpacked_size(archive) == len(SERVER) + len(LIBRARY)


def test_a_tar_unpacks_its_files_and_inner_links(tmp_path: Path) -> None:
    archive = make_tar(
        tmp_path / "llama.tar.gz",
        {"build/bin/llama-server": SERVER, "build/bin/libggml.so.0": LIBRARY},
        {"build/bin/libggml.so": "libggml.so.0"},
    )
    target = tmp_path / "runtime"

    unpack([archive], target)

    assert (target / "build/bin/llama-server").read_bytes() == SERVER
    assert (target / "build/bin/libggml.so").read_bytes() == LIBRARY
    assert unpacked_size(archive) == len(SERVER) + len(LIBRARY)


def test_a_build_and_its_cuda_runtime_share_one_directory(tmp_path: Path) -> None:
    build = make_zip(tmp_path / "llama.zip", {"llama-server.exe": SERVER})
    cudart = make_zip(tmp_path / "cudart.zip", {"cudart64_12.dll": LIBRARY})
    target = tmp_path / "runtime"

    unpack([build, cudart], target)

    assert sorted(p.name for p in target.iterdir()) == [
        "cudart64_12.dll",
        "llama-server.exe",
    ]


def test_unpacking_again_replaces_the_old_install_and_stale_staging(
    tmp_path: Path,
) -> None:
    target = tmp_path / "runtime"
    target.mkdir()
    (target / "old.dll").write_bytes(b"old")
    stale = tmp_path / f"runtime{STAGING_SUFFIX}"
    stale.mkdir()
    (stale / "half.dll").write_bytes(b"half")

    unpack([make_zip(tmp_path / "new.zip", {"new.dll": LIBRARY})], target)

    assert [p.name for p in target.iterdir()] == ["new.dll"]
    assert not stale.exists()


@pytest.mark.parametrize(
    "name",
    [
        "../evil.dll",
        "lib/../../evil.dll",
        "/etc/evil",
        "C:/Windows/evil.dll",
        "C:evil.dll",
        "..\\evil.dll",
    ],
)
def test_a_zip_member_outside_the_directory_is_refused(
    tmp_path: Path, name: str
) -> None:
    archive = make_zip(tmp_path / "bad.zip", {"ok.dll": LIBRARY, name: b"x"})
    target = tmp_path / "runtime"

    with pytest.raises(ArchiveError, match="would land outside"):
        unpack([archive], target)
    assert not target.exists()
    assert not (tmp_path / f"runtime{STAGING_SUFFIX}").exists()
    assert not (tmp_path / "evil.dll").exists()


@pytest.mark.parametrize(
    ("members", "links"),
    [
        ({"../evil": b"x"}, {}),
        ({"/etc/evil": b"x"}, {}),
        ({}, {"escape": "../../outside"}),
        ({}, {"escape": "/etc/passwd"}),
    ],
)
def test_a_tar_member_outside_the_directory_is_refused(
    tmp_path: Path, members: dict[str, bytes], links: dict[str, str]
) -> None:
    archive = make_tar(tmp_path / "bad.tar.gz", members, links)
    target = tmp_path / "runtime"

    with pytest.raises(ArchiveError, match="refusing to unpack it"):
        unpack([archive], target)
    assert not target.exists()


def test_a_failed_second_archive_leaves_the_old_install(tmp_path: Path) -> None:
    target = tmp_path / "runtime"
    target.mkdir()
    (target / "working.dll").write_bytes(b"ok")
    good = make_zip(tmp_path / "good.zip", {"new.dll": LIBRARY})
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip at all")

    with pytest.raises(ArchiveError, match=r"bad\.zip is not a readable archive"):
        unpack([good, bad], target)
    assert [p.name for p in target.iterdir()] == ["working.dll"]


@pytest.mark.parametrize("name", ["bad.zip", "bad.tar.gz"])
def test_an_unreadable_archive_says_so(tmp_path: Path, name: str) -> None:
    archive = tmp_path / name
    archive.write_bytes(b"\x1f\x8b truncated")

    with pytest.raises(ArchiveError, match="is not a readable archive"):
        unpacked_size(archive)
    with pytest.raises(ArchiveError, match="is not a readable archive"):
        unpack([archive], tmp_path / "runtime")


def test_an_unknown_archive_type_is_refused(tmp_path: Path) -> None:
    archive = tmp_path / "llama.7z"
    archive.write_bytes(b"7z")

    with pytest.raises(ArchiveError, match="unknown archive type"):
        unpacked_size(archive)
    with pytest.raises(ArchiveError, match="unknown archive type"):
        unpack([archive], tmp_path / "runtime")
