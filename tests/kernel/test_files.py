from pathlib import Path

import pytest

from synthia.kernel.files import write_text_atomic


def test_a_new_file_and_its_directories_are_made(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "file.json"

    write_text_atomic(target, "one\ntwo\n")

    assert target.read_bytes() == b"one\ntwo\n"
    assert [p.name for p in target.parent.iterdir()] == ["file.json"]


def test_a_failed_replace_leaves_the_old_file_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "file.json"
    target.write_text("old", encoding="utf-8")

    def refuse(_self: Path, _target: Path) -> Path:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "replace", refuse)

    with pytest.raises(PermissionError):
        write_text_atomic(target, "new")

    assert target.read_text(encoding="utf-8") == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["file.json"]
