import re
from pathlib import Path

import public_guard
import pytest
from hypothesis import given
from hypothesis import strategies as st


@pytest.fixture
def patterns() -> list[re.Pattern[str]]:
    return [re.compile("acme-corp", re.IGNORECASE), re.compile(r"C:[\\/]+Work")]


def test_missing_pattern_file_means_nothing_to_guard(tmp_path: Path) -> None:
    assert public_guard.load_patterns(tmp_path / "absent") == []


def test_comments_and_blank_lines_are_not_patterns(tmp_path: Path) -> None:
    pattern_file = tmp_path / "public-guard"
    pattern_file.write_text("# a comment\n\n  acme  \n   # indented comment\n")

    loaded = public_guard.load_patterns(pattern_file)

    assert [p.pattern for p in loaded] == ["acme"]


def test_finds_a_match_in_content_with_its_line(
    tmp_path: Path, patterns: list[re.Pattern[str]]
) -> None:
    source = tmp_path / "notes.py"
    source.write_text("ok\nbuilt at ACME-Corp\n")

    findings = public_guard.scan([source], patterns)

    assert [(f.line, f.pattern) for f in findings] == [(2, "acme-corp")]
    assert str(findings[0]).endswith(":2: matches private pattern 'acme-corp'")


def test_finds_a_match_in_the_path_itself(
    tmp_path: Path, patterns: list[re.Pattern[str]]
) -> None:
    folder = tmp_path / "acme-corp"
    folder.mkdir()
    source = folder / "clean.txt"
    source.write_text("nothing here\n")

    findings = public_guard.scan([source], patterns)

    assert [f.line for f in findings] == [0]
    assert "clean.txt: matches" in str(findings[0])


def test_sees_text_inside_a_binary_file(
    tmp_path: Path, patterns: list[re.Pattern[str]]
) -> None:
    blob = tmp_path / "blob.bin"
    blob.write_bytes(b"\xff\xfe\x00C:\\\\Work\\\\secret\x00\xff")

    assert public_guard.scan([blob], patterns)


def test_skips_paths_that_are_not_files(
    tmp_path: Path, patterns: list[re.Pattern[str]]
) -> None:
    assert public_guard.scan([tmp_path, tmp_path / "deleted.py"], patterns) == []


@given(st.text())
def test_no_patterns_never_finds_anything(text: str) -> None:
    assert public_guard.scan([Path(text)], []) == []


def test_main_fails_when_a_file_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    info = tmp_path / ".git" / "info"
    info.mkdir(parents=True)
    (info / "public-guard").write_text("acme-corp\n")
    dirty = tmp_path / "dirty.md"
    dirty.write_text("ACME-CORP\n")
    clean = tmp_path / "clean.md"
    clean.write_text("fine\n")
    monkeypatch.setattr(public_guard, "git_dir", lambda: tmp_path / ".git")

    assert public_guard.main([str(clean)]) == 0
    assert public_guard.main([str(dirty), str(clean)]) == 1
    assert "dirty.md:1" in capsys.readouterr().err


def test_git_dir_resolves_inside_this_repository() -> None:
    assert (public_guard.git_dir() / "HEAD").is_file()
