"""What every reader's outline must give the chunker, checked on real files."""

import shutil
import subprocess
from pathlib import Path, PurePosixPath

import pytest

from synthia.knowledge.chunk import Drafted, TokenCounter
from synthia.knowledge.text import Outline, Text

ROOT = Path(__file__).parents[2]


class Words:
    """Counts whitespace-separated words: simple enough to check by eye."""

    def count(self, text: str) -> int:
        return len(text.split())


WORDS = Words()


def tracked(pattern: str) -> list[PurePosixPath]:
    """Return the repository's files git lists for ``pattern``."""
    git = shutil.which("git")
    if git is None or not (ROOT / ".git").exists():
        pytest.skip("the repository's files are listed by git")
    listed = subprocess.run(  # noqa: S603 - a resolved git with fixed arguments
        [git, "ls-files", pattern],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return [PurePosixPath(line) for line in listed.stdout.splitlines()]


def assert_whole(
    text: Text, shape: Outline, drafted: Drafted, counter: TokenCounter, budget: int
) -> None:
    """Check the chunker's promises for a one-page text."""
    covered: list[int] = []
    for ordinal, draft in enumerate(drafted.drafts):
        assert draft.ordinal == ordinal
        assert draft.page is None
        assert draft.body == text.span(None, draft.start_line, draft.end_line)
        lines = draft.body.split("\n")
        assert lines[0].strip()
        assert lines[-1].strip()
        cost = counter.count(draft.header) + sum(counter.count(line) for line in lines)
        assert (cost > budget) is draft.cut
        assert not draft.cut or draft.start_line == draft.end_line
        covered.extend(range(draft.start_line, draft.end_line + 1))
    assert covered == sorted(set(covered))
    lines = text.lines()
    assert {n for n in covered if lines[n - 1].strip()} == {
        n for n, line in enumerate(lines, start=1) if line.strip()
    }
    assert sum(len(d.links) for d in drafted.drafts) == len(shape.links)
