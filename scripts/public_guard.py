"""Reject staged files that contain private text before it reaches a public history.

The patterns are regular expressions read from ``.git/info/public-guard``, one per
line, matched case-insensitively against each file's path and content. The file
lives inside ``.git`` so the patterns are never committed: publishing the list of
things that must not be published would defeat it. With no pattern file there is
nothing private to guard, and every file passes.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

PATTERN_FILE = Path("info") / "public-guard"
COMMENT_PREFIX = "#"


@dataclass(frozen=True, slots=True)
class Finding:
    """One pattern match in one file."""

    path: str
    line: int
    pattern: str

    def __str__(self) -> str:
        """Return the finding as ``path:line``, the form editors can jump to."""
        where = f"{self.path}:{self.line}" if self.line else self.path
        return f"{where}: matches private pattern {self.pattern!r}"


def load_patterns(pattern_file: Path) -> list[re.Pattern[str]]:
    """Return the compiled patterns, or none if the file does not exist.

    Raises:
        re.error: If a line is not a valid regular expression.
    """
    if not pattern_file.is_file():
        return []
    lines = pattern_file.read_text(encoding="utf-8").splitlines()
    return [
        re.compile(line.strip(), re.IGNORECASE)
        for line in lines
        if line.strip() and not line.lstrip().startswith(COMMENT_PREFIX)
    ]


def scan_file(path: Path, patterns: Sequence[re.Pattern[str]]) -> list[Finding]:
    """Return every match of any pattern in the path itself or its text."""
    findings = [
        Finding(path.as_posix(), 0, pattern.pattern)
        for pattern in patterns
        if pattern.search(path.as_posix())
    ]
    # Binary files are decoded leniently so a secret embedded in one is still seen.
    text = path.read_bytes().decode("utf-8", errors="ignore")
    for number, line in enumerate(text.splitlines(), start=1):
        findings.extend(
            Finding(path.as_posix(), number, pattern.pattern)
            for pattern in patterns
            if pattern.search(line)
        )
    return findings


def scan(paths: Iterable[Path], patterns: Sequence[re.Pattern[str]]) -> list[Finding]:
    """Return the findings across all regular files among ``paths``."""
    if not patterns:
        return []
    return [
        finding
        for path in paths
        if path.is_file()
        for finding in scan_file(path, patterns)
    ]


def git_dir() -> Path:
    """Return this repository's git directory, which differs inside a worktree.

    Raises:
        subprocess.CalledProcessError: If the working directory is not in a repository.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--git-dir"],  # noqa: S607 - git is resolved from PATH on purpose
        capture_output=True,
        check=True,
        text=True,
    )
    return Path(result.stdout.strip())


def main(argv: Sequence[str]) -> int:
    """Scan the files pre-commit passes and return the process exit code."""
    findings = scan(
        (Path(arg) for arg in argv), load_patterns(git_dir() / PATTERN_FILE)
    )
    for finding in findings:
        print(finding, file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
