"""Print each failed test of a pytest JUnit report as a GitHub Actions error annotation.

    uv run python scripts/ci_annotations.py pytest-report.xml

A public repository's job logs can be read only by a signed-in user, but its
annotations are shown to everyone, so a red run names the tests that failed
without a login. Each annotation carries the test's id and the first line of
its failure, at most ``MESSAGE_LIMIT`` characters.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

MESSAGE_LIMIT = 300
FAILED_TAGS = ("failure", "error")
# GitHub workflow command escaping: data needs these three, properties also ":" and ",".
_DATA_ESCAPES = (("%", "%25"), ("\r", "%0D"), ("\n", "%0A"))
_PROPERTY_ESCAPES = (*_DATA_ESCAPES, (":", "%3A"), (",", "%2C"))


def _escape(text: str, escapes: Sequence[tuple[str, str]]) -> str:
    for raw, escaped in escapes:
        text = text.replace(raw, escaped)
    return text


def failures(report: Path) -> Iterator[tuple[str, str]]:
    """Yield each failed or errored test's id and the first line of its message."""
    root = ET.parse(report).getroot()  # noqa: S314 - written by pytest in this job
    for case in root.iter("testcase"):
        for tag in FAILED_TAGS:
            found = case.find(tag)
            if found is not None:
                test_id = f"{case.get('classname', '')}::{case.get('name', '')}"
                first_line = (found.get("message") or found.text or tag).strip()
                yield (
                    test_id,
                    first_line.splitlines()[0][:MESSAGE_LIMIT] if first_line else tag,
                )
                break


def annotation(test_id: str, message: str) -> str:
    """Return the workflow command that shows one failed test as an error."""
    title = _escape(test_id, _PROPERTY_ESCAPES)
    return f"::error title={title}::{_escape(message, _DATA_ESCAPES)}"


def main(argv: Sequence[str]) -> int:
    """Print one annotation per failed test in the report ``argv`` names."""
    (report,) = argv
    path = Path(report)
    if not path.exists():
        print(f"::error title=pytest::no report at {report}; pytest stopped early")
        return 0
    for test_id, message in failures(path):
        print(annotation(test_id, message))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
