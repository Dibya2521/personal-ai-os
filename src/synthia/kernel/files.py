"""Writing a file so a reader sees the old content or the new, never half of it."""

from __future__ import annotations

import tempfile
from pathlib import Path


def write_text_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` as UTF-8 with the line endings given.

    The text goes to a temporary file in the same directory, which then
    replaces ``path`` in one step; a crash or a full disk leaves the old file
    whole. Missing parent directories are made.

    Raises:
        OSError: If the directory, the temporary file or the replace fails.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=path.name,
        suffix=".part",
        delete_on_close=False,
    ) as part:
        part.write(text)
        part.close()
        Path(part.name).replace(path)
