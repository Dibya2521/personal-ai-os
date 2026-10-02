"""Reading files, only inside the folders the user has allowed.

Every path is resolved to the real file it names, following ``..``, links and
junctions, before it is compared with the allowed folders, so no spelling of a
path can reach outside them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final

from pydantic import Field

from synthia.agent.tools import Effect, FunctionTool, Reach, ToolError

if TYPE_CHECKING:
    from collections.abc import Iterable

# About 5,000 tokens at 4 characters each: four reads still leave the
# conversation over 12,000 of the local model's 32,768 tokens.
MAX_READ_CHARS: Final = 20_000
SNIFF_BYTES: Final = 8192
DEFAULT_LIST_LIMIT: Final = 50
MAX_LIST_LIMIT: Final = 500


@dataclass(frozen=True, slots=True)
class AllowedRoots:
    """The folders file tools may read, each resolved to its real location.

    A relative path is taken from the first folder.
    """

    roots: tuple[Path, ...]

    @classmethod
    def of(cls, folders: Iterable[Path]) -> AllowedRoots:
        """Return the allowed roots for ``folders``."""
        return cls(tuple(Path(f).expanduser().resolve() for f in folders))

    def resolve(self, path: str) -> Path:
        """Return the real path ``path`` names, if it is inside an allowed folder.

        Raises:
            ToolError: If no folder is allowed, or the path leads outside them.
        """
        if not self.roots:
            message = (
                "no folders are allowed for reading; "
                "SYNTHIA_FILE_ROOTS names the ones that are"
            )
            raise ToolError(message)
        given = Path(path).expanduser()
        real = (given if given.is_absolute() else self.roots[0] / given).resolve()
        if not any(real.is_relative_to(root) for root in self.roots):
            message = f"{path} is outside the allowed folders"
            raise ToolError(message)
        return real


def file_tools(roots: AllowedRoots) -> tuple[FunctionTool, FunctionTool]:
    """Return ``read_file`` and ``list_files``, confined to ``roots``."""

    def read_file(path: str) -> str:
        """Return the text of a file inside the allowed folders."""
        real = roots.resolve(path)
        if not real.is_file():
            message = f"{path} is not a file"
            raise ToolError(message)
        with real.open("rb") as handle:
            if b"\0" in handle.read(SNIFF_BYTES):
                message = f"{path} is a binary file, not text"
                raise ToolError(message)
        text = real.read_text(encoding="utf-8", errors="replace")
        if len(text) <= MAX_READ_CHARS:
            return text
        return (
            f"{text[:MAX_READ_CHARS]}\n"
            f"[the first {MAX_READ_CHARS} of {len(text)} characters]"
        )

    def list_files(
        directory: str,
        limit: Annotated[
            int, Field(ge=1, le=MAX_LIST_LIMIT, description="most entries to return")
        ] = DEFAULT_LIST_LIMIT,
    ) -> str:
        """List a folder inside the allowed folders; subfolders end in /."""
        real = roots.resolve(directory)
        if not real.is_dir():
            message = f"{directory} is not a folder"
            raise ToolError(message)
        entries = sorted(f"{p.name}/" if p.is_dir() else p.name for p in real.iterdir())
        shown = "\n".join(entries[:limit]) or "(empty)"
        if len(entries) > limit:
            shown += f"\n[{limit} of {len(entries)} entries]"
        return shown

    return (
        FunctionTool.of(read_file, reach=Reach.LOCAL, effect=Effect.READ),
        FunctionTool.of(list_files, reach=Reach.LOCAL, effect=Effect.READ),
    )
