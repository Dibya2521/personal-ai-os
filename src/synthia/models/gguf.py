"""Read what a GGUF model file says about itself, with the standard library only.

A GGUF file opens with a header of typed key and value pairs before the
tensors. The context a model was trained for is the key
``<architecture>.context_length``, where the architecture is the value of
``general.architecture``; for Qwen3.5-4B that is ``qwen35.context_length``,
262,144 tokens. Only the header is walked, skipping what is not needed, so a
3 GB file costs a few small reads. Anything unexpected gives ``None`` instead
of an error, because the caller has a safe default.

Format: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import logging
import os
import struct
from typing import TYPE_CHECKING, BinaryIO, Final

if TYPE_CHECKING:
    from pathlib import Path

MAGIC: Final = b"GGUF"
ARCHITECTURE_KEY: Final = "general.architecture"
CONTEXT_SUFFIX: Final = ".context_length"
# Version 1 used 32-bit counts and lengths; files in use today are 2 or 3.
SUPPORTED_VERSIONS: Final = frozenset({2, 3})
# Keys and the architecture name are short; a longer one means a corrupt file,
# and reading it would allocate whatever the length field claims.
MAX_TEXT_BYTES: Final = 1 << 16

_STRING: Final = 8
_ARRAY: Final = 9
_INTEGERS: Final = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    10: "<Q",
    11: "<q",
}
_OTHER_SCALARS: Final = {6: "<f", 7: "<?", 12: "<d"}
_SCALARS: Final = _INTEGERS | _OTHER_SCALARS
_PAST_THE_END: Final = "the header runs past the end of the file"

logger = logging.getLogger(__name__)


class _MalformedError(Exception):
    pass


def context_length(path: Path) -> int | None:
    """Return the context the model in ``path`` was trained for, if it says."""
    try:
        with path.open("rb") as file:
            return _Header(file).context_length()
    except (OSError, _MalformedError, UnicodeDecodeError):
        logger.warning("could not read the GGUF header of %s", path.name)
        return None


class _Header:
    def __init__(self, file: BinaryIO) -> None:
        self._file = file
        self._size = file.seek(0, os.SEEK_END)
        file.seek(0)

    def context_length(self) -> int | None:
        if self._read(len(MAGIC)) != MAGIC:
            return None
        if self._integer("<I") not in SUPPORTED_VERSIONS:
            return None
        self._integer("<Q")  # tensor count, not needed
        architecture: str | None = None
        for _ in range(self._integer("<Q")):
            key = self._text()
            kind = self._integer("<I")
            if key == ARCHITECTURE_KEY and kind == _STRING:
                architecture = self._text()
            elif architecture and key == architecture + CONTEXT_SUFFIX:
                if kind not in _INTEGERS:
                    return None
                value = self._integer(_INTEGERS[kind])
                return value if value > 0 else None
            else:
                self._skip(kind)
        return None

    def _skip(self, kind: int) -> None:
        if kind in _SCALARS:
            self._skip_bytes(struct.calcsize(_SCALARS[kind]))
        elif kind == _STRING:
            self._skip_bytes(self._integer("<Q"))
        elif kind == _ARRAY:
            item = self._integer("<I")
            count = self._integer("<Q")
            if item in _SCALARS:
                self._skip_bytes(struct.calcsize(_SCALARS[item]) * count)
            else:
                for _ in range(count):
                    self._skip(item)
        else:
            message = f"unknown GGUF value type {kind}"
            raise _MalformedError(message)

    def _text(self) -> str:
        length = self._integer("<Q")
        if length > MAX_TEXT_BYTES:
            message = f"a GGUF string of {length} bytes"
            raise _MalformedError(message)
        return self._read(length).decode("utf-8")

    def _integer(self, layout: str) -> int:
        return int(struct.unpack(layout, self._read(struct.calcsize(layout)))[0])

    def _skip_bytes(self, count: int) -> None:
        if self._file.tell() + count > self._size:
            raise _MalformedError(_PAST_THE_END)
        self._file.seek(count, os.SEEK_CUR)

    def _read(self, count: int) -> bytes:
        data = self._file.read(count)
        if len(data) != count:
            raise _MalformedError(_PAST_THE_END)
        return data
