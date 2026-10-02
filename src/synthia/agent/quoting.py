"""Tool output goes back to the model as quoted data, never as instructions.

A file or a web page can hold text written to steer the model ("ignore your
instructions"). Quoting marks where such text begins and ends, and defuses
the markup that could end the quote or open a turn of its own. It lowers how
often a model follows injected text; the permission policy, which no text can
change, is what guarantees a refused tool still never runs.
"""

from __future__ import annotations

import html
import re
from typing import Final

CLOSE: Final = "</tool_result>"
_CLOSING: Final = re.compile(r"</(\s*tool_result)", re.IGNORECASE)
# Chat templates mark turns with <|...|> tokens; text spelled the same way
# could be read as one and start a turn of its own.
_CONTROL: Final = re.compile(r"<\|")
_FLAGS: Final = (
    (
        "asks to ignore instructions",
        re.compile(
            r"\b(ignore|disregard|forget)\b.{0,40}\b(instructions|rules|prompt)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "tries to change the role",
        re.compile(r"\byou are now\b|\bnew instructions\b", re.IGNORECASE),
    ),
    (
        "speaks as a role",
        re.compile(
            r"^\s*(system|assistant|developer)\s*:", re.IGNORECASE | re.MULTILINE
        ),
    ),
    ("holds a chat control token", re.compile(r"<\|[\w-]+\|>")),
)


def quote(name: str, result: str) -> str:
    """Return ``result`` wrapped as the untrusted output of the tool ``name``."""
    safe = _CONTROL.sub("< |", _CLOSING.sub(r"<\\/\1", result))
    return f'<tool_result name="{html.escape(name)}" trusted="false">\n{safe}\n{CLOSE}'


def flags(result: str) -> tuple[str, ...]:
    """Return what in ``result`` is shaped like an instruction to the model."""
    return tuple(label for label, pattern in _FLAGS if pattern.search(result))
