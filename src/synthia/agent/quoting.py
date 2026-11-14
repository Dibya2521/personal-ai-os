"""Tool output goes back to the model as quoted data, never as instructions.

A file or a web page can hold text written to steer the model ("ignore your
instructions"). Quoting marks where such text begins and ends, and defuses
the markup that could end the quote or open a turn of its own. It lowers how
often a model follows injected text; the permission policy, which no text can
change, is what guarantees a refused tool still never runs.

Characters a person cannot see but a model reads (zero-width spaces and
joiners, bidi controls, U+FEFF, variation selectors) can split a closing tag
or an instruction so a plain pattern misses it. They also carry meaning in
real text (Odia, Devanagari and Persian need ZWJ and ZWNJ, emoji need ZWJ and
U+FE0F), so they are kept, and the patterns look through them instead. Only
the Unicode tag block is removed: it renders as nothing and can spell a whole
hidden prompt.

Defusing catches the closing tags it knows, but Unicode has thousands of
characters that look like ``/`` or a Latin letter. So the opening tag also
names the quote's end mark, a hash of the quoted text, and only a closing tag
carrying that mark ends it. Text cannot carry the mark of a quote that holds
it: that is finding a fixed point of SHA-256 cut to 64 bits. The mark depends
on the text alone, so the same result is quoted the same way every time.
"""

from __future__ import annotations

import functools
import hashlib
import html
import re
from typing import Final

END_MARK_HEX: Final = 16
_TAG_BLOCK: Final = re.compile(r"[\U000e0000-\U000e007f]")
# Unicode's format characters (category Cf), the same 170 in Unicode 15.0 and
# 16.0, as inclusive runs. Listed, not scanned: walking all 1.1 million code
# points at the first quote held a daemon's event loop for 0.15 to 0.33 s.
FORMAT_CHARACTERS: Final = (
    (0x00AD, 0x00AD),
    (0x0600, 0x0605),
    (0x061C, 0x061C),
    (0x06DD, 0x06DD),
    (0x070F, 0x070F),
    (0x0890, 0x0891),
    (0x08E2, 0x08E2),
    (0x180E, 0x180E),
    (0x200B, 0x200F),
    (0x202A, 0x202E),
    (0x2060, 0x2064),
    (0x2066, 0x206F),
    (0xFEFF, 0xFEFF),
    (0xFFF9, 0xFFFB),
    (0x110BD, 0x110BD),
    (0x110CD, 0x110CD),
    (0x13430, 0x1343F),
    (0x1BCA0, 0x1BCA3),
    (0x1D173, 0x1D17A),
    (0xE0001, 0xE0001),
    (0xE0020, 0xE007F),
)
_VARIATION_SELECTORS: Final = ((0xFE00, 0xFE0F), (0xE0100, 0xE01EF))
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
        re.compile(r"\byou\s+are\s+now\b|\bnew\s+instructions\b", re.IGNORECASE),
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
    shown = _TAG_BLOCK.sub("", result)
    safe = _CONTROL.sub("< |", _closing().sub(r"<\\/\1", shown))
    end = _end_mark(safe)
    return (
        f'<tool_result name="{html.escape(name)}" trusted="false" end="{end}">\n'
        f'{safe}\n</tool_result end="{end}">'
    )


def _end_mark(text: str) -> str:
    """Return the mark that ends the quote of ``text``."""
    # A tool can return a lone surrogate (decoded JSON), which strict UTF-8 refuses.
    digest = hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()
    return digest[:END_MARK_HEX]


def flags(result: str) -> tuple[str, ...]:
    """Return what in ``result`` is shaped like an instruction to the model."""
    if result.isascii():
        views = (result,)
    else:
        hidden = _hidden()
        # Removed, a hidden character inside a word joins the word back;
        # spaced, one between two words keeps them apart.
        views = (hidden.sub("", result), hidden.sub(" ", result))
    return tuple(
        label
        for label, pattern in _FLAGS
        if any(pattern.search(view) for view in views)
    )


@functools.cache
def _hidden_class() -> str:
    # As ranges: the same class written as single characters made each pass
    # about 10 times slower.
    return "".join(
        f"{re.escape(chr(low))}-{re.escape(chr(high))}"
        for low, high in FORMAT_CHARACTERS + _VARIATION_SELECTORS
    )


@functools.cache
def _hidden() -> re.Pattern[str]:
    return re.compile(f"[{_hidden_class()}]+")


@functools.cache
def _closing() -> re.Pattern[str]:
    gap = f"[\\s{_hidden_class()}]*"
    name = f"[{_hidden_class()}]*".join("tool_result")
    return re.compile(f"<{gap}/({gap}{name})", re.IGNORECASE)
