"""SQLite FTS5 full-text indexes: how they split text, and safe queries for them.

Every index uses the same tokenizer, so a word is found the same way wherever it
is kept, and :mod:`synthia.retrieval.bm25` scores exactly as these indexes rank.
"""

from __future__ import annotations

import re
from typing import Final

FTS_TOKENIZER: Final = "porter unicode61"
_WORD: Final = re.compile(r"\w+")


def match_expression(query: str) -> str:
    """Return ``query`` as an FTS5 expression matching any of its words.

    Each word is quoted, so nothing a person types is read as FTS5 syntax.
    """
    return " OR ".join(f'"{word}"' for word in _WORD.findall(query))


def match_all_expression(query: str) -> str:
    """Return ``query`` as an FTS5 expression matching only text with all its words."""
    return " AND ".join(f'"{word}"' for word in _WORD.findall(query))
