"""Choose how much a model thinks, from what the request asks.

``auto`` becomes a level by fixed, tested rules on the latest user message, so
the same question always gets the same depth and costs no extra request. Each
sign below adds points once, however often it appears:

- code (a fenced block, a line shaped like code, a traceback): 2;
- mathematics words (solve, integral, equation, prime, ...): 2, or else
  arithmetic between numbers, such as ``17*23``: 1;
- reasoning words (why, how does, compare, explain, trade-off, ...): 2;
- proof or depth words (prove, derive, step by step, in depth): 3;
- two or more questions, or a numbered or bulleted list of them: 1;
- more than 80 words: 1.

0 points is ``off`` for 8 words or fewer (a greeting, a quick lookup) and
``low`` above that; 1 is ``low``, 2 is ``medium``, 3 or more is ``high``.
Length alone never goes past ``low``, so a long pasted log with a trivial
question does not buy a long think. The word lists are English only.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING, Final

from synthia.gateway.types import Reasoning, Role

if TYPE_CHECKING:
    from synthia.gateway.types import ChatRequest

SHORT_WORDS: Final = 8
LONG_WORDS: Final = 80
MEDIUM_POINTS: Final = 2
HIGH_POINTS: Final = 3
SEVERAL_QUESTIONS: Final = 2

_FLAGS: Final = re.IGNORECASE | re.MULTILINE
_CODE: Final = re.compile(
    r"```|Traceback \(most recent call last\)"
    r"|^\s*(?:def\s+\w+\s*\(|class\s+\w+\s*[:(]|import\s+[\w.]+\s*$"
    r"|from\s+[\w.]+\s+import\s|function\s+\w+\s*\(|(?:const|let)\s+\w+\s*="
    r"|fn\s+\w+\s*\(|#include\s*<)"
    r"|[;{]\s*$",
    _FLAGS,
)
_MATHS: Final = re.compile(
    r"\b(?:solve|integra(?:l|te)|derivative|differentiate|equations?|probability"
    r"|matrix|eigen\w*|factori[sz]e|prime(?!\s+minister)|logarithm)\b",
    _FLAGS,
)
_REASONING: Final = re.compile(
    r"\b(?:why|how\s+(?:do|does|did|can|could|would|should|to|come)"
    r"|compare|comparison|difference\s+between|versus|vs\.?|explain"
    r"|trade-?offs?|pros\s+and\s+cons|debug|optimi[sz]e|analy[sz]e|design)\b",
    _FLAGS,
)
_DEPTH: Final = re.compile(
    r"\b(?:prove|proof|derive|step\s+by\s+step|in\s+depth|thoroughly)\b", _FLAGS
)
_ARITHMETIC: Final = re.compile(
    r"\d\s*[+*/^=\N{MULTIPLICATION SIGN}\N{DIVISION SIGN}]\s*\d"
)
_LIST_ITEM: Final = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+\S", re.MULTILINE)

_SIGNS: Final = ((_CODE, 2), (_MATHS, 2), (_REASONING, 2), (_DEPTH, 3))


def points(text: str) -> int:
    """Return how much thinking ``text`` calls for, by the rules above."""
    score = sum(weight for pattern, weight in _SIGNS if pattern.search(text))
    if not _MATHS.search(text) and _ARITHMETIC.search(text):
        score += 1
    questions = text.count("?")
    if questions >= SEVERAL_QUESTIONS or (
        questions and len(_LIST_ITEM.findall(text)) >= SEVERAL_QUESTIONS
    ):
        score += 1
    if len(text.split()) > LONG_WORDS:
        score += 1
    return score


def choose(request: ChatRequest) -> Reasoning:
    """Return the level for ``request``'s latest user message; never ``AUTO``."""
    text = next((m.text for m in reversed(request.messages) if m.role is Role.USER), "")
    score = points(text)
    if score >= HIGH_POINTS:
        return Reasoning.HIGH
    if score >= MEDIUM_POINTS:
        return Reasoning.MEDIUM
    if score == 0 and len(text.split()) <= SHORT_WORDS:
        return Reasoning.OFF
    return Reasoning.LOW


def resolve(request: ChatRequest) -> ChatRequest:
    """Return ``request`` with ``AUTO`` replaced by the level chosen for it."""
    if request.reasoning is not Reasoning.AUTO:
        return request
    return replace(request, reasoning=choose(request))
