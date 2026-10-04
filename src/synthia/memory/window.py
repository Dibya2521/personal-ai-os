"""What fits in the model's context window, and which old turns must leave it.

A request is counted in characters and turned into tokens at a rate learned
from the model itself: each answer reports how many tokens its prompt took,
and the rate follows what is reported. Until the first report the rate is
deliberately low, so the first estimate errs towards too many tokens.

A quarter of the window is kept free for the turn's own growth: tool results
read during the turn, the thinking and the answer. Old turns leave whole,
oldest first, so the model never sees a tool call without its result.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Final

from synthia.gateway.types import ImagePart, TextPart

if TYPE_CHECKING:
    from collections.abc import Sequence

    from synthia.gateway.types import Message, ToolSpec

# English runs near four characters a token for Qwen-style tokenizers; three
# overcounts on purpose until the model has reported a real figure.
START_CHARS_PER_TOKEN: Final = 3.0
MIN_CHARS_PER_TOKEN: Final = 2.0
MAX_CHARS_PER_TOKEN: Final = 6.0
LEARNING_WEIGHT: Final = 0.3
# Role markers and separators the chat template adds around each message.
MESSAGE_TOKENS: Final = 8
# Not measured: a vision model's cost per image depends on its size; this
# stands for a large one.
IMAGE_TOKENS: Final = 1500
ANSWER_SHARE: Final = 4


def message_chars(message: Message) -> int:
    """Return the characters of ``message`` the model reads, images aside."""
    text = sum(len(p.text) for p in message.parts if isinstance(p, TextPart))
    calls = sum(len(c.name) + len(c.arguments) for c in message.tool_calls)
    return text + calls


def images(messages: Sequence[Message]) -> int:
    """Return how many images ``messages`` carry."""
    return sum(isinstance(p, ImagePart) for m in messages for p in m.parts)


def tools_chars(tools: Sequence[ToolSpec]) -> int:
    """Return the characters the tool definitions take in a request."""
    return sum(
        len(t.name) + len(t.description) + len(json.dumps(t.parameters)) for t in tools
    )


class ContextWindow:
    """A model's context window, and a token count learned from the model."""

    def __init__(
        self, limit: int, chars_per_token: float = START_CHARS_PER_TOKEN
    ) -> None:
        self.limit = limit
        self.chars_per_token = chars_per_token

    @property
    def budget(self) -> int:
        """Return the tokens a request may take before the turn begins."""
        return self.limit - self.limit // ANSWER_SHARE

    def tokens(self, messages: Sequence[Message], extra_chars: int = 0) -> int:
        """Return the estimated tokens of ``messages`` plus ``extra_chars``."""
        chars = extra_chars + sum(message_chars(m) for m in messages)
        return (
            math.ceil(chars / self.chars_per_token)
            + MESSAGE_TOKENS * len(messages)
            + IMAGE_TOKENS * images(messages)
        )

    def overflow(
        self,
        fixed: Sequence[Message],
        turns: Sequence[Sequence[Message]],
        extra_chars: int = 0,
    ) -> int:
        """Return how many of the oldest ``turns`` must leave for the rest to fit.

        ``fixed`` (the system message and the new question) always stays; if
        it alone does not fit, every turn leaves and the model says so.
        """
        total = self.tokens(fixed, extra_chars) + sum(self.tokens(t) for t in turns)
        leaving = 0
        while leaving < len(turns) and total > self.budget:
            total -= self.tokens(turns[leaving])
            leaving += 1
        return leaving

    def learn(self, chars: int, prompt_tokens: int | None) -> None:
        """Move the rate towards what the model reported for ``chars`` characters."""
        if not prompt_tokens or chars <= 0:
            return
        observed = chars / prompt_tokens
        rate = (1 - LEARNING_WEIGHT) * self.chars_per_token + LEARNING_WEIGHT * observed
        self.chars_per_token = min(max(rate, MIN_CHARS_PER_TOKEN), MAX_CHARS_PER_TOKEN)
