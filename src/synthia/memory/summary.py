"""The rolling summary of the turns that no longer fit in the context window.

The summary is written on this machine: the request never allows the remote,
so a router sends it to the local model or nowhere. The old summary and the
turns leaving the window go in, and a new summary comes out, no longer than
:data:`SUMMARY_TOKENS`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from synthia.gateway.protocol import collect
from synthia.gateway.types import ChatRequest, Message, Reasoning, Role

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from synthia.gateway.protocol import ChatModel

type Summarizer = Callable[[str, Sequence[Message]], Awaitable[str]]
"""Given the summary so far and the turns leaving the window, the new summary."""

SUMMARY_TOKENS: Final = 1500
MAX_LINE_CHARS: Final = 1000
INSTRUCTIONS: Final = (
    "You keep a short running summary of a conversation between a person and "
    "SYNTHIA, their assistant. Merge the summary so far with the new part of "
    "the conversation. Keep names, facts, decisions, promises and open "
    "questions; drop small talk. Write plain sentences, at most 300 words, "
    "and nothing but the summary."
)
_SPEAKERS: Final = {Role.USER: "person", Role.ASSISTANT: "SYNTHIA", Role.TOOL: "tool"}


def transcript(messages: Sequence[Message]) -> str:
    """Return ``messages`` as lines of who said what, each cut to a length."""
    lines: list[str] = []
    for message in messages:
        speaker = _SPEAKERS.get(message.role)
        text = " ".join(message.text.split())
        if speaker is None or not text:
            continue
        lines.append(f"{speaker}: {text[:MAX_LINE_CHARS]}")
    return "\n".join(lines)


async def summarize(
    model: ChatModel, previous: str, messages: Sequence[Message]
) -> str:
    """Return the summary of ``previous`` and ``messages``, written by ``model``.

    Raises:
        GatewayError: If no model on this machine could write it.
    """
    so_far = previous or "(nothing yet)"
    request = ChatRequest(
        (
            Message.system(INSTRUCTIONS),
            Message.user(
                f"Summary so far:\n{so_far}\n\nNew part:\n{transcript(messages)}"
            ),
        ),
        max_tokens=SUMMARY_TOKENS,
        reasoning=Reasoning.OFF,
        use_remote=False,
    )
    response = await collect(model.stream(request))
    return response.text.strip()
