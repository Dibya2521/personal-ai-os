"""The one interface every model sits behind, local or remote."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from synthia.gateway.errors import IncompleteResponseError
from synthia.gateway.types import (
    ChatResponse,
    FinishReason,
    ToolCall,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, AsyncIterator

    from synthia.gateway.types import ChatChunk, ChatRequest, ModelInfo, Usage


class ChatModel(Protocol):
    """A model that streams a completion for a request.

    Streaming is the primitive rather than a whole response, because speech has
    to start on the first sentence while the rest is still being generated. A
    whole response is a stream passed through :func:`collect`.
    """

    @property
    def info(self) -> ModelInfo:
        """Return what this model can do."""
        ...

    def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        """Yield the completion for ``request`` as it is generated."""
        ...


@dataclass(slots=True)
class _PendingCall:
    id: str | None = None
    name: str | None = None
    arguments: str = ""


async def collect(chunks: AsyncIterable[ChatChunk]) -> ChatResponse:
    """Join a stream of chunks into one response.

    Raises:
        IncompleteResponseError: If the stream ends without a finish reason, or
            a tool call is missing its id or name.
    """
    text: list[str] = []
    reasoning: list[str] = []
    calls: dict[int, _PendingCall] = {}
    finish: FinishReason | None = None
    usage: Usage | None = None
    model: str | None = None
    async for chunk in chunks:
        text.append(chunk.text)
        reasoning.append(chunk.reasoning)
        for delta in chunk.tool_calls:
            pending = calls.setdefault(delta.index, _PendingCall())
            pending.id = pending.id or delta.id
            pending.name = pending.name or delta.name
            pending.arguments += delta.arguments
        finish = chunk.finish_reason or finish
        usage = chunk.usage or usage
        model = model or chunk.model
    if finish is None:
        message = "the stream ended without a finish reason"
        raise IncompleteResponseError(message)
    return ChatResponse(
        text="".join(text),
        tool_calls=tuple(_complete(index, calls[index]) for index in sorted(calls)),
        finish_reason=finish,
        usage=usage,
        model=model,
        reasoning="".join(reasoning),
    )


def _complete(index: int, pending: _PendingCall) -> ToolCall:
    if pending.id is None or pending.name is None:
        message = f"tool call {index} ended without an id or a name"
        raise IncompleteResponseError(message)
    return ToolCall(pending.id, pending.name, pending.arguments)
