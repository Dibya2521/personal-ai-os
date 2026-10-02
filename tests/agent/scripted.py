"""A model that answers from a script, for agent tests."""

from collections.abc import AsyncGenerator, Callable

from synthia.gateway.types import (
    ChatChunk,
    ChatRequest,
    FinishReason,
    ModelInfo,
    ToolCallDelta,
)

INFO = ModelInfo("scripted", 4096, vision=False, tools=True)

type Reply = list[ChatChunk] | Callable[[ChatRequest], list[ChatChunk]]


def says(text: str) -> list[ChatChunk]:
    return [ChatChunk(text=text), ChatChunk(finish_reason=FinishReason.STOP)]


def calls(*wanted: tuple[str, str]) -> list[ChatChunk]:
    return [
        ChatChunk(
            tool_calls=tuple(
                ToolCallDelta(i, f"call_{i}", name, arguments)
                for i, (name, arguments) in enumerate(wanted)
            )
        ),
        ChatChunk(finish_reason=FinishReason.TOOL_CALLS),
    ]


class Scripted:
    """Answers each request with the next reply, and keeps every request."""

    def __init__(self, *replies: Reply) -> None:
        self._replies = list(replies)
        self.requests: list[ChatRequest] = []

    @property
    def info(self) -> ModelInfo:
        return INFO

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        self.requests.append(request)
        reply = self._replies.pop(0)
        for chunk in reply(request) if callable(reply) else reply:
            yield chunk
