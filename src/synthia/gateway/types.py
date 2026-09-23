"""The vocabulary every model provider is translated to and from.

These types are provider-neutral on purpose: an adapter maps them onto one
provider's wire format, and nothing above the gateway ever sees that format.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

IMAGE_MEDIA_TYPES = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})
MAX_TEMPERATURE = 2.0


class Role(StrEnum):
    """Who wrote a message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class TextPart:
    """Plain text inside a message."""

    text: str


@dataclass(frozen=True, slots=True)
class ImagePart:
    """An image inside a message, carried as bytes.

    Raises:
        ValueError: If the media type is not one vision models accept, or the
            image is empty.
    """

    data: bytes = field(repr=False)
    media_type: str

    def __post_init__(self) -> None:
        if self.media_type not in IMAGE_MEDIA_TYPES:
            message = f"unsupported image type {self.media_type!r}"
            raise ValueError(message)
        if not self.data:
            message = "an image part needs data"
            raise ValueError(message)

    def data_url(self) -> str:
        """Return the image as a ``data:`` URL, the form chat APIs accept inline."""
        encoded = base64.b64encode(self.data).decode("ascii")
        return f"data:{self.media_type};base64,{encoded}"


type Part = TextPart | ImagePart


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A request from the model to run a tool.

    ``arguments`` is the raw JSON text the model produced. It is parsed and
    validated by whoever runs the tool, because a model can emit invalid JSON.
    """

    id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class Message:
    """One turn in a conversation.

    Raises:
        ValueError: If the fields do not fit the role: a tool result without the
            id of the call it answers, tool calls outside an assistant message,
            or an image outside a user message.
    """

    role: Role
    parts: tuple[Part, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if (self.role is Role.TOOL) != (self.tool_call_id is not None):
            message = "tool_call_id is required on, and only on, a tool message"
            raise ValueError(message)
        if self.tool_calls and self.role is not Role.ASSISTANT:
            message = "only an assistant message can carry tool calls"
            raise ValueError(message)
        if self.role is not Role.USER and any(
            isinstance(p, ImagePart) for p in self.parts
        ):
            message = "only a user message can carry images"
            raise ValueError(message)

    @property
    def text(self) -> str:
        """Return the text parts joined, ignoring images."""
        return "".join(p.text for p in self.parts if isinstance(p, TextPart))

    @classmethod
    def system(cls, text: str) -> Message:
        """Build a system message."""
        return cls(Role.SYSTEM, (TextPart(text),))

    @classmethod
    def user(cls, text: str, *images: ImagePart) -> Message:
        """Build a user message, optionally with images after the text."""
        return cls(Role.USER, (TextPart(text), *images))

    @classmethod
    def assistant(cls, text: str, *tool_calls: ToolCall) -> Message:
        """Build an assistant message, optionally with the tool calls it made."""
        parts = (TextPart(text),) if text else ()
        return cls(Role.ASSISTANT, parts, tool_calls)

    @classmethod
    def tool_result(cls, tool_call_id: str, content: str) -> Message:
        """Build the message that answers one tool call."""
        return cls(Role.TOOL, (TextPart(content),), tool_call_id=tool_call_id)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool the model may call. ``parameters`` is a JSON Schema object."""

    name: str
    description: str
    parameters: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """Everything a model needs for one completion.

    ``model`` overrides the provider's default model for this request only.
    ``response_schema`` asks for JSON matching that schema.

    Raises:
        ValueError: If there are no messages, or a sampling setting is out of
            range.
    """

    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...] = ()
    temperature: float | None = None
    max_tokens: int | None = None
    response_schema: Mapping[str, object] | None = None
    model: str | None = None

    def __post_init__(self) -> None:
        if not self.messages:
            message = "a request needs at least one message"
            raise ValueError(message)
        if (
            self.temperature is not None
            and not 0 <= self.temperature <= MAX_TEMPERATURE
        ):
            message = f"temperature must be between 0 and {MAX_TEMPERATURE:g}"
            raise ValueError(message)
        if self.max_tokens is not None and self.max_tokens <= 0:
            message = "max_tokens must be positive"
            raise ValueError(message)

    @property
    def has_images(self) -> bool:
        """Return whether any message carries an image."""
        return any(isinstance(p, ImagePart) for m in self.messages for p in m.parts)


class FinishReason(StrEnum):
    """Why the model stopped. ``OTHER`` covers any value a provider adds later."""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    OTHER = "other"

    @classmethod
    def parse(cls, value: str) -> FinishReason:
        """Map a provider's finish reason onto this enum, never failing."""
        try:
            return cls(value)
        except ValueError:
            return cls.OTHER


@dataclass(frozen=True, slots=True)
class Usage:
    """Tokens counted by the provider for one completion."""

    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        """Return prompt and completion tokens together."""
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    """A fragment of a tool call as it streams.

    Fragments of one call share an ``index``; ``id`` and ``name`` usually arrive
    once, in the first fragment, and ``arguments`` arrives in pieces.
    """

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str = ""


@dataclass(frozen=True, slots=True)
class ChatChunk:
    """One piece of a streamed completion.

    ``model`` is the model that actually answered, which can differ from the one
    requested when a router such as ``openrouter/free`` picks it. ``reasoning``
    is a thinking model's working, kept apart from ``text`` so it is never spoken
    or shown as the answer.
    """

    text: str = ""
    reasoning: str = ""
    tool_calls: tuple[ToolCallDelta, ...] = ()
    finish_reason: FinishReason | None = None
    usage: Usage | None = None
    model: str | None = None


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """A completion collected from its stream."""

    text: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: FinishReason
    usage: Usage | None
    model: str | None
    reasoning: str = ""

    def as_message(self) -> Message:
        """Return the completion as an assistant message for the history."""
        return Message.assistant(self.text, *self.tool_calls)


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """What a model can do, used to decide where a request may go."""

    id: str
    context_window: int
    vision: bool
    tools: bool
