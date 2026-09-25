"""A :class:`~synthia.gateway.protocol.ChatModel` for OpenAI-compatible chat APIs.

OpenRouter and the llama.cpp server both speak this protocol, so one adapter
serves both: a provider is a base URL, headers and a capability record. The
translation in each direction is a pure function (:func:`build_payload`,
:func:`parse_chunk`, :func:`error_for`) so it is tested without a network.
"""

from __future__ import annotations

import json
import math
import time
from contextlib import aclosing
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast

import httpx

from synthia.gateway.errors import (
    AuthError,
    BadRequestError,
    ConnectionFailedError,
    GatewayError,
    IncompleteResponseError,
    MalformedStreamError,
    PaymentRequiredError,
    ProviderError,
    RateLimitedError,
)
from synthia.gateway.sse import aiter_events
from synthia.gateway.types import (
    ChatChunk,
    FinishReason,
    ImagePart,
    Reasoning,
    Role,
    TextPart,
    ToolCallDelta,
    Usage,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Mapping

    from pydantic import SecretStr

    from synthia.gateway.types import ChatRequest, Message, ModelInfo

DONE = "[DONE]"
EVENT_STREAM = "text/event-stream"
MAX_ERROR_DETAIL = 300
SCHEMA_NAME = "response"
REDACTED = "**********"
# Thinking models stream their working under one of these, depending on the server.
REASONING_KEYS = ("reasoning_content", "reasoning")

type JSON = dict[str, Any]


class Dialect(StrEnum):
    """How a server takes what the OpenAI chat API has no field for, like thinking.

    OpenRouter takes ``reasoning.effort``, where ``none`` turns thinking off.
    llama.cpp's server passes ``chat_template_kwargs`` to the model's chat
    template, and ``enable_thinking`` is the one switch Qwen3.5's template has.
    """

    OPENROUTER = "openrouter"
    LLAMA_CPP = "llama.cpp"


def _reasoning(level: Reasoning | None, dialect: Dialect) -> JSON:
    if level is None or level is Reasoning.AUTO:
        return {}
    if dialect is Dialect.LLAMA_CPP:
        thinking = level is not Reasoning.OFF
        return {"chat_template_kwargs": {"enable_thinking": thinking}}
    effort = "none" if level is Reasoning.OFF else level.value
    return {"reasoning": {"effort": effort}}


def _content(message: Message) -> str | list[JSON]:
    """Return plain text when there is no image, which every server accepts."""
    if not any(isinstance(p, ImagePart) for p in message.parts):
        return message.text
    return [
        {"type": "text", "text": p.text}
        if isinstance(p, TextPart)
        else {"type": "image_url", "image_url": {"url": p.data_url()}}
        for p in message.parts
    ]


def _message(message: Message) -> JSON:
    wire: JSON = {"role": message.role.value}
    if message.role is Role.TOOL:
        return wire | {"tool_call_id": message.tool_call_id, "content": message.text}
    if message.tool_calls:
        wire["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in message.tool_calls
        ]
        wire["content"] = message.text or None
        return wire
    return wire | {"content": _content(message)}


def build_payload(
    request: ChatRequest, default_model: str, dialect: Dialect = Dialect.OPENROUTER
) -> JSON:
    """Return the JSON body of a streaming chat completion request."""
    payload: JSON = {
        "model": request.model or default_model,
        "messages": [_message(m) for m in request.messages],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if request.tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": dict(t.parameters),
                },
            }
            for t in request.tools
        ]
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.max_tokens is not None:
        payload["max_tokens"] = request.max_tokens
    if request.response_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": SCHEMA_NAME,
                "schema": dict(request.response_schema),
                "strict": True,
            },
        }
    return payload | _reasoning(request.reasoning, dialect)


def parse_chunk(data: JSON) -> ChatChunk:
    """Translate one decoded ``chat.completion.chunk`` into a :class:`ChatChunk`.

    Raises:
        ProviderError: If the chunk reports an error, which OpenRouter does
            mid-stream when the upstream model fails after the headers were sent.
        MalformedStreamError: If the chunk does not have the expected shape.
    """
    if "error" in data:
        raise ProviderError(_error_detail(data) or "the provider failed mid-stream")
    try:
        usage_data = data.get("usage")
        usage = (
            Usage(
                int(usage_data["prompt_tokens"]), int(usage_data["completion_tokens"])
            )
            if usage_data
            else None
        )
        choices = cast("list[JSON]", data.get("choices") or [{}])
        choice = choices[0]
        delta = cast("JSON", choice.get("delta") or {})
        reason = choice.get("finish_reason")
        return ChatChunk(
            text=delta.get("content") or "",
            reasoning=next((delta[k] for k in REASONING_KEYS if delta.get(k)), ""),
            tool_calls=tuple(
                _tool_delta(t)
                for t in cast("list[JSON]", delta.get("tool_calls") or [])
            ),
            finish_reason=FinishReason.parse(reason) if reason else None,
            usage=usage,
            model=data.get("model"),
        )
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as error:
        message = f"unexpected chunk shape: {type(error).__name__}"
        raise MalformedStreamError(message) from error


def _tool_delta(data: JSON) -> ToolCallDelta:
    function = cast("JSON", data.get("function") or {})
    return ToolCallDelta(
        index=int(data["index"]),
        id=data.get("id"),
        name=function.get("name"),
        arguments=function.get("arguments") or "",
    )


def _error_detail(body: object) -> str | None:
    if isinstance(body, dict):
        error = cast("JSON", body).get("error")
        if isinstance(error, dict):
            message = cast("JSON", error).get("message")
            if isinstance(message, str):
                return message[:MAX_ERROR_DETAIL]
        if isinstance(error, str):
            return error[:MAX_ERROR_DETAIL]
    return None


def retry_after_seconds(value: str | None, now: float | None = None) -> float | None:
    """Parse ``Retry-After``, given as seconds or as an HTTP date."""
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = parsedate_to_datetime(value).timestamp()
        except (TypeError, ValueError, IndexError):
            return None
        seconds = when - (now if now is not None else time.time())
    if not math.isfinite(seconds):
        return None
    return max(seconds, 0.0)


def error_for(status: int, body: bytes, headers: Mapping[str, str]) -> GatewayError:
    """Return the error for an HTTP failure. The body may be JSON, HTML or empty."""
    try:
        detail = _error_detail(json.loads(body))
    except (ValueError, UnicodeDecodeError):
        detail = None
    message = f"HTTP {status}" + (f": {detail}" if detail else "")
    if status in {401, 403}:
        return AuthError(message)
    if status == 402:  # noqa: PLR2004 - HTTP status codes read best as numbers
        return PaymentRequiredError(message)
    if status == 429:  # noqa: PLR2004
        return RateLimitedError(
            message, retry_after_seconds(headers.get("retry-after"))
        )
    if status == 408 or status >= 500:  # noqa: PLR2004
        return ProviderError(message)
    return BadRequestError(message)


@dataclass(frozen=True, slots=True)
class Endpoint:
    """Where a model is served and how to address it.

    ``model`` is the default model id sent when a request names none.
    ``dialect`` defaults to OpenRouter's, the only remote provider.
    """

    base_url: str
    model: str
    api_key: SecretStr | None = None
    headers: Mapping[str, str] = field(default_factory=dict[str, str])
    dialect: Dialect = Dialect.OPENROUTER

    @property
    def completions_url(self) -> str:
        """Return the chat completions URL under ``base_url``."""
        return self.base_url.rstrip("/") + "/chat/completions"


class OpenAICompatibleModel:
    """Stream completions from any server that speaks the OpenAI chat API."""

    def __init__(
        self, *, client: httpx.AsyncClient, endpoint: Endpoint, info: ModelInfo
    ) -> None:
        self._client = client
        self._url = endpoint.completions_url
        self._model = endpoint.model
        self._info = info
        self._api_key = endpoint.api_key
        self._headers = dict(endpoint.headers)
        self._dialect = endpoint.dialect

    @property
    def info(self) -> ModelInfo:
        """Return what this model can do."""
        return self._info

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        """Yield the completion for ``request`` as the server generates it.

        Raises:
            GatewayError: A subclass describing why the request failed; its
                ``retryable`` says whether sending it again may succeed.
        """
        headers = dict(self._headers)
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key.get_secret_value()}"
        payload = build_payload(request, self._model, self._dialect)
        try:
            async with self._client.stream(
                "POST", self._url, json=payload, headers=headers
            ) as response:
                if response.status_code >= httpx.codes.BAD_REQUEST:
                    body = await response.aread()
                    raise self._scrub(
                        error_for(response.status_code, body, response.headers)
                    )
                _expect_event_stream(response)
                async with aclosing(self._chunks(response)) as chunks:
                    async for chunk in chunks:
                        yield chunk
        except httpx.TransportError as error:
            message = f"no response from {self._url}: {type(error).__name__}"
            raise ConnectionFailedError(message) from error

    async def _chunks(self, response: httpx.Response) -> AsyncGenerator[ChatChunk]:
        finished = False
        events = aiter_events(response.aiter_bytes())
        async with aclosing(events):
            async for event in events:
                if event.data == DONE:
                    return
                try:
                    data = json.loads(event.data)
                except ValueError as error:
                    message = "a stream event was not JSON"
                    raise MalformedStreamError(message) from error
                if not isinstance(data, dict):
                    message = "a stream event was not a JSON object"
                    raise MalformedStreamError(message)
                chunk = parse_chunk(cast("JSON", data))
                finished = finished or chunk.finish_reason is not None
                yield chunk
        if not finished:
            message = "the stream ended without a finish reason or [DONE]"
            raise IncompleteResponseError(message)

    def _scrub(self, error: GatewayError) -> GatewayError:
        return scrub(error, self._api_key)


def _expect_event_stream(response: httpx.Response) -> None:
    # A captive portal or a proxy answers 200 with its own page; say so, rather
    # than reporting an answer that never finished.
    content_type = response.headers.get("content-type", "").partition(";")[0].strip()
    if content_type and content_type != EVENT_STREAM:
        message = f"expected an event stream, got {content_type}"
        raise MalformedStreamError(message)


def scrub(error: GatewayError, api_key: SecretStr | None) -> GatewayError:
    """Remove the API key from a provider's message; some echo it back."""
    if api_key is None:
        return error
    secret = api_key.get_secret_value()
    if secret and secret in str(error):
        error.args = (str(error).replace(secret, REDACTED),)
    return error
