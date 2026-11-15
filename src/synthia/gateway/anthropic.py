"""A :class:`~synthia.gateway.protocol.ChatModel` for Anthropic's Messages API.

Anthropic also serves an OpenAI-compatible endpoint, but it drops what this
adapter keeps: prompt caching, signed thinking blocks and the cache counts in
the usage. The request is ``POST <base_url>/v1/messages`` with the key in
``x-api-key`` and ``anthropic-version: 2026-10-21``; the answer streams as
named events (``message_start``, ``content_block_start``,
``content_block_delta``, ``content_block_stop``, ``message_delta``,
``message_stop``), each one JSON object. Errors map onto the same classes as
the OpenAI adapter's, so retry, budget and circuit treat both alike.
"""

from __future__ import annotations

import base64
import json
import logging
from contextlib import aclosing
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, cast

import httpx

from synthia.gateway.errors import (
    BadRequestError,
    ConnectionFailedError,
    GatewayError,
    IncompleteResponseError,
    MalformedStreamError,
)
from synthia.gateway.openai_compat import (
    PLAIN_BODY,
    Dialect,
    error_for,
    expect_event_stream,
    read_error_body,
    scrub,
)
from synthia.gateway.sse import aiter_events
from synthia.gateway.types import (
    ChatChunk,
    FinishReason,
    ImagePart,
    Reasoning,
    Replay,
    Role,
    TextPart,
    ToolCallDelta,
    Usage,
)

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        Callable,
        Iterable,
        Mapping,
        Sequence,
    )

    from synthia.gateway.openai_compat import Endpoint
    from synthia.gateway.types import (
        ChatRequest,
        Message,
        ModelInfo,
        ToolCall,
        ToolSpec,
    )

MESSAGES_PATH: Final = "/v1/messages"
API_VERSION: Final = "2026-10-21"
PROVIDER: Final = Dialect.ANTHROPIC.value
# The API requires a limit, and thinking counts against it; Anthropic's
# migration guide reports 64K working well for long agentic turns.
DEFAULT_MAX_TOKENS: Final = 64_000
# Thinking cannot be turned off on Claude Opus 5.5 (a 400 at every effort),
# so OFF asks for the least thinking there is. AUTO sends no effort at all.
EFFORT: Final = MappingProxyType(
    {
        Reasoning.OFF: "low",
        Reasoning.LOW: "low",
        Reasoning.MEDIUM: "medium",
        Reasoning.HIGH: "high",
    }
)
STOP_REASONS: Final = MappingProxyType(
    {
        "end_turn": FinishReason.STOP,
        "stop_sequence": FinishReason.STOP,
        "max_tokens": FinishReason.LENGTH,
        "tool_use": FinishReason.TOOL_CALLS,
        "refusal": FinishReason.CONTENT_FILTER,
    }
)
# The HTTP status each error type in Anthropic's error reference stands for,
# so an error event inside a 200 stream maps like the status would.
STREAM_ERROR_STATUSES: Final = MappingProxyType(
    {
        "invalid_request_error": 400,
        "authentication_error": 401,
        "billing_error": 402,
        "permission_error": 403,
        "not_found_error": 404,
        "request_too_large": 413,
        "rate_limit_error": 429,
        "api_error": 500,
        "overloaded_error": 529,
    }
)
UNKNOWN_STREAM_ERROR_STATUS: Final = 500
# Blocks a cache breakpoint may sit on, per Anthropic's prompt caching guide.
CACHEABLE_BLOCKS: Final = frozenset(
    {"text", "image", "tool_use", "tool_result", "document"}
)
# Blocks a Message holds in full; any other kind exists only in a Replay.
REBUILT_BLOCKS: Final = frozenset({"text", "tool_use"})
# The field each streamed delta kind adds to; any other kind is refused, so the
# replay holds nothing the answer did not.
DELTA_FIELDS: Final = MappingProxyType(
    {
        "text_delta": "text",
        "thinking_delta": "thinking",
        "signature_delta": "signature",
    }
)
# 64 000 tokens at about 4 characters each is 256 000; this leaves 32 times that
# for thinking, signatures and tool input before a stream counts as runaway.
MAX_ANSWER_CHARS: Final = 8 * 1024 * 1024
SHOWN_KIND_CHARS: Final = 40
EMPTY_INPUT: Final = "{}"

type JSON = dict[str, Any]

logger = logging.getLogger(__name__)


def _ephemeral() -> JSON:
    return {"type": "ephemeral"}


def _text(text: str) -> JSON:
    return {"type": "text", "text": text}


def _image(image: ImagePart) -> JSON:
    data = base64.b64encode(image.data).decode("ascii")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": image.media_type, "data": data},
    }


def _tool_use(call: ToolCall) -> JSON:
    """Return a tool call as a block; the API takes its input as an object.

    Raises:
        BadRequestError: If the call's arguments are not a JSON object, which
            only a call written by another model can be.
    """
    try:
        given: object = json.loads(call.arguments or EMPTY_INPUT)
    except ValueError:
        given = None
    if not isinstance(given, dict):
        message = (
            f"tool call {call.id!r} has arguments that are not a JSON object, "
            "which the Messages API cannot carry"
        )
        raise BadRequestError(message)
    return {"type": "tool_use", "id": call.id, "name": call.name, "input": given}


def _replayed(replay: Replay | None) -> list[JSON] | None:
    """Return the blocks this adapter recorded, or None for anyone else's."""
    if replay is None or replay.provider != PROVIDER:
        return None
    try:
        blocks: object = json.loads(replay.content)
    except ValueError:
        return None
    if not isinstance(blocks, list):
        return None
    listed = cast("list[object]", blocks)
    if not all(isinstance(b, dict) for b in listed):
        return None
    return cast("list[JSON]", listed)


def _blocks(message: Message) -> list[JSON]:
    if message.role is Role.TOOL:
        return [
            {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": message.text,
            }
        ]
    if message.role is Role.ASSISTANT:
        replayed = _replayed(message.replay)
        if replayed is not None:
            return replayed
        text = [_text(message.text)] if message.text else []
        return text + [_tool_use(c) for c in message.tool_calls]
    return [
        _text(p.text) if isinstance(p, TextPart) else _image(p)
        for p in message.parts
        if not isinstance(p, TextPart) or p.text
    ]


def _turns(messages: Iterable[Message]) -> list[JSON]:
    """Return the conversation as alternating user and assistant turns.

    A tool result is a block in a user turn, and the API takes roles strictly
    in turn, so the results of one round, and any text after them, share one
    user message.
    """
    turns: list[JSON] = []
    for message in messages:
        if message.role is Role.SYSTEM:
            continue
        role = Role.ASSISTANT.value if message.role is Role.ASSISTANT else "user"
        blocks = _blocks(message)
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"].extend(blocks)
        else:
            turns.append({"role": role, "content": blocks})
    return turns


def _tool(spec: ToolSpec) -> JSON:
    return {
        "name": spec.name,
        "description": spec.description,
        "input_schema": dict(spec.parameters),
    }


def _mark_last(blocks: Sequence[JSON]) -> None:
    if blocks and blocks[-1].get("type") in CACHEABLE_BLOCKS:
        blocks[-1]["cache_control"] = _ephemeral()


def _output_config(request: ChatRequest) -> JSON:
    config: JSON = {}
    effort = None if request.reasoning is None else EFFORT.get(request.reasoning)
    if effort is not None:
        config["effort"] = effort
    if request.response_schema is not None:
        config["format"] = {
            "type": "json_schema",
            "schema": dict(request.response_schema),
        }
    return config


def build_payload(request: ChatRequest, default_model: str) -> JSON:
    """Return the JSON body of a streaming Messages API request.

    System messages become the top-level ``system``. Two cache breakpoints
    are placed: after the tools and system prompt, which rarely change, and
    after the last message, so the next turn reads the whole conversation
    from the cache. ``temperature`` and ``reasoning_tokens`` are never sent:
    API version 2026-10-21 rejects the first on every model, and Claude Opus
    5.5 rejects a thinking budget.
    """
    system = [_text(m.text) for m in request.messages if m.role is Role.SYSTEM]
    tools = [_tool(t) for t in request.tools]
    if system:
        _mark_last(system)
    elif tools:
        tools[-1]["cache_control"] = _ephemeral()
    messages = _turns(request.messages)
    if messages:
        _mark_last(messages[-1]["content"])
    payload: JSON = {
        "model": request.model or default_model,
        "max_tokens": request.max_tokens or DEFAULT_MAX_TOKENS,
        "stream": True,
        "messages": messages,
    }
    if system:
        payload["system"] = system
    if tools:
        payload["tools"] = tools
    output = _output_config(request)
    if output:
        payload["output_config"] = output
    return payload


@dataclass(slots=True)
class _Block:
    """One content block as it streams, kept to be replayed as it came.

    Streamed pieces are joined once, when the block stops: adding each to a
    string held in ``data`` copied the whole string every time.
    """

    data: JSON
    tool: int | None = None
    arguments: list[str] = field(default_factory=list[str])
    pieces: dict[str, list[str]] = field(default_factory=dict[str, list[str]])

    def join(self) -> None:
        for name, pieces in self.pieces.items():
            self.data[name] = str(self.data.get(name, "")) + "".join(pieces)
        self.pieces.clear()


@dataclass(slots=True)
class _Answer:
    """One streamed answer, assembled event by event.

    Tool calls are numbered among themselves, as :class:`ToolCallDelta`
    expects, while content blocks are numbered among all blocks.
    """

    blocks: dict[int, _Block] = field(default_factory=dict[int, _Block])
    usage: JSON = field(default_factory=dict[str, Any])
    tools: int = 0
    chars: int = 0
    stopped: bool = False

    def read(self, data: JSON) -> ChatChunk | None:
        """Take one event; return what it adds to the answer, if anything.

        Raises:
            GatewayError: If the event is an error, the same class an HTTP
                status with that error type would give.
            MalformedStreamError: If the event does not have the expected shape.
        """
        try:
            return self._read(data)
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            message = f"unexpected event shape: {type(error).__name__}"
            raise MalformedStreamError(message) from error

    def _read(self, data: JSON) -> ChatChunk | None:
        kind = data.get("type")
        if kind == "error":
            raise _stream_error(data.get("error"))
        handler = _HANDLERS.get(str(kind))
        # A ping, or an event type added later, adds nothing.
        return None if handler is None else handler(self, data)

    def begin(self, data: JSON) -> ChatChunk:
        message = cast("JSON", data["message"])
        self._count(message.get("usage"))
        return ChatChunk(model=message.get("model"), id=message.get("id"))

    def add(self, data: JSON) -> ChatChunk | None:
        """Add a delta to its block.

        Raises:
            MalformedStreamError: If the answer grows past ``MAX_ANSWER_CHARS``.
        """
        block = self.blocks[int(data["index"])]
        chunk, added = _apply(block, cast("JSON", data["delta"]))
        self.chars += added
        if self.chars > MAX_ANSWER_CHARS:
            message = f"the answer passed {MAX_ANSWER_CHARS} characters"
            raise MalformedStreamError(message)
        return chunk

    def end_block(self, data: JSON) -> ChatChunk | None:
        return _close(self.blocks[int(data["index"])])

    def finish(self, data: JSON) -> ChatChunk:
        self._count(data.get("usage"))
        reason = cast("JSON", data["delta"]).get("stop_reason")
        return ChatChunk(
            finish_reason=(
                STOP_REASONS.get(reason, FinishReason.OTHER) if reason else None
            ),
            usage=self._usage(),
        )

    def stop(self, _data: JSON) -> ChatChunk | None:
        self.stopped = True
        replay = self._replay()
        return None if replay is None else ChatChunk(replay=replay)

    def start_block(self, data: JSON) -> ChatChunk | None:
        """Open a content block.

        Raises:
            MalformedStreamError: If a block with that index was already opened.
        """
        index = int(data["index"])
        if index in self.blocks:
            message = f"content block {index} was started twice"
            raise MalformedStreamError(message)
        block = dict(cast("JSON", data["content_block"]))
        kind = block["type"]
        if kind == "tool_use":
            ordinal = self.tools
            self.tools += 1
            self.blocks[index] = _Block(block, ordinal)
            call = ToolCallDelta(ordinal, id=str(block["id"]), name=str(block["name"]))
            return ChatChunk(tool_calls=(call,))
        self.blocks[index] = _Block(block)
        text = block.get("text") if kind == "text" else None
        return ChatChunk(text=text) if isinstance(text, str) and text else None

    def _count(self, usage: object) -> None:
        if isinstance(usage, dict):
            given = cast("JSON", usage)
            self.usage |= {k: v for k, v in given.items() if v is not None}

    def _usage(self) -> Usage:
        read = int(self.usage.get("cache_read_input_tokens", 0))
        written = int(self.usage.get("cache_creation_input_tokens", 0))
        uncached = int(self.usage.get("input_tokens", 0))
        return Usage(
            prompt_tokens=uncached + read + written,
            completion_tokens=int(self.usage.get("output_tokens", 0)),
            cached_prompt_tokens=read,
            cache_write_tokens=written,
        )

    def _replay(self) -> Replay | None:
        for block in self.blocks.values():
            block.join()
        content = [self.blocks[i].data for i in sorted(self.blocks)]
        if all(b.get("type") in REBUILT_BLOCKS for b in content):
            return None
        text = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
        return Replay(PROVIDER, text)


_HANDLERS: Final[Mapping[str, Callable[[_Answer, JSON], ChatChunk | None]]] = (
    MappingProxyType(
        {
            "message_start": _Answer.begin,
            "content_block_start": _Answer.start_block,
            "content_block_delta": _Answer.add,
            "content_block_stop": _Answer.end_block,
            "message_delta": _Answer.finish,
            "message_stop": _Answer.stop,
        }
    )
)


def _apply(block: _Block, delta: JSON) -> tuple[ChatChunk | None, int]:
    """Add a delta to its block; return what it shows and how many characters it added.

    Text is shown, tool input is passed on as a fragment, and thinking and its
    signature are kept for the replay but never shown.

    Raises:
        MalformedStreamError: If the delta is of a kind this adapter does not
            know or does not fit its block.
    """
    kind = str(delta["type"])
    if kind == "input_json_delta" and block.tool is not None:
        fragment = str(delta["partial_json"])
        block.arguments.append(fragment)
        call = ToolCallDelta(block.tool, arguments=fragment)
        return ChatChunk(tool_calls=(call,)), len(fragment)
    name = DELTA_FIELDS.get(kind)
    value = None if name is None else delta.get(name)
    if name is None or not isinstance(value, str):
        message = (
            f"cannot add a {kind[:SHOWN_KIND_CHARS]!r} delta "
            f"to a {str(block.data.get('type'))[:SHOWN_KIND_CHARS]!r} block"
        )
        raise MalformedStreamError(message)
    block.pieces.setdefault(name, []).append(value)
    return (ChatChunk(text=value) if name == "text" else None), len(value)


def _close(block: _Block) -> ChatChunk | None:
    """Finish a block; a tool call with no fragments gets the start's input.

    Raises:
        MalformedStreamError: If a tool call's input is not a JSON object.
    """
    block.join()
    if block.tool is None:
        return None
    if not block.arguments:
        given = block.data.get("input", {})
        if not isinstance(given, dict):
            message = "a tool call's input was not a JSON object"
            raise MalformedStreamError(message)
        whole = json.dumps(given)
        return ChatChunk(tool_calls=(ToolCallDelta(block.tool, arguments=whole),))
    try:
        given: object = json.loads("".join(block.arguments))
    except ValueError:
        given = None
    if not isinstance(given, dict):
        message = "a tool call's input was not a JSON object"
        raise MalformedStreamError(message)
    block.data["input"] = given
    return None


def _stream_error(error: object) -> GatewayError:
    kind = cast("JSON", error).get("type") if isinstance(error, dict) else None
    status = STREAM_ERROR_STATUSES.get(str(kind), UNKNOWN_STREAM_ERROR_STATUS)
    return error_for(status, json.dumps({"error": error}).encode(), {})


class AnthropicModel:
    """Stream completions from Anthropic's Messages API."""

    def __init__(
        self, *, client: httpx.AsyncClient, endpoint: Endpoint, info: ModelInfo
    ) -> None:
        self._client = client
        self._url = endpoint.base_url.rstrip("/") + MESSAGES_PATH
        self._model = endpoint.model
        self._info = info
        self._api_key = endpoint.api_key
        self._headers = dict(endpoint.headers)
        self._timeout = (
            httpx.USE_CLIENT_DEFAULT if endpoint.timeout is None else endpoint.timeout
        )

    @property
    def info(self) -> ModelInfo:
        """Return what this model can do."""
        return self._info

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        """Yield the completion for ``request`` as the API generates it.

        Raises:
            GatewayError: A subclass describing why the request failed; its
                ``retryable`` says whether sending it again may succeed. No
                message ever holds the key.
        """
        if request.temperature is not None:
            logger.info(
                "temperature %s not sent: the Anthropic API rejects it",
                request.temperature,
            )
        headers = {
            **self._headers,
            **PLAIN_BODY,
            "anthropic-version": API_VERSION,
        }
        if self._api_key is not None:
            headers["x-api-key"] = self._api_key.get_secret_value()
        payload = build_payload(request, self._model)
        try:
            async with self._client.stream(
                "POST",
                self._url,
                json=payload,
                headers=headers,
                timeout=self._timeout,
                follow_redirects=False,
            ) as response:
                _refuse_redirect(response)
                if response.status_code >= httpx.codes.BAD_REQUEST:
                    body = await read_error_body(response)
                    raise error_for(response.status_code, body, response.headers)
                expect_event_stream(response)
                async with aclosing(_chunks(response)) as chunks:
                    async for chunk in chunks:
                        yield chunk
        except httpx.TransportError as error:
            message = f"no response from {self._url}: {type(error).__name__}"
            raise ConnectionFailedError(message) from error
        except GatewayError as error:
            scrub(error, self._api_key)
            raise


def _refuse_redirect(response: httpx.Response) -> None:
    """Refuse a redirect: following one could carry the key to another host.

    Raises:
        BadRequestError: If the response is a redirect.
    """
    status = response.status_code
    if httpx.codes.MULTIPLE_CHOICES <= status < httpx.codes.BAD_REQUEST:
        message = (
            f"HTTP {status}: the API answered with a redirect, which is not "
            "followed so the key reaches no other address; check base_url"
        )
        raise BadRequestError(message)


async def _chunks(response: httpx.Response) -> AsyncGenerator[ChatChunk]:
    answer = _Answer()
    events = aiter_events(response.aiter_bytes())
    async with aclosing(events):
        async for event in events:
            try:
                data: object = json.loads(event.data)
            except ValueError as error:
                message = "a stream event was not JSON"
                raise MalformedStreamError(message) from error
            if not isinstance(data, dict):
                message = "a stream event was not a JSON object"
                raise MalformedStreamError(message)
            chunk = answer.read(cast("JSON", data))
            if chunk is not None:
                yield chunk
            if answer.stopped:
                return
    message = "the stream ended before message_stop"
    raise IncompleteResponseError(message)
