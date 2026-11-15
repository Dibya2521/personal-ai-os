import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from synthia.gateway.anthropic import AnthropicModel, build_payload
from synthia.gateway.assemble import remote_adapter
from synthia.gateway.errors import (
    AuthError,
    BadRequestError,
    GatewayError,
    IncompleteResponseError,
    ProviderError,
    RateLimitedError,
)
from synthia.gateway.openai_compat import (
    MAX_ERROR_BODY_BYTES,
    MAX_ERROR_MESSAGE,
    Dialect,
    Endpoint,
)
from synthia.gateway.protocol import collect
from synthia.gateway.providers import ANTHROPIC, ANTHROPIC_OPUS, load_providers
from synthia.gateway.types import (
    ChatRequest,
    ChatResponse,
    FinishReason,
    ImagePart,
    Message,
    ModelInfo,
    Reasoning,
    Replay,
    ToolCall,
    ToolSpec,
    Usage,
)
from synthia.memory.window import message_chars
from tests.endless_stream import EndlessBody

KEY = "sk-ant-api03-adapter-test-key-000"  # pragma: allowlist secret
BASE = "https://api.example.test"
INFO = ModelInfo("claude-test", 1_000_000, vision=True, tools=True)
HELLO = ChatRequest((Message.user("hello"),))
WEATHER = ToolSpec(
    "get_weather",
    "Return the weather for a city.",
    {"type": "object", "properties": {"city": {"type": "string"}}},
)

type Handler = Callable[[httpx.Request], httpx.Response]


def sse(*events: dict[str, object]) -> bytes:
    return "".join(
        f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events
    ).encode()


def start(**usage: int | None) -> dict[str, object]:
    message = {"id": "msg_1", "type": "message", "model": "claude-test-actual"}
    return {"type": "message_start", "message": message | {"usage": usage}}


def block(index: int, content: dict[str, object]) -> dict[str, object]:
    return {"type": "content_block_start", "index": index, "content_block": content}


def delta(index: int, **fields: object) -> dict[str, object]:
    return {"type": "content_block_delta", "index": index, "delta": fields}


def stop_block(index: int) -> dict[str, object]:
    return {"type": "content_block_stop", "index": index}


def finish(reason: str | None, output_tokens: int = 5) -> dict[str, object]:
    return {
        "type": "message_delta",
        "delta": {"stop_reason": reason},
        "usage": {"output_tokens": output_tokens},
    }


STOP: dict[str, object] = {"type": "message_stop"}
TEXT_ANSWER = sse(
    start(input_tokens=10, output_tokens=1),
    block(0, {"type": "text", "text": ""}),
    {"type": "ping"},
    delta(0, type="text_delta", text="Hel"),
    delta(0, type="text_delta", text="lo"),
    stop_block(0),
    finish("end_turn"),
    STOP,
)
TOOL_TURN = sse(
    start(input_tokens=10),
    block(0, {"type": "thinking", "thinking": "", "signature": ""}),
    delta(0, type="thinking_delta", thinking="check the weather tool"),
    delta(0, type="signature_delta", signature="sig-abc"),
    stop_block(0),
    block(1, {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}}),
    delta(1, type="input_json_delta", partial_json='{"city": '),
    delta(1, type="input_json_delta", partial_json='"Paris"}'),
    stop_block(1),
    finish("tool_use"),
    STOP,
)


class Recorder:
    """Answer each request with the next body, keeping what was sent."""

    def __init__(self, *bodies: bytes, status: int = 200) -> None:
        self.bodies = list(bodies)
        self.status = status
        self.sent: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.sent.append(request)
        return httpx.Response(
            self.status,
            headers={"content-type": "text/event-stream"},
            content=self.bodies.pop(0),
        )

    def body(self, n: int = -1) -> dict[str, object]:
        return json.loads(self.sent[n].content)


async def ask(
    handler: Handler, request: ChatRequest = HELLO, key: str | None = KEY
) -> ChatResponse:
    endpoint = Endpoint(
        BASE,
        "claude-test",
        None if key is None else SecretStr(key),
        dialect=Dialect.ANTHROPIC,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = AnthropicModel(client=client, endpoint=endpoint, info=INFO)
        return await collect(model.stream(request))


async def test_a_text_answer_streams_with_model_usage_and_stop() -> None:
    reply = await ask(Recorder(TEXT_ANSWER))

    assert reply.text == "Hello"
    assert reply.finish_reason is FinishReason.STOP
    assert reply.model == "claude-test-actual"
    assert reply.usage == Usage(10, 5)
    assert reply.tool_calls == ()
    assert reply.replay is None


async def test_the_key_goes_in_x_api_key_and_temperature_is_never_sent() -> None:
    recorder = Recorder(TEXT_ANSWER)

    await ask(recorder, ChatRequest((Message.user("hi"),), temperature=0.4))

    sent = recorder.sent[0]
    assert str(sent.url) == f"{BASE}/v1/messages"
    assert sent.headers["x-api-key"] == KEY
    assert sent.headers["anthropic-version"] == "2026-10-21"
    assert sent.headers["accept-encoding"] == "identity"
    assert "authorization" not in sent.headers
    body = recorder.body()
    assert "temperature" not in body
    assert body["stream"] is True
    assert body["model"] == "claude-test"
    assert body["max_tokens"] == 64_000


async def test_a_tool_call_streamed_as_json_fragments_arrives_whole() -> None:
    reply = await ask(Recorder(TOOL_TURN))

    assert reply.tool_calls == (
        ToolCall("toolu_1", "get_weather", '{"city": "Paris"}'),
    )
    assert reply.finish_reason is FinishReason.TOOL_CALLS


async def test_thinking_is_never_shown_and_is_sent_back_with_its_signature() -> None:
    recorder = Recorder(TOOL_TURN, TEXT_ANSWER)
    first = await ask(recorder, ChatRequest((Message.user("weather?"),)))
    history = (
        Message.user("weather?"),
        first.as_message(),
        Message.tool_result("toolu_1", "18 C, clear"),
    )

    await ask(recorder, ChatRequest(history, tools=(WEATHER,)))

    assert first.text == ""
    assert first.reasoning == ""
    sent = recorder.body()["messages"]
    assert isinstance(sent, list)
    assert sent[1] == {
        "role": "assistant",
        "content": [
            {
                "type": "thinking",
                "thinking": "check the weather tool",
                "signature": "sig-abc",
            },
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {"city": "Paris"},
            },
        ],
    }
    assert sent[2]["content"][0]["type"] == "tool_result"


async def test_cache_reads_and_writes_count_into_usage() -> None:
    body = sse(
        start(
            input_tokens=3,
            cache_read_input_tokens=2000,
            cache_creation_input_tokens=100,
        ),
        block(0, {"type": "text", "text": "ok"}),
        stop_block(0),
        finish("end_turn", output_tokens=7),
        STOP,
    )

    reply = await ask(Recorder(body))

    assert reply.text == "ok"
    assert reply.usage == Usage(
        prompt_tokens=2103,
        completion_tokens=7,
        cached_prompt_tokens=2000,
        cache_write_tokens=100,
    )


async def test_null_cache_fields_count_as_none_cached() -> None:
    body = sse(
        start(input_tokens=4, cache_read_input_tokens=None),
        finish("end_turn", output_tokens=1),
        STOP,
    )

    reply = await ask(Recorder(body))

    assert reply.usage == Usage(4, 1)


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("end_turn", FinishReason.STOP),
        ("max_tokens", FinishReason.LENGTH),
        ("tool_use", FinishReason.TOOL_CALLS),
        ("refusal", FinishReason.CONTENT_FILTER),
        ("a_reason_added_later", FinishReason.OTHER),
    ],
)
async def test_stop_reasons_map_to_finish_reasons(
    reason: str, expected: FinishReason
) -> None:
    reply = await ask(Recorder(sse(start(input_tokens=1), finish(reason), STOP)))

    assert reply.finish_reason is expected


def failing(status: int, headers: dict[str, str] | None = None) -> Handler:
    def handler(_request: httpx.Request) -> httpx.Response:
        error = {"type": "error", "error": {"type": "x", "message": f"bad key {KEY}"}}
        return httpx.Response(status, headers=headers, json=error)

    return handler


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (400, BadRequestError),
        (401, AuthError),
        (500, ProviderError),
        (529, ProviderError),
    ],
)
async def test_http_errors_map_to_typed_errors_without_the_key(
    status: int, kind: type[GatewayError]
) -> None:
    with pytest.raises(kind) as raised:
        await ask(failing(status))

    assert KEY not in str(raised.value)
    assert f"HTTP {status}" in str(raised.value)


async def test_a_rate_limit_carries_retry_after() -> None:
    with pytest.raises(RateLimitedError) as raised:
        await ask(failing(429, {"retry-after": "7"}))

    assert raised.value.retry_after_s == pytest.approx(7.0)
    assert raised.value.retryable
    assert KEY not in str(raised.value)


async def test_an_overload_inside_the_stream_is_a_retryable_provider_error() -> None:
    error = {"type": "overloaded_error", "message": f"Overloaded for {KEY}"}
    body = sse(start(input_tokens=1), {"type": "error", "error": error})

    with pytest.raises(ProviderError) as raised:
        await ask(Recorder(body))

    assert raised.value.retryable
    assert KEY not in str(raised.value)


async def test_a_stream_error_with_the_key_at_the_cut_leaves_no_part_of_it() -> None:
    padding = "x" * (MAX_ERROR_MESSAGE - len("HTTP 529: ") - len(KEY) + 1)
    error = {"type": "overloaded_error", "message": padding + KEY}
    body = sse(start(input_tokens=1), {"type": "error", "error": error})

    with pytest.raises(ProviderError) as raised:
        await ask(Recorder(body))

    assert KEY[:-1] not in str(raised.value)


async def test_a_stream_that_ends_before_message_stop_is_incomplete() -> None:
    body = sse(start(input_tokens=1), block(0, {"type": "text", "text": "Hi"}))

    with pytest.raises(IncompleteResponseError):
        await ask(Recorder(body))


async def test_an_endless_error_body_is_read_only_up_to_its_cap() -> None:
    body = EndlessBody()

    with pytest.raises(ProviderError, match=r"^HTTP 503$"):
        await ask(lambda _request: httpx.Response(503, stream=body))

    assert body.pieces_read * len(body.piece) == MAX_ERROR_BODY_BYTES


async def test_a_redirect_is_not_followed() -> None:
    seen: list[str] = []

    def redirect(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(
            302, headers={"location": "https://other.test/v1/messages"}
        )

    with pytest.raises(BadRequestError, match=r"HTTP 302"):
        await ask(redirect)

    assert seen == [f"{BASE}/v1/messages"]


def test_tool_results_of_one_round_go_in_one_user_message() -> None:
    calls = (
        ToolCall("toolu_a", "get_weather", '{"city": "Paris"}'),
        ToolCall("toolu_b", "get_weather", '{"city": "Rome"}'),
    )
    history = (
        Message.user("weather in Paris and Rome?"),
        Message.assistant("Checking both.", *calls),
        Message.tool_result("toolu_a", "18 C"),
        Message.tool_result("toolu_b", "24 C"),
    )

    messages = build_payload(ChatRequest(history), "m")["messages"]

    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"] == [
        {"type": "text", "text": "Checking both."},
        {
            "type": "tool_use",
            "id": "toolu_a",
            "name": "get_weather",
            "input": {"city": "Paris"},
        },
        {
            "type": "tool_use",
            "id": "toolu_b",
            "name": "get_weather",
            "input": {"city": "Rome"},
        },
    ]
    assert [
        (b["type"], b["tool_use_id"], b["content"]) for b in messages[2]["content"]
    ] == [
        ("tool_result", "toolu_a", "18 C"),
        ("tool_result", "toolu_b", "24 C"),
    ]


def test_system_goes_top_level_with_breakpoints_after_it_and_the_last_message() -> None:
    request = ChatRequest(
        (Message.system("You are SYNTHIA."), Message.user("hi")),
        tools=(WEATHER,),
        reasoning=Reasoning.OFF,
        response_schema={"type": "object"},
    )

    payload = build_payload(request, "m")

    assert payload["system"] == [
        {
            "type": "text",
            "text": "You are SYNTHIA.",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert payload["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
            ],
        }
    ]
    assert "cache_control" not in payload["tools"][0]
    assert payload["tools"][0]["input_schema"] == WEATHER.parameters
    assert payload["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": {"type": "object"}},
    }


def test_without_a_system_prompt_the_tools_carry_the_breakpoint() -> None:
    payload = build_payload(ChatRequest((Message.user("hi"),), tools=(WEATHER,)), "m")

    assert payload["tools"][0]["cache_control"] == {"type": "ephemeral"}
    assert "system" not in payload
    assert "output_config" not in payload


@pytest.mark.parametrize(
    ("level", "effort"),
    [
        (Reasoning.LOW, "low"),
        (Reasoning.MEDIUM, "medium"),
        (Reasoning.HIGH, "high"),
    ],
)
def test_reasoning_levels_become_effort(level: Reasoning, effort: str) -> None:
    payload = build_payload(ChatRequest((Message.user("hi"),), reasoning=level), "m")

    assert payload["output_config"] == {"effort": effort}


def test_auto_reasoning_and_a_token_budget_send_no_thinking_setting() -> None:
    request = ChatRequest(
        (Message.user("hi"),), reasoning=Reasoning.AUTO, reasoning_tokens=500
    )

    payload = build_payload(request, "m")

    assert "output_config" not in payload
    assert "thinking" not in payload


def test_an_image_is_sent_as_base64() -> None:
    image = ImagePart(b"\x89PNG-bytes", "image/png")

    payload = build_payload(ChatRequest((Message.user("what?", image),)), "m")

    assert payload["messages"][0]["content"][1] == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "iVBORy1ieXRlcw==",
        },
        "cache_control": {"type": "ephemeral"},
    }


def test_another_providers_replay_is_ignored() -> None:
    foreign = Replay("someone-else", '[{"type": "secret"}]')
    history = (
        Message.user("hi"),
        Message.assistant("hello", replay=foreign),
        Message.user("again"),
    )

    messages = build_payload(ChatRequest(history), "m")["messages"]

    assert messages[1]["content"] == [{"type": "text", "text": "hello"}]


def test_tool_arguments_that_are_not_an_object_are_refused() -> None:
    history = (
        Message.user("hi"),
        Message.assistant("", ToolCall("c1", "get_weather", "not json")),
        Message.tool_result("c1", "?"),
    )

    with pytest.raises(BadRequestError, match=r"not a JSON object"):
        build_payload(ChatRequest(history), "m")


def test_anthropic_is_built_in_on_claude_opus_5_5(tmp_path: Path) -> None:
    built_in = load_providers(tmp_path)[ANTHROPIC]

    assert built_in.model == ModelInfo(
        "claude-opus-5-5", 1_000_000, vision=True, tools=True
    )
    assert built_in.dialect is Dialect.ANTHROPIC
    assert built_in.base_url == "https://api.anthropic.com"


async def test_the_anthropic_dialect_selects_the_messages_api() -> None:
    recorder = Recorder(TEXT_ANSWER)
    async with httpx.AsyncClient(transport=httpx.MockTransport(recorder)) as client:
        model = remote_adapter(client, ANTHROPIC_OPUS, SecretStr(KEY))
        reply = await collect(model.stream(HELLO))

    assert reply.text == "Hello"
    assert str(recorder.sent[0].url) == "https://api.anthropic.com/v1/messages"
    assert recorder.body()["model"] == "claude-opus-5-5"


def thinking_stream(pieces: int, piece: str) -> bytes:
    return sse(
        start(input_tokens=10),
        block(0, {"type": "thinking", "thinking": "", "signature": ""}),
        *(delta(0, type="thinking_delta", thinking=piece) for _ in range(pieces)),
        delta(0, type="signature_delta", signature="sig"),
        stop_block(0),
        block(1, {"type": "text", "text": ""}),
        delta(1, type="text_delta", text="done"),
        stop_block(1),
        finish("end_turn"),
        STOP,
    )


async def test_long_thinking_is_kept_whole_and_in_linear_time() -> None:
    pieces = 50_000
    reply = await ask(Recorder(thinking_stream(pieces, "think ")))

    assert reply.replay is not None
    replayed = json.loads(reply.replay.content)
    assert replayed[0]["thinking"] == "think " * pieces
    assert replayed[0]["signature"] == "sig"
    assert replayed[1] == {"type": "text", "text": "done"}


async def test_a_runaway_answer_is_cut_off() -> None:
    stream = thinking_stream(9, "x" * 1024 * 1024)

    with pytest.raises(GatewayError, match="passed 8388608 characters"):
        await ask(Recorder(stream))


@pytest.mark.parametrize(
    "events",
    [
        pytest.param(
            (
                block(0, {"type": "tool_use", "id": "t", "name": "list", "input": {}}),
                delta(0, type="name_delta", name="list_and_delete"),
            ),
            id="unknown-delta-kind",
        ),
        pytest.param(
            (
                block(0, {"type": "tool_use", "id": "t1", "name": "a", "input": {}}),
                block(0, {"type": "tool_use", "id": "t2", "name": "b", "input": {}}),
            ),
            id="index-started-twice",
        ),
        pytest.param(
            (
                block(0, {"type": "tool_use", "id": "t", "name": "a", "input": ["x"]}),
                stop_block(0),
            ),
            id="non-object-input",
        ),
    ],
)
async def test_a_stream_the_replay_could_not_match_is_refused(
    events: tuple[dict[str, object], ...],
) -> None:
    stream = sse(start(input_tokens=1), *events, finish("tool_use"), STOP)

    with pytest.raises(GatewayError):
        await ask(Recorder(stream))


async def test_a_refused_delta_kind_is_named_briefly() -> None:
    kind = "k" * 100_000 + "_delta"
    stream = sse(
        start(input_tokens=1),
        block(0, {"type": "text", "text": ""}),
        delta(0, type=kind),
        STOP,
    )

    with pytest.raises(GatewayError) as caught:
        await ask(Recorder(stream))

    assert len(str(caught.value)) < 200


def test_the_window_counts_a_replay_as_what_is_sent() -> None:
    content = json.dumps([{"type": "thinking", "thinking": "t" * 5000}])
    message = Message.assistant("ok", replay=Replay(ANTHROPIC, content))

    assert message_chars(message) == len(content)
