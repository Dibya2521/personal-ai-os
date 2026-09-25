import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest
from pydantic import SecretStr

from synthia.gateway.errors import (
    AuthError,
    BadRequestError,
    ConnectionFailedError,
    IncompleteResponseError,
    MalformedStreamError,
    PaymentRequiredError,
    ProviderError,
    RateLimitedError,
)
from synthia.gateway.openai_compat import (
    Dialect,
    Endpoint,
    OpenAICompatibleModel,
    build_payload,
    error_for,
    parse_chunk,
    retry_after_seconds,
)
from synthia.gateway.protocol import collect
from synthia.gateway.types import (
    ChatRequest,
    FinishReason,
    ImagePart,
    Message,
    ModelInfo,
    Reasoning,
    ToolCall,
    ToolSpec,
    Usage,
)

KEY = "sk-or-v1-adapter-test-key-000"  # pragma: allowlist secret
INFO = ModelInfo("test/model", 32_000, vision=True, tools=True)
HELLO = ChatRequest((Message.user("hello"),))


def sse(*events: object, done: bool = True) -> bytes:
    lines = [f"data: {json.dumps(e)}\n\n" for e in events]
    if done:
        lines.append("data: [DONE]\n\n")
    return (": OPENROUTER PROCESSING\n\n" + "".join(lines)).encode()


def delta(**fields: object) -> dict[str, object]:
    return {"model": "vendor/actual", "choices": [{"delta": fields}]}


def finish(
    reason: str = "stop", usage: tuple[int, int] | None = (9, 3)
) -> dict[str, object]:
    body: dict[str, object] = {"choices": [{"delta": {}, "finish_reason": reason}]}
    if usage:
        body["usage"] = {"prompt_tokens": usage[0], "completion_tokens": usage[1]}
    return body


def model_answering(
    handler: Callable[[httpx.Request], httpx.Response],
) -> OpenAICompatibleModel:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    endpoint = Endpoint(
        "https://example.test/api/v1/",
        "default/model",
        SecretStr(KEY),
        {"X-Title": "SYNTHIA"},
    )
    return OpenAICompatibleModel(client=client, endpoint=endpoint, info=INFO)


def streaming(
    body: bytes, status: int = 200, **headers: str
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=body, headers=headers)

    return handler


# Payload: request -> wire


def test_text_only_content_is_a_plain_string() -> None:
    payload = build_payload(HELLO, "default/model")

    assert payload["model"] == "default/model"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert "temperature" not in payload
    assert "tools" not in payload


def test_an_image_becomes_a_content_list_with_a_data_url() -> None:
    image = ImagePart(b"\x89PNG", "image/png")
    request = ChatRequest((Message.user("what is this", image),), model="vision/model")

    payload = build_payload(request, "default/model")

    assert payload["model"] == "vision/model"
    (content,) = [m["content"] for m in payload["messages"]]
    assert content[0] == {"type": "text", "text": "what is this"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_tools_tool_calls_and_tool_results_map_to_the_wire_shape() -> None:
    call = ToolCall("call_1", "clock", '{"zone":"IST"}')
    request = ChatRequest(
        (
            Message.system("be precise"),
            Message.user("time?"),
            Message.assistant("", call),
            Message.tool_result("call_1", "15:21"),
        ),
        tools=(ToolSpec("clock", "Current time", {"type": "object"}),),
        temperature=0.2,
        max_tokens=64,
        response_schema={"type": "object"},
    )

    payload = build_payload(request, "m")

    assistant, tool = payload["messages"][2], payload["messages"][3]
    assert assistant == {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "clock", "arguments": '{"zone":"IST"}'},
            }
        ],
        "content": None,
    }
    assert tool == {"role": "tool", "tool_call_id": "call_1", "content": "15:21"}
    assert payload["tools"][0]["function"]["name"] == "clock"
    assert payload["temperature"] == 0.2
    assert payload["max_tokens"] == 64
    assert payload["response_format"]["json_schema"]["strict"] is True


@pytest.mark.parametrize(
    ("level", "openrouter", "llama_cpp"),
    [
        (
            Reasoning.OFF,
            {"reasoning": {"effort": "none"}},
            {"chat_template_kwargs": {"enable_thinking": False}},
        ),
        (
            Reasoning.LOW,
            {"reasoning": {"effort": "low"}},
            {"chat_template_kwargs": {"enable_thinking": True}},
        ),
        (
            Reasoning.MEDIUM,
            {"reasoning": {"effort": "medium"}},
            {"chat_template_kwargs": {"enable_thinking": True}},
        ),
        (
            Reasoning.HIGH,
            {"reasoning": {"effort": "high"}},
            {"chat_template_kwargs": {"enable_thinking": True}},
        ),
        (Reasoning.AUTO, {}, {}),
        (None, {}, {}),
    ],
)
def test_a_reasoning_level_is_sent_in_each_servers_dialect(
    level: Reasoning | None,
    openrouter: dict[str, object],
    llama_cpp: dict[str, object],
) -> None:
    request = ChatRequest(HELLO.messages, reasoning=level)
    plain = build_payload(HELLO, "m")

    for dialect, added in (
        (Dialect.OPENROUTER, openrouter),
        (Dialect.LLAMA_CPP, llama_cpp),
    ):
        payload = build_payload(request, "m", dialect)
        assert payload == plain | added


def test_no_reasoning_level_leaves_the_body_byte_identical() -> None:
    # Recorded cassettes match on the body's hash, so a request that sets no
    # level must serialise exactly as it did before levels existed.
    before = {
        "model": "m",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    for dialect in Dialect:
        assert json.dumps(build_payload(HELLO, "m", dialect)) == json.dumps(before)


async def test_the_endpoints_dialect_shapes_what_is_sent() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=sse(delta(content="hi"), finish()))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    endpoint = Endpoint("http://127.0.0.1:8080/v1", "m", dialect=Dialect.LLAMA_CPP)
    model = OpenAICompatibleModel(client=client, endpoint=endpoint, info=INFO)

    await collect(model.stream(ChatRequest(HELLO.messages, reasoning=Reasoning.OFF)))

    assert bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning" not in bodies[0]


# Chunks: wire -> chunk


def test_parse_chunk_reads_text_reasoning_tools_finish_usage_and_model() -> None:
    chunk = parse_chunk(
        {
            "model": "qwen/qwen3.5-4b",
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            "choices": [
                {
                    "delta": {
                        "content": "Hi",
                        "reasoning_content": "user greets",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c",
                                "function": {"name": "n", "arguments": "{"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
    )

    assert chunk.text == "Hi"
    assert chunk.reasoning == "user greets"
    assert chunk.tool_calls[0].name == "n"
    assert chunk.finish_reason is FinishReason.TOOL_CALLS
    assert chunk.usage == Usage(4, 2)
    assert chunk.model == "qwen/qwen3.5-4b"


def test_openrouter_reasoning_field_is_read_too() -> None:
    assert parse_chunk(delta(reasoning="thinking")).reasoning == "thinking"


def test_a_usage_only_chunk_with_no_choices_is_accepted() -> None:
    chunk = parse_chunk(
        {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    )

    assert chunk.usage == Usage(1, 1)
    assert chunk.text == ""


MALFORMED: list[dict[str, object]] = [
    {"choices": "not a list"},
    {"choices": [{"delta": {"tool_calls": [{"function": {}}]}}]},
    {"usage": {"prompt_tokens": 1}},
]


@pytest.mark.parametrize("data", MALFORMED)
def test_a_chunk_of_the_wrong_shape_is_malformed(data: dict[str, object]) -> None:
    with pytest.raises(MalformedStreamError, match="unexpected chunk shape"):
        parse_chunk(data)


def test_an_error_inside_the_stream_is_a_provider_error() -> None:
    with pytest.raises(ProviderError, match="upstream timed out"):
        parse_chunk({"error": {"message": "upstream timed out", "code": 502}})


# HTTP errors


@pytest.mark.parametrize(
    ("status", "error_type", "retryable"),
    [
        (400, BadRequestError, False),
        (401, AuthError, False),
        (402, PaymentRequiredError, False),
        (403, AuthError, False),
        (404, BadRequestError, False),
        (408, ProviderError, True),
        (429, RateLimitedError, True),
        (500, ProviderError, True),
        (503, ProviderError, True),
    ],
)
def test_statuses_map_to_typed_errors(
    status: int, error_type: type[Exception], retryable: bool
) -> None:
    error = error_for(status, b'{"error":{"message":"nope"}}', {})

    assert isinstance(error, error_type)
    assert error.retryable is retryable
    assert str(error) == f"HTTP {status}: nope"


def test_an_html_error_page_still_gives_a_clean_error() -> None:
    error = error_for(502, b"<html><body>Bad Gateway</body></html>", {})

    assert str(error) == "HTTP 502"


def test_a_very_long_provider_message_is_truncated() -> None:
    error = error_for(400, json.dumps({"error": {"message": "x" * 5000}}).encode(), {})

    assert len(str(error)) < 400


def test_retry_after_in_seconds_and_as_a_date() -> None:
    now = datetime(2026, 9, 23, 10, 0, tzinfo=UTC)
    later = format_datetime(now + timedelta(seconds=30), usegmt=True)

    assert retry_after_seconds("7") == 7.0
    assert retry_after_seconds(later, now=now.timestamp()) == 30.0
    assert retry_after_seconds("in a bit") is None
    assert retry_after_seconds("nan") is None
    assert retry_after_seconds("-5") == 0.0
    assert retry_after_seconds(None) is None
    error = error_for(429, b"{}", {"retry-after": "12"})
    assert isinstance(error, RateLimitedError)
    assert error.retry_after_s == 12.0


# The adapter end to end, over an in-process transport


async def test_a_streamed_answer_collects_with_the_real_model_name() -> None:
    body = sse(delta(content="Namas"), delta(content="te"), finish())
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=body)

    response = await collect(model_answering(handler).stream(HELLO))

    assert response.text == "Namaste"
    assert response.model == "vendor/actual"
    assert response.usage == Usage(9, 3)
    (sent,) = requests
    assert sent.url == "https://example.test/api/v1/chat/completions"
    assert sent.headers["authorization"] == f"Bearer {KEY}"
    assert sent.headers["x-title"] == "SYNTHIA"
    assert json.loads(sent.content)["model"] == "default/model"


async def test_no_key_means_no_authorization_header() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=sse(finish()))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    endpoint = Endpoint("http://127.0.0.1:8080/v1", "m")
    local = OpenAICompatibleModel(client=client, endpoint=endpoint, info=INFO)

    await collect(local.stream(HELLO))

    assert "authorization" not in seen[0].headers
    assert local.info is INFO


async def test_a_tool_call_streamed_in_fragments_arrives_whole() -> None:
    body = sse(
        delta(
            tool_calls=[
                {"index": 0, "id": "c1", "function": {"name": "clock", "arguments": ""}}
            ]
        ),
        delta(tool_calls=[{"index": 0, "function": {"arguments": '{"zone":'}}]),
        delta(tool_calls=[{"index": 0, "function": {"arguments": '"IST"}'}}]),
        finish("tool_calls"),
    )

    response = await collect(model_answering(streaming(body)).stream(HELLO))

    assert response.tool_calls == (ToolCall("c1", "clock", '{"zone":"IST"}'),)
    assert response.finish_reason is FinishReason.TOOL_CALLS


async def test_a_stream_that_stops_without_finishing_is_incomplete() -> None:
    body = sse(delta(content="the answer is"), done=False)

    with pytest.raises(IncompleteResponseError, match="without a finish reason"):
        await collect(model_answering(streaming(body)).stream(HELLO))


async def test_a_finished_stream_without_done_is_still_complete() -> None:
    body = sse(delta(content="ok"), finish(), done=False)

    assert (await collect(model_answering(streaming(body)).stream(HELLO))).text == "ok"


@pytest.mark.parametrize("event", [b"data: {not json\n\n", b"data: [1, 2]\n\n"])
async def test_an_event_that_is_not_a_json_object_is_malformed(event: bytes) -> None:
    with pytest.raises(MalformedStreamError):
        await collect(model_answering(streaming(event)).stream(HELLO))


async def test_an_http_error_is_raised_before_any_chunk() -> None:
    body = b'{"error":{"message":"Rate limit exceeded: free-models-per-day"}}'
    model = model_answering(streaming(body, 429, **{"Retry-After": "60"}))

    with pytest.raises(RateLimitedError) as caught:
        await collect(model.stream(HELLO))

    assert caught.value.retry_after_s == 60.0
    assert "free-models-per-day" in str(caught.value)


async def test_a_provider_that_echoes_the_key_never_gets_it_into_the_error() -> None:
    body = json.dumps({"error": {"message": f"Invalid API key {KEY}"}}).encode()

    with pytest.raises(AuthError) as caught:
        await collect(model_answering(streaming(body, 401)).stream(HELLO))

    assert KEY not in str(caught.value)
    assert "Invalid API key" in str(caught.value)


async def test_an_unreachable_server_is_a_retryable_connection_failure() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        message = "connection refused"
        raise httpx.ConnectError(message, request=request)

    with pytest.raises(ConnectionFailedError) as caught:
        await collect(model_answering(refuse).stream(HELLO))

    assert caught.value.retryable
    assert KEY not in str(caught.value)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (b'{"error": "model overloaded"}', "HTTP 500: model overloaded"),
        (b'{"error": {"code": 500}}', "HTTP 500"),
        (b'["not", "an", "object"]', "HTTP 500"),
        (b"", "HTTP 500"),
    ],
)
def test_error_bodies_of_every_shape_give_a_clean_message(
    body: bytes, expected: str
) -> None:
    assert str(error_for(500, body, {})) == expected


async def test_a_keyless_local_server_error_is_reported_unchanged() -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(streaming(b"{}", 500)))
    endpoint = Endpoint("http://127.0.0.1:8080/v1", "m")
    local = OpenAICompatibleModel(client=client, endpoint=endpoint, info=INFO)

    with pytest.raises(ProviderError, match="HTTP 500"):
        await collect(local.stream(HELLO))
