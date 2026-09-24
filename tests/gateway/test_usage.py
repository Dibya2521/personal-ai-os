from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from synthia.gateway.errors import (
    AuthError,
    ConnectionFailedError,
    GatewayError,
    ProviderError,
)
from synthia.gateway.protocol import collect
from synthia.gateway.types import (
    ChatChunk,
    ChatRequest,
    FinishReason,
    Message,
    ModelInfo,
    Usage,
)
from synthia.gateway.usage import (
    AccountingModel,
    ModelTotals,
    UsageLog,
    UsageRecorded,
    free_requests_today,
)
from synthia.kernel.bus import Event

KEY = "sk-or-v1-usage-test-key-000"  # pragma: allowlist secret
BASE = "https://openrouter.ai/api/v1"
INFO = ModelInfo("openrouter/free", 32_768, vision=True, tools=True)
HELLO = ChatRequest((Message.user("hello"),))
TEN_IN_TWO_OUT = Usage(10, 2)


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 9, 24, 23, 59, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


class Answering:
    """Answer as ``model`` with ``usage``, or fail, or stop without a finish."""

    def __init__(
        self, model: str | None = "vendor/a", usage: Usage | None = TEN_IN_TWO_OUT
    ) -> None:
        self.model = model
        self.usage = usage
        self.error: GatewayError | None = None
        self.finish = True

    @property
    def info(self) -> ModelInfo:
        return INFO

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        del request
        yield ChatChunk(text="hi", model=self.model)
        if self.error:
            raise self.error
        if self.finish:
            yield ChatChunk(finish_reason=FinishReason.STOP, usage=self.usage)


class Ticks:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 0.5
        return self.now


async def _ignore(_: Event) -> None:
    return None


def accounting(
    tmp_path: Path, inner: Answering, clock: Clock | None = None
) -> tuple[AccountingModel, UsageLog, list[Event]]:
    events: list[Event] = []

    async def publish(event: Event) -> None:
        events.append(event)

    log = UsageLog(tmp_path / "gateway.db", clock or Clock())
    return AccountingModel(inner, log, publish, Ticks()), log, events


async def test_each_finished_call_adds_to_the_answering_models_totals(
    tmp_path: Path,
) -> None:
    first, log, events = accounting(tmp_path, Answering("vendor/a"))
    second = AccountingModel(Answering("vendor/b", Usage(5, 1)), log, _ignore, Ticks())

    await collect(first.stream(HELLO))
    await collect(first.stream(HELLO))
    await collect(second.stream(HELLO))

    assert await log.today() == (
        ModelTotals("vendor/a", 2, 20, 4, 1.0),
        ModelTotals("vendor/b", 1, 5, 1, 0.5),
    )
    assert first.info is INFO
    assert events[0] == UsageRecorded(
        model="vendor/a",
        prompt_tokens=10,
        completion_tokens=2,
        seconds=0.5,
        event_id=events[0].event_id,
        created_at=events[0].created_at,
        correlation_id=events[0].correlation_id,
    )


async def test_a_call_without_usage_or_model_counts_under_the_model_asked(
    tmp_path: Path,
) -> None:
    model, log, events = accounting(tmp_path, Answering(model=None, usage=None))

    await collect(model.stream(HELLO))

    assert await log.today() == (ModelTotals("openrouter/free", 1, 0, 0, 0.5),)
    (event,) = events
    assert isinstance(event, UsageRecorded)
    assert (event.prompt_tokens, event.completion_tokens) == (None, None)


async def test_failed_unfinished_or_interrupted_calls_are_not_recorded(
    tmp_path: Path,
) -> None:
    failing, unfinished, interrupted = Answering(), Answering(), Answering()
    failing.error = ProviderError("down")
    unfinished.finish = False
    log = UsageLog(tmp_path / "gateway.db", Clock())

    with pytest.raises(ProviderError):
        await collect(AccountingModel(failing, log, _ignore).stream(HELLO))
    chunks = [c async for c in AccountingModel(unfinished, log, _ignore).stream(HELLO)]
    stream = AccountingModel(interrupted, log, _ignore).stream(HELLO)
    await anext(stream)
    await stream.aclose()

    assert [c.text for c in chunks] == ["hi"]
    assert await log.today() == ()


async def test_totals_are_per_utc_day(tmp_path: Path) -> None:
    clock = Clock()
    model, log, _ = accounting(tmp_path, Answering(), clock)
    await collect(model.stream(HELLO))

    clock.now += timedelta(minutes=2)  # 00:01 UTC, still 05:31 in India

    assert await log.today() == ()


def key_endpoint(status: int = 200, body: object = None) -> httpx.MockTransport:
    if body is None:
        body = {
            "data": {
                "is_free_tier": True,
                "free_model_daily_requests": {"used": 3, "limit": 50, "remaining": 47},
            }
        }

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{BASE}/key"
        assert request.headers["Authorization"] == f"Bearer {KEY}"
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handler)


async def test_the_providers_count_is_read_from_the_key_endpoint() -> None:
    async with httpx.AsyncClient(transport=key_endpoint()) as client:
        count = await free_requests_today(client, SecretStr(KEY), BASE + "/")

    assert (count.used, count.limit, count.remaining, count.paid) == (3, 50, 47, False)


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"data": {}},
        {"data": {"is_free_tier": True}},
        {"data": {"is_free_tier": True, "free_model_daily_requests": {"used": "many"}}},
        [],
    ],
)
async def test_an_unknown_shape_is_a_provider_error(body: object) -> None:
    async with httpx.AsyncClient(transport=key_endpoint(body=body)) as client:
        with pytest.raises(ProviderError, match="unknown shape"):
            await free_requests_today(client, SecretStr(KEY), BASE)


async def test_a_refusal_is_typed_and_never_carries_the_key() -> None:
    body = {"error": {"message": f"key {KEY} is not valid"}}
    async with httpx.AsyncClient(transport=key_endpoint(401, body)) as client:
        with pytest.raises(AuthError) as caught:
            await free_requests_today(client, SecretStr(KEY), BASE)

    assert KEY not in str(caught.value)


async def test_no_connection_is_a_connection_failure() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        message = "refused"
        raise httpx.ConnectError(message, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(refuse)) as client:
        with pytest.raises(ConnectionFailedError, match="no response from"):
            await free_requests_today(client, SecretStr(KEY), BASE)
