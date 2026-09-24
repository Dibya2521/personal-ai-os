import json
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from synthia.gateway.assemble import GATEWAY_DB, build_gateway
from synthia.gateway.protocol import collect
from synthia.gateway.providers import (
    APP_TITLE,
    APP_URL,
    OPENROUTER,
    OPENROUTER_FREE,
    OPENROUTER_FREE_CONTEXT,
    openrouter_endpoint,
    openrouter_info,
)
from synthia.gateway.router import Route, RouteDecided, RouteReason
from synthia.gateway.types import ChatRequest, Message
from synthia.kernel.bus import Event
from synthia.kernel.config import Settings
from synthia.kernel.errors import ConfigError

KEY = "sk-or-v1-assemble-test-key-000"  # pragma: allowlist secret
HELLO = ChatRequest((Message.user("hello"),))


def answer(_: httpx.Request) -> httpx.Response:
    events: list[dict[str, object]] = [
        {"model": "vendor/free-model", "choices": [{"delta": {"content": "hi"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    return httpx.Response(200, content=body.encode())


def test_the_openrouter_endpoint_sends_the_attribution_headers() -> None:
    endpoint = openrouter_endpoint(SecretStr(KEY))

    assert endpoint.completions_url == "https://openrouter.ai/api/v1/chat/completions"
    assert endpoint.model == OPENROUTER_FREE == "openrouter/free"
    assert dict(endpoint.headers) == {"HTTP-Referer": APP_URL, "X-Title": APP_TITLE}


def test_the_free_router_sees_and_calls_tools_and_others_get_text_only() -> None:
    free = openrouter_info()
    other = openrouter_info("vendor/some-model")

    assert (free.vision, free.tools, free.context_window) == (
        True,
        True,
        OPENROUTER_FREE_CONTEXT,
    )
    assert (other.id, other.vision, other.tools) == ("vendor/some-model", False, False)


async def test_without_a_key_there_is_no_gateway_and_the_variable_is_named(
    tmp_path: Path,
) -> None:
    async with httpx.AsyncClient() as client:
        with pytest.raises(ConfigError, match="SYNTHIA_OPENROUTER_API_KEY"):
            build_gateway(Settings(home=tmp_path), client, publish=_ignore)


async def _ignore(_: Event) -> None:
    return None


async def test_a_request_goes_out_metered_to_openrouter_with_the_key_only_in_its_header(
    tmp_path: Path,
) -> None:
    sent: list[httpx.Request] = []
    events: list[Event] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return answer(request)

    async def publish(event: Event) -> None:
        events.append(event)

    settings = Settings(home=tmp_path, openrouter_api_key=SecretStr(KEY))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway = build_gateway(settings, client, publish)
        reply = await collect(gateway.router.stream(HELLO))
        status = await gateway.health.ledger.status(OPENROUTER)

    assert reply.text == "hi"
    assert reply.model == "vendor/free-model"
    (request,) = sent
    assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    assert request.headers["X-Title"] == APP_TITLE
    assert KEY not in request.content.decode()
    assert json.loads(request.content)["model"] == OPENROUTER_FREE
    assert (status.used, status.cap) == (1, 50)
    assert (tmp_path / GATEWAY_DB).is_file()
    decided = [e for e in events if isinstance(e, RouteDecided)]
    assert [(e.route, e.reason) for e in decided] == [
        (Route.REMOTE, RouteReason.PREFERRED)
    ]
