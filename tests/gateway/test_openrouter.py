import json
import re
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr
from record_cassettes import OPENROUTER_CASSETTES, SCENARIOS, STRUCTURED, Capital

from synthia.gateway.assemble import GATEWAY_DB, build_gateway
from synthia.gateway.cassette import CassetteTransport
from synthia.gateway.openai_compat import OpenAICompatibleModel
from synthia.gateway.protocol import collect
from synthia.gateway.providers import OPENROUTER, openrouter_endpoint, openrouter_info
from synthia.gateway.structured import generate
from synthia.gateway.types import ChatResponse, FinishReason
from synthia.kernel.bus import Event
from synthia.kernel.config import Settings

KEY = "sk-or-v1-replay-test-key-000"  # pragma: allowlist secret
FREE_SUFFIX = ":free"
CREDENTIAL = re.compile(
    r"sk-or-[A-Za-z0-9-]{8,}|bearer\s+\S|authorization", re.IGNORECASE
)


async def _ignore(_event: Event) -> None:
    return None


async def replay(name: str) -> ChatResponse:
    cassette = CassetteTransport.replaying(OPENROUTER_CASSETTES / f"{name}.json")
    async with httpx.AsyncClient(transport=cassette) as client:
        model = OpenAICompatibleModel(
            client=client,
            endpoint=openrouter_endpoint(SecretStr(KEY)),
            info=openrouter_info(),
        )
        response = await collect(model.stream(SCENARIOS[name]))
    assert cassette.unplayed == 0
    return response


async def test_a_plain_turn_streams_text_and_usage_from_a_free_model() -> None:
    reply = await replay("plain")

    assert reply.text.strip()
    assert reply.finish_reason is FinishReason.STOP
    assert reply.usage is not None
    assert reply.usage.prompt_tokens > 0
    assert reply.model is not None
    assert reply.model.endswith(FREE_SUFFIX)


async def test_an_image_reaches_the_model_as_pixels() -> None:
    reply = await replay("image")

    assert reply.finish_reason is FinishReason.STOP
    assert reply.text.strip().lower().rstrip(".") == "red"


async def test_a_tool_call_arrives_whole_with_its_arguments() -> None:
    reply = await replay("tool")

    assert reply.finish_reason is FinishReason.TOOL_CALLS
    (call,) = reply.tool_calls
    assert call.name == "get_weather"
    assert call.id
    assert json.loads(call.arguments) == {"city": "Paris"}


async def test_a_structured_reply_validates_against_its_schema() -> None:
    cassette = CassetteTransport.replaying(OPENROUTER_CASSETTES / "structured.json")
    async with httpx.AsyncClient(transport=cassette) as client:
        model = OpenAICompatibleModel(
            client=client,
            endpoint=openrouter_endpoint(SecretStr(KEY)),
            info=openrouter_info(),
        )
        answer = await generate(model, SCENARIOS["structured"], Capital)

    assert answer == Capital(city="Paris", country="France")
    assert cassette.unplayed == 0


@pytest.mark.parametrize("name", sorted(set(SCENARIOS) - set(STRUCTURED)))
async def test_the_assembled_gateway_sends_what_was_recorded(
    tmp_path: Path, name: str
) -> None:
    settings = Settings(home=tmp_path, openrouter_api_key=SecretStr(KEY))
    cassette = CassetteTransport.replaying(OPENROUTER_CASSETTES / f"{name}.json")
    async with httpx.AsyncClient(transport=cassette) as client:
        gateway = build_gateway(settings, client, _ignore)
        await collect(gateway.model.stream(SCENARIOS[name]))
        status = await gateway.health.ledger.status(OPENROUTER)

    assert cassette.unplayed == 0
    assert status.used == 1
    assert (tmp_path / GATEWAY_DB).is_file()


def test_every_scenario_has_a_committed_cassette() -> None:
    recorded = {path.stem for path in OPENROUTER_CASSETTES.glob("*.json")}

    assert recorded == set(SCENARIOS)


@pytest.mark.parametrize("path", sorted(OPENROUTER_CASSETTES.glob("*.json")))
def test_no_cassette_holds_a_credential(path: Path) -> None:
    assert CREDENTIAL.search(path.read_text(encoding="utf-8")) is None


@pytest.mark.parametrize(
    "leak",
    [KEY, f"Bearer {KEY}", '"authorization": "x"'],
    ids=["key", "bearer", "header"],
)
def test_the_credential_check_sees_a_leak(leak: str) -> None:
    assert CREDENTIAL.search(f'{{"text": "{leak}"}}') is not None
