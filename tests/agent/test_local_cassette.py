import json

import httpx
from pydantic import SecretStr
from record_local_cassettes import (
    AGENT_CASSETTE,
    AGENT_MODEL,
    agent_request,
    agent_tools,
)

from synthia.agent.loop import Agent, Finished, Outcome, ToolFinished
from synthia.gateway.cassette import CassetteTransport
from synthia.gateway.openai_compat import Dialect, Endpoint, OpenAICompatibleModel
from synthia.gateway.types import ModelInfo

# Any origin will do: a cassette matches by path, and the recorded port is gone.
REPLAY_BASE = "http://127.0.0.1:1/v1"
# From the recorded run (scripts/record_local_cassettes.py, qwen3.5-4b on llama-server).
RECORDED_CALLS = [
    ("current_time", "{}", "Saturday 2026-10-03 09:30:00 UTC+0530", True),
    ("calculate", '{"expression":"1234 * 5678"}', "7006652", True),
]
RECORDED_ANSWER = (
    "It is Saturday, October 3, 2026, at 09:30:00 UTC+0530, "
    "and 1234 multiplied by 5678 equals 7006652."
)


def recorded_model(client: httpx.AsyncClient) -> OpenAICompatibleModel:
    """Return a model that asks exactly as the local model asked when recorded."""
    info = json.loads(AGENT_MODEL.read_text(encoding="utf-8"))
    model = ModelInfo(
        info["id"], info["context_window"], vision=info["vision"], tools=True
    )
    endpoint = Endpoint(
        REPLAY_BASE, model.id, SecretStr("replay"), dialect=Dialect.LLAMA_CPP
    )
    return OpenAICompatibleModel(client=client, endpoint=endpoint, info=model)


async def test_a_local_run_with_two_tool_calls_at_once_replays() -> None:
    cassette = CassetteTransport.replaying(AGENT_CASSETTE)
    async with httpx.AsyncClient(transport=cassette) as client:
        agent = Agent(recorded_model(client), agent_tools())
        events = [event async for event in agent.run(agent_request())]

    calls = [
        (e.call.name, e.call.arguments, e.result, e.ok)
        for e in events
        if isinstance(e, ToolFinished)
    ]
    (finished,) = [e for e in events if isinstance(e, Finished)]
    assert calls == RECORDED_CALLS
    assert (finished.outcome, finished.text) == (Outcome.ANSWERED, RECORDED_ANSWER)
    assert cassette.unplayed == 0
