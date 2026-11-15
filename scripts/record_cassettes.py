"""Record the OpenRouter and Anthropic cassettes the gateway tests replay.

Run once with a working key in ``.env`` or the key store, and the provider
granted with ``synthia consent grant <provider>``, then commit the files:

    uv run python scripts/record_cassettes.py                  # every scenario
    uv run python scripts/record_cassettes.py image            # just one
    uv run python scripts/record_cassettes.py anthropic-cache  # one on Anthropic

Each scenario is one real request through the assembled gateway, so it is
claimed from the daily budget like any other; the Anthropic cache scenario is
two. ``openrouter/free`` picks the answering model at random, so a re-recording
may come from another model. The Anthropic scenarios are paid requests to
Claude Opus 5.5 on the key's account.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Final

import httpx
from pydantic import BaseModel, ConfigDict

from synthia.gateway.assemble import build_gateway
from synthia.gateway.cassette import CassetteTransport
from synthia.gateway.protocol import collect
from synthia.gateway.providers import (
    ANTHROPIC_OPUS,
    OPENROUTER_FREE,
    REMOTE_KEY_HINT,
    remote_key,
)
from synthia.gateway.structured import generate
from synthia.gateway.types import (
    ChatRequest,
    ImagePart,
    Message,
    Role,
    TextPart,
    ToolSpec,
)
from synthia.kernel.config import load_settings
from synthia.kernel.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from synthia.gateway.protocol import ChatModel
    from synthia.kernel.bus import Event

CASSETTES: Final = Path(__file__).resolve().parent.parent / "tests" / "cassettes"
OPENROUTER_CASSETTES: Final = CASSETTES / "openrouter"
ANTHROPIC_CASSETTES: Final = CASSETTES / "anthropic"
TIMEOUT: Final = httpx.Timeout(120.0, connect=10.0)
RED_PNG: Final = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452"
    "00000010000000100802000000909168"
    "360000001749444154789c63f8cfc040"  # pragma: allowlist secret
    "12224df5a886510d434a030090f9ff01"  # pragma: allowlist secret
    "f9e1fa780000000049454e44ae426082"  # pragma: allowlist secret
)
"""A 16 by 16 PNG filled with pure red, as recorded in the image cassette.

Kept as bytes, not compressed at run time: zlib builds compress the same
pixels to different bytes (Python 3.14 on Windows ships zlib-ng), which
would change the request body and miss the recording."""


def _ask(*parts: TextPart | ImagePart) -> tuple[Message, ...]:
    return (Message(Role.USER, parts),)


WEATHER_TOOL: Final = ToolSpec(
    "get_weather",
    "Return the current weather for a city.",
    {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "The city name."}},
        "required": ["city"],
    },
)


class Capital(BaseModel):
    """The reply the structured scenario asks for."""

    # Strict structured outputs require additionalProperties false.
    model_config = ConfigDict(extra="forbid")

    city: str
    country: str


SCENARIOS: Final = {
    "plain": ChatRequest(
        _ask(TextPart("Say hello in one short sentence.")), use_remote=True
    ),
    "image": ChatRequest(
        _ask(
            TextPart("What colour fills this image? Answer with one word."),
            ImagePart(RED_PNG, "image/png"),
        ),
        use_remote=True,
    ),
    "tool": ChatRequest(
        _ask(TextPart("What is the weather in Paris right now? Use the tool.")),
        tools=(WEATHER_TOOL,),
        use_remote=True,
    ),
    "structured": ChatRequest(
        _ask(TextPart("Name the capital of France and its country.")),
        use_remote=True,
    ),
}
# Scenarios sent through structured.generate, which adds the schema itself.
STRUCTURED: Final[dict[str, type[BaseModel]]] = {"structured": Capital}

# Claude Opus 5.5 caches nothing shorter than 512 tokens; 120 rules of over a
# dozen tokens each put the system prompt well above that minimum.
CACHED_RULE_COUNT: Final = 120
CACHED_SYSTEM: Final = " ".join(
    f"Rule {n}: answer in plain words, one short sentence at a time."
    for n in range(1, CACHED_RULE_COUNT + 1)
)
ANTHROPIC_SCENARIOS: Final = {
    "anthropic-text": ChatRequest(
        _ask(TextPart("Say hello in one short sentence.")), use_remote=True
    ),
    "anthropic-tool": ChatRequest(
        _ask(TextPart("What is the weather in Paris right now? Use the tool.")),
        tools=(WEATHER_TOOL,),
        use_remote=True,
    ),
    "anthropic-cache": ChatRequest(
        (Message.system(CACHED_SYSTEM), *_ask(TextPart("Say hello."))),
        use_remote=True,
    ),
}
# Sent twice into one cassette: the first request writes the cache, the second
# reads it.
REPEATED: Final = frozenset({"anthropic-cache"})
# Answered and sent again, so the cassette holds the follow-up that carries the
# tool results and the first answer's thinking blocks back.
ROUND_TRIP: Final = frozenset({"anthropic-tool"})
WEATHER_ANSWER: Final = '{"city": "Paris", "sky": "clear", "celsius": 18}'
ALL_SCENARIOS: Final = SCENARIOS | ANTHROPIC_SCENARIOS


async def _ignore(_event: Event) -> None:
    return None


async def _send(model: ChatModel, name: str) -> str:
    """Send scenario ``name`` once, twice, or as a tool round; describe each answer."""
    request = ALL_SCENARIOS[name]
    sends = 2 if name in REPEATED | ROUND_TRIP else 1
    results: list[str] = []
    for _ in range(sends):
        response = await collect(model.stream(request))
        results.append(f"{response.model} {response.finish_reason} {response.usage}")
        if name in ROUND_TRIP:
            answers = tuple(
                Message.tool_result(call.id, WEATHER_ANSWER)
                for call in response.tool_calls
            )
            messages = (*request.messages, response.as_message(), *answers)
            request = replace(request, messages=messages)
    return "; ".join(results)


async def record(names: Sequence[str]) -> None:
    """Record ``names``, one file each, under the provider's cassette folder.

    Raises:
        ConfigError: If the provider's key is not set.
    """
    settings = load_settings()
    for name in names:
        on_anthropic = name in ANTHROPIC_SCENARIOS
        remote = ANTHROPIC_OPUS if on_anthropic else OPENROUTER_FREE
        key = remote_key(settings, remote)
        if key is None:
            message = f"no key for {remote.name} to record with: {REMOTE_KEY_HINT}"
            raise ConfigError(message)
        folder = ANTHROPIC_CASSETTES if on_anthropic else OPENROUTER_CASSETTES
        path = folder / f"{name}.json"
        transport = CassetteTransport.recording(path, secrets=[key.get_secret_value()])
        async with httpx.AsyncClient(transport=transport, timeout=TIMEOUT) as client:
            gateway = build_gateway(settings, client, _ignore, remote=remote)
            if name in STRUCTURED:
                kind = STRUCTURED[name]
                result = repr(await generate(gateway.model, SCENARIOS[name], kind))
            else:
                result = await _send(gateway.model, name)
        print(f"{name}: {result} -> {path.name}")


def main(argv: Sequence[str]) -> int:
    """Record the scenarios named in ``argv``, or all of them."""
    unknown = [name for name in argv if name not in ALL_SCENARIOS]
    if unknown:
        known = ", ".join(ALL_SCENARIOS)
        print(f"unknown scenario: {', '.join(unknown)}; known: {known}")
        return 2
    asyncio.run(record(argv or list(ALL_SCENARIOS)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
