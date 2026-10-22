"""Record the OpenRouter cassettes the gateway tests replay.

Run once with a working key in ``.env``, then commit the files:

    uv run python scripts/record_cassettes.py            # every scenario
    uv run python scripts/record_cassettes.py image      # just one

Each scenario is one real request through the assembled gateway, so it is
claimed from the daily budget like any other. ``openrouter/free`` picks the
answering model at random, so a re-recording may come from another model.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final

import httpx
from pydantic import BaseModel, ConfigDict

from synthia.gateway.assemble import build_gateway
from synthia.gateway.cassette import CassetteTransport
from synthia.gateway.protocol import collect
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

    from synthia.kernel.bus import Event

CASSETTES: Final = Path(__file__).resolve().parent.parent / "tests" / "cassettes"
OPENROUTER_CASSETTES: Final = CASSETTES / "openrouter"
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


async def _ignore(_event: Event) -> None:
    return None


async def record(names: Sequence[str]) -> None:
    """Record ``names`` into ``tests/cassettes/openrouter``, one file each.

    Raises:
        ConfigError: If no key is set.
    """
    settings = load_settings()
    key = settings.openrouter_api_key
    if key is None:
        message = "set SYNTHIA_OPENROUTER_API_KEY in .env to record"
        raise ConfigError(message)
    for name in names:
        path = OPENROUTER_CASSETTES / f"{name}.json"
        transport = CassetteTransport.recording(path, secrets=[key.get_secret_value()])
        async with httpx.AsyncClient(transport=transport, timeout=TIMEOUT) as client:
            gateway = build_gateway(settings, client, _ignore)
            if name in STRUCTURED:
                kind = STRUCTURED[name]
                result = repr(await generate(gateway.model, SCENARIOS[name], kind))
            else:
                response = await collect(gateway.model.stream(SCENARIOS[name]))
                result = f"{response.model} {response.finish_reason}"
        print(f"{name}: {result} -> {path.name}")


def main(argv: Sequence[str]) -> int:
    """Record the scenarios named in ``argv``, or all of them."""
    unknown = [name for name in argv if name not in SCENARIOS]
    if unknown:
        print(f"unknown scenario: {', '.join(unknown)}; known: {', '.join(SCENARIOS)}")
        return 2
    asyncio.run(record(argv or list(SCENARIOS)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
