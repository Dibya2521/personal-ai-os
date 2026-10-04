"""Record the local-model cassette the agent test replays.

Needs the local model installed (``synthia models install``). It runs
llama-server on this machine, so it reaches no outside service and spends no
budget:

    uv run python scripts/record_local_cassettes.py

The scenario asks for two tools at once (the clock, fixed at one moment so
the replay sends the same bytes, and the calculator) and then an answer,
with thinking off. The model's id and context are written beside it.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

import httpx

from synthia.agent.loop import Agent, Finished, ToolFinished
from synthia.agent.tools import Toolbox
from synthia.gateway.cassette import CassetteTransport
from synthia.gateway.types import ChatRequest, Message, Reasoning
from synthia.interfaces.cli import local_service
from synthia.kernel.config import load_settings
from synthia.tools.basic import calculator_tool, clock_tool

LOCAL_CASSETTES: Final = (
    Path(__file__).resolve().parent.parent / "tests" / "cassettes" / "local"
)
AGENT_CASSETTE: Final = LOCAL_CASSETTES / "agent_two_tools.json"
AGENT_MODEL: Final = LOCAL_CASSETTES / "agent_two_tools.model.json"
FIXED_NOW: Final = datetime(
    2026, 10, 3, 9, 30, tzinfo=timezone(timedelta(hours=5, minutes=30))
)
QUESTION: Final = (
    "What is the time now, and what is 1234 * 5678? Call both tools in one turn, "
    "then answer in one sentence."
)
# Few threads, so a recording leaves the machine to whatever else it is doing.
RECORDING_THREADS: Final = 2
# Loading on a busy machine has taken over 200 s on the CPU build.
TIMEOUT: Final = httpx.Timeout(600.0, connect=10.0)


def agent_tools() -> Toolbox:
    """Return the scenario's tools: a clock stopped at ``FIXED_NOW``, the calculator."""
    return Toolbox((clock_tool(lambda: FIXED_NOW), calculator_tool()))


def agent_request() -> ChatRequest:
    """Return the scenario's request: the question, thinking off, local only."""
    return ChatRequest((Message.user(QUESTION),), reasoning=Reasoning.OFF)


async def record() -> int:
    """Run the scenario on the local model and write the cassette."""
    service = local_service(
        load_settings(),
        command=lambda launch: [*launch.command(), "--threads", str(RECORDING_THREADS)],
    )
    if service is None:
        sys.stdout.write("no local model is installed: run `synthia models install`\n")
        return 1
    # llama-server reports its speed in every answer and in its progress while
    # it reads the prompt: this machine's, not for git.
    recorder = CassetteTransport.recording(
        AGENT_CASSETTE,
        httpx.AsyncHTTPTransport(),
        dropped=("timings", "prompt_progress"),
    )
    async with httpx.AsyncClient(transport=recorder, timeout=TIMEOUT) as client:
        model = service.model(client)
        service.start()
        try:
            async for event in Agent(model, agent_tools()).run(agent_request()):
                if isinstance(event, ToolFinished):
                    call = event.call
                    sys.stdout.write(
                        f"tool {call.name} {call.arguments} -> {event.result}\n"
                    )
                elif isinstance(event, Finished):
                    sys.stdout.write(f"answer ({event.outcome}): {event.text}\n")
        finally:
            service.stop()
    info = {
        "id": model.info.id,
        "context_window": model.info.context_window,
        "vision": model.info.vision,
    }
    AGENT_MODEL.write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(record()))
