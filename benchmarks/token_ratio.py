"""Measure the characters per token the context window learns in a real chat.

    uv run python benchmarks/token_ratio.py --threads 2

SYNTHIA counts a request in characters and turns them into tokens at a rate
it learns from the model: after each turn it moves the rate towards the
characters it sent over the prompt tokens the model reported. This holds a
short scripted chat through :class:`synthia.server.session.ChatSession`, the
code every conversation runs, on the installed local model, as the default
persona, with SYNTHIA's own tools that are the same on every machine (clock,
calculator, files, Python, memory search on an empty store, web fetch; no
outside agents or MCP servers, whose presence varies). Thinking is off to
keep the run short.

After each turn it prints the prompt tokens reported, the rate the window
holds, and the turn's own figure, worked back from the rate's change (the
window weighs each new figure 0.3 against the rate before it). Answers are
printed cut short, for a reader to see the chat was a real one. It reaches
no outside service and spends no budget.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final

import httpx
import psutil

from synthia.gateway.types import ChatChunk, Reasoning
from synthia.interfaces.cli import local_service
from synthia.kernel.config import load_settings
from synthia.memory.hybrid import Recall
from synthia.memory.recall import memory_tool
from synthia.memory.store import MEMORY_FILE, MemoryStore
from synthia.memory.window import (
    LEARNING_WEIGHT,
    MAX_CHARS_PER_TOKEN,
    MIN_CHARS_PER_TOKEN,
    START_CHARS_PER_TOKEN,
)
from synthia.persona.library import PersonaLibrary
from synthia.server.host import PERSONAS_DIR
from synthia.server.session import ChatSession, TurnReport
from synthia.tools import local_tools
from synthia.tools.web import fetch_tool

if TYPE_CHECKING:
    from synthia.kernel.config import Settings
    from synthia.models.server import Launch
    from synthia.models.service import LocalService

LOAD_TIMEOUT_S: Final = 900.0
TIMEOUT: Final = httpx.Timeout(LOAD_TIMEOUT_S, connect=10.0)
WAIT_S: Final = 0.2
SHOWN_ANSWER_CHARS: Final = 120
SCRIPT: Final = (
    (
        "Hi! I'm Sam. I work as a nurse in Bhubaneswar, and in my spare time "
        "I'm learning to play the guitar."
    ),
    "Which three chord changes should a beginner practise first? Keep it short.",
    "What time is it now?",
    "If I practise 25 minutes a day, how many hours is that over six weeks?",
    "Give me one tip for practising after a long night shift, in one sentence.",
    "What did I tell you about my job? One sentence.",
)


def turn_rate(before: float, after: float) -> float | None:
    """Return the figure a turn taught the window.

    None if it taught nothing (the model reported no prompt size) or the
    rate was clamped, since the figure cannot be worked back then.
    """
    if after == before or after in {MIN_CHARS_PER_TOKEN, MAX_CHARS_PER_TOKEN}:
        return None
    return (after - (1 - LEARNING_WEIGHT) * before) / LEARNING_WEIGHT


async def chat(session: ChatSession) -> None:
    """Say each line of :data:`SCRIPT` and print what each turn showed."""
    for number, text in enumerate(SCRIPT, 1):
        before = session.window.chars_per_token
        answer: list[str] = []
        report: TurnReport | None = None
        async for item in session.turn(text):
            if isinstance(item, TurnReport):
                report = item
            elif isinstance(item, ChatChunk):
                answer.append(item.text)
        after = session.window.chars_per_token
        figure = turn_rate(before, after)
        printed(
            {
                "turn": number,
                "prompt_tokens": None if report is None else report.prompt_tokens,
                "seconds": None if report is None else round(report.seconds, 1),
                "rate_after": round(after, 3),
                "turn_figure": None if figure is None else round(figure, 3),
                "answer": "".join(answer)[:SHOWN_ANSWER_CHARS],
            }
        )


def printed(line: dict[str, object]) -> None:
    """Print ``line`` as JSON at once, so a long run shows its progress."""
    sys.stdout.write(json.dumps(line) + "\n")
    sys.stdout.flush()


async def measure(service: LocalService, settings: Settings) -> None:
    """Start the local model, hold the chat on it, print the rate it ended at."""
    with tempfile.TemporaryDirectory() as home:
        tools = local_tools(settings.file_roots)
        tools.add(memory_tool(Recall(MemoryStore(Path(home) / MEMORY_FILE))))
        tools.add(fetch_tool())
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            model = service.model(client)
            started = time.perf_counter()
            service.start()
            try:
                async with asyncio.timeout(LOAD_TIMEOUT_S):
                    # The server's state lives on the service's own thread and loop.
                    while not model.ready():  # noqa: ASYNC110
                        await asyncio.sleep(WAIT_S)
                printed(
                    {
                        "model": model.info.id,
                        "load_s": round(time.perf_counter() - started, 1),
                    }
                )
                session = ChatSession(
                    model,
                    PersonaLibrary(settings.home / PERSONAS_DIR),
                    settings.persona,
                    tools=tools,
                )
                session.reasoning = Reasoning.OFF
                await chat(session)
                printed(
                    {
                        "context_window": session.window.limit,
                        "start_rate": START_CHARS_PER_TOKEN,
                        "rate_after_chat": round(session.window.chars_per_token, 3),
                    }
                )
            finally:
                service.stop()


async def main(arguments: list[str]) -> int:
    """Hold the scripted chat and print one JSON line per turn, then the rate."""
    parser = argparse.ArgumentParser(description="Characters per token, learned")
    parser.add_argument("--threads", type=int, default=None)
    options = parser.parse_args(arguments)
    settings = load_settings()

    def command(launch: Launch) -> list[str]:
        extra = [] if options.threads is None else ["--threads", str(options.threads)]
        return [*launch.command(), *extra]

    service = local_service(settings, command=command)
    if service is None:
        sys.stderr.write("no local model is installed: run `synthia models install`\n")
        return 1
    load = await asyncio.to_thread(psutil.cpu_percent, interval=3)
    printed({"cpu_busy_percent_before": load, "threads": options.threads})
    await measure(service, settings)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
