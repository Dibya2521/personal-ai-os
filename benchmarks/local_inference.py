"""Measure the local model the way the chat uses it, on each installed build.

For every backend asked for, the server is started through SYNTHIA's own
launch path (same model, context and key handling as ``synthia chat``), and
the same streamed request is sent several times with thinking off:

    uv run python benchmarks/local_inference.py --backend cpu --backend vulkan

Each run reports the time to the first answer token, prompt tokens per second
(prompt tokens over that time, so it includes the server's own overhead),
generation tokens per second after the first token, and the server's peak
resident memory. It reaches no outside service and spends no budget. The
numbers describe the machine it runs on and how busy it is, so the load is
printed with them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Final

import httpx
import psutil

from synthia.gateway.types import ChatRequest, Message, Reasoning
from synthia.interfaces.cli import local_service
from synthia.kernel.config import LocalBackend, load_settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from synthia.gateway.types import ChatChunk, Usage
    from synthia.models.server import Launch
    from synthia.models.service import LocalService

DESCRIPTION: Final = "Measure the local model the way the chat uses it."
RUNS: Final = 5
MAX_ANSWER_TOKENS: Final = 128
SAMPLE_EVERY_S: Final = 0.2
# Loading on a busy machine has taken over 200 s on the CPU build.
LOAD_TIMEOUT_S: Final = 900.0
TIMEOUT: Final = httpx.Timeout(LOAD_TIMEOUT_S, connect=10.0)
SERVER_NAME: Final = "llama-server"
# About 300 prompt tokens with the system turn: long enough that prompt
# processing shows, short enough for five runs per build in minutes.
PROMPT: Final = (
    "Read this note and answer in exactly three short sentences: what it is "
    "about, what the main risk is, and what to do first.\n\n"
    + " ".join(
        [
            "The team moved the nightly backup from the old file server to object",
            "storage last week. Restores were tested once, from a single folder,",
            "and took four minutes. Nobody has tried a full restore yet. The old",
            "server is due to be switched off on Friday, and its disks will be",
            "wiped the same day. Two people know the storage credentials, and one",
            "of them is on leave until the end of the month. The backup job sends",
            "an email when it fails, but the address it uses belongs to someone",
            "who left the company in the spring.",
        ]
        * 3
    )
)


@dataclass(frozen=True, slots=True)
class Run:
    """One request's measurements."""

    first_token_s: float
    prompt_tokens: int
    completion_tokens: int
    total_s: float

    @property
    def prompt_per_s(self) -> float:
        """Return prompt tokens over the time to the first answer token."""
        return self.prompt_tokens / self.first_token_s

    @property
    def generation_per_s(self) -> float:
        """Return answer tokens after the first over the time they took."""
        generating = self.total_s - self.first_token_s
        return (self.completion_tokens - 1) / generating if generating > 0 else 0.0


def server_memory() -> int:
    """Return the resident bytes of every llama-server this process started."""
    total = 0
    for child in psutil.Process().children(recursive=True):
        try:
            if child.name().lower().startswith(SERVER_NAME):
                total += child.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    return total


async def watch_memory(peak: list[int]) -> None:
    """Keep ``peak[0]`` at the highest server memory seen until cancelled."""
    while True:
        peak[0] = max(peak[0], server_memory())
        await asyncio.sleep(SAMPLE_EVERY_S)


async def one_run(
    stream: Callable[[ChatRequest], AsyncIterator[ChatChunk]], request: ChatRequest
) -> Run:
    """Send ``request`` once and time it."""
    started = time.perf_counter()
    first: float | None = None
    usage: Usage | None = None
    async for chunk in stream(request):
        if chunk.text and first is None:
            first = time.perf_counter() - started
        if chunk.usage is not None:
            usage = chunk.usage
    total = time.perf_counter() - started
    if first is None or usage is None:
        message = "the server answered with no text or no token counts"
        raise RuntimeError(message)
    return Run(first, usage.prompt_tokens, usage.completion_tokens, total)


async def bench(backend: LocalBackend, threads: int | None) -> dict[str, object]:
    """Start ``backend``, time its load, then measure :data:`RUNS` requests."""
    settings = load_settings().model_copy(update={"local_backend": backend})

    def command(launch: Launch) -> list[str]:
        extra = [] if threads is None else ["--threads", str(threads)]
        return [*launch.command(), *extra]

    service = local_service(settings, command=command)
    if service is None:
        message = "no local model is installed: run `synthia models install`"
        raise RuntimeError(message)
    peak = [0]
    watcher = asyncio.create_task(watch_memory(peak))
    try:
        ran_on, load_s, runs = await measure(service)
    finally:
        watcher.cancel()
    return {
        "asked": backend.value,
        "ran_on": ran_on,
        "threads": threads,
        "load_s": round(load_s, 1),
        "runs": [asdict(r) for r in runs],
        "first_token_s": spread([r.first_token_s for r in runs]),
        "prompt_per_s": spread([r.prompt_per_s for r in runs]),
        "generation_per_s": spread([r.generation_per_s for r in runs]),
        "peak_server_mb": round(peak[0] / 2**20),
    }


def request_for(run: int) -> ChatRequest:
    """Return run ``run``'s request, which opens with its own number.

    llama-server reuses the cached start of an earlier prompt, so identical
    requests would skip prompt processing after the first.
    """
    return ChatRequest(
        (Message.user(f"Request {run}. {PROMPT}"),),
        temperature=0.0,
        max_tokens=MAX_ANSWER_TOKENS,
        reasoning=Reasoning.OFF,
    )


async def measure(service: LocalService) -> tuple[str, float, list[Run]]:
    """Start ``service`` and return the build it ran on, its load time and runs."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        model = service.model(client)
        started = time.perf_counter()
        service.start()
        try:
            async with asyncio.timeout(LOAD_TIMEOUT_S):
                # The server's state lives on the service's own thread and loop,
                # so there is no event on this loop to wait for.
                while not model.ready():  # noqa: ASYNC110
                    await asyncio.sleep(SAMPLE_EVERY_S)
            load_s = time.perf_counter() - started
            running = service.server.running
            ran_on = "unknown" if running is None else running.backend.value
            # Not counted: the first request after loading also fills caches.
            await one_run(model.stream, request_for(0))
            runs = [
                await one_run(model.stream, request_for(n)) for n in range(1, RUNS + 1)
            ]
        finally:
            service.stop()
    return ran_on, load_s, runs


def spread(values: list[float]) -> dict[str, float]:
    """Return the lowest, median and highest of ``values``."""
    return {
        "min": round(min(values), 2),
        "median": round(statistics.median(values), 2),
        "max": round(max(values), 2),
    }


async def main(arguments: list[str]) -> int:
    """Run the benchmark for each ``--backend`` and print one JSON line each."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--backend",
        action="append",
        type=LocalBackend,
        choices=[LocalBackend.CPU, LocalBackend.VULKAN],
        required=True,
    )
    parser.add_argument("--threads", type=int, default=None)
    options = parser.parse_args(arguments)
    load = psutil.cpu_percent(interval=3)
    sys.stdout.write(json.dumps({"cpu_busy_percent_before": load}) + "\n")
    for backend in options.backend:
        result = await bench(backend, options.threads)
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
