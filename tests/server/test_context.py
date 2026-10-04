import asyncio
import logging
from collections.abc import AsyncGenerator, Sequence

import pytest

from synthia.agent.tools import Toolbox
from synthia.gateway.errors import ProviderError
from synthia.gateway.types import (
    ChatChunk,
    ChatRequest,
    FinishReason,
    Message,
    ModelInfo,
    Role,
    ToolCallDelta,
    Usage,
)
from synthia.persona.library import PersonaLibrary
from synthia.server.session import ChatSession
from synthia.tools.basic import clock_tool

PADDING = "x" * 1200


class Model:
    """Answers "ok", or calls the clock on the turns it is told to."""

    def __init__(
        self, window: int, *, clock_on: int | None = None, usage: Usage | None = None
    ) -> None:
        self.requests: list[ChatRequest] = []
        self._window = window
        self._clock_on = clock_on
        self._usage = usage

    @property
    def info(self) -> ModelInfo:
        return ModelInfo("m", self._window, vision=False, tools=True)

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        self.requests.append(request)
        asks_clock = len(self.requests) == self._clock_on
        if asks_clock:
            yield ChatChunk(tool_calls=(ToolCallDelta(0, "c0", "current_time", "{}"),))
            yield ChatChunk(finish_reason=FinishReason.TOOL_CALLS)
            return
        yield ChatChunk(text="ok")
        yield ChatChunk(finish_reason=FinishReason.STOP, usage=self._usage)


class Summaries:
    """Writes "summary N" from what it is given, and keeps what it was given."""

    def __init__(self) -> None:
        self.given: list[tuple[str, list[Message]]] = []

    async def __call__(self, previous: str, messages: Sequence[Message]) -> str:
        self.given.append((previous, list(messages)))
        return f"summary {len(self.given)}"


def chat(model: Model, **options: object) -> ChatSession:
    return ChatSession(model, PersonaLibrary(), "synthia", **options)  # type: ignore[arg-type]


async def say(session: ChatSession, text: str) -> None:
    async for _ in session.turn(text):
        pass
    # Let a summary started by the turn finish, as the time between turns would.
    await asyncio.sleep(0)


async def test_every_request_fits_and_old_turns_leave_oldest_first() -> None:
    model = Model(window=6000)
    session = chat(model)

    for n in range(12):
        await say(session, f"question {n} {PADDING}")

    fitted = [session.window.tokens(r.messages) for r in model.requests]
    assert max(fitted) <= session.window.budget
    last = model.requests[-1].messages
    assert "question 0 " not in "".join(m.text for m in last)
    assert "question 11 " in last[-1].text
    assert last[1].role is Role.USER


async def test_a_turn_with_a_tool_call_leaves_whole() -> None:
    model = Model(window=6000, clock_on=1)
    session = chat(model, tools=Toolbox([clock_tool()]))

    for n in range(12):
        await say(session, f"question {n} {PADDING}")

    assert all(request.messages[1].role is Role.USER for request in model.requests[2:])


async def test_turns_that_leave_are_summarised_and_the_summary_is_kept_in_view() -> (
    None
):
    model = Model(window=6000)
    summaries = Summaries()
    session = chat(model, summarizer=summaries)

    for n in range(12):
        await say(session, f"question {n} {PADDING}")

    first_previous, first_left = summaries.given[0]
    assert first_previous == ""
    assert first_left[0].text.startswith("question 0 ")
    assert summaries.given[1][0] == "summary 1"
    assert session.summary == "summary 3"
    # Summary 3 is written after the last request, from the turns it pushed out.
    system = model.requests[-1].messages[0].text
    assert system.endswith("Earlier in this conversation, in short: summary 2")


async def test_a_summary_that_fails_is_logged_and_the_chat_goes_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def broken(previous: str, messages: Sequence[Message]) -> str:
        del previous, messages
        message = "no local model"
        raise ProviderError(message)

    session = chat(Model(window=6000), summarizer=broken)

    with caplog.at_level(logging.WARNING, "synthia.server.session"):
        for n in range(12):
            await say(session, f"question {n} {PADDING}")

    assert "were not summarised" in caplog.text
    assert session.summary == ""


async def test_closing_stops_a_summary_still_being_written() -> None:
    started = asyncio.Event()

    async def slow(previous: str, messages: Sequence[Message]) -> str:
        del previous, messages
        started.set()
        await asyncio.Event().wait()
        return "never"

    session = chat(Model(window=6000), summarizer=slow)
    for n in range(12):
        await say(session, f"question {n} {PADDING}")
    await asyncio.wait_for(started.wait(), 5)

    await session.close()

    assert session.summary == ""


async def test_the_rate_is_learned_from_the_first_answer_of_a_turn() -> None:
    session = chat(
        Model(window=6000, usage=Usage(prompt_tokens=200, completion_tokens=1))
    )

    await say(session, "hello")

    assert session.window.chars_per_token != 3.0
