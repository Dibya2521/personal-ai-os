import asyncio
import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing

import pytest

from synthia.agent.loop import (
    Agent,
    Finished,
    Limits,
    ModelChunk,
    ModelTurn,
    Outcome,
    Step,
    ToolFinished,
    ToolStarted,
)
from synthia.agent.tools import Effect, FunctionTool, Reach, Toolbox, ToolFunction
from synthia.gateway.types import (
    ChatChunk,
    ChatRequest,
    FinishReason,
    Message,
    ModelInfo,
    Reasoning,
    Role,
    ToolCallDelta,
)
from tests.timing import HANG_TIMEOUT_S

INFO = ModelInfo("scripted", 4096, vision=False, tools=True)


def says(text: str) -> list[ChatChunk]:
    return [ChatChunk(text=text), ChatChunk(finish_reason=FinishReason.STOP)]


def calls(*wanted: tuple[str, str]) -> list[ChatChunk]:
    return [
        ChatChunk(
            tool_calls=tuple(
                ToolCallDelta(i, f"call_{i}", name, arguments)
                for i, (name, arguments) in enumerate(wanted)
            )
        ),
        ChatChunk(finish_reason=FinishReason.TOOL_CALLS),
    ]


class Scripted:
    """A model that answers each request with the next scripted reply."""

    def __init__(
        self, *replies: list[ChatChunk] | Callable[[ChatRequest], list[ChatChunk]]
    ) -> None:
        self._replies = list(replies)
        self.requests: list[ChatRequest] = []

    @property
    def info(self) -> ModelInfo:
        return INFO

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        self.requests.append(request)
        reply = self._replies.pop(0)
        for chunk in reply(request) if callable(reply) else reply:
            yield chunk


def tool(function: ToolFunction) -> FunctionTool:
    return FunctionTool.of(function, reach=Reach.LOCAL, effect=Effect.READ)


def question(text: str = "go") -> ChatRequest:
    return ChatRequest((Message.user(text),), reasoning=Reasoning.LOW, use_remote=True)


async def steps(agent: Agent, request: ChatRequest) -> list[Step]:
    async def gather() -> list[Step]:
        async with aclosing(agent.run(request)) as run:
            return [s async for s in run]

    return await asyncio.wait_for(gather(), HANG_TIMEOUT_S)


def end(events: list[Step]) -> Finished:
    last = events[-1]
    assert isinstance(last, Finished)
    return last


def finished_calls(events: list[Step]) -> list[ToolFinished]:
    return [e for e in events if isinstance(e, ToolFinished)]


async def test_an_answer_without_tools_ends_the_run() -> None:
    model = Scripted(says("hello"))

    events = await steps(Agent(model, Toolbox()), question())

    assert [type(e) for e in events] == [ModelChunk, ModelChunk, ModelTurn, Finished]
    assert end(events).text == "hello"
    assert end(events).outcome is Outcome.ANSWERED


async def test_every_step_keeps_the_request_settings_and_offers_the_tools() -> None:
    def now() -> str:
        """Return the time."""
        return "noon"

    model = Scripted(calls(("now", "{}")), says("it is noon"))
    box = Toolbox([tool(now)])

    await steps(Agent(model, box), question())

    for sent in model.requests:
        assert sent.reasoning is Reasoning.LOW
        assert sent.use_remote is True
        assert sent.tools == box.specs()


async def test_a_tool_result_goes_back_and_the_model_answers() -> None:
    def now() -> str:
        """Return the time."""
        return "noon"

    model = Scripted(calls(("now", "{}")), says("it is noon"))

    events = await steps(Agent(model, Toolbox([tool(now)])), question("time?"))

    second = model.requests[1].messages
    assert [m.role for m in second] == [Role.USER, Role.ASSISTANT, Role.TOOL]
    assert second[2].tool_call_id == "call_0"
    assert second[2].text == "noon"
    assert end(events).text == "it is noon"
    assert end(events).messages[-1].text == "it is noon"


async def test_calls_in_one_answer_run_at_the_same_time() -> None:
    a_started, b_started = asyncio.Event(), asyncio.Event()

    async def a() -> str:
        """A."""
        a_started.set()
        await b_started.wait()
        return "a"

    async def b() -> str:
        """B."""
        b_started.set()
        await a_started.wait()
        return "b"

    model = Scripted(calls(("a", "{}"), ("b", "{}")), says("both"))

    # Run one after the other, each would wait for the other forever.
    events = await steps(Agent(model, Toolbox([tool(a), tool(b)])), question())

    assert end(events).text == "both"


async def test_results_go_back_in_call_order_whatever_finishes_first() -> None:
    slow_may_finish = asyncio.Event()

    async def slow() -> str:
        """Slow."""
        await slow_may_finish.wait()
        return "slow"

    async def fast() -> str:
        """Fast."""
        slow_may_finish.set()
        return "fast"

    model = Scripted(calls(("slow", "{}"), ("fast", "{}")), says("done"))

    events = await steps(Agent(model, Toolbox([tool(slow), tool(fast)])), question())

    assert [f.result for f in finished_calls(events)] == ["fast", "slow"]
    sent = model.requests[1].messages
    assert [(m.tool_call_id, m.text) for m in sent[2:]] == [
        ("call_0", "slow"),
        ("call_1", "fast"),
    ]
    started = [e.call.name for e in events if isinstance(e, ToolStarted)]
    assert started == ["slow", "fast"]


async def test_bad_arguments_go_back_as_the_error_and_a_retry_succeeds() -> None:
    def double(n: int) -> str:
        """Double a number."""
        return str(2 * n)

    model = Scripted(
        calls(("double", '{"n": "many"}')),
        calls(("double", '{"n": 21}')),
        says("42"),
    )

    events = await steps(Agent(model, Toolbox([tool(double)])), question())

    first, second = finished_calls(events)
    assert not first.ok
    assert first.result == (
        "invalid arguments for double: n: Input should be a valid integer, "
        "unable to parse string as an integer"
    )
    assert model.requests[1].messages[-1].text == first.result
    assert (second.ok, second.result) == (True, "42")
    assert end(events).outcome is Outcome.ANSWERED


async def test_an_unknown_tool_is_answered_not_fatal() -> None:
    model = Scripted(calls(("teleport", "{}")), says("cannot"))

    events = await steps(Agent(model, Toolbox()), question())

    (only,) = finished_calls(events)
    assert (only.ok, only.result) == (False, "there is no tool named teleport")
    assert end(events).text == "cannot"


async def test_a_crashing_tool_is_reported_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def broken() -> str:
        """Break."""
        message = "disk on fire"
        raise RuntimeError(message)

    model = Scripted(calls(("broken", "{}")), says("sorry"))

    with caplog.at_level(logging.ERROR, logger="synthia.agent.loop"):
        events = await steps(Agent(model, Toolbox([tool(broken)])), question())

    (only,) = finished_calls(events)
    # The exception's own text may hold anything, so the model sees only its type.
    assert (only.ok, only.result) == (False, "broken failed: RuntimeError")
    assert "tool broken failed" in caplog.text
    assert "disk on fire" in caplog.text
    assert end(events).text == "sorry"


async def test_a_call_that_runs_too_long_is_stopped() -> None:
    stopped = asyncio.Event()

    async def forever() -> str:
        """Never end."""
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
        return ""

    model = Scripted(calls(("forever", "{}")), says("gave up"))
    agent = Agent(model, Toolbox([tool(forever)]), Limits(call_timeout_s=0.01))

    events = await steps(agent, question())

    (only,) = finished_calls(events)
    assert (only.ok, only.result) == (
        False,
        "forever did not finish within 0.01 s and was stopped",
    )
    assert stopped.is_set()


async def test_the_last_step_offers_no_tools_so_the_model_must_answer() -> None:
    def again() -> str:
        """Again."""
        return "more"

    def reply(request: ChatRequest) -> list[ChatChunk]:
        return calls(("again", "{}")) if request.tools else says("enough")

    model = Scripted(reply, reply, reply)
    agent = Agent(model, Toolbox([tool(again)]), Limits(max_steps=3))

    events = await steps(agent, question())

    assert [bool(r.tools) for r in model.requests] == [True, True, False]
    assert end(events).text == "enough"
    assert end(events).outcome is Outcome.STEP_LIMIT


async def test_one_step_allowed_is_a_plain_answer() -> None:
    model = Scripted(says("hi"))

    events = await steps(Agent(model, Toolbox(), Limits(max_steps=1)), question())

    assert model.requests[0].tools == ()
    assert end(events).outcome is Outcome.ANSWERED


async def test_the_deadline_ends_the_run_before_the_next_step() -> None:
    now = [0.0]

    def slow_clock_tool() -> str:
        """Use up time."""
        now[0] += 301.0
        return "late"

    model = Scripted(calls(("slow_clock_tool", "{}")))
    agent = Agent(model, Toolbox([tool(slow_clock_tool)]), clock=lambda: now[0])

    events = await steps(agent, question())

    assert len(model.requests) == 1
    assert end(events).outcome is Outcome.DEADLINE
    assert end(events).messages[-1].role is Role.TOOL


async def test_a_model_slower_than_the_deadline_is_stopped() -> None:
    class Stalled(Scripted):
        async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
            self.requests.append(request)
            await asyncio.Event().wait()
            yield ChatChunk()

    agent = Agent(Stalled(), Toolbox(), Limits(deadline_s=0.01))

    events = await steps(agent, question())

    assert end(events).outcome is Outcome.DEADLINE
    assert end(events).text == ""


async def test_cancelling_the_run_cancels_the_running_tools() -> None:
    running, cancelled = asyncio.Event(), asyncio.Event()

    async def wait() -> str:
        """Wait."""
        running.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return ""

    model = Scripted(calls(("wait", "{}")))
    agent = Agent(model, Toolbox([tool(wait)]))

    async def consume() -> None:
        async with aclosing(agent.run(question())) as run:
            async for _ in run:
                pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(running.wait(), HANG_TIMEOUT_S)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


@pytest.mark.parametrize(
    "limits",
    [{"max_steps": 0}, {"deadline_s": 0.0}, {"call_timeout_s": -1.0}],
)
def test_limits_must_be_positive(limits: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        Limits(**limits)  # type: ignore[arg-type]
