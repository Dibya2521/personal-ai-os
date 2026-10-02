import asyncio
import itertools
import logging
from contextlib import aclosing

import pytest

from synthia.agent.loop import Agent, Finished, Limits, Step, ToolFinished
from synthia.agent.policy import Decision, Policy, nobody_approves
from synthia.agent.tools import Effect, FunctionTool, Reach, Tool, Toolbox
from synthia.gateway.types import ChatRequest, Message
from tests.agent.scripted import Scripted, calls, says
from tests.timing import HANG_TIMEOUT_S


def recorded(ran: list[str], reach: Reach, effect: Effect) -> FunctionTool:
    def act(target: str) -> str:
        """Act on a target."""
        ran.append(target)
        return f"did {target}"

    return FunctionTool.of(act, reach=reach, effect=effect)


async def run(agent: Agent) -> list[Step]:
    async def gather() -> list[Step]:
        async with aclosing(agent.run(ChatRequest((Message.user("go"),)))) as steps:
            return [s async for s in steps]

    return await asyncio.wait_for(gather(), HANG_TIMEOUT_S)


def results(events: list[Step]) -> list[ToolFinished]:
    return [e for e in events if isinstance(e, ToolFinished)]


@pytest.mark.parametrize(
    ("reach", "effect", "decision"),
    [
        (Reach.LOCAL, Effect.READ, Decision.ALLOW),
        (Reach.LOCAL, Effect.CHANGE, Decision.ASK),
        (Reach.OUTSIDE, Effect.READ, Decision.ASK),
        (Reach.OUTSIDE, Effect.CHANGE, Decision.ASK),
    ],
)
def test_only_a_local_look_runs_without_asking(
    reach: Reach, effect: Effect, decision: Decision
) -> None:
    assert Policy().decide(recorded([], reach, effect)) is decision


@pytest.mark.parametrize(("reach", "effect"), list(itertools.product(Reach, Effect)))
def test_a_denied_name_is_denied_whatever_it_declares(
    reach: Reach, effect: Effect
) -> None:
    assert Policy(denied=frozenset({"act"})).decide(recorded([], reach, effect)) is (
        Decision.DENY
    )


async def test_nobody_approves_says_no() -> None:
    assert await nobody_approves(recorded([], Reach.LOCAL, Effect.READ), "{}") is False


@pytest.mark.parametrize(
    ("reach", "effect", "denied", "answer"),
    list(itertools.product(Reach, Effect, [False, True], [False, True])),
)
async def test_a_call_runs_only_when_allowed_or_approved(
    reach: Reach, effect: Effect, denied: bool, answer: bool
) -> None:
    ran: list[str] = []
    asked: list[tuple[str, str]] = []

    async def approver(tool: Tool, arguments: str) -> bool:
        asked.append((tool.spec.name, arguments))
        return answer

    tool = recorded(ran, reach, effect)
    policy = Policy(denied=frozenset({"act"}) if denied else frozenset())
    model = Scripted(calls(("act", '{"target": "x"}')), says("done"))

    events = await run(Agent(model, Toolbox([tool]), policy=policy, approver=approver))

    needs_asking = not denied and (reach is Reach.OUTSIDE or effect is Effect.CHANGE)
    should_run = not denied and (not needs_asking or answer)
    assert ran == (["x"] if should_run else [])
    assert asked == ([("act", '{"target": "x"}')] if needs_asking else [])
    (finished,) = results(events)
    if should_run:
        assert (finished.ok, finished.result) == (True, "did x")
    elif denied:
        assert (finished.ok, finished.result) == (False, "act is not permitted")
    else:
        assert (finished.ok, finished.result) == (False, "running act was not approved")


async def test_without_an_approver_a_change_never_runs() -> None:
    ran: list[str] = []
    model = Scripted(calls(("act", '{"target": "x"}')), says("ok"))

    events = await run(
        Agent(model, Toolbox([recorded(ran, Reach.LOCAL, Effect.CHANGE)]))
    )

    assert ran == []
    assert results(events)[0].result == "running act was not approved"


async def test_a_model_asking_again_and_again_never_gets_a_denied_tool() -> None:
    ran: list[str] = []
    model = Scripted(*(calls(("act", '{"target": "x"}')) for _ in range(4)), says("ok"))
    agent = Agent(
        model,
        Toolbox([recorded(ran, Reach.LOCAL, Effect.READ)]),
        Limits(max_steps=5),
        policy=Policy(denied=frozenset({"act"})),
    )

    events = await run(agent)

    assert ran == []
    assert [r.result for r in results(events)] == ["act is not permitted"] * 4
    assert isinstance(events[-1], Finished)
    # The refusal reaches the model as the call's result.
    assert model.requests[1].messages[-1].text == "act is not permitted"


async def test_a_failing_approver_counts_as_no_and_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ran: list[str] = []

    async def broken(_tool: Tool, _arguments: str) -> bool:
        message = "terminal gone"
        raise OSError(message)

    model = Scripted(calls(("act", '{"target": "x"}')), says("ok"))

    with caplog.at_level(logging.ERROR, logger="synthia.agent.loop"):
        events = await run(
            Agent(
                model,
                Toolbox([recorded(ran, Reach.OUTSIDE, Effect.READ)]),
                approver=broken,
            )
        )

    assert ran == []
    assert results(events)[0].result == "running act was not approved"
    assert "asking to run act failed" in caplog.text


async def test_time_spent_deciding_is_not_the_tools_time() -> None:
    decided = asyncio.Event()

    async def instant(target: str) -> str:
        """Finish without ever waiting, so only time outside it can time it out."""
        return target

    async def slow_yes(_tool: Tool, _arguments: str) -> bool:
        # Five times the call's own limit: a person takes their time.
        await asyncio.sleep(0.05)
        decided.set()
        return True

    tool = FunctionTool.of(instant, reach=Reach.LOCAL, effect=Effect.CHANGE)
    model = Scripted(calls(("instant", '{"target": "x"}')), says("ok"))
    agent = Agent(
        model, Toolbox([tool]), Limits(call_timeout_s=0.01), approver=slow_yes
    )

    events = await run(agent)

    assert decided.is_set()
    assert (results(events)[0].ok, results(events)[0].result) == (True, "x")
