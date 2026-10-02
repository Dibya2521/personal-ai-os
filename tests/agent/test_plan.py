import json

import pytest

from synthia.agent.loop import Agent, Finished, Limits, Outcome, ToolFinished
from synthia.agent.plan import (
    MAX_RESULT_CHARS,
    PlanEvent,
    Planned,
    Planner,
    PlanStepBegun,
)
from synthia.agent.tools import Effect, FunctionTool, Reach, Toolbox
from synthia.gateway.types import ChatChunk, ChatRequest, Message, Reasoning, Role
from tests.agent.scripted import Scripted, calls, says


def note(text: str) -> str:
    """Note something down."""
    return f"noted {text}"


def planned(*steps: str) -> list[ChatChunk]:
    return says(json.dumps({"steps": list(steps)}))


def question() -> ChatRequest:
    return ChatRequest(
        (Message.user("plan it"),), reasoning=Reasoning.LOW, use_remote=True
    )


def planner(model: Scripted, limits: Limits | None = None, repairs: int = 2) -> Planner:
    tools = Toolbox([FunctionTool.of(note, reach=Reach.LOCAL, effect=Effect.READ)])
    return Planner(Agent(model, tools, limits), repairs=repairs)


async def events(planner: Planner) -> list[PlanEvent]:
    return [event async for event in planner.run(question())]


def asked(model: Scripted, index: int) -> str:
    """Return the last user message of request ``index``."""
    last = model.requests[index].messages[-1]
    assert last.role is Role.USER
    return last.text


def only_finished(seen: list[PlanEvent]) -> Finished:
    (finished,) = [e for e in seen if isinstance(e, Finished)]
    return finished


@pytest.mark.parametrize("steps", [(), ("just answer",)], ids=["no-steps", "one-step"])
async def test_a_task_of_one_step_runs_as_a_plain_loop(steps: tuple[str, ...]) -> None:
    model = Scripted(planned(*steps), says("hi"))

    seen = await events(planner(model))

    assert only_finished(seen) == Finished(
        "hi", Outcome.ANSWERED, (*question().messages, Message.assistant("hi"))
    )
    assert not [e for e in seen if isinstance(e, Planned | PlanStepBegun)]
    plan, run = model.requests
    assert (plan.tools, plan.response_schema is not None) == ((), True)
    assert "- note: Note something down." in asked(model, 0)
    assert [t.name for t in run.tools] == ["note"]


async def test_a_plan_runs_step_by_step_and_ends_with_one_answer() -> None:
    model = Scripted(
        planned("find a", "use a"),
        calls(("note", '{"text": "a"}')),
        says("found a"),
        says("used a"),
        says("All done."),
    )

    seen = await events(planner(model))

    marks = [
        e.result if isinstance(e, ToolFinished) else e
        for e in seen
        if isinstance(e, Planned | PlanStepBegun | ToolFinished)
    ]
    assert marks == [
        Planned(("find a", "use a"), revised=False),
        PlanStepBegun(1, "find a"),
        "noted a",
        PlanStepBegun(2, "use a"),
    ]
    assert only_finished(seen) == Finished(
        "All done.",
        Outcome.ANSWERED,
        (*question().messages, Message.assistant("All done.")),
    )
    assert asked(model, 3) == (
        "Work through this plan for the request above:\n1. find a\n2. use a\n"
        "\nDone so far:\n1. find a: found a\n"
        "\nNow do only this step: use a\nReply with what it found."
    )
    assert asked(model, 4) == (
        "Each step of the plan found this:\n1. find a: found a\n2. use a: used a\n\n"
        "Now answer the request above in full, using these results."
    )


async def test_a_step_that_does_not_finish_is_planned_again_but_only_once() -> None:
    model = Scripted(
        planned("a", "b"),
        calls(("note", '{"text": "a"}')),
        says("partial a"),
        planned("c"),
        calls(("note", '{"text": "c"}')),
        says("partial c"),
        says("done"),
    )

    seen = await events(planner(model, Limits(max_steps=2)))

    assert [e for e in seen if isinstance(e, Planned | PlanStepBegun)] == [
        Planned(("a", "b"), revised=False),
        PlanStepBegun(1, "a"),
        Planned(("c",), revised=True),
        PlanStepBegun(2, "c"),
    ]
    assert asked(model, 3).endswith(
        "A step did not finish. Done so far:\n1. a: partial a\n"
        "Plan only the steps still needed."
    )
    assert only_finished(seen).text == "done"
    assert len(model.requests) == 7


async def test_a_new_plan_that_cannot_be_read_goes_straight_to_the_answer() -> None:
    model = Scripted(
        planned("a", "b"),
        calls(("note", '{"text": "a"}')),
        says("partial a"),
        says("not a plan"),
        says("done anyway"),
    )

    seen = await events(planner(model, Limits(max_steps=2), repairs=0))

    assert Planned((), revised=True) in seen
    assert only_finished(seen).text == "done anyway"
    assert asked(model, 4).startswith(
        "Each step of the plan found this:\n1. a: partial a\n"
    )


async def test_no_valid_plan_means_a_plain_loop() -> None:
    model = Scripted(says("I will not write JSON"), says("answer"))

    seen = await events(planner(model, repairs=0))

    assert only_finished(seen).text == "answer"
    assert len(model.requests) == 2


async def test_a_long_step_result_is_cut_before_later_steps_see_it() -> None:
    long = "x" * (MAX_RESULT_CHARS + 10)
    model = Scripted(planned("a", "b"), says(long), says(""), says("done"))

    await events(planner(model))

    assert f"1. a: {'x' * MAX_RESULT_CHARS} [cut]\n" in asked(model, 2)
    assert "2. b: (no result)" in asked(model, 3)
