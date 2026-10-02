"""Plan and execute: ask for a plan first, then run each step as its own loop.

One request, with no tools offered, asks the model whether the task needs
several steps and which. Fewer than two means it does not, and the task
runs as a plain agent loop, so a simple request pays one extra model call
and nothing more. Otherwise each step runs as its own loop with the plan and
the results so far in view, and a last loop writes the answer. A step that
ends without an answer (step limit or deadline) leads to one new plan from
what is done; a second such step is not re-planned. A plan the model cannot
give in the right shape is no plan: the task runs as a plain loop.
"""

from __future__ import annotations

import dataclasses
from contextlib import aclosing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Final

from pydantic import BaseModel, ConfigDict, Field

from synthia.agent.loop import Finished, Outcome
from synthia.gateway.structured import DEFAULT_REPAIRS, StructuredOutputError, generate
from synthia.gateway.types import Message

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from synthia.agent.loop import Agent, Step
    from synthia.gateway.types import ChatRequest

MAX_PLAN_STEPS: Final = 6
MIN_PLAN_STEPS: Final = 2
MAX_STEP_CHARS: Final = 300
# A step's result is shown to later steps; past this it is cut, so six long
# results cannot fill the context window.
MAX_RESULT_CHARS: Final = 2000


class Plan(BaseModel):
    """The steps a task needs, in order; none or one means just do it."""

    model_config = ConfigDict(extra="forbid")

    steps: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=MAX_STEP_CHARS)]],
        Field(max_length=MAX_PLAN_STEPS),
    ]


@dataclass(frozen=True, slots=True)
class Planned:
    """A plan was made, or made again after a step failed."""

    steps: tuple[str, ...]
    revised: bool


@dataclass(frozen=True, slots=True)
class PlanStepBegun:
    """Step ``number`` (from 1) of the plan begins."""

    number: int
    text: str


type PlanEvent = Step | Planned | PlanStepBegun


@dataclass(frozen=True, slots=True)
class _Done:
    text: str
    result: str
    answered: bool


class Planner:
    """Runs a task as a plan of steps, each through ``agent``."""

    def __init__(self, agent: Agent, *, repairs: int = DEFAULT_REPAIRS) -> None:
        self._agent = agent
        self._repairs = repairs

    async def run(self, request: ChatRequest) -> AsyncGenerator[PlanEvent]:
        """Yield every step of answering ``request``, ending with :class:`Finished`.

        The final :class:`Finished` carries the request's messages plus the
        answer, so a conversation keeps none of the plan's own prompts.

        Raises:
            GatewayError: What the model raised.
        """
        steps = await self._plan(request, ())
        if steps is None or len(steps) < MIN_PLAN_STEPS:
            async for event in self._loop(request):
                yield event
            return
        yield Planned(steps, revised=False)
        done: list[_Done] = []
        remaining, revised = list(steps), False
        while remaining:
            text = remaining.pop(0)
            yield PlanStepBegun(len(done) + 1, text)
            async for event in self._work(
                self._step(request, steps, done, text), text, done
            ):
                yield event
            if not done[-1].answered and not revised:
                revised = True
                steps = await self._plan(request, done) or ()
                remaining = list(steps)
                yield Planned(steps, revised=True)
        async for event in self._loop(self._final(request, done)):
            if isinstance(event, Finished):
                yield Finished(
                    event.text,
                    event.outcome,
                    (*request.messages, Message.assistant(event.text)),
                )
            else:
                yield event

    async def _loop(self, request: ChatRequest) -> AsyncGenerator[Step]:
        async with aclosing(self._agent.run(request)) as events:
            async for event in events:
                yield event

    async def _work(
        self, request: ChatRequest, text: str, done: list[_Done]
    ) -> AsyncGenerator[Step]:
        """Run one step, yielding all but its end; its result is added to ``done``."""
        result, answered = "", False
        async for event in self._loop(request):
            if isinstance(event, Finished):
                result, answered = event.text, event.outcome is Outcome.ANSWERED
            else:
                yield event
        done.append(_Done(text, result, answered))

    async def _plan(
        self, request: ChatRequest, done: Sequence[_Done]
    ) -> tuple[str, ...] | None:
        """Return the steps the model plans, or None if it gave no valid plan."""
        tools = "\n".join(
            f"- {spec.name}: {spec.description}" for spec in self._agent.toolbox.specs()
        )
        prompt = (
            "Before doing anything, decide whether the request above needs "
            "several separate steps. If it does, list them in order, at most "
            f"{MAX_PLAN_STEPS}, each short and doable with these tools:\n"
            f"{tools or '- none'}\n"
            "If one step will do, give an empty list."
        )
        if done:
            prompt += f"\n\nA step did not finish. Done so far:\n{_results(done)}\n"
            prompt += "Plan only the steps still needed."
        ask = dataclasses.replace(
            request, messages=(*request.messages, Message.user(prompt)), tools=()
        )
        try:
            plan = await generate(self._agent.model, ask, Plan, repairs=self._repairs)
        except StructuredOutputError:
            return None
        return tuple(plan.steps)

    def _step(
        self,
        request: ChatRequest,
        steps: Sequence[str],
        done: Sequence[_Done],
        text: str,
    ) -> ChatRequest:
        plan = "\n".join(f"{n}. {s}" for n, s in enumerate(steps, 1))
        prompt = f"Work through this plan for the request above:\n{plan}\n"
        if done:
            prompt += f"\nDone so far:\n{_results(done)}\n"
        prompt += f"\nNow do only this step: {text}\nReply with what it found."
        return dataclasses.replace(
            request, messages=(*request.messages, Message.user(prompt))
        )

    def _final(self, request: ChatRequest, done: Sequence[_Done]) -> ChatRequest:
        prompt = (
            f"Each step of the plan found this:\n{_results(done)}\n\n"
            "Now answer the request above in full, using these results."
        )
        return dataclasses.replace(
            request, messages=(*request.messages, Message.user(prompt))
        )


def _results(done: Sequence[_Done]) -> str:
    return "\n".join(
        f"{n}. {d.text}: {_cut(d.result) or '(no result)'}"
        for n, d in enumerate(done, 1)
    )


def _cut(text: str) -> str:
    return text if len(text) <= MAX_RESULT_CHARS else text[:MAX_RESULT_CHARS] + " [cut]"
