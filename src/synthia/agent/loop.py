"""The agent loop: ask the model, run the tools it calls, repeat until it answers.

Every call the model makes in one answer runs at the same time, and their
results go back in the order the calls were made. A failing tool never ends
the loop: its error goes back to the model as the result, so the model can try
again or answer without it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from contextlib import aclosing
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from synthia.agent.policy import Decision, Policy, nobody_approves
from synthia.agent.quoting import flags, quote
from synthia.agent.tools import ToolError
from synthia.gateway.protocol import join
from synthia.gateway.types import Message

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from synthia.agent.policy import Approver
    from synthia.agent.tools import Tool, Toolbox
    from synthia.gateway.protocol import ChatModel
    from synthia.gateway.types import ChatChunk, ChatRequest, ChatResponse, ToolCall

logger = logging.getLogger(__name__)

# Derived, not measured: a few tools with one repair each.
DEFAULT_MAX_STEPS: Final = 8
# The local model's slow end per step (about 16 s at 2 threads) times the
# steps, rounded down to the most a person waits.
DEFAULT_DEADLINE_S: Final = 300.0
DEFAULT_CALL_TIMEOUT_S: Final = 60.0


class Outcome(StrEnum):
    """Why the loop stopped."""

    ANSWERED = "answered"
    STEP_LIMIT = "step limit"
    DEADLINE = "deadline"


@dataclass(frozen=True, slots=True)
class Limits:
    """How far one run may go.

    ``max_steps`` counts model turns; the last one is offered no tools, so the
    model has to answer with what it has. ``call_timeout_s`` bounds a tool
    call unless the tool sets its own ``time_limit_s``. ``deadline_s`` of None sets no
    deadline: right for a chat, where the answer streams in view and the
    person stops it when they choose.

    Raises:
        ValueError: If a limit is not positive.
    """

    max_steps: int = DEFAULT_MAX_STEPS
    deadline_s: float | None = DEFAULT_DEADLINE_S
    call_timeout_s: float = DEFAULT_CALL_TIMEOUT_S

    def __post_init__(self) -> None:
        if (
            self.max_steps < 1
            or (self.deadline_s is not None and self.deadline_s <= 0)
            or self.call_timeout_s <= 0
        ):
            message = "every agent limit must be positive"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class ModelChunk:
    """A piece of the model's answer as it streams."""

    chunk: ChatChunk


@dataclass(frozen=True, slots=True)
class ModelTurn:
    """The model's whole answer for one step."""

    response: ChatResponse
    seconds: float


@dataclass(frozen=True, slots=True)
class ToolStarted:
    """A call began."""

    call: ToolCall


@dataclass(frozen=True, slots=True)
class ToolFinished:
    """A call ended.

    ``result`` is the tool's own text; the model receives it quoted as
    untrusted data. ``flags`` names what in it looks like an instruction.
    """

    call: ToolCall
    result: str
    ok: bool
    seconds: float
    flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Finished:
    """The run ended with ``text`` as the answer."""

    text: str
    outcome: Outcome
    messages: tuple[Message, ...]


type Step = ModelChunk | ModelTurn | ToolStarted | ToolFinished | Finished


class Agent:
    """Runs a request to its answer, calling tools from ``toolbox`` on the way.

    The request's own settings (thinking level, permission to use the remote)
    apply to every step, so an agent never leaves the machine unless the
    request allows it. ``policy`` decides each call before it runs; a call it
    asks about runs only if ``approver`` says yes, and the default approver
    always says no.
    """

    def __init__(  # noqa: PLR0913
        self,
        model: ChatModel,
        toolbox: Toolbox,
        limits: Limits | None = None,
        *,
        policy: Policy | None = None,
        approver: Approver = nobody_approves,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._model = model
        self._toolbox = toolbox
        self._limits = limits or Limits()
        self._policy = policy or Policy()
        self._approver = approver
        self._clock = clock

    async def run(self, request: ChatRequest) -> AsyncGenerator[Step]:
        """Yield every step of answering ``request``, ending with :class:`Finished`.

        Cancelling the consumer cancels the tools still running.

        Raises:
            GatewayError: What the model raised.
        """
        limit = self._limits.deadline_s
        deadline = None if limit is None else self._clock() + limit
        messages = request.messages
        for step in range(1, self._limits.max_steps + 1):
            last = step == self._limits.max_steps
            tools = () if last else self._toolbox.specs()
            ask = dataclasses.replace(request, messages=messages, tools=tools)
            response: ChatResponse | None = None
            async for event in self._turn(ask, deadline):
                if isinstance(event, ModelTurn):
                    response = event.response
                yield event
            if response is None:
                yield Finished("", Outcome.DEADLINE, messages)
                return
            messages = (*messages, response.as_message())
            if not response.tool_calls:
                outcome = Outcome.STEP_LIMIT if last and step > 1 else Outcome.ANSWERED
                yield Finished(response.text, outcome, messages)
                return
            results: list[ToolFinished | None] = [None] * len(response.tool_calls)
            async for event in self._calls(response.tool_calls, results):
                yield event
            messages = (
                *messages,
                *(
                    Message.tool_result(r.call.id, quote(r.call.name, r.result))
                    for r in results
                    if r
                ),
            )
        # Unreachable: the last step offers no tools, so it ends above.
        raise AssertionError  # pragma: no cover

    async def _turn(
        self, ask: ChatRequest, deadline: float | None
    ) -> AsyncGenerator[ModelChunk | ModelTurn]:
        """Stream one model answer; no :class:`ModelTurn` if the deadline came first."""
        remaining = None if deadline is None else deadline - self._clock()
        if remaining is not None and remaining <= 0:
            return
        started = self._clock()
        chunks: list[ChatChunk] = []
        try:
            async with asyncio.timeout(remaining):
                async with aclosing(self._model.stream(ask)) as stream:
                    async for chunk in stream:
                        chunks.append(chunk)
                        yield ModelChunk(chunk)
        except TimeoutError:
            return
        yield ModelTurn(join(chunks), self._clock() - started)

    async def _calls(
        self, calls: tuple[ToolCall, ...], results: list[ToolFinished | None]
    ) -> AsyncGenerator[ToolStarted | ToolFinished]:
        """Run ``calls`` at once, yield each as it ends, fill ``results`` in order."""
        for call in calls:
            yield ToolStarted(call)

        async def place(index: int, call: ToolCall) -> ToolFinished:
            results[index] = finished = await self._call(call)
            return finished

        tasks = [asyncio.create_task(place(i, c)) for i, c in enumerate(calls)]
        try:
            for done in asyncio.as_completed(tasks):
                yield await done
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _call(self, call: ToolCall) -> ToolFinished:
        tool = self._toolbox.get(call.name)
        refusal = (
            f"there is no tool named {call.name}"
            if tool is None
            else await self._refusal(tool, call.arguments)
        )
        # Started after any approval, so a person's time to answer is not
        # counted against the tool's own limit.
        started = self._clock()
        ok = False
        if tool is None or refusal is not None:
            result = refusal or ""
        else:
            limit = tool.time_limit_s or self._limits.call_timeout_s
            try:
                async with asyncio.timeout(limit):
                    result = await tool.run(call.arguments)
                ok = True
            except ToolError as error:
                result = str(error)
            except TimeoutError:
                result = (
                    f"{call.name} did not finish within {limit:g} s and was stopped"
                )
            except Exception as error:
                logger.exception("tool %s failed", call.name)
                result = f"{call.name} failed: {type(error).__name__}"
        return ToolFinished(call, result, ok, self._clock() - started, flags(result))

    async def _refusal(self, tool: Tool, arguments: str) -> str | None:
        """Return why ``tool`` may not run with ``arguments``, or None if it may."""
        name = tool.spec.name
        decision = self._policy.decide(tool)
        if decision is Decision.DENY:
            return f"{name} is not permitted"
        if decision is Decision.ASK:
            try:
                approved = await self._approver(tool, arguments)
            except Exception:
                logger.exception("asking to run %s failed", name)
                approved = False
            if not approved:
                return f"running {name} was not approved"
        return None
