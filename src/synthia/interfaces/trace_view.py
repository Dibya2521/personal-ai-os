"""``synthia trace``: show what the agent did in a chat, as a tree.

Each turn is a branch; under it, each model step and, under the step, the
tool calls it asked for with what came back. A planned turn shows its plan,
then each plan step with its model steps, then the answer written from them.
A call that began and has no end was still running when the turn stopped.
The daemon's trace shows each start, end and restart of a supervised service.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
from rich.tree import Tree

from synthia.agent.trace import (
    ModelAnswered,
    PlanAnswerStarted,
    PlanMade,
    PlanStepStarted,
    ServiceChanged,
    ToolBegan,
    ToolEnded,
    TurnBegan,
    TurnEnded,
    TurnFailed,
    read_trace,
)
from synthia.interfaces.chat import shorten

if TYPE_CHECKING:
    from datetime import tzinfo
    from pathlib import Path

    from synthia.agent.trace import Record


def _preview(text: str) -> str:
    """Return ``text`` on one line, shortened, so each node stays one line."""
    return shorten(" ".join(text.split()))


def _turn(record: TurnBegan, tz: tzinfo | None) -> str:
    at = record.at.astimezone(tz).strftime("%H:%M:%S")
    where = "remote allowed" if record.use_remote else "local only"
    thinking = (
        f"thinking {record.reasoning}" if record.reasoning else "no thinking level"
    )
    return (
        f"turn {record.turn} at {at} | you: {_preview(record.question.text)} | "
        f"{record.persona}, {thinking}, {where}"
    )


def _model(record: ModelAnswered) -> str:
    if record.prompt_tokens is None or record.completion_tokens is None:
        tokens = "tokens not reported"
    else:
        tokens = f"{record.prompt_tokens} in, {record.completion_tokens} out"
    thought = (
        [f"thought {record.reasoning_chars} characters"]
        if record.reasoning_chars
        else []
    )
    level = f", thinking {record.thinking}" if record.thinking else ""
    parts = [
        f"model {record.route} {record.model or 'unnamed'}{level}",
        record.finish,
        tokens,
        *thought,
        f"{record.seconds:.1f} s",
    ]
    return " | ".join(parts)


def _plan(record: PlanMade) -> str:
    made = "planned again" if record.revised else "planned"
    steps = " | ".join(f"{n}. {s}" for n, s in enumerate(record.steps, 1))
    return f"{made}: {_preview(steps) or 'no steps left'}"


def _service(record: ServiceChanged, tz: tzinfo | None) -> str:
    at = record.at.astimezone(tz).strftime("%H:%M:%S")
    named = f"service {record.service}"
    match record.change:
        case "started":
            again = f", attempt {record.attempt}" if (record.attempt or 1) > 1 else ""
            return f"{named} started at {at}{again}"
        case "exited":
            why = f": {_preview(record.error.text)}" if record.error else ""
            after = (
                "not restarted"
                if record.restart_in_s is None
                else f"restart in {record.restart_in_s:.1f} s"
            )
            return f"{named} ended at {at}{why} | {after}"
        case _:
            return f"{named} stopped at {at}"


def _call(record: ToolBegan) -> str:
    return f"tool {record.name} {_preview(record.arguments.text or '{}')}"


def _end(calls: dict[str, tuple[Tree, str]], step: Tree, record: ToolEnded) -> None:
    # The loop always records a start first; if that line was lost, the end
    # still shows, under the latest step.
    node, began = calls.pop(record.call_id, (None, f"tool {record.name}"))
    if node is None:
        node = step.add("")
    outcome = "ok" if record.ok else "failed"
    flagged = f" | flagged: {', '.join(record.flags)}" if record.flags else ""
    node.label = Text(f"{began} | {outcome}{flagged} | {record.seconds:.1f} s")
    node.add(Text(_preview(record.result.text)))


class _Branches:
    """Where the next record goes while a session's records are read in order."""

    def __init__(self, root: Tree, tz: tzinfo | None) -> None:
        self._root = root
        self._tz = tz
        self._turn = root
        # The turn, or the plan step being worked on: where model steps go.
        self._section = root
        self._step = root
        self._calls: dict[str, tuple[Tree, str]] = {}

    def add(self, record: Record) -> None:
        match record:
            case TurnBegan():
                self._turn = self._section = self._step = self._root.add(
                    Text(_turn(record, self._tz))
                )
                self._calls = {}
            case ModelAnswered():
                self._step = self._section.add(Text(_model(record)))
            case ToolBegan():
                began = _call(record)
                node = self._step.add(Text(f"{began} | did not finish"))
                self._calls[record.call_id] = (node, began)
            case ToolEnded():
                _end(self._calls, self._step, record)
            case TurnEnded():
                self._turn.add(Text(f"{record.outcome} | {_preview(record.text.text)}"))
            case TurnFailed():
                self._turn.add(Text(f"failed | {_preview(record.reason)}"))
            case ServiceChanged():
                self._root.add(Text(_service(record, self._tz)))
            case _:
                self._plan(record)

    def _plan(self, record: PlanMade | PlanStepStarted | PlanAnswerStarted) -> None:
        match record:
            case PlanMade():
                self._turn.add(Text(_plan(record)))
            case PlanStepStarted():
                text = f"step {record.number}: {_preview(record.text)}"
                self._section = self._step = self._turn.add(Text(text))
            case _:
                self._section = self._step = self._turn.add(
                    Text("answer from the steps")
                )


def build_tree(
    name: str, records: list[Record], unreadable: int = 0, tz: tzinfo | None = None
) -> Tree:
    """Return the tree of the session ``name``, with times in ``tz`` (local if None)."""
    root = Tree(Text(f"session {name}"))
    branches = _Branches(root, tz)
    for record in records:
        branches.add(record)
    if unreadable:
        lines = "line" if unreadable == 1 else "lines"
        root.add(Text(f"{unreadable} {lines} could not be read"))
    return root


def summary(path: Path) -> str:
    """Return one line about the trace at ``path``: its name, turns and tool calls.

    The daemon's trace also counts the restarts of the services it supervises.
    """
    records, _ = read_trace(path)
    turns = sum(isinstance(r, TurnBegan) for r in records)
    calls = sum(isinstance(r, ToolBegan) for r in records)
    line = f"{path.stem} | turns {turns} | tool calls {calls}"
    changes = [r for r in records if isinstance(r, ServiceChanged)]
    if changes:
        restarts = sum(r.change == "started" and (r.attempt or 1) > 1 for r in changes)
        line += f" | service restarts {restarts}"
    return line
