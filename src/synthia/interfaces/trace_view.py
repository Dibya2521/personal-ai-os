"""``synthia trace``: show what the agent did in a chat, as a tree.

Each turn is a branch; under it, each model step and, under the step, the
tool calls it asked for with what came back. A call that began and has no end
was still running when the turn stopped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
from rich.tree import Tree

from synthia.agent.trace import (
    ModelAnswered,
    ToolBegan,
    ToolEnded,
    TurnBegan,
    TurnEnded,
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


def build_tree(
    name: str, records: list[Record], unreadable: int = 0, tz: tzinfo | None = None
) -> Tree:
    """Return the tree of the session ``name``, with times in ``tz`` (local if None)."""
    root = Tree(Text(f"session {name}"))
    turn = root
    step = root
    calls: dict[str, tuple[Tree, str]] = {}
    for record in records:
        match record:
            case TurnBegan():
                turn = step = root.add(Text(_turn(record, tz)))
                calls = {}
            case ModelAnswered():
                step = turn.add(Text(_model(record)))
            case ToolBegan():
                began = _call(record)
                calls[record.call_id] = (
                    step.add(Text(f"{began} | did not finish")),
                    began,
                )
            case ToolEnded():
                _end(calls, step, record)
            case TurnEnded():
                turn.add(Text(f"{record.outcome} | {_preview(record.text.text)}"))
            case _:
                turn.add(Text(f"failed | {_preview(record.reason)}"))
    if unreadable:
        lines = "line" if unreadable == 1 else "lines"
        root.add(Text(f"{unreadable} {lines} could not be read"))
    return root


def summary(path: Path) -> str:
    """Return one line about the trace at ``path``: its name, turns and tool calls."""
    records, _ = read_trace(path)
    turns = sum(isinstance(r, TurnBegan) for r in records)
    calls = sum(isinstance(r, ToolBegan) for r in records)
    return f"{path.stem} | turns {turns} | tool calls {calls}"
