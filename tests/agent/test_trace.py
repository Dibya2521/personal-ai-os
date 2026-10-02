import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from synthia.agent.tools import Effect, FunctionTool, Reach, Toolbox
from synthia.agent.trace import (
    MAX_TRACED_CHARS,
    Clip,
    ModelAnswered,
    Trace,
    TurnFailed,
    read_trace,
    sessions,
)
from synthia.gateway.errors import ProviderError
from synthia.gateway.protocol import ChatModel
from synthia.gateway.router import Route, RouteDecided, RouteReason
from synthia.gateway.types import ChatChunk, ChatRequest, ModelInfo, Reasoning
from synthia.interfaces.session import ChatSession
from synthia.persona.library import PersonaLibrary
from tests.agent.scripted import Scripted, calls, says

AT = datetime(2026, 10, 2, 15, 15, tzinfo=UTC)


def at() -> datetime:
    return AT


def note_tool(ran: list[str]) -> FunctionTool:
    def note(text: str) -> str:
        """Note something down."""
        ran.append(text)
        return f"noted {text}"

    return FunctionTool.of(note, reach=Reach.LOCAL, effect=Effect.READ)


def traced(model: ChatModel, path: Path, *tools: FunctionTool) -> ChatSession:
    return ChatSession(
        model,
        PersonaLibrary(),
        "synthia",
        tools=Toolbox(tools),
        trace=Trace(path, at),
    )


async def drain(session: ChatSession, text: str) -> None:
    async for _ in session.turn(text):
        pass


def dumps(path: Path) -> list[dict[str, object]]:
    records, unreadable = read_trace(path)
    assert unreadable == 0
    return [r.model_dump(mode="json", exclude={"seconds"}) for r in records]


async def test_a_turn_with_a_tool_call_is_recorded_step_by_step(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s.jsonl"
    ran: list[str] = []
    model = Scripted(calls(("note", '{"text": "milk"}')), says("Noted."))

    await drain(traced(model, path, note_tool(ran)), "note milk")

    stamp = "2026-10-02T15:15:00Z"
    step = {"turn": 1, "at": stamp}
    unused = {
        "thinking": None,
        "model": None,
        "prompt_tokens": None,
        "completion_tokens": None,
    }
    assert dumps(path) == [
        step
        | {
            "kind": "turn",
            "question": {"text": "note milk", "chars": 9},
            "persona": "SYNTHIA",
            "reasoning": "auto",
            "use_remote": False,
        },
        step
        | unused
        | {
            "kind": "model",
            "route": "direct",
            "text": {"text": "", "chars": 0},
            "reasoning_chars": 0,
            "finish": "tool_calls",
        },
        step
        | {
            "kind": "tool_started",
            "call_id": "call_0",
            "name": "note",
            "arguments": {"text": '{"text": "milk"}', "chars": 16},
        },
        step
        | {
            "kind": "tool_finished",
            "call_id": "call_0",
            "name": "note",
            "result": {"text": "noted milk", "chars": 10},
            "ok": True,
            "flags": [],
        },
        step
        | unused
        | {
            "kind": "model",
            "route": "direct",
            "text": {"text": "Noted.", "chars": 6},
            "reasoning_chars": 0,
            "finish": "stop",
        },
        step
        | {
            "kind": "finished",
            "outcome": "answered",
            "text": {"text": "Noted.", "chars": 6},
        },
    ]
    assert ran == ["milk"]


async def test_turns_are_numbered_and_streamed_pieces_are_not_recorded(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s.jsonl"
    many = [ChatChunk(text=c) for c in "a long answer"] + says("")
    session = traced(Scripted(says("one"), many), path)

    await drain(session, "first")
    await drain(session, "second")

    kinds = [(r["turn"], r["kind"]) for r in dumps(path)]
    assert kinds == [
        (1, "turn"),
        (1, "model"),
        (1, "finished"),
        (2, "turn"),
        (2, "model"),
        (2, "finished"),
    ]


async def test_each_model_step_records_where_the_router_sent_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s.jsonl"

    def routed(request: ChatRequest) -> list[ChatChunk]:
        del request
        session.routes.decision = RouteDecided(
            route=Route.LOCAL,
            reason=RouteReason.LOCAL_FIRST,
            model="qwen3.5-4b",
            reasoning=Reasoning.OFF,
        )
        return [ChatChunk(model="qwen3.5-4b"), *says("hi")]

    session = traced(Scripted(routed), path)

    await drain(session, "hello")

    model = next(r for r in read_trace(path)[0] if isinstance(r, ModelAnswered))
    assert (model.route, model.thinking, model.model) == ("local", "off", "qwen3.5-4b")


class Failing:
    info = ModelInfo("failing", 1000, vision=False, tools=True)

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        del request
        yield ChatChunk(text="half")
        message = "the provider broke"
        raise ProviderError(message)


async def test_a_failed_turn_is_recorded_with_its_reason_and_still_raises(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s.jsonl"

    with pytest.raises(ProviderError):
        await drain(traced(Failing(), path), "hello")

    records, _ = read_trace(path)
    assert records[-1] == TurnFailed(
        turn=1, at=AT, reason="no answer: the provider broke"
    )


async def test_a_turn_stopped_while_a_tool_runs_is_recorded_as_stopped(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s.jsonl"
    entered = asyncio.Event()

    async def wait(text: str) -> str:
        """Wait for ever."""
        entered.set()
        await asyncio.Event().wait()
        return text

    tool = FunctionTool.of(wait, reach=Reach.LOCAL, effect=Effect.READ)
    session = traced(Scripted(calls(("wait", '{"text": "x"}'))), path, tool)
    turn = asyncio.create_task(drain(session, "go"))
    await asyncio.wait_for(entered.wait(), 1)

    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    kinds = [r["kind"] for r in dumps(path)]
    assert kinds == ["turn", "model", "tool_started", "failed"]
    assert dumps(path)[-1]["reason"] == "stopped"


async def test_a_trace_that_cannot_be_written_never_breaks_the_chat(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    blocked = tmp_path / "a file, not a folder"
    blocked.write_text("")
    session = traced(Scripted(says("one"), says("two")), blocked / "s.jsonl")

    with caplog.at_level(logging.ERROR, logger="synthia.agent.trace"):
        await drain(session, "first")
        await drain(session, "second")

    assert session.history[-1].text == "two"
    assert [r.message for r in caplog.records] == [
        "trace not written; tracing stops for this session"
    ]


def test_a_long_text_is_cut_and_keeps_its_full_length() -> None:
    exact = Clip.of("x" * MAX_TRACED_CHARS)
    over = Clip.of("x" * MAX_TRACED_CHARS + "yz")

    assert (len(exact.text), exact.chars) == (2000, 2000)
    assert (len(over.text), over.chars) == (2000, 2002)
    assert "y" not in over.text


async def test_half_an_emoji_from_the_model_is_traced_as_its_escape(
    tmp_path: Path,
) -> None:
    path = tmp_path / "s.jsonl"
    # json.loads is how the gateway decodes a model's stream, and it accepts a
    # lone surrogate escape, which UTF-8 cannot encode.
    broken = json.loads('"half \\ud83d"')
    session = traced(Scripted(says(broken)), path)

    await drain(session, "hello")

    assert session.history[-1].text == broken
    assert dumps(path)[-1]["text"] == {"text": "half \\ud83d", "chars": 6}


def test_a_cut_last_line_is_counted_and_the_rest_is_read(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    whole = TurnFailed(turn=1, at=AT, reason="stopped").model_dump_json()
    path.write_text(f"{whole}\n\n{whole}\n{whole[:25]}", encoding="utf-8")

    records, unreadable = read_trace(path)

    assert records == [TurnFailed(turn=1, at=AT, reason="stopped")] * 2
    assert unreadable == 1


def test_a_session_is_named_by_its_start_time_and_written_only_when_used(
    tmp_path: Path,
) -> None:
    trace = Trace.start(tmp_path / "traces", at)

    assert re.fullmatch(r"20261002T151500Z-[0-9a-f]{6}\.jsonl", trace.path.name)
    assert not (tmp_path / "traces").exists()
    assert Trace.start(tmp_path, at).path != trace.path


def test_sessions_are_listed_oldest_first(tmp_path: Path) -> None:
    for name in ("20261002T151500Z-bb", "20261001T090000Z-ff", "notes"):
        (tmp_path / f"{name}.jsonl").write_text("")
    (tmp_path / "other.txt").write_text("")

    assert [p.stem for p in sessions(tmp_path)] == [
        "20261001T090000Z-ff",
        "20261002T151500Z-bb",
        "notes",
    ]
    assert sessions(tmp_path / "missing") == []
