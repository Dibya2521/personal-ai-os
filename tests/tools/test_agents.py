import json
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from synthia.agent.tools import Effect, Reach, ToolError
from synthia.tools.agents import (
    AGENT_TIME_LIMIT_S,
    KNOWN_AGENTS,
    ON_STDIN,
    OutsideAgent,
    agent_tools,
)

FAKE = OutsideAgent(
    "fake",
    "Fake Agent",
    "fake-agent",
    (str(Path(__file__).with_name("fake_agent.py")), "-p", ON_STDIN),
)


def found(*programs: str) -> Callable[[str], str | None]:
    def find(program: str) -> str | None:
        return f"/bin/{program}" if program in programs else None

    return find


def test_every_agent_found_becomes_a_tool_that_asks_and_may_take_minutes(
    tmp_path: Path,
) -> None:
    tools = agent_tools(tmp_path, find=found("claude", "gemini"))

    assert [(t.spec.name, t.reach, t.effect, t.time_limit_s) for t in tools] == [
        ("ask_claude", Reach.OUTSIDE, Effect.CHANGE, AGENT_TIME_LIMIT_S),
        ("ask_gemini", Reach.OUTSIDE, Effect.CHANGE, AGENT_TIME_LIMIT_S),
    ]
    assert tools[0].spec.parameters["required"] == ["task"]  # type: ignore[index]
    assert tools[0].spec.description.startswith(
        "Give a task to Claude Code, an outside"
    )


def test_an_agent_that_is_not_installed_is_not_offered(tmp_path: Path) -> None:
    assert [t.spec.name for t in agent_tools(tmp_path, find=found("gemini"))] == [
        "ask_gemini"
    ]
    assert agent_tools(tmp_path, find=found()) == []


def test_every_known_agent_gets_its_task_on_stdin_never_as_an_argument() -> None:
    assert [(a.program, a.arguments) for a in KNOWN_AGENTS] == [
        ("claude", ("-p", ON_STDIN)),
        ("gemini", ("-p", ON_STDIN)),
    ]


async def test_the_task_goes_in_on_stdin_and_the_answer_comes_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SYNTHIA_OPENROUTER_API_KEY", "not-a-real-key")
    work = tmp_path / "work"
    work.mkdir()
    (tool,) = agent_tools(
        tmp_path, cwd=work, find=lambda _: sys.executable, agents=(FAKE,)
    )
    task = 'fix "it" & echo pwned | more'

    answer = json.loads(await tool.run(json.dumps({"task": task})))

    assert answer == {
        "args": ["-p", ON_STDIN],
        "cwd": str(work),
        "own": [],
        "task": task,
    }
    assert (tmp_path / "logs" / "agents" / "fake.log").read_text().split() == [
        "working"
    ]


async def test_an_agent_past_its_time_limit_is_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SLEEP", "60")
    (tool,) = agent_tools(
        tmp_path, find=lambda _: sys.executable, agents=(FAKE,), time_limit_s=1
    )

    with pytest.raises(
        ToolError, match=r"^stopped after 1 s; printed so far:\n\(nothing printed\)$"
    ):
        await tool.run('{"task": "wait"}')
