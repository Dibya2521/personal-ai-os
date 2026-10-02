"""Outside coding agents found on PATH, each offered as one tool, on command.

An outside agent is another program that answers with a cloud model (Claude
Code, Gemini CLI). Each one found on PATH becomes a tool
``ask_<name>`` that runs it once, without a session, on the task the model
gives. It is OUTSIDE and CHANGE, so every call asks first: the task leaves
the machine, and the agent may change files in the folder it starts in.
None that are missing appear, and none of them is needed for SYNTHIA to work.

The agent runs with its own default permissions, which in its
non-interactive mode do not approve its own edits, and without any
``SYNTHIA_*`` variable. What it prints on standard error (progress) goes to
``<home>/logs/agents/<name>.log``; what it prints on standard output is the
answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING, Annotated, Final

from pydantic import Field

from synthia.agent.tools import Effect, FunctionTool, Reach
from synthia.tools.run import (
    describe_run,
    environment_without_own,
    open_log,
    run_once,
)

if TYPE_CHECKING:
    from collections.abc import Callable

AGENT_LOGS: Final = Path("logs") / "agents"
# A coding task takes minutes; past ten the person is better asked again.
AGENT_TIME_LIMIT_S: Final = 600.0
MAX_TASK_CHARS: Final = 8000
MAX_ANSWER_BYTES: Final = 20_000


@dataclass(frozen=True, slots=True)
class OutsideAgent:
    """How to give one task to an agent: fixed arguments, the task on stdin.

    The task never goes on the command line: on Windows an npm-installed
    agent is a ``.cmd`` file, whose arguments ``cmd.exe`` parses again, so a
    task holding ``&`` could run a command of its own.
    """

    name: str
    title: str
    program: str
    arguments: tuple[str, ...]


ON_STDIN: Final = "Do the task given above."
# Both read the task from stdin with -p (Gemini CLI's --help: the -p text is
# "Appended to input on stdin"). opencode and codex are left out until their
# stdin form has been checked on a machine that has them.
KNOWN_AGENTS: Final = (
    OutsideAgent("claude", "Claude Code", "claude", ("-p", ON_STDIN)),
    OutsideAgent("gemini", "Gemini CLI", "gemini", ("-p", ON_STDIN)),
)


def agent_tools(
    home: Path,
    *,
    cwd: Path | None = None,
    find: Callable[[str], str | None] | None = None,
    agents: tuple[OutsideAgent, ...] = KNOWN_AGENTS,
    time_limit_s: float = AGENT_TIME_LIMIT_S,
) -> list[FunctionTool]:
    """Return a tool for every agent in ``agents`` that ``find`` locates on PATH.

    Each agent works in ``cwd``, the current directory when not given.
    """
    work = Path.cwd() if cwd is None else cwd
    locate = which if find is None else find
    tools: list[FunctionTool] = []
    for agent in agents:
        path = locate(agent.program)
        if path is not None:
            tools.append(_tool(agent, path, work, home / AGENT_LOGS, time_limit_s))
    return tools


def _tool(
    agent: OutsideAgent, path: str, work: Path, log_dir: Path, time_limit_s: float
) -> FunctionTool:
    async def ask(
        task: Annotated[
            str,
            Field(
                min_length=1,
                max_length=MAX_TASK_CHARS,
                description="the whole task, written for someone with no context",
            ),
        ],
    ) -> str:
        with open_log(log_dir, agent.name) as log:
            run = await run_once(
                [path, *agent.arguments],
                cwd=work,
                env=environment_without_own({}),
                stdin=task.encode(),
                timeout_s=time_limit_s,
                max_output_bytes=MAX_ANSWER_BYTES,
                errors=log,
            )
        return describe_run(run, time_limit_s, MAX_ANSWER_BYTES)

    ask.__doc__ = (
        f"Give a task to {agent.title}, an outside coding agent that uses a cloud "
        f"model and works in {work}; return its answer.\n\n"
        "Use it only for work you cannot do with your own tools."
    )
    return FunctionTool.of(
        ask,
        reach=Reach.OUTSIDE,
        effect=Effect.CHANGE,
        name=f"ask_{agent.name}",
        time_limit_s=time_limit_s,
    )
