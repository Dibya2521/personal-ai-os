import json
import sys
from pathlib import Path

import httpx
import pytest

from synthia.agent.loop import ToolFinished
from synthia.agent.plan import PlanAnswerBegun, Planned, PlanStepBegun
from synthia.agent.policy import Policy
from synthia.agent.quoting import quote
from synthia.agent.tools import Effect, FunctionTool, Reach, Toolbox
from synthia.gateway.types import ChatChunk, Role, ToolCall, Usage
from synthia.interfaces.chat import (
    MAX_SHOWN,
    approval_question,
    describe_call,
    describe_mark,
    run_chat,
    terminal_approver,
)
from synthia.interfaces.commands import Invalid, ShowTools, parse
from synthia.persona.library import PersonaLibrary
from synthia.server.session import ChatSession, PlanMark, TurnReport
from synthia.tools import agents
from tests.agent.scripted import Scripted, calls, says
from tests.interfaces.test_chat import (
    answer,
    app_for,
    console,
    event,
    scripted,
    settings,
)


def recorded(
    ran: list[str], effect: Effect = Effect.READ, name: str = "note"
) -> FunctionTool:
    def note(text: str) -> str:
        """Note something down."""
        ran.append(text)
        return f"noted {text}"

    return FunctionTool.of(note, reach=Reach.LOCAL, effect=effect, name=name)


def finished(arguments: str, *, ok: bool, result: str = "fine") -> ToolFinished:
    return ToolFinished(ToolCall("call_0", "note", arguments), result, ok, 0.04)


def test_a_tool_line_names_the_call_its_outcome_and_time() -> None:
    assert describe_call(finished('{"text": "milk"}', ok=True)) == (
        'tool note {"text": "milk"} | ok | 0.0 s'
    )
    assert describe_call(
        finished("", ok=False, result="running note was not approved")
    ) == ("tool note {} | running note was not approved | 0.0 s")


def test_a_flagged_result_says_so_on_its_line() -> None:
    call = ToolFinished(
        ToolCall("call_0", "read_file", "{}"),
        result="text",
        ok=True,
        seconds=0.0,
        flags=("asks to ignore instructions", "speaks as a role"),
    )

    assert describe_call(call) == (
        "tool read_file {} | ok | flagged: asks to ignore instructions, "
        "speaks as a role | 0.0 s"
    )


def test_long_arguments_and_results_are_shortened_to_80_characters() -> None:
    line = describe_call(
        finished(json.dumps({"text": "x" * 200}), ok=False, result="e" * 200)
    )

    arguments, outcome, _ = line.removeprefix("tool note ").split(" | ")
    assert len(arguments) == len(outcome) == MAX_SHOWN
    assert arguments.endswith("...")
    assert outcome == "e" * 77 + "..."


@pytest.mark.parametrize(
    ("typed", "approved"),
    [
        ("y", True),
        ("yes", True),
        (" YES ", True),
        ("", False),
        ("n", False),
        ("sure", False),
    ],
)
async def test_the_terminal_approver_says_yes_only_to_y_or_yes(
    typed: str, approved: bool
) -> None:
    read = scripted(typed)
    approve = terminal_approver(read)

    assert await approve(recorded([]), '{"text": "milk"}') is approved
    assert read.prompts == ['run note {"text": "milk"}? [y/N] ']  # type: ignore[attr-defined]


async def test_the_end_of_input_at_the_question_is_no() -> None:
    assert await terminal_approver(scripted())(recorded([]), "") is False


async def test_long_arguments_are_shown_whole_before_the_question() -> None:
    code = "import os\n" + "\n".join(f"print({n})" for n in range(20))
    read = scripted("y")

    assert await terminal_approver(read)(recorded([]), json.dumps({"code": code}))
    assert read.prompts == [  # type: ignore[attr-defined]
        "run note with:\n  code:\n    import os\n"
        + "".join(f"    print({n})\n" for n in range(20))
        + "[y/N] "
    ]


def test_terminal_controls_in_arguments_are_shown_escaped() -> None:
    hidden = '{"code": "rm()\\u001b[1A\\u001b[2Kprint(1)"}'
    raw = '{"code": "rm()\x1b[1A\x1b[2Kprint(1)"}'

    assert approval_question("note", raw) == (
        'run note {"code": "rm()\\x1b[1A\\x1b[2Kprint(1)"}? [y/N] '
    )
    assert approval_question("note", hidden.replace("print(1)", "x" * 80)) == (
        "run note with:\n  code:\n    rm()\\x1b[1A\\x1b[2K" + "x" * 80 + "\n[y/N] "
    )


def test_arguments_that_are_not_an_object_are_shown_as_sent() -> None:
    sent = "[" + ", ".join(["1"] * 40) + "]"

    assert approval_question("note", sent) == f"run note with:\n{sent}\n[y/N] "
    assert approval_question("note", "x" * 90 + "{") == (
        "run note with:\n" + "x" * 90 + "{\n[y/N] "
    )


def test_values_other_than_text_are_shown_as_json() -> None:
    arguments = json.dumps({"path": "a" * 70, "limit": 5})

    assert approval_question("note", arguments) == (
        "run note with:\n  path:\n    " + "a" * 70 + "\n  limit:\n    5\n[y/N] "
    )


async def test_a_turn_with_a_tool_shows_the_call_and_keeps_the_whole_exchange() -> None:
    ran: list[str] = []
    model = Scripted(calls(("note", '{"text": "milk"}')), says("Noted."))
    chat = ChatSession(
        model, PersonaLibrary(), "synthia", tools=Toolbox([recorded(ran)])
    )

    items = [item async for item in chat.turn("remember milk")]

    assert ran == ["milk"]
    calls_seen = [i for i in items if isinstance(i, ToolFinished)]
    assert [(c.call.name, c.result, c.ok) for c in calls_seen] == [
        ("note", "noted milk", True)
    ]
    assert "".join(i.text for i in items if isinstance(i, ChatChunk)) == "Noted."
    assert [m.role for m in chat.history] == [
        Role.USER,
        Role.ASSISTANT,
        Role.TOOL,
        Role.ASSISTANT,
    ]
    assert chat.history[2].text == quote("note", "noted milk")
    # The next turn sends the tool exchange back, after a fresh system message.
    assert model.requests[1].messages[0].role is Role.SYSTEM
    assert model.requests[1].tools == chat.tools.specs()


async def test_a_change_runs_only_when_the_person_says_yes() -> None:
    for typed, expected in (("n", []), ("y", ["milk"])):
        ran: list[str] = []
        model = Scripted(calls(("note", '{"text": "milk"}')), says("ok"))
        chat = ChatSession(
            model,
            PersonaLibrary(),
            "synthia",
            tools=Toolbox([recorded(ran, Effect.CHANGE)]),
            approver=terminal_approver(scripted(typed)),
        )

        _ = [item async for item in chat.turn("remember milk")]

        assert ran == expected


async def test_the_report_adds_up_the_tokens_of_every_step() -> None:
    def with_usage(
        chunks: list[ChatChunk], prompt: int, completion: int
    ) -> list[ChatChunk]:
        *body, last = chunks
        return [
            *body,
            ChatChunk(
                finish_reason=last.finish_reason, usage=Usage(prompt, completion)
            ),
        ]

    model = Scripted(
        with_usage(calls(("note", '{"text": "a"}')), 100, 10),
        with_usage(says("done"), 130, 4),
    )
    chat = ChatSession(
        model, PersonaLibrary(), "synthia", tools=Toolbox([recorded([])])
    )

    (report,) = [i async for i in chat.turn("go") if isinstance(i, TurnReport)]

    assert (report.prompt_tokens, report.completion_tokens) == (230, 14)


async def test_without_tools_a_turn_sends_no_tools() -> None:
    model = Scripted(says("hi"))
    chat = ChatSession(model, PersonaLibrary(), "synthia")

    _ = [item async for item in chat.turn("hello")]

    assert model.requests[0].tools == ()
    assert len(model.requests) == 1


def test_slash_tools_is_a_command_without_arguments() -> None:
    assert parse("/tools") == ShowTools()
    assert parse("/tools now") == Invalid("/tools takes no arguments")


async def test_slash_tools_lists_each_tool_and_whether_it_asks(tmp_path: Path) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(404))
    ) as client:
        chat, out = await app_for(tmp_path, client)
        chat.conversation.session.tools = Toolbox(
            [
                recorded([]),
                recorded([], Effect.CHANGE, name="edit"),
                recorded([], name="gone"),
            ]
        )
        chat.conversation.session.policy = Policy(denied=frozenset({"gone"}))
        await chat.handle(ShowTools())
        chat.conversation.session.tools = Toolbox()
        await chat.handle(ShowTools())

    assert out.getvalue().splitlines() == [
        "note | local, read | Note something down.",
        "edit | local, change | asks first | Note something down.",
        "gone | local, read | never runs | Note something down.",
        "no tools",
    ]


async def test_an_answer_with_nothing_in_it_says_none_came_back(
    tmp_path: Path,
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(404))
    ) as client:
        chat, out = await app_for(tmp_path, client)
        chat.conversation.session = ChatSession(
            Scripted([]), PersonaLibrary(), "synthia"
        )
        await chat.handle(parse("hello"))

    assert out.getvalue() == "no answer came back\n"
    assert chat.conversation.session.history == []


async def test_plan_shows_the_plan_each_step_and_the_answer(tmp_path: Path) -> None:
    model = Scripted(
        says(json.dumps({"steps": ["find a", "use a"]})),
        says("found a"),
        says("used a"),
        says("All done."),
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(404))
    ) as client:
        chat, out = await app_for(tmp_path, client)
        chat.conversation.session = ChatSession(model, PersonaLibrary(), "synthia")
        await chat.handle(parse("/plan find a, then use it"))

    *shown, report = out.getvalue().splitlines()
    assert shown == [
        "plan: 1. find a | 2. use a",
        "step 1: find a",
        "found a",
        "step 2: use a",
        "used a",
        "answer:",
        "All done.",
    ]
    assert report.startswith("direct | ")
    assert [m.text for m in chat.conversation.session.history] == [
        "find a, then use it",
        "All done.",
    ]


@pytest.mark.parametrize(
    ("mark", "line"),
    [
        (Planned((), revised=True), "plan again: no steps left"),
        (Planned(("a\x1b[2Jb",), revised=False), "plan: 1. a\\x1b[2Jb"),
        (PlanStepBegun(3, "check\rthe log"), "step 3: check\\rthe log"),
        (PlanAnswerBegun(), "answer:"),
    ],
    ids=["no-steps-left", "escape-in-plan", "return-in-step", "answer"],
)
def test_a_plan_line_escapes_what_a_terminal_would_act_on(
    mark: PlanMark, line: str
) -> None:
    assert describe_mark(mark) == line


@pytest.fixture(autouse=True)
def no_outside_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the tool lists below the same on every machine, agents or not."""

    def nowhere(_: str) -> None:
        return None

    monkeypatch.setattr(agents, "which", nowhere)


def called(name: str, arguments: str) -> bytes:
    call = {
        "index": 0,
        "id": "call_0",
        "function": {"name": name, "arguments": arguments},
    }
    return (
        event({"model": "vendor/free", "choices": [{"delta": {"tool_calls": [call]}}]})
        + event({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
        + b"data: [DONE]\n\n"
    )


def test_the_chat_calls_a_tool_of_an_mcp_server_after_a_yes(tmp_path: Path) -> None:
    server = Path(__file__).parent.parent / "mcp" / "fake_mcp_server.py"
    command = json.dumps([sys.executable, str(server)])
    (tmp_path / "mcp.toml").write_text(f"[servers.fake]\ncommand = {command}\n")
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(404)
        sent.append(json.loads(request.content))
        reply = (
            called("fake__echo", '{"text": "hi"}')
            if len(sent) == 1
            else answer("Done.")
        )
        return httpx.Response(200, content=reply)

    screen, out = console()
    read = scripted("say hi through the server", "y")

    run_chat(
        settings(tmp_path),
        "synthia",
        screen,
        read,
        httpx.MockTransport(respond),
        use_remote=True,
    )

    names = [t["function"]["name"] for t in sent[0]["tools"]]  # type: ignore[index]
    assert names[5:8] == ["fetch_url", "fake__echo", "fake__add"]
    assert read.prompts[1] == 'run fake__echo {"text": "hi"}? [y/N] '  # type: ignore[attr-defined]
    assert sent[1]["messages"][-1]["content"] == quote("fake__echo", "hi")  # type: ignore[index]
    assert "tool fake__echo" in out.getvalue()
    assert (
        tmp_path / "logs" / "mcp" / "fake.log"
    ).read_text() == "fake mcp server started\n"


def test_a_broken_server_list_is_named_and_the_chat_goes_on(tmp_path: Path) -> None:
    (tmp_path / "mcp.toml").write_text("[servers.fake\n")

    def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(404)
        return httpx.Response(200, content=answer("Hello."))

    screen, out = console()
    run_chat(
        settings(tmp_path),
        "synthia",
        screen,
        scripted("hello"),
        httpx.MockTransport(respond),
        use_remote=True,
    )

    text = out.getvalue()
    assert text.startswith(f"no MCP servers: {tmp_path / 'mcp.toml'} is not a valid")
    assert "Hello." in text


def test_the_chat_calls_the_calculator_and_shows_the_call(tmp_path: Path) -> None:
    tool_call = (
        event(
            {
                "model": "vendor/free",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_0",
                                    "function": {
                                        "name": "calculate",
                                        "arguments": '{"expression": "6 * 7"}',
                                    },
                                }
                            ]
                        }
                    }
                ],
            }
        )
        + event({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
        + b"data: [DONE]\n\n"
    )
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(404)
        sent.append(json.loads(request.content))
        return httpx.Response(
            200, content=tool_call if len(sent) == 1 else answer("42.")
        )

    screen, out = console()

    run_chat(
        settings(tmp_path),
        "synthia",
        screen,
        scripted("what is 6 times 7?"),
        httpx.MockTransport(respond),
        use_remote=True,
    )

    text = out.getvalue()
    # The time is the tool's own and varies with load.
    assert 'tool calculate {"expression": "6 * 7"} | ok | ' in text
    assert "42." in text
    assert [t["function"]["name"] for t in sent[0]["tools"]] == [  # type: ignore[index]
        "current_time",
        "calculate",
        "read_file",
        "list_files",
        "run_python",
        "fetch_url",
    ]
    assert sent[1]["messages"][-1] == {  # type: ignore[index]
        "role": "tool",
        "tool_call_id": "call_0",
        "content": quote("calculate", "42"),
    }
