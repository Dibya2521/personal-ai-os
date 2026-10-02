import json
from pathlib import Path

import httpx
import pytest

from synthia.agent.loop import ToolFinished
from synthia.agent.policy import Policy
from synthia.agent.quoting import quote
from synthia.agent.tools import Effect, FunctionTool, Reach, Toolbox
from synthia.gateway.types import ChatChunk, Role, ToolCall, Usage
from synthia.interfaces.chat import (
    MAX_SHOWN,
    describe_call,
    run_chat,
    terminal_approver,
)
from synthia.interfaces.commands import Invalid, ShowTools, parse
from synthia.interfaces.session import ChatSession, TurnReport
from synthia.persona.library import PersonaLibrary
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
        chat.session.tools = Toolbox(
            [
                recorded([]),
                recorded([], Effect.CHANGE, name="edit"),
                recorded([], name="gone"),
            ]
        )
        chat.session.policy = Policy(denied=frozenset({"gone"}))
        await chat.handle(ShowTools())
        chat.session.tools = Toolbox()
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
        chat.session = ChatSession(Scripted([]), PersonaLibrary(), "synthia")
        await chat.handle(parse("hello"))

    assert out.getvalue() == "no answer came back\n"
    assert chat.session.history == []


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
    ]
    assert sent[1]["messages"][-1] == {  # type: ignore[index]
        "role": "tool",
        "tool_call_id": "call_0",
        "content": quote("calculate", "42"),
    }
