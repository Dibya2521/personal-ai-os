from pathlib import Path

from synthia.agent.tools import Toolbox
from synthia.interfaces.chat import (
    HOLDS_PRIVATE,
    STAYS_LOCAL,
    ChatApp,
    approval_question,
    terminal_answer,
    terminal_approver,
)
from synthia.interfaces.commands import parse
from synthia.interfaces.daemon_client import conversation
from synthia.persona.library import PersonaLibrary
from synthia.server.session import ChatSession
from tests.agent.scripted import Scripted, calls, says
from tests.gateway.test_confinement import private_reader
from tests.interfaces.test_chat import console
from tests.server.test_daemon import Local, daemon, edit_tool

EDIT = '{"text": "x"}'


def test_the_question_before_a_call_in_a_private_chat_says_so_first() -> None:
    assert approval_question("edit", EDIT, confined=True) == (
        f"{HOLDS_PRIVATE}\n{approval_question('edit', EDIT)}"
    )
    assert approval_question("edit", "{}") == "run edit {}? [y/N] "


async def test_a_chat_in_this_process_asks_with_the_private_text_line() -> None:
    prompts: list[str] = []

    def read(prompt: str) -> str:
        prompts.append(prompt)
        return "n"

    model = Scripted(
        calls(("edit", EDIT)),
        calls(("read_note", "{}")),
        calls(("edit", EDIT)),
        says("done"),
    )
    chat = ChatSession(
        model,
        PersonaLibrary(),
        "synthia",
        tools=Toolbox([private_reader(), edit_tool([])]),
        approver=terminal_approver(read),
    )

    _ = [item async for item in chat.turn("edit, read my note, edit")]

    question = f"run edit {EDIT}? [y/N] "
    assert prompts == [question, f"{HOLDS_PRIVATE}\n{question}"]


async def test_a_chat_that_reads_private_text_is_told_once_until_reset(
    tmp_path: Path,
) -> None:
    prompts: list[str] = []

    def read(prompt: str) -> str:
        prompts.append(prompt)
        return "y"

    model = Local(
        calls(("read_note", "{}")),
        calls(("edit", EDIT)),
        says("done"),
        says("still here"),
        calls(("read_note", "{}")),
        says("read again"),
    )
    tools = Toolbox([private_reader(), edit_tool([])])
    screen, out = console()
    async with (
        daemon(tmp_path, model, tools) as info,
        conversation(info, terminal_answer(read)) as talk,
    ):
        chat = ChatApp(talk, screen)
        for line in ["read my note, then edit", "thanks", "/reset", "read it again"]:
            await chat.handle(parse(line))

    assert out.getvalue().count(STAYS_LOCAL) == 2
    assert out.getvalue().index(STAYS_LOCAL) > out.getvalue().index("done")
    assert prompts == [f"{HOLDS_PRIVATE}\nrun edit {EDIT}? [y/N] "]
