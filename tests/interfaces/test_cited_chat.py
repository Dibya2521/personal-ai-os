from pathlib import Path

from synthia.agent.tools import Effect, FunctionTool, Reach, Toolbox
from synthia.gateway.assemble import Gateway
from synthia.gateway.router import Router
from synthia.gateway.usage import UsageLog
from synthia.interfaces.chat import SOURCES, ChatApp
from synthia.interfaces.commands import parse
from synthia.interfaces.daemon_client import conversation
from synthia.knowledge.tool import CITED, knowledge_tool
from synthia.persona.library import PersonaLibrary
from synthia.server.conversation import Conversation
from synthia.server.session import ChatSession
from tests.agent.scripted import Scripted, calls, says
from tests.gateway.test_router import LOCAL, Fake
from tests.interfaces.test_chat import console
from tests.knowledge.test_tool import KESTREL, PUNE, knowing
from tests.server.test_confinement import nothing_learned
from tests.server.test_daemon import Local, daemon


def stays_local() -> bool:
    return True


async def nobody_answers(tool: str, arguments: str, *, confined: bool) -> bool:
    del tool, arguments, confined
    return False


async def test_the_chat_lists_the_sources_under_the_answer_once(
    tmp_path: Path,
) -> None:
    knowledge = await knowing(tmp_path, private=False)
    tools = Toolbox([knowledge_tool(knowledge, can_stay_local=stays_local)])
    model = Local(
        calls(("search_knowledge", '{"query": "kestrel"}')),
        says("Kestrel is a host."),
        says("hello"),
    )
    screen, out = console()
    async with (
        daemon(tmp_path, model, tools) as info,
        conversation(info, nobody_answers) as talk,
    ):
        chat = ChatApp(talk, screen)
        for line in ["what is kestrel?", "hi"]:
            await chat.handle(parse(line))

    lines = out.getvalue().splitlines()
    (at,) = [i for i, line in enumerate(lines) if line.startswith(SOURCES)]
    assert lines[at - 1].endswith(" s")
    assert lines[at - 2] == "Kestrel is a host."
    assert lines[at] == f"{SOURCES}{KESTREL}, {PUNE}"


async def test_a_control_character_in_a_source_is_shown_escaped(
    tmp_path: Path,
) -> None:
    def cite() -> str:
        """Cite a file whose name holds an escape sequence."""
        cited = CITED.get()
        assert cited is not None
        cited.append("notes/\x1b[2Jwiped.md:1")
        return "cited"

    tools = Toolbox([FunctionTool.of(cite, reach=Reach.LOCAL, effect=Effect.READ)])
    model = Scripted(calls(("cite", "{}")), says("done"))
    session = ChatSession(model, PersonaLibrary(), "synthia", tools=tools)
    router = Router(Fake(LOCAL))
    gateway = Gateway(router, router, UsageLog(tmp_path / "usage.db"), nothing_learned)
    screen, out = console()

    await ChatApp(Conversation(session, gateway), screen).handle(parse("cite it"))

    assert f"{SOURCES}notes/\\x1b[2Jwiped.md:1\n" in out.getvalue()
    assert "\x1b" not in out.getvalue()
