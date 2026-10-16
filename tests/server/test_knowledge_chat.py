import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from synthia.agent.loop import ToolFinished
from synthia.agent.tools import Toolbox
from synthia.gateway.router import Remote, Route, Router, RouteReason
from synthia.kernel.config import Settings
from synthia.knowledge.service import Knowledge
from synthia.knowledge.store import KNOWLEDGE_FILE
from synthia.knowledge.tool import knowledge_tool
from synthia.persona.library import PersonaLibrary
from synthia.retrieval.embed import TextEmbedder
from synthia.server.daemon import run_daemon
from synthia.server.session import ChatSession, TurnItem, TurnReport
from tests.agent.scripted import Scripted, calls, says
from tests.gateway.test_router import Setup
from tests.interfaces.daemons import toy_embedder
from tests.knowledge.corpus import planted
from tests.knowledge.test_tool import DOOR, PRIVATE, knowing, shown
from tests.retrieval.toy import Network, embedder
from tests.server.test_confinement import Serving
from tests.server.test_daemon import Local, client, written
from tests.timing import HANG_TIMEOUT_S

SEARCH = "search_knowledge"


def search(query: str) -> tuple[str, str]:
    return SEARCH, json.dumps({"query": query})


@pytest.fixture
def setup(tmp_path: Path) -> Setup:
    return Setup(tmp_path)


def chat_over(knowledge: Knowledge, router: Router) -> ChatSession:
    tools = Toolbox(
        [knowledge_tool(knowledge, can_stay_local=lambda: router.can_stay_local)]
    )
    chat = ChatSession(router, PersonaLibrary(), "synthia", tools=tools)
    chat.use_remote = True
    return chat


async def turned(chat: ChatSession, text: str) -> list[TurnItem]:
    return [item async for item in chat.turn(text)]


def report_of(items: list[TurnItem]) -> TurnReport:
    (report,) = [i for i in items if isinstance(i, TurnReport)]
    return report


def results(items: list[TurnItem]) -> list[str]:
    return [i.result for i in items if isinstance(i, ToolFinished)]


async def test_a_private_passage_keeps_the_rest_of_the_turn_local(
    tmp_path: Path, setup: Setup
) -> None:
    knowledge = await knowing(tmp_path)
    remote = Scripted(calls(search("kestrel")))
    local = Scripted(calls(search("pune")), says("The door code is 4021."))
    router = Router(local, Remote(remote, setup.health), setup.publish)

    items = await turned(chat_over(knowledge, router), "what is the door code?")

    report = report_of(items)
    first, second = results(items)
    assert DOOR in shown(first)
    assert len(remote.requests) == 1
    assert len(local.requests) == 2
    assert [route[:2] for route in setup.routes()] == [
        (Route.REMOTE, RouteReason.REMOTE_ASKED),
        (Route.LOCAL, RouteReason.CONFINED),
        (Route.LOCAL, RouteReason.CONFINED),
    ]
    assert report.confined
    assert report.sources == tuple(dict.fromkeys(shown(first) + shown(second)))
    assert len(report.sources) < len(shown(first) + shown(second))


async def test_a_public_search_leaves_the_turn_remote(
    tmp_path: Path, setup: Setup
) -> None:
    knowledge = await knowing(tmp_path, private=False)
    remote = Scripted(calls(search("kestrel")), says("It is a host."))
    local = Scripted()
    router = Router(local, Remote(remote, setup.health), setup.publish)

    items = await turned(chat_over(knowledge, router), "what is kestrel?")

    report = report_of(items)
    (result,) = results(items)
    assert (len(remote.requests), len(local.requests)) == (2, 0)
    assert {route[0] for route in setup.routes()} == {Route.REMOTE}
    assert not report.confined
    assert report.sources == tuple(shown(result))


async def test_a_turn_with_no_search_has_no_sources(
    tmp_path: Path, setup: Setup
) -> None:
    knowledge = await knowing(tmp_path)
    router = Router(None, Remote(Scripted(says("hi")), setup.health), setup.publish)

    items = await turned(chat_over(knowledge, router), "hello")

    assert report_of(items).sources == ()


def no_embedder(_settings: Settings, _warn: Callable[[str], None]) -> None:
    return None


async def seeded_index(home: Path) -> None:
    knowledge = await Knowledge.open(home / KNOWLEDGE_FILE, embedder(Network()))
    await knowledge.add(planted(home / "diary", PRIVATE), private=True)
    await knowledge.index()


@pytest.mark.parametrize(
    ("opener", "offered"), [(toy_embedder, True), (no_embedder, False)]
)
async def test_the_daemon_offers_the_search_only_with_knowledge_open(
    tmp_path: Path,
    opener: Callable[[Settings, Callable[[str], None]], TextEmbedder | None],
    offered: bool,
) -> None:
    await seeded_index(tmp_path)
    model = Local(calls(search("code")), says("4021."))
    stop = asyncio.Event()
    running = asyncio.create_task(
        run_daemon(Settings(home=tmp_path), Serving(model), stop=stop, embedder=opener)
    )
    info = await written(tmp_path, running)
    async with client(info) as talk:
        tools = await talk.call("tools")
        report = await talk.call("turn", {"text": "what is the door code?"})
    stop.set()
    await asyncio.wait_for(running, HANG_TIMEOUT_S)

    names = [
        line.split(" | ")[0] for line in cast("dict[str, list[str]]", tools)["notes"]
    ]
    assert (SEARCH in names) is offered
    fields = cast("dict[str, object]", report)
    found = [str(p["result"]) for name, p in talk.notes if name == "tool"]
    if offered:
        # The local model takes tools, so the private note is read and cited.
        assert "4021" in found[0]
        assert fields["sources"] == shown(found[0])
        assert fields["confined"] is True
    else:
        assert fields["sources"] == []
