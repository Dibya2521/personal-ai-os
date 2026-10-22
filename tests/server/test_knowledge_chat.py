import asyncio
import json
import re
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Never, cast, override

import pytest

from synthia.agent.loop import ToolFinished
from synthia.agent.quoting import quote
from synthia.agent.tools import Toolbox, ToolError
from synthia.agent.trace import ToolBegan, ToolEnded, Trace, read_trace, sessions
from synthia.gateway.errors import GatewayError, LocalUnavailableError
from synthia.gateway.router import Remote, Route, Router, RouteReason
from synthia.gateway.types import ChatChunk, ChatRequest, FinishReason, Role, Usage
from synthia.kernel.config import Settings
from synthia.knowledge.deep import NO_LOCAL_MODEL
from synthia.knowledge.service import Knowledge, KnowledgeError
from synthia.knowledge.store import KNOWLEDGE_FILE
from synthia.knowledge.tool import DEEP, knowledge_tool
from synthia.memory.window import MAX_CHARS_PER_TOKEN, START_CHARS_PER_TOKEN
from synthia.persona.library import PersonaLibrary
from synthia.retrieval.embed import TextEmbedder
from synthia.server.daemon import run_daemon
from synthia.server.session import ChatSession, NoKnowledgeError, TurnItem, TurnReport
from tests.agent.scripted import Reply, Scripted, calls, says
from tests.agent.test_plan import planned
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


def chat_over(
    knowledge: Knowledge, router: Router, trace: Trace | None = None
) -> ChatSession:
    tools = Toolbox(
        [
            knowledge_tool(
                knowledge, can_stay_local=lambda: router.can_stay_local, model=router
            )
        ]
    )
    chat = ChatSession(router, PersonaLibrary(), "synthia", tools=tools, trace=trace)
    chat.use_remote = True
    return chat


async def turned(chat: ChatSession, text: str, *, deep: bool = False) -> list[TurnItem]:
    return [item async for item in chat.turn(text, deep=deep)]


def rewrites(*queries: str) -> list[ChatChunk]:
    return says(json.dumps({"queries": queries, "passage": "the code is 4021"}))


async def test_a_deep_turn_searches_before_the_model_is_first_asked(
    tmp_path: Path, setup: Setup
) -> None:
    knowledge = await knowing(tmp_path)
    remote = Scripted()
    local = Scripted(
        rewrites("door code"), calls(search("pune")), says("The door code is 4021.")
    )
    router = Router(local, Remote(remote, setup.health), setup.publish)
    chat = chat_over(knowledge, router)
    question = "what is the door code?"

    items = await turned(chat, question, deep=True)

    first, second = [i for i in items if isinstance(i, ToolFinished)]
    assert (first.call.name, first.call.arguments) == (
        SEARCH,
        json.dumps({"query": question}),
    )
    assert DOOR in shown(first.result)
    expansion, step, answer = local.requests
    assert expansion.response_schema is not None
    assert step.response_schema is None
    assert answer.response_schema is None
    *_, asked, called, result = step.messages
    assert asked.text == question
    assert called.tool_calls == (first.call,)
    assert (result.tool_call_id, result.text) == (
        first.call.id,
        quote(SEARCH, first.result),
    )
    assert step.tools == chat.tools.specs()
    assert "The door code is 4021." in [
        i.text for i in items if isinstance(i, ChatChunk)
    ]
    report = report_of(items)
    assert report.confined
    assert report.sources == tuple(
        dict.fromkeys(shown(first.result) + shown(second.result))
    )
    assert remote.requests == []
    assert [m.role for m in chat.history] == [
        Role.USER,
        Role.ASSISTANT,
        Role.TOOL,
        Role.ASSISTANT,
        Role.TOOL,
        Role.ASSISTANT,
    ]


async def test_a_public_deep_turn_is_expanded_locally_and_answered_remotely(
    tmp_path: Path, setup: Setup
) -> None:
    knowledge = await knowing(tmp_path, private=False)
    remote = Scripted(says("It is a host."))
    local = Scripted(rewrites("kestrel host"))
    router = Router(local, Remote(remote, setup.health), setup.publish)

    items = await turned(chat_over(knowledge, router), "what is kestrel?", deep=True)

    assert (len(local.requests), len(remote.requests)) == (1, 1)
    assert [route[:2] for route in setup.routes()] == [
        (Route.LOCAL, RouteReason.LOCAL_FIRST),
        (Route.REMOTE, RouteReason.REMOTE_ASKED),
    ]
    assert not report_of(items).confined


async def test_a_deep_turn_with_no_local_model_fails_and_keeps_nothing(
    tmp_path: Path, setup: Setup
) -> None:
    knowledge = await knowing(tmp_path)
    remote = Scripted()
    chat = chat_over(knowledge, Router(None, Remote(remote, setup.health)))

    with pytest.raises(LocalUnavailableError, match=f"^{re.escape(NO_LOCAL_MODEL)}$"):
        await turned(chat, "what is the door code?", deep=True)

    assert (remote.requests, chat.history) == ([], [])


async def test_a_deep_turn_in_a_chat_with_no_knowledge_asks_no_model() -> None:
    model = Scripted()
    chat = ChatSession(model, PersonaLibrary(), "synthia")

    with pytest.raises(NoKnowledgeError, match="no knowledge index"):
        await turned(chat, "what is the door code?", deep=True)

    assert (model.requests, chat.history) == ([], [])


async def test_the_trace_shows_the_deep_search_as_a_tool_call(
    tmp_path: Path, setup: Setup
) -> None:
    knowledge = await knowing(tmp_path)
    local = Scripted(rewrites("door code"), says("4021."))
    router = Router(local, None, setup.publish)
    chat = chat_over(knowledge, router, Trace.start(tmp_path / "traces"))

    await turned(chat, "what is the door code?", deep=True)

    (path,) = sessions(tmp_path / "traces")
    records, _ = read_trace(path)
    began, ended = [r for r in records if isinstance(r, ToolBegan | ToolEnded)]
    assert isinstance(began, ToolBegan)
    assert isinstance(ended, ToolEnded)
    assert (began.name, ended.name, ended.ok) == (SEARCH, SEARCH, True)
    assert began.call_id == ended.call_id


class Held(Scripted):
    """Holds its first answer until ``release`` is set, and sets ``asked`` then."""

    def __init__(
        self, *replies: Reply, asked: asyncio.Event, release: asyncio.Event
    ) -> None:
        super().__init__(*replies)
        self._asked = asked
        self._release = release

    @override
    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        if not self.requests:
            self._asked.set()
            await self._release.wait()
        async for chunk in super().stream(request):
            yield chunk


def local_chat(
    knowledge: Knowledge, model: Scripted, setup: Setup, trace: Trace | None = None
) -> ChatSession:
    chat = chat_over(knowledge, Router(model, None, setup.publish), trace)
    chat.use_remote = False
    return chat


def schemas(requests: list[ChatRequest]) -> list[bool]:
    return [r.response_schema is not None for r in requests]


async def test_a_deep_turn_leaves_every_other_search_plain(
    tmp_path: Path, setup: Setup
) -> None:
    knowledge = await knowing(tmp_path)
    asked, release = asyncio.Event(), asyncio.Event()
    local = Held(
        rewrites("door code"),
        says("4021."),
        calls(search("door code")),
        says("4021."),
        asked=asked,
        release=release,
    )
    chat = local_chat(knowledge, local, setup)
    beside = Scripted(calls(search("door code")), says("4021."))
    other = local_chat(knowledge, beside, setup)

    async def meanwhile() -> list[TurnItem]:
        await asked.wait()
        try:
            return await turned(other, "what is the door code?")
        finally:
            release.set()

    running = asyncio.create_task(meanwhile())
    # In this task, not a task of its own, so a flag left set would show here.
    async with asyncio.timeout(HANG_TIMEOUT_S):
        await turned(chat, "what is the door code?", deep=True)
    side = await asyncio.wait_for(running, HANG_TIMEOUT_S)

    assert schemas(beside.requests) == [False, False]
    assert DOOR in shown(results(side)[0])
    assert not DEEP.get()
    await turned(chat, "and the code again?")
    assert schemas(local.requests) == [True, False, False, False]


async def test_a_deep_planned_turn_plans_with_the_search_in_hand(
    tmp_path: Path, setup: Setup
) -> None:
    knowledge = await knowing(tmp_path)
    local = Scripted(rewrites("door code"), planned(), says("4021."))
    chat = local_chat(knowledge, local, setup)
    question = "what is the door code?"

    items = [i async for i in chat.turn(question, plan=True, deep=True)]

    (found,) = [i for i in items if isinstance(i, ToolFinished)]
    _, planning, _ = local.requests
    assert planning.response_schema is not None
    *_, asked, called, result, _prompt = planning.messages
    assert asked.text == question
    assert called.tool_calls == (found.call,)
    assert result.text == quote(SEARCH, found.result)
    assert report_of(items).sources == tuple(shown(found.result))
    assert chat.history[-1].text == "4021."


def counted(text: str) -> list[ChatChunk]:
    # One prompt token pulls any learned rate to the most it may be.
    usage = Usage(prompt_tokens=1, completion_tokens=1)
    return [
        ChatChunk(text=text),
        ChatChunk(finish_reason=FinishReason.STOP, usage=usage),
    ]


async def test_a_deep_turn_teaches_the_window_nothing(
    tmp_path: Path, setup: Setup
) -> None:
    knowledge = await knowing(tmp_path)
    local = Scripted(rewrites("door code"), counted("4021."), counted("4021."))
    chat = local_chat(knowledge, local, setup)

    await turned(chat, "what is the door code?", deep=True)
    after_deep = chat.window.chars_per_token
    await turned(chat, "and the code again?")

    assert (after_deep, chat.window.chars_per_token) == (
        START_CHARS_PER_TOKEN,
        MAX_CHARS_PER_TOKEN,
    )


@pytest.mark.parametrize(
    ("error", "said"),
    [
        (KnowledgeError("the index is unreadable"), f"{SEARCH} failed: KnowledgeError"),
        (ToolError("the index is busy"), "the index is busy"),
    ],
    ids=["unexpected", "tool-error"],
)
async def test_a_deep_search_that_fails_is_answered_from_its_failure(
    tmp_path: Path,
    setup: Setup,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    said: str,
) -> None:
    knowledge = await knowing(tmp_path)

    async def broken(*_args: object, **_kwargs: object) -> Never:
        raise error

    monkeypatch.setattr(knowledge, "search", broken)
    local = Scripted(rewrites("door code"), says("I could not search."))
    chat = local_chat(knowledge, local, setup, Trace.start(tmp_path / "traces"))

    items = await turned(chat, "what is the door code?", deep=True)

    (found,) = [i for i in items if isinstance(i, ToolFinished)]
    assert (found.result, found.ok) == (said, False)
    assert local.requests[-1].messages[-1].text == quote(SEARCH, said)
    assert chat.history[-1].text == "I could not search."
    (path,) = sessions(tmp_path / "traces")
    records, _ = read_trace(path)
    ended = [(r.call_id, r.ok) for r in records if isinstance(r, ToolEnded)]
    assert ended == [(found.call.id, False)]


async def test_a_deep_search_the_gateway_fails_fails_the_turn(
    tmp_path: Path, setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    knowledge = await knowing(tmp_path)

    gone = "the model went away"

    async def refused(*_args: object, **_kwargs: object) -> Never:
        raise GatewayError(gone)

    monkeypatch.setattr(knowledge, "search", refused)
    local = Scripted(rewrites("door code"))
    chat = local_chat(knowledge, local, setup)

    with pytest.raises(GatewayError, match=f"^{gone}$"):
        await turned(chat, "what is the door code?", deep=True)

    assert (len(local.requests), chat.history) == (1, [])


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
