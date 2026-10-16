import asyncio
from pathlib import Path
from typing import cast

import httpx
import pytest

from synthia.agent.tools import Effect, FunctionTool, Reach, Toolbox
from synthia.agent.trace import Trace, TurnBegan, read_trace
from synthia.gateway.assemble import Gateway
from synthia.gateway.router import Remote, Route, Router, RouteReason
from synthia.gateway.usage import UsageLog
from synthia.kernel.config import Settings
from synthia.memory.hybrid import Recall
from synthia.memory.remembering import REMINDER_HEADER, Remembering
from synthia.memory.store import MEMORY_FILE, MemoryStore, Turn
from synthia.persona.library import PersonaLibrary
from synthia.server.conversation import (
    ALREADY_TALKING,
    PRIVATE_NEEDS_LOCAL,
    RESUMED_PRIVATE,
    TURN_RUNNING,
    Conversation,
)
from synthia.server.daemon import run_daemon
from synthia.server.session import ChatSession, LastRoute, TurnReport
from tests.agent.scripted import Scripted, calls, says
from tests.gateway.test_confinement import READ_NOTE, private_reader
from tests.gateway.test_router import LOCAL, REMOTE, Fake, Setup
from tests.interfaces.daemons import toy_embedder
from tests.memory.test_hybrid import MONDAY
from tests.retrieval.toy import Network, embedder
from tests.server.test_daemon import Local, Service, client, written
from tests.timing import HANG_TIMEOUT_S

PRIVATE_QUESTION = "what does my note say?"


@pytest.fixture
def setup(tmp_path: Path) -> Setup:
    return Setup(tmp_path)


async def reported(chat: ChatSession | Conversation, text: str) -> TurnReport:
    (report,) = [i async for i in chat.turn(text) if isinstance(i, TurnReport)]
    return report


def reading(
    model: Scripted, trace: Trace | None = None, memory: Remembering | None = None
) -> ChatSession:
    return ChatSession(
        model,
        PersonaLibrary(),
        "synthia",
        tools=Toolbox([private_reader()]),
        trace=trace,
        memory=memory,
    )


async def nothing_learned() -> int | None:
    return None


def conversation(
    tmp_path: Path,
    router: Router,
    routes: LastRoute,
    memory: Remembering | None = None,
) -> Conversation:
    session = ChatSession(router, PersonaLibrary(), "synthia", routes, memory=memory)
    gateway = Gateway(router, router, UsageLog(tmp_path / "usage.db"), nothing_learned)
    return Conversation(session, gateway)


async def seeded(store: MemoryStore, *, confined: bool) -> int:
    chat = await store.begin_conversation("SYNTHIA", MONDAY)
    turn = Turn(
        "my cat is called miso",
        "Miso is a lovely name.",
        MONDAY,
        "SYNTHIA",
        "local",
        "m",
        confined=confined,
    )
    await store.add_turn(chat, turn)
    return chat


async def test_reading_private_text_keeps_the_chat_local_until_reset(
    tmp_path: Path,
) -> None:
    model = Scripted(
        calls(("read_note", "{}")), says("it says 4021"), says("ok"), says("hi")
    )
    store = MemoryStore(tmp_path / "memory.db")
    trace = Trace(tmp_path / "trace.jsonl")
    chat = reading(model, trace=trace, memory=Remembering(Recall(store)))
    chat.use_remote = True

    first = await reported(chat, PRIVATE_QUESTION)
    left_after_reading = chat.may_leave
    second = await reported(chat, "thanks")
    chat.reset()
    third = await reported(chat, "hello")

    assert [r.confined for r in (first, second, third)] == [True, True, False]
    assert (left_after_reading, chat.may_leave) == (False, True)
    # The turn's second step still asks: the router keeps it local.
    assert [r.use_remote for r in model.requests] == [True, True, False, True]
    began = [r for r in read_trace(trace.path)[0] if isinstance(r, TurnBegan)]
    assert [b.use_remote for b in began] == [True, False, True]
    kept = {t.question: t.confined for t in await store.search("")}
    assert kept == {PRIVATE_QUESTION: True, "thanks": True, "hello": False}


async def test_each_chat_puts_its_own_latch_in_context() -> None:
    private = reading(Scripted(calls(("read_note", "{}")), says("4021")))
    other = reading(Scripted(says("hello")))

    await reported(private, PRIVATE_QUESTION)
    report = await reported(other, "hi")

    assert private.confinement.reason == READ_NOTE
    assert (report.confined, other.confinement.confined) == (False, False)


async def test_a_reminder_of_a_private_turn_confines_the_turn_before_it_asks(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    await seeded(store, confined=True)
    recall = Recall(store, embedder(Network()))
    await recall.backfill()
    model = Scripted(says("Miso."))
    chat = ChatSession(
        model,
        PersonaLibrary(),
        "synthia",
        memory=Remembering(recall, can_stay_local=lambda: True),
    )
    chat.use_remote = True

    report = await reported(chat, "what is the name of the pet")

    (request,) = model.requests
    assert request.messages[-1].text.startswith(REMINDER_HEADER)
    assert "my cat is called miso" in request.messages[-1].text
    assert (request.use_remote, report.confined) == (False, True)


async def test_remote_on_in_a_confined_chat_stays_local_until_reset(
    tmp_path: Path, setup: Setup
) -> None:
    remote, local, routes = Fake(REMOTE), Fake(LOCAL), LastRoute()
    talk = conversation(
        tmp_path, Router(local, Remote(remote, setup.health), routes), routes
    )
    talk.session.confinement.confine(READ_NOTE)

    said = await talk.remote(on=True)
    shown = await talk.remote(on=None)
    await reported(talk, "hi")
    kept_local = routes.decision
    await talk.reset()
    await reported(talk, "hi again")
    await talk.close()

    stays = (
        f"remote: stays off in this chat: {READ_NOTE}; "
        "/reset starts a chat that may leave this machine"
    )
    assert said.notes == shown.notes == (stays,)
    assert kept_local is not None
    assert (kept_local.route, kept_local.reason) == (Route.LOCAL, RouteReason.CONFINED)
    assert routes.decision is not None
    assert routes.decision.route is Route.REMOTE
    assert (local.calls, remote.calls) == (1, 1)


async def test_a_resumed_private_conversation_confines_the_chat(
    tmp_path: Path, setup: Setup
) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    await seeded(store, confined=True)
    remote, local, routes = Fake(REMOTE), Fake(LOCAL), LastRoute()
    router = Router(local, Remote(remote, setup.health), routes)
    talk = conversation(tmp_path, router, routes, Remembering(Recall(store)))
    talk.session.use_remote = True

    reply = await talk.resume()
    report = await reported(talk, "and her sister?")

    assert reply.errors == ()
    assert talk.session.confinement.reason == RESUMED_PRIVATE
    assert routes.decision is not None
    assert routes.decision.reason is RouteReason.CONFINED
    assert (report.confined, remote.calls) == (True, 0)


@pytest.mark.parametrize("confined", [True, False])
async def test_without_a_local_model_a_private_conversation_is_not_resumed(
    tmp_path: Path, setup: Setup, confined: bool
) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    earlier = await seeded(store, confined=confined)
    remote, routes = Fake(REMOTE), LastRoute()
    router = Router(None, Remote(remote, setup.health), routes)
    talk = conversation(tmp_path, router, routes, Remembering(Recall(store)))
    talk.session.use_remote = True

    reply = await talk.resume()
    await reported(talk, "hello")

    assert reply.errors == ((PRIVATE_NEEDS_LOCAL,) if confined else ())
    last = await store.last_conversation()
    assert last is not None
    # Refused, the chat starts a conversation of its own.
    assert (last[0] == earlier) is not confined
    assert remote.calls == 1


async def test_reset_and_forget_are_refused_while_a_turn_runs(
    tmp_path: Path, setup: Setup
) -> None:
    holding, let_go = asyncio.Event(), asyncio.Event()

    async def hold() -> str:
        """Wait until the test lets go."""
        holding.set()
        await let_go.wait()
        return "held"

    tools = Toolbox([FunctionTool.of(hold, reach=Reach.LOCAL, effect=Effect.READ)])
    session = ChatSession(
        Scripted(calls(("hold", "{}")), says("done")),
        PersonaLibrary(),
        "synthia",
        tools=tools,
        memory=Remembering(Recall(MemoryStore(tmp_path / "memory.db"))),
    )
    router = Router(Fake(LOCAL), Remote(Fake(REMOTE), setup.health), LastRoute())
    gateway = Gateway(router, router, UsageLog(tmp_path / "usage.db"), nothing_learned)
    talk = Conversation(session, gateway)
    session.confinement.confine(READ_NOTE)

    turn = asyncio.create_task(reported(talk, "hi"))
    await asyncio.wait_for(holding.wait(), HANG_TIMEOUT_S)
    refused = [await talk.reset(), await talk.forget(), await talk.resume()]
    held_closed = session.confinement.confined
    let_go.set()
    await asyncio.wait_for(turn, HANG_TIMEOUT_S)
    after = await talk.reset()

    assert [reply.errors for reply in refused] == [
        (TURN_RUNNING,),
        (TURN_RUNNING,),
        (ALREADY_TALKING,),
    ]
    assert held_closed
    assert (after.errors, session.confinement.confined) == ((), False)
    assert session.history == []


class Serving(Service):
    def __init__(self, served: Local) -> None:
        super().__init__()
        self.served = served

    def model(self, client: httpx.AsyncClient) -> Local:
        del client
        return self.served


async def test_the_daemon_shows_private_turns_to_a_chat_that_stays_local(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / MEMORY_FILE)
    await seeded(store, confined=True)
    await Recall(store, embedder(Network())).backfill()
    model = Local(calls(("search_memory", '{"query": "cat"}')), says("Miso."))
    stop = asyncio.Event()
    running = asyncio.create_task(
        run_daemon(
            Settings(home=tmp_path), Serving(model), stop=stop, embedder=toy_embedder
        )
    )
    info = await written(tmp_path, running)
    async with client(info) as talk:
        report = await talk.call("turn", {"text": "what is the name of the pet"})
    stop.set()
    await asyncio.wait_for(running, HANG_TIMEOUT_S)

    assert cast("dict[str, object]", report)["confined"] is True
    assert "my cat is called miso" in model.requests[0].messages[-1].text
    (found,) = [p["result"] for name, p in talk.notes if name == "tool"]
    assert "my cat is called miso" in str(found)
