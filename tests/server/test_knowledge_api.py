import asyncio
import datetime as dt
import json
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import cast

import httpx
import pytest
from pydantic import SecretStr

from synthia.agent.tools import Toolbox
from synthia.agent.trace import TRACES, IndexRan, read_trace, sessions
from synthia.gateway.assemble import build_gateway
from synthia.gateway.types import ChatChunk, ChatRequest
from synthia.interfaces.daemon_client import (
    KnownSource,
    ask_daemon,
    indexed_from,
    removed_from,
    searched_from,
    source_from,
    sources_from,
)
from synthia.kernel.config import Settings
from synthia.kernel.jsonrpc import RpcError
from synthia.knowledge.deep import NO_LOCAL_MODEL, NO_REWRITES
from synthia.knowledge.model import SourceKind
from synthia.memory.store import MEMORY_FILE, MemoryStore, Turn
from synthia.persona.library import PersonaLibrary
from synthia.server.api import (
    ANSWER_FAILED,
    INVALID_PARAMS,
    KNOWLEDGE_REFUSED,
    NO_KNOWLEDGE,
    NO_MEMORY,
    NOT_INDEXING,
    NOT_REMEMBERING,
)
from synthia.server.daemon import serve
from synthia.server.host import Host, publish_route
from synthia.server.upkeep import Activity
from tests.agent.scripted import says
from tests.interfaces.daemons import private_daemon, toy_embedder
from tests.interfaces.test_chat import KEY
from tests.knowledge.corpus import NOTES, planted
from tests.server.test_confinement import Serving
from tests.server.test_daemon import Local, client, written
from tests.timing import HANG_TIMEOUT_S


def keyed(home: Path) -> Settings:
    return Settings(home=home, openrouter_api_key=SecretStr(KEY))


@pytest.fixture
def notes(tmp_path: Path) -> Path:
    return planted(tmp_path / "notes", NOTES)


async def refused(home: Path, method: str, params: dict[str, object]) -> RpcError:
    with pytest.raises(RpcError) as raised:
        await ask_daemon(home, method, params)
    return raised.value


async def test_a_folder_is_added_listed_indexed_searched_and_removed(
    tmp_path: Path, notes: Path
) -> None:
    with private_daemon(keyed(tmp_path), embedder=toy_embedder):
        added = source_from(
            await ask_daemon(
                tmp_path,
                "knowledge_add",
                {"root": str(notes), "private": True, "excludes": ["drafts\\"]},
            )
        )
        listed = sources_from(await ask_daemon(tmp_path, "knowledge_sources", {}))
        indexed = indexed_from(await ask_daemon(tmp_path, "knowledge_index", {}))
        found = searched_from(
            await ask_daemon(
                tmp_path, "knowledge_search", {"query": "kestrel", "variant": "words"}
            )
        )
        removed = [
            removed_from(
                await ask_daemon(tmp_path, "knowledge_remove", {"root": str(notes)})
            )
            for _ in range(2)
        ]
        left = sources_from(await ask_daemon(tmp_path, "knowledge_sources", {}))

    assert added == KnownSource(
        "notes", notes.resolve(), SourceKind.FILES, private=True, excludes=("drafts",)
    )
    assert listed == [added]
    assert indexed.error is None
    assert indexed.report is not None
    assert (indexed.report.added, indexed.report.embedded) == (3, 3)
    (passage,) = found.passages
    assert (passage.citation, passage.body, passage.private) == (
        "notes/server.txt:1",
        "the database server host kestrel",
        True,
    )
    assert passage.score > 0
    assert found.withheld == 0
    assert removed == [True, False]
    assert left == []


async def test_an_asked_pass_goes_to_the_daemon_trace(
    tmp_path: Path, notes: Path
) -> None:
    with private_daemon(keyed(tmp_path), embedder=toy_embedder):
        await ask_daemon(tmp_path, "knowledge_add", {"root": str(notes)})
        await ask_daemon(tmp_path, "knowledge_index", {})

    passes = [
        record
        for path in sessions(tmp_path / TRACES)
        for record in read_trace(path)[0]
        if isinstance(record, IndexRan)
    ]
    (traced,) = passes
    assert (traced.scheduled, traced.error) == (False, None)
    assert traced.counts is not None
    assert (traced.counts.added, traced.counts.chunks) == (3, 3)


async def test_what_knowledge_or_the_params_refuse_is_an_error_with_its_code(
    tmp_path: Path,
) -> None:
    with private_daemon(keyed(tmp_path), embedder=toy_embedder):
        missing = await refused(
            tmp_path, "knowledge_add", {"root": str(tmp_path / "missing")}
        )
        variant = await refused(
            tmp_path, "knowledge_search", {"query": "cat", "variant": "loud"}
        )
        unloaded = await refused(
            tmp_path,
            "knowledge_search",
            {"query": "cat", "variant": "fused+rerank:ms-marco-minilm-l6-int8@10"},
        )
        limit = await refused(
            tmp_path, "knowledge_search", {"query": "cat", "limit": 51}
        )
        relative = [
            await refused(tmp_path, method, {"root": "notes"})
            for method in ("knowledge_add", "knowledge_remove")
        ]

    assert missing.code == KNOWLEDGE_REFUSED
    assert "missing cannot be added" in str(missing)
    assert variant.code == INVALID_PARAMS
    assert str(variant).startswith("variant: 'loud' is not a variant")
    assert unloaded.code == INVALID_PARAMS
    assert str(unloaded).endswith("needs the reranker ms-marco-minilm-l6-int8")
    assert limit.code == INVALID_PARAMS
    assert str(limit).startswith("limit: ")
    assert [(e.code, str(e)) for e in relative] == 2 * [
        (INVALID_PARAMS, "root: Value error, must be an absolute path")
    ]


async def test_a_search_can_leave_out_private_sources_and_keep_to_named_ones(
    tmp_path: Path, notes: Path
) -> None:
    public = planted(tmp_path / "public", {"bird.txt": "a kestrel over the field\n"})
    asked: dict[str, object] = {"query": "kestrel", "variant": "words"}
    narrowed: list[dict[str, object]] = [
        {"sources": ["public"]},
        {"private": False},
        {"private": False, "sources": ["notes"]},
    ]
    with private_daemon(keyed(tmp_path), embedder=toy_embedder):
        await ask_daemon(
            tmp_path, "knowledge_add", {"root": str(notes), "private": True}
        )
        await ask_daemon(tmp_path, "knowledge_add", {"root": str(public)})
        await ask_daemon(tmp_path, "knowledge_index", {})
        searches = [
            searched_from(await ask_daemon(tmp_path, "knowledge_search", asked | more))
            for more in narrowed
        ]
        too_many = await refused(
            tmp_path,
            "knowledge_search",
            asked | {"sources": [str(n) for n in range(257)]},
        )

    assert [([p.citation for p in s.passages], s.withheld) for s in searches] == [
        (["public/bird.txt:1"], 0),
        (["public/bird.txt:1"], 1),
        ([], 1),
    ]
    assert too_many.code == INVALID_PARAMS


async def test_memory_is_searched_with_confined_turns_left_out_when_asked(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / MEMORY_FILE)
    for day, confined in ((1, False), (5, True)):
        at = dt.datetime(2026, 11, day, 12, tzinfo=dt.UTC)
        turn = Turn(
            "where is the kestrel",
            f"on day {day}",
            at,
            "synthia",
            "local",
            "m",
            confined=confined,
        )
        await store.add_turn(await store.begin_conversation("synthia", at), turn)
    asked: list[dict[str, object]] = [
        {"query": "kestrel"},
        {"query": "kestrel", "private": False},
        {"query": "kestrel", "since": "2026-11-03"},
    ]
    with private_daemon(keyed(tmp_path), embedder=toy_embedder):
        searches = [
            cast("dict[str, object]", await ask_daemon(tmp_path, "memory_search", a))
            for a in asked
        ]

    answers = [
        [t["answer"] for t in cast("list[dict[str, object]]", s["turns"])]
        for s in searches
    ]
    assert sorted(cast("list[str]", answers[0])) == ["on day 1", "on day 5"]
    assert answers[1:] == [["on day 1"], ["on day 5"]]
    assert [s["withheld"] for s in searches] == [0, 1, 0]


async def test_a_daemon_whose_memory_cannot_open_says_so(tmp_path: Path) -> None:
    (tmp_path / MEMORY_FILE).mkdir()
    with private_daemon(keyed(tmp_path), embedder=toy_embedder):
        error = await refused(tmp_path, "memory_search", {"query": "kestrel"})

    assert (error.code, str(error)) == (NO_MEMORY, NOT_REMEMBERING)


async def indexed_daemon_search(
    home: Path, notes: Path, model: Local | None, params: dict[str, object]
) -> dict[str, object]:
    """Index ``notes`` in a daemon whose local model is ``model``, then search."""
    local = None if model is None else Serving(model)
    with private_daemon(keyed(home), local=local, embedder=toy_embedder):
        await ask_daemon(home, "knowledge_add", {"root": str(notes)})
        await ask_daemon(home, "knowledge_index", {})
        result = await ask_daemon(home, "knowledge_search", params)
    return cast("dict[str, object]", result)


async def test_a_deep_search_says_what_the_local_model_added_and_its_seconds(
    tmp_path: Path, notes: Path
) -> None:
    model = Local(says(json.dumps({"queries": ["miso"], "passage": "a cat"})))
    asked: dict[str, object] = {"query": "kestrel", "variant": "words", "deep": True}

    result = await indexed_daemon_search(tmp_path, notes, model, asked)

    assert result["expansion"] == {"queries": ["miso"], "passage": "a cat"}
    seconds = cast("dict[str, float]", result["seconds"])
    assert sorted(seconds) == ["expand", "search"]
    assert min(seconds.values()) >= 0
    assert result["note"] is None
    found = searched_from(result)
    assert sorted(p.citation for p in found.passages) == [
        "notes/cat.txt:1",
        "notes/server.txt:1",
    ]
    (request,) = model.requests
    assert (request.use_remote, request.messages[-1].text) == (False, "kestrel")


async def test_a_deep_search_with_no_valid_rewrites_is_the_plain_one_with_a_note(
    tmp_path: Path, notes: Path
) -> None:
    model = Local(says("no"), says("still no"), says("never"))
    asked: dict[str, object] = {"query": "kestrel", "variant": "words", "deep": True}

    result = await indexed_daemon_search(tmp_path, notes, model, asked)

    assert (result["expansion"], result["note"]) == (None, NO_REWRITES)
    assert [p.citation for p in searched_from(result).passages] == [
        "notes/server.txt:1"
    ]
    assert len(model.requests) == 3


@pytest.mark.parametrize("variant", ["loud", "fused+rerank:ms-marco-minilm-l6-int8@10"])
async def test_a_bad_variant_is_refused_before_the_local_model_is_asked(
    tmp_path: Path, variant: str
) -> None:
    model = Local()
    asked: dict[str, object] = {"query": "kestrel", "variant": variant, "deep": True}
    with private_daemon(keyed(tmp_path), local=Serving(model), embedder=toy_embedder):
        error = await refused(tmp_path, "knowledge_search", asked)

    assert error.code == INVALID_PARAMS
    assert str(error).startswith("variant: ")
    assert model.requests == []


async def test_a_deep_search_with_no_local_model_is_refused(
    tmp_path: Path, notes: Path
) -> None:
    with pytest.raises(RpcError) as raised:
        await indexed_daemon_search(
            tmp_path, notes, None, {"query": "kestrel", "deep": True}
        )

    assert raised.value.code == ANSWER_FAILED
    assert str(raised.value) == f"deep search failed: {NO_LOCAL_MODEL}"


async def test_a_daemon_without_an_embedding_model_has_no_knowledge(
    tmp_path: Path,
) -> None:
    asked: list[tuple[str, dict[str, object]]] = [
        ("knowledge_add", {"root": str(tmp_path)}),
        ("knowledge_remove", {"root": str(tmp_path)}),
        ("knowledge_sources", {}),
        ("knowledge_index", {}),
        ("knowledge_search", {"query": "cat"}),
    ]
    with private_daemon(keyed(tmp_path)):
        errors = [await refused(tmp_path, method, params) for method, params in asked]

    assert {(error.code, str(error)) for error in errors} == {
        (NO_KNOWLEDGE, NOT_INDEXING)
    }


class Gated(Local):
    """A local model that starts its answer and holds it until let go."""

    def __init__(self) -> None:
        super().__init__()
        self.answering = asyncio.Event()
        self.let_go = asyncio.Event()

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        del request
        self.answering.set()
        await self.let_go.wait()
        for chunk in says("Hello."):
            yield chunk


async def test_a_turn_holds_the_daemon_busy_until_it_ends(tmp_path: Path) -> None:
    model = Gated()
    activity = Activity()
    async with httpx.AsyncClient() as web:
        gateway = build_gateway(Settings(home=tmp_path), web, publish_route, model)
        host = Host(gateway, PersonaLibrary(), "synthia", Toolbox(), activity=activity)
        stop = asyncio.Event()
        serving = asyncio.create_task(serve(host, tmp_path, stop=stop))
        info = await written(tmp_path, serving)
        async with client(info) as talk:
            turn = asyncio.create_task(talk.call("turn", {"text": "hi"}))
            await asyncio.wait_for(model.answering.wait(), HANG_TIMEOUT_S)
            busy = activity.running
            model.let_go.set()
            await turn
            after = activity.running
        stop.set()
        await asyncio.wait_for(serving, HANG_TIMEOUT_S)

    assert (busy, after) == (True, False)
