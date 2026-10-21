import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

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
from synthia.knowledge.model import SourceKind
from synthia.persona.library import PersonaLibrary
from synthia.server.api import (
    INVALID_PARAMS,
    KNOWLEDGE_REFUSED,
    NO_KNOWLEDGE,
    NOT_INDEXING,
)
from synthia.server.daemon import serve
from synthia.server.host import Host, publish_route
from synthia.server.upkeep import Activity
from tests.agent.scripted import says
from tests.interfaces.daemons import private_daemon, toy_embedder
from tests.interfaces.test_chat import KEY
from tests.knowledge.corpus import NOTES, planted
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
