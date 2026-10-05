import asyncio
import logging
from pathlib import Path

import pytest

from synthia.kernel.config import Settings
from synthia.memory.hybrid import Recall
from synthia.memory.store import MEMORY_FILE, MemoryStore
from synthia.models.install import BYTES_PER_GB, Installer
from synthia.server.host import fill_in, filling_in, open_embedder, open_memory
from tests.memory.test_hybrid import Network, embedder, remember
from tests.memory.test_remembering import FailsAfterStart
from tests.models.fakes import EMBEDDER, NETWORK, VOCABULARY


def test_with_no_embedding_model_installed_memory_is_searched_by_words(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    warnings: list[str] = []

    with caplog.at_level(logging.INFO, "synthia.server.host"):
        found = open_embedder(Settings(home=tmp_path), warnings.append, (EMBEDDER,))

    assert found is None
    assert warnings == []
    assert "no embedding model installed" in caplog.text


def test_an_installed_model_that_cannot_load_is_warned_about(tmp_path: Path) -> None:
    folder = Installer(tmp_path, BYTES_PER_GB).path_of(EMBEDDER)
    folder.mkdir(parents=True)
    (folder / EMBEDDER.model.name).write_bytes(NETWORK)
    (folder / EMBEDDER.tokenizer.name).write_bytes(VOCABULARY)
    warnings: list[str] = []

    found = open_embedder(Settings(home=tmp_path), warnings.append, (EMBEDDER,))

    assert found is None
    (warning,) = warnings
    assert warning.startswith(
        "memory is searched by words only: "
        "cannot load the embedding model tiny-embedder"
    )


async def test_memory_comes_back_with_the_vectors_already_kept(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / MEMORY_FILE)
    await remember(store, "my cat is called miso", "my sister lives in pune")
    await Recall(store, embedder(Network())).backfill()
    warnings: list[str] = []

    memory = await open_memory(
        Settings(home=tmp_path), warnings.append, embedder(Network())
    )

    assert memory is not None
    assert warnings == []
    found = await memory.find("pet", limit=1)
    assert [r.turn.question for r in found] == ["my cat is called miso"]


async def test_memory_that_cannot_be_opened_is_warned_about(tmp_path: Path) -> None:
    (tmp_path / MEMORY_FILE).mkdir()
    warnings: list[str] = []

    assert await open_memory(Settings(home=tmp_path), warnings.append) is None
    (warning,) = warnings
    assert warning.startswith("nothing will be remembered: ")


async def test_earlier_turns_are_embedded_and_counted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = MemoryStore(tmp_path / MEMORY_FILE)
    await remember(store, "first", "second")

    with caplog.at_level(logging.INFO, "synthia.server.host"):
        await fill_in(Recall(store, embedder(Network())))

    assert await store.unembedded(EMBEDDER.id, 5) == []
    assert "embedded 2 earlier turns" in caplog.text
    caplog.clear()

    with caplog.at_level(logging.INFO, "synthia.server.host"):
        await fill_in(Recall(store, embedder(Network())))

    assert "embedded" not in caplog.text


async def test_a_failing_model_is_logged_and_the_daemon_goes_on(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = MemoryStore(tmp_path / MEMORY_FILE)
    await remember(store, "first")

    with caplog.at_level(logging.ERROR, "synthia.server.host"):
        await fill_in(Recall(store, embedder(FailsAfterStart())))

    assert "earlier turns were not all embedded" in caplog.text
    assert len(await store.unembedded(EMBEDDER.id, 5)) == 1


class Stuck(Recall):
    """Memory whose backfill runs until it is cancelled."""

    def __init__(self, store: MemoryStore) -> None:
        super().__init__(store)
        self.started = asyncio.Event()
        self.cancelled = False

    async def backfill(self) -> int:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return 0


async def test_filling_in_is_stopped_when_serving_ends(tmp_path: Path) -> None:
    memory = Stuck(MemoryStore(tmp_path / MEMORY_FILE))

    async with filling_in(memory):
        await asyncio.wait_for(memory.started.wait(), timeout=1)

    assert memory.cancelled


async def test_without_memory_there_is_nothing_to_fill_in() -> None:
    async with filling_in(None):
        pass
