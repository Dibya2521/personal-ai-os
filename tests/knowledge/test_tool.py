"""The toy embedder gives every chunk some meaning in common with any query.

So a search here returns every chunk of the index, best first, up to six,
and what a test sees is decided by which sources the index holds.
"""

import json
import re
from pathlib import Path

import pytest

from synthia.agent.tools import Effect, Reach
from synthia.gateway.confinement import CONFINEMENT, Confinement
from synthia.knowledge.chunk import Chunking
from synthia.knowledge.service import Knowledge
from synthia.knowledge.tool import (
    CITED,
    MAX_BODY_CHARS,
    NOTHING_FOUND,
    knowledge_tool,
)
from tests.knowledge.corpus import planted
from tests.retrieval.toy import Network, embedder

PUBLIC = {
    "kestrel.md": (
        "# Kestrel\n\n## Hosts\n\nthe database server host kestrel\n"
        "see [[pune]], [[cat]], [[pune]] and [[nowhere]]\n"
    ),
    "pune.md": "# Pune trip\n\nmy sister lives in pune city\n",
}
PRIVATE = {"door.txt": "the door code of kestrel is 4021\n"}
KESTREL = "notes/kestrel.md:1-6"
PUNE = "notes/pune.md:1-3"
DOOR = "diary/door.txt:1"
CONFINED = "private files from diary were read"
ONE_WITHHELD = "1 private passage was withheld: this chat cannot stay on this machine"
CITATION = re.compile(r"^\[([^\]]+)\] ", re.MULTILINE)


def stays_local() -> bool:
    return True


def cannot_stay_local() -> bool:
    return False


async def knowing(
    tmp_path: Path, *, files: dict[str, str] = PUBLIC, private: bool = True
) -> Knowledge:
    opened = await Knowledge.open(
        tmp_path / "knowledge.db", embedder(Network()), chunking=Chunking(4096)
    )
    if files:
        await opened.add(planted(tmp_path / "notes", files))
    if private:
        await opened.add(planted(tmp_path / "diary", PRIVATE), private=True)
    await opened.index()
    return opened


@pytest.fixture
async def knowledge(tmp_path: Path) -> Knowledge:
    return await knowing(tmp_path)


def shown(found: str) -> list[str]:
    return CITATION.findall(found)


async def searched(
    knowledge: Knowledge, *, can_stay_local: bool = True, query: str = "kestrel"
) -> str:
    tool = knowledge_tool(
        knowledge, can_stay_local=stays_local if can_stay_local else cannot_stay_local
    )
    return await tool.run(json.dumps({"query": query}))


async def test_the_tool_reads_without_asking_and_says_how_to_use_it(
    knowledge: Knowledge,
) -> None:
    tool = knowledge_tool(knowledge, can_stay_local=stays_local)

    assert (tool.spec.name, tool.reach, tool.effect) == (
        "search_knowledge",
        Reach.LOCAL,
        Effect.READ,
    )
    assert "citations you used under your answer" in tool.spec.description
    assert "search for one of those titles" in tool.spec.description


async def test_a_chat_that_can_stay_local_reads_private_text_and_is_confined(
    knowledge: Knowledge,
) -> None:
    latch = Confinement()
    CONFINEMENT.set(latch)

    found = await searched(knowledge)

    assert f"[{DOOR}] door.txt\nthe door code of kestrel is 4021\n" in found
    assert f"\n\n[{KESTREL}] Kestrel > Hosts\n# Kestrel\n" in found
    assert "withheld" not in found
    assert latch.reason == CONFINED


async def test_a_chat_that_cannot_stay_local_has_private_text_withheld(
    knowledge: Knowledge,
) -> None:
    latch = Confinement()
    CONFINEMENT.set(latch)

    found = await searched(knowledge, can_stay_local=False)

    assert "4021" not in found
    assert found.startswith(f"[{KESTREL}]")
    assert found.endswith(f"\n\n{ONE_WITHHELD}")
    assert not latch.confined


async def test_without_a_latch_in_context_private_text_is_withheld(
    knowledge: Knowledge,
) -> None:
    found = await searched(knowledge)

    assert "4021" not in found
    assert found.endswith(f"\n\n{ONE_WITHHELD}")


async def test_withheld_passages_are_counted(tmp_path: Path) -> None:
    knowledge = await knowing(tmp_path, files={"safe.txt": "the safe code\n"})
    await knowledge.add(
        planted(tmp_path / "more", {"lock.txt": "the lock code\n"}), private=True
    )
    await knowledge.index()

    found = await searched(knowledge)

    assert found.endswith(
        "\n\n2 private passages were withheld: this chat cannot stay on this machine"
    )


async def test_only_private_text_withheld_says_only_that(tmp_path: Path) -> None:
    knowledge = await knowing(tmp_path, files={})

    assert await searched(knowledge) == ONE_WITHHELD


async def test_public_text_leaves_the_latch_open(tmp_path: Path) -> None:
    knowledge = await knowing(tmp_path, private=False)
    latch = Confinement()
    CONFINEMENT.set(latch)

    found = await searched(knowledge)

    assert f"[{KESTREL}]" in found
    assert f"[{PUNE}] Pune trip\n" in found
    assert "withheld" not in found
    assert not latch.confined


async def test_a_question_nothing_answers_says_so(tmp_path: Path) -> None:
    knowledge = await knowing(tmp_path, files={}, private=False)
    CONFINEMENT.set(Confinement())

    assert await searched(knowledge) == NOTHING_FOUND


async def test_a_passage_lists_the_titles_it_links_to_once_in_order(
    knowledge: Knowledge,
) -> None:
    found = await searched(knowledge)

    assert "[[nowhere]]\nlinks to: Pune trip\n" in found
    assert found.count("links to:") == 1


async def test_links_reach_every_document_of_the_source_they_name(
    tmp_path: Path,
) -> None:
    files = {**PUBLIC, "cat.md": "# Cat\n\nmy cat is miso\n"}
    knowledge = await knowing(tmp_path, files=files, private=False)

    found = await searched(knowledge)

    assert "[[nowhere]]\nlinks to: Pune trip, Cat\n" in found
    assert found.count("links to:") == 1


async def test_a_private_document_is_never_named_by_a_public_link(
    tmp_path: Path,
) -> None:
    files = {"kestrel.md": "# Kestrel\n\nthe host kestrel, see [[door]]\n"}
    knowledge = await knowing(tmp_path, files=files)

    found = await searched(knowledge, can_stay_local=False)

    assert "links to" not in found


async def test_the_citations_shown_are_added_to_the_turns_list(
    knowledge: Knowledge,
) -> None:
    CONFINEMENT.set(Confinement())
    cited = ["earlier/note.md:1"]
    CITED.set(cited)

    first = await searched(knowledge)
    second = await searched(knowledge, query="pune")

    assert cited == ["earlier/note.md:1", *shown(first), *shown(second)]
    assert sorted(shown(first)) == sorted([DOOR, KESTREL, PUNE])


async def test_withheld_citations_are_not_added(knowledge: Knowledge) -> None:
    cited: list[str] = []
    CITED.set(cited)

    await searched(knowledge)

    assert cited == [KESTREL, PUNE]


LINE = "x" * 99 + "\n"
WHOLE = "x" * MAX_BODY_CHARS


@pytest.mark.parametrize(
    ("text", "span", "body"),
    [
        pytest.param(LINE * 30, "1-30", LINE * 24 + "...", id="cut on a line end"),
        pytest.param(WHOLE + "xy", "1", WHOLE + "...", id="one line past the limit"),
        pytest.param(WHOLE, "1", WHOLE, id="2,400 characters kept whole"),
        pytest.param(
            LINE * 24, "1-24", LINE * 23 + "x" * 99, id="2,399 characters kept whole"
        ),
        pytest.param(
            f"{WHOLE}\ny\n", "1-2", f"{WHOLE}\n...", id="a line ending at the limit"
        ),
        pytest.param(
            "short\n" + "y" * 3000,
            "1-2",
            ("short\n" + "y" * 3000)[:MAX_BODY_CHARS] + "...",
            id="an early line end is not where it is cut",
        ),
    ],
)
async def test_a_long_body_is_cut(
    tmp_path: Path, text: str, span: str, body: str
) -> None:
    knowledge = await knowing(tmp_path, files={"long.txt": text}, private=False)

    found = await searched(knowledge)

    assert found == f"[notes/long.txt:{span}] long.txt\n{body}"
