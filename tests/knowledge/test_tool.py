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
from synthia.gateway.types import ToolSpec
from synthia.knowledge.chunk import Chunking
from synthia.knowledge.deep import NO_REWRITES
from synthia.knowledge.service import Knowledge
from synthia.knowledge.tool import (
    CITED,
    DEEP,
    MAX_BODY_CHARS,
    NOTHING_FOUND,
    SEARCH_KNOWLEDGE,
    knowledge_tool,
)
from tests.agent.scripted import Scripted, says
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
    knowledge: Knowledge,
    *,
    can_stay_local: bool = True,
    query: str = "kestrel",
    model: Scripted | None = None,
) -> str:
    """Search as the model does; a model with no reply fails the test if asked."""
    tool = knowledge_tool(
        knowledge,
        can_stay_local=stays_local if can_stay_local else cannot_stay_local,
        model=model or Scripted(),
    )
    return await tool.run(json.dumps({"query": query}))


async def deeply(
    knowledge: Knowledge, model: Scripted, *, can_stay_local: bool = True
) -> str:
    """Search deep, as the session does for ``/deep``."""
    deep = DEEP.set(True)
    try:
        return await searched(knowledge, can_stay_local=can_stay_local, model=model)
    finally:
        DEEP.reset(deep)


def rewritten(*queries: str, passage: str = "") -> Scripted:
    return Scripted(says(json.dumps({"queries": queries, "passage": passage})))


async def test_the_tool_reads_without_asking_and_says_how_to_use_it(
    knowledge: Knowledge,
) -> None:
    tool = knowledge_tool(knowledge, can_stay_local=stays_local, model=Scripted())

    assert (tool.spec.name, tool.reach, tool.effect) == (
        SEARCH_KNOWLEDGE,
        Reach.LOCAL,
        Effect.READ,
    )
    assert "citations you used under your answer" in tool.spec.description
    assert "search for one of those titles" in tool.spec.description


async def test_what_the_model_sees_of_the_tool_has_no_way_to_ask_for_deep(
    knowledge: Knowledge,
) -> None:
    tool = knowledge_tool(knowledge, can_stay_local=stays_local, model=Scripted())

    assert tool.spec == ToolSpec(
        "search_knowledge",
        "Search the person's own files (notes, documents and code) for passages "
        "that answer a question about their projects, notes or code. Each passage "
        "comes under its citation in square brackets: name the citations you used "
        "under your answer. A passage may end with the titles of documents it "
        "links to; search for one of those titles to follow the link.",
        {
            "additionalProperties": False,
            "properties": {
                "query": {
                    "description": "what to look for, in any words",
                    "minLength": 1,
                    "type": "string",
                }
            },
            "required": ["query"],
            "type": "object",
        },
    )


async def test_a_deep_search_asks_the_local_model_once_and_shows_what_it_finds(
    tmp_path: Path,
) -> None:
    knowledge = await knowing(tmp_path, private=False)
    model = rewritten("sister pune", passage="my sister lives in pune city")

    found = await deeply(knowledge, model)

    (request,) = model.requests
    assert (request.use_remote, request.messages[-1].text) == (False, "kestrel")
    expected = await knowledge.search(
        "kestrel", also=("sister pune", "my sister lives in pune city")
    )
    assert shown(found) == [passage.citation() for passage in expected.passages]


async def test_a_deep_search_over_a_private_file_confines_the_chat(
    knowledge: Knowledge,
) -> None:
    latch = Confinement()
    CONFINEMENT.set(latch)

    found = await deeply(knowledge, rewritten("door code"))

    assert DOOR in shown(found)
    assert latch.reason == CONFINED


async def test_a_deep_search_in_a_chat_that_cannot_stay_local_withholds_private(
    knowledge: Knowledge,
) -> None:
    latch = Confinement()
    CONFINEMENT.set(latch)

    found = await deeply(knowledge, rewritten("door code"), can_stay_local=False)

    assert found.endswith(f"\n\n{ONE_WITHHELD}")
    assert not latch.confined


async def test_a_deep_search_with_no_valid_rewrites_says_it_searched_plainly(
    knowledge: Knowledge,
) -> None:
    model = Scripted(says("no"), says("still no"), says("never"))

    found = await deeply(knowledge, model)

    assert found == f"{await searched(knowledge)}\n\n{NO_REWRITES}"


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
