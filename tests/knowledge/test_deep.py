import json
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from synthia.gateway.structured import StructuredOutputError, schema_of
from synthia.gateway.types import (
    ChatChunk,
    ChatRequest,
    FinishReason,
    ModelInfo,
    Reasoning,
    Role,
)
from synthia.knowledge.deep import (
    EXAMPLE,
    INSTRUCTIONS,
    MAX_PASSAGE_CHARS,
    MAX_QUERIES,
    MAX_QUERY_CHARS,
    Expansion,
    Rewrites,
    deep_search,
    expand,
    expansion_request,
)
from synthia.knowledge.retrieve import Merge, Side, Variant
from synthia.knowledge.service import Knowledge
from tests.knowledge.corpus import NOTES, citations, opened, planted

INFO = ModelInfo("local", 4096, vision=False, tools=True)
QUESTION = "Who looks after the bird?"


class Local:
    """A local model replying with the scripted texts in turn, recording requests."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.requests: list[ChatRequest] = []

    @property
    def info(self) -> ModelInfo:
        return INFO

    def ready(self) -> bool:
        return True

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        self.requests.append(request)
        yield ChatChunk(text=self.replies.pop(0))
        yield ChatChunk(finish_reason=FinishReason.STOP)


def reply(queries: list[str], passage: str = "The kestrel is fed daily.") -> str:
    return json.dumps({"queries": queries, "passage": passage})


async def test_expand_sends_one_local_request_without_reasoning() -> None:
    model = Local(reply(["kestrel keeper", "bird care rota"]))

    expansion = await expand(model, QUESTION)

    assert expansion == Expansion(
        ("kestrel keeper", "bird care rota"), "The kestrel is fed daily."
    )
    (request,) = model.requests
    assert request == ChatRequest(
        expansion_request(QUESTION).messages,
        temperature=0.0,
        response_schema=schema_of(Rewrites),
        reasoning=Reasoning.OFF,
        use_remote=False,
    )
    assert [m.role for m in request.messages] == [
        Role.SYSTEM,
        Role.USER,
        Role.ASSISTANT,
        Role.USER,
    ]
    assert request.messages[0].text == INSTRUCTIONS
    assert request.messages[-1].text == QUESTION


def test_the_worked_example_is_a_full_valid_reply() -> None:
    _, answer = EXAMPLE
    example = Rewrites.model_validate_json(answer.text)

    assert len(example.queries) == MAX_QUERIES
    assert example.passage.count(". ") >= 2


@pytest.mark.parametrize(
    ("queries", "passage", "expected"),
    [
        (
            [" Kestrel keeper ", "kestrel KEEPER", "who looks after the BIRD?"],
            "  fed daily  ",
            Expansion(("Kestrel keeper",), "fed daily"),
        ),
        (["", "   ", "rota"], "p", Expansion(("rota",), "p")),
        (["  who looks after the bird?  "], "   ", Expansion((), None)),
        ([], "", Expansion((), None)),
        (["a", "B", "c"], "p", Expansion(("a", "B", "c"), "p")),
    ],
    ids=["strip_and_dedupe", "blanks", "only_the_question", "nothing", "three"],
)
def test_an_expansion_is_stripped_and_drops_blanks_and_repeats(
    queries: list[str], passage: str, expected: Expansion
) -> None:
    rewrites = Rewrites(queries=queries, passage=passage)

    assert Expansion.of(QUESTION, rewrites) == expected


@pytest.mark.parametrize(
    "bad",
    [
        reply(["a", "b", "c", "d"]),
        reply(["a" * (MAX_QUERY_CHARS + 1)]),
        reply(["a"], "p" * (MAX_PASSAGE_CHARS + 1)),
        '{"queries": ["a"], "passage": "p", "answer": "x"}',
        '{"queries": ["a"]}',
        "kestrel keeper, bird care rota",
    ],
    ids=["four_queries", "long_query", "long_passage", "extra", "no_passage", "prose"],
)
async def test_a_bad_reply_is_repaired_not_cut(bad: str) -> None:
    model = Local(bad, reply(["kestrel keeper"]))

    expansion = await expand(model, QUESTION)

    assert expansion.queries == ("kestrel keeper",)
    assert len(model.requests) == 2
    assert model.requests[1].messages[-2].text == bad


async def test_an_expansion_never_repaired_fails() -> None:
    model = Local("no", "still no", "never")

    with pytest.raises(StructuredOutputError, match="no valid Rewrites"):
        await expand(model, QUESTION)


@pytest.mark.parametrize(
    ("queries", "passage", "message"),
    [
        (("a", "b", "c", "d"), None, r"^4 queries, over MAX_QUERIES=3$"),
        (("",), None, r"^'' is empty"),
        ((" a",), None, r"^' a' is empty or has space"),
        ((), "p ", r"^'p ' is empty or has space"),
        ((), "", r"^'' is empty"),
        (("a", "A"), None, r"repeat one another$"),
    ],
)
def test_an_expansion_refuses_what_was_not_cleaned(
    queries: tuple[str, ...], passage: str | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Expansion(queries, passage)


def test_rewrites_refuse_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Rewrites.model_validate({"queries": [], "passage": "", "more": 1})


@pytest.mark.parametrize(
    ("expansion", "also"),
    [
        (Expansion(("a", "b"), "p"), ("a", "b", "p")),
        (Expansion(("a",), None), ("a",)),
        (Expansion((), "p"), ("p",)),
        (Expansion((), None), ()),
    ],
)
def test_also_is_the_queries_then_the_passage(
    expansion: Expansion, also: tuple[str, ...]
) -> None:
    assert expansion.also() == also


@pytest.fixture
async def notes(tmp_path: Path) -> Knowledge:
    knowledge = await opened(tmp_path / "home" / "knowledge.db")
    await knowledge.add(planted(tmp_path / "notes", NOTES))
    await knowledge.index()
    return knowledge


def ticks(*times: float) -> Iterator[float]:
    return iter(times)


SERVER, SISTER, CAT = "notes/server.txt:1", "notes/sister.txt:1", "notes/cat.txt:1"


@pytest.mark.parametrize(
    ("merge", "ranked"),
    [
        # By words: "kestrel" finds server; "pet" nothing; the passage finds
        # sister by four words and cat by "my" alone. Best keeps sister's
        # larger BM25; by rank, server and sister are both first and tie.
        (Merge.BEST, [SISTER, SERVER, CAT]),
        (Merge.RRF, [SERVER, SISTER, CAT]),
    ],
)
async def test_deep_search_searches_the_question_its_rewrites_and_passage(
    notes: Knowledge, merge: Merge, ranked: list[str]
) -> None:
    model = Local(reply(["pet"], "my sister lives in pune"))
    clock = ticks(10.0, 12.5, 13.0)
    words = Variant(Side.WORDS)

    deep = await deep_search(
        notes,
        model,
        "kestrel",
        limit=3,
        variant=words,
        merge=merge,
        clock=lambda: next(clock),
    )

    expected = await notes.search(
        "kestrel",
        limit=3,
        variant=words,
        also=("pet", "my sister lives in pune"),
        merge=merge,
    )
    assert deep.found == expected
    assert citations(deep.found) == ranked
    assert deep.expansion == Expansion(("pet",), "my sister lives in pune")
    assert (deep.expand_s, deep.search_s) == (2.5, 0.5)
    (request,) = model.requests
    assert request.use_remote is False


async def test_deep_search_with_nothing_added_is_the_plain_search(
    notes: Knowledge,
) -> None:
    model = Local(reply(["kestrel"], ""))

    deep = await deep_search(notes, model, "kestrel", private=False)

    assert deep.expansion == Expansion((), None)
    assert deep.found == await notes.search("kestrel", private=False)
    assert deep.expand_s >= 0
    assert deep.search_s >= 0


@pytest.mark.parametrize("merge", list(Merge))
async def test_deep_search_withholds_what_its_rewrites_find_in_private_files(
    notes: Knowledge, tmp_path: Path, merge: Merge
) -> None:
    await notes.add(planted(tmp_path / "diary", {"pet.txt": "my pet\n"}), private=True)
    await notes.index()
    words = Variant(Side.WORDS)

    hidden = await deep_search(
        notes,
        Local(reply(["pet"])),
        "kestrel",
        variant=words,
        private=False,
        merge=merge,
    )
    shown = await deep_search(
        notes, Local(reply(["pet"])), "kestrel", variant=words, merge=merge
    )

    assert hidden.found.withheld == 1
    assert not any(passage.private for passage in hidden.found.passages)
    assert "diary/pet.txt:1" in citations(shown.found)
    assert shown.found.withheld == 0
