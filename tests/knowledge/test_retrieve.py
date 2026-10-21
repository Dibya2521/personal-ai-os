import re
from pathlib import Path, PurePosixPath

import pytest
from hypothesis import given
from hypothesis import strategies as st

from synthia.knowledge.model import (
    Chunk,
    ChunkId,
    Document,
    DocumentId,
    DocumentKind,
    SourceId,
)
from synthia.knowledge.retrieve import (
    CANDIDATES,
    DEFAULT_VARIANT,
    WORD_WEIGHT,
    Found,
    Passage,
    Side,
    Variant,
    VariantError,
)
from synthia.knowledge.service import Knowledge
from synthia.knowledge.store import KnowledgeStore
from synthia.retrieval.fusion import word_scale
from tests.knowledge.corpus import (
    CAT,
    KESTREL,
    KESTREL_BM25,
    NOTES,
    PET,
    PET_KESTREL,
    SERVER,
    SISTER,
    WORD_SCALE,
    citations,
    cosine,
    opened,
    planted,
    wired,
)

CAT_AT = "notes/cat.txt:1"
SERVER_AT = "notes/server.txt:1"
SISTER_AT = "notes/sister.txt:1"


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "home" / "knowledge.db"


@pytest.fixture
async def notes(tmp_path: Path, db: Path) -> Knowledge:
    knowledge = await opened(db)
    await knowledge.add(planted(tmp_path / "notes", NOTES))
    await knowledge.index()
    return knowledge


@pytest.mark.parametrize(
    ("variant", "name"),
    [
        (Variant(), "fused"),
        (Variant(Side.WORDS), "words"),
        (Variant(Side.MEANING), "meaning"),
        (Variant(word_weight=0.25), "fused:0.25"),
        (Variant(word_weight=1e-5), "fused:1e-05"),
        (Variant(word_weight=0.0), "fused:0.0"),
        (Variant(word_weight=1), "fused:1.0"),
        (Variant(hops=1, link_weight=0.3), "fused+hop1x0.3"),
        (Variant(word_weight=0.05, hops=2, link_weight=1), "fused:0.05+hop2x1.0"),
        (Variant(Side.WORDS, hops=1, link_weight=0.0), "words+hop1x0.0"),
        (Variant(Side.MEANING, hops=2, link_weight=2.5), "meaning+hop2x2.5"),
        (Variant(word_weight=1e16), "fused:1e+16"),
        (
            Variant(word_weight=1e16, hops=1, link_weight=1e16),
            "fused:1e+16+hop1x1e+16",
        ),
    ],
)
def test_variant_name_and_parse_round_trip(variant: Variant, name: str) -> None:
    assert variant.name() == name
    assert Variant.parse(name) == variant


@pytest.mark.parametrize(
    "name",
    [
        "fused:-1",
        "fused:nan",
        "fused:inf",
        "fused:-0.0",
        "words:0.1",
        "meaning:0.2",
        "fused:0.2",
        "fused:1",
        "fused:+0.5",
        "fused:1_0",
        "fused:0.25:1",
        "fused:",
        "hops",
        "",
        "FUSED",
        "fused+",
        "fused+hop1",
        "fused+hop1x",
        "fused+hopx0.3",
        "fused+hop0x0.3",
        "fused+hop0x0.0",
        "fused+hop3x0.3",
        "fused+hop01x0.3",
        "fused+hop1x-0.3",
        "fused+hop1x-0.0",
        "fused+hop1xnan",
        "fused+hop1x0.3+hop1x0.3",
        "fused+HOP1x0.3",
        "fused:0.2+hop1x0.3",
        "words:0.1+hop1x0.3",
        "fused+hop1x 0.3",
    ],
)
def test_variant_parse_refuses_a_name_it_would_not_write(name: str) -> None:
    with pytest.raises(VariantError, match=re.escape(repr(name))):
        Variant.parse(name)


@pytest.mark.parametrize(
    ("side", "weight"),
    [
        (Side.FUSED, -1.0),
        (Side.FUSED, -0.0),
        (Side.FUSED, float("nan")),
        (Side.FUSED, float("inf")),
        (Side.WORDS, 0.1),
        (Side.MEANING, 0.0),
    ],
)
def test_variant_refuses_a_weight_it_cannot_use(side: Side, weight: float) -> None:
    with pytest.raises(ValueError, match=r"^word_weight="):
        Variant(side, weight)


@pytest.mark.parametrize(
    ("hops", "weight", "message"),
    [
        (1, -1.0, r"^link_weight=-1\.0 is not"),
        (1, -0.0, r"^link_weight=-0\.0 is not"),
        (1, float("nan"), r"^link_weight=nan is not"),
        (2, float("inf"), r"^link_weight=inf is not"),
        (3, 0.3, r"^hops=3 is not 0 to 2$"),
        (-1, 0.0, r"^hops=-1 is not 0 to 2$"),
        (0, 0.3, r"^link_weight=0\.3 with no hops"),
    ],
)
def test_variant_refuses_hops_it_cannot_follow(
    hops: int, weight: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Variant(hops=hops, link_weight=weight)


@given(
    st.one_of(
        st.sampled_from([Variant(Side.WORDS), Variant(Side.MEANING)]),
        st.builds(
            Variant,
            word_weight=st.floats(min_value=0.0, allow_nan=False, allow_infinity=False),
        ),
        st.builds(
            Variant,
            side=st.sampled_from(Side),
            hops=st.integers(1, 2),
            link_weight=st.floats(min_value=0.0, allow_nan=False, allow_infinity=False),
        ),
    )
)
def test_variant_name_parses_back_to_the_same_variant(variant: Variant) -> None:
    name = variant.name()

    assert Variant.parse(name) == variant
    assert Variant.parse(name).name() == name


def test_the_default_variant_fuses_at_the_word_weight() -> None:
    assert (DEFAULT_VARIANT.side, DEFAULT_VARIANT.word_weight) == (
        Side.FUSED,
        WORD_WEIGHT,
    )
    assert DEFAULT_VARIANT.name() == "fused"


def passage(page: int | None, start: int, end: int) -> Passage:
    chunk = Chunk(ChunkId(1), DocumentId(1), 0, page, start, end, "", "text")
    document = Document(
        DocumentId(1),
        SourceId(1),
        PurePosixPath("adr/0012 fusion.md"),
        DocumentKind.MARKDOWN,
        0,
        0,
        "sha",
        "recipe",
        "Fusion",
    )
    return Passage(chunk, document, "notes", 1.0, private=False)


@pytest.mark.parametrize(
    ("page", "start", "end", "citation"),
    [
        (None, 3, 9, "notes/adr/0012 fusion.md:3-9"),
        (None, 4, 4, "notes/adr/0012 fusion.md:4"),
        (2, 1, 7, "notes/adr/0012 fusion.md:p.2:1-7"),
        (2, 5, 5, "notes/adr/0012 fusion.md:p.2:5"),
    ],
)
def test_passage_citation_names_source_path_page_and_lines(
    page: int | None, start: int, end: int, citation: str
) -> None:
    assert passage(page, start, end).citation() == citation


@pytest.mark.parametrize(
    ("name", "ranked"),
    [
        ("words", [SERVER_AT]),
        ("meaning", [CAT_AT, SISTER_AT, SERVER_AT]),
        ("fused", [CAT_AT, SERVER_AT, SISTER_AT]),
        ("fused:0.0", [CAT_AT, SISTER_AT, SERVER_AT]),
        ("fused:10.0", [SERVER_AT, CAT_AT, SISTER_AT]),
    ],
)
async def test_search_ranks_as_each_variant_says(
    notes: Knowledge, name: str, ranked: list[str]
) -> None:
    found = await notes.search("pet kestrel", limit=3, variant=Variant.parse(name))

    assert citations(found) == ranked
    assert found.withheld == 0


async def test_search_scores_are_bm25_cosine_and_their_fusion(notes: Knowledge) -> None:
    words = await notes.search("pet kestrel", variant=Variant(Side.WORDS))
    meaning = await notes.search("pet kestrel", variant=Variant(Side.MEANING))
    both = await notes.search("pet kestrel")

    assert [p.score for p in words.passages] == pytest.approx([KESTREL_BM25])
    assert [p.score for p in meaning.passages] == pytest.approx(
        [
            cosine(PET_KESTREL, CAT),
            cosine(PET_KESTREL, SISTER),
            cosine(PET_KESTREL, SERVER),
        ]
    )
    # The default weight is 0.2: server is 0.0066 + 0.2 * 0.5108 / (ln 7 * 2.2)
    # = 0.0305, below cat's 1.0 and above sister's 0.0132.
    server = cosine(PET_KESTREL, SERVER) + 0.2 * KESTREL_BM25 / WORD_SCALE
    assert [p.score for p in both.passages] == pytest.approx(
        [cosine(PET_KESTREL, CAT), server, cosine(PET_KESTREL, SISTER)]
    )


async def test_a_word_only_one_file_holds_is_found_by_words_not_meaning(
    notes: Knowledge,
) -> None:
    words = await notes.search("kestrel", variant=Variant(Side.WORDS))
    meaning = await notes.search("kestrel", variant=Variant(Side.MEANING))

    assert citations(words) == [SERVER_AT]
    assert citations(meaning) == [SISTER_AT, CAT_AT, SERVER_AT]
    assert [p.score for p in meaning.passages] == pytest.approx(
        [cosine(KESTREL, SISTER), cosine(KESTREL, CAT), cosine(KESTREL, SERVER)]
    )


async def test_a_word_in_other_words_is_found_by_meaning_not_words(
    notes: Knowledge,
) -> None:
    words = await notes.search("pet", variant=Variant(Side.WORDS))
    meaning = await notes.search("pet", variant=Variant(Side.MEANING), limit=1)

    assert words == Found((), 0)
    assert citations(meaning) == [CAT_AT]
    assert meaning.passages[0].score == pytest.approx(cosine(PET, CAT))


async def test_fused_finds_what_each_side_alone_misses(notes: Knowledge) -> None:
    words = await notes.search("pet kestrel", limit=2, variant=Variant(Side.WORDS))
    meaning = await notes.search("pet kestrel", limit=2, variant=Variant(Side.MEANING))
    both = await notes.search("pet kestrel", limit=2)

    assert citations(words) == [SERVER_AT]
    assert citations(meaning) == [CAT_AT, SISTER_AT]
    assert citations(both) == [CAT_AT, SERVER_AT]


async def test_fused_counts_the_cosine_of_a_word_match_past_the_nearest(
    tmp_path: Path, db: Path
) -> None:
    # CANDIDATES chunks are nearer "pet kestrel" than server.txt, the only
    # one holding a query word: it is past the meaning's candidates.
    files = {f"cat{n:02}.txt": "my cat\n" for n in range(CANDIDATES)}
    files["server.txt"] = NOTES["server.txt"]
    knowledge = await opened(db)
    await knowledge.add(planted(tmp_path / "notes", files))
    await knowledge.index()
    heavy = Variant(word_weight=10.0)

    words = await knowledge.search("pet kestrel", variant=Variant(Side.WORDS))
    both = await knowledge.search("pet kestrel", limit=1, variant=heavy)

    (bm25,) = [p.score for p in words.passages]
    word = heavy.word_weight * bm25 / word_scale(len(files))
    assert citations(both) == [SERVER_AT]
    assert both.passages[0].score == pytest.approx(cosine(PET_KESTREL, SERVER) + word)


@pytest.mark.parametrize("name", ["words", "meaning", "fused"])
async def test_equal_scores_go_to_the_lower_chunk_id(
    tmp_path: Path, db: Path, name: str
) -> None:
    knowledge = await opened(db)
    root = planted(
        tmp_path / "notes", {"a.txt": "my cat is miso\n", "b.txt": "my cat is miso\n"}
    )
    await knowledge.add(root)
    await knowledge.index()
    # The same words in another order (and a size that differs, so the pass
    # reads it): the same vector and score, under a newer chunk id.
    (root / "a.txt").write_bytes(b"miso is my cat\n\n")
    await knowledge.index()

    found = await knowledge.search("cat", variant=Variant.parse(name))

    assert citations(found) == ["notes/b.txt:1", "notes/a.txt:1"]
    assert [p.chunk.id for p in found.passages] == [2, 3]
    assert found.passages[0].score == found.passages[1].score


async def test_a_tie_wider_than_the_candidates_goes_to_the_lower_chunk_ids(
    tmp_path: Path, db: Path
) -> None:
    files = {f"{n:03}.txt": "my cat is miso\n" for n in range(CANDIDATES + 20)}
    root = planted(tmp_path / "notes", files)
    knowledge = await opened(db)
    await knowledge.add(root)
    await knowledge.index()
    # Each vector let go of is replaced in its place by the last one held, so
    # the held order is no longer the chunk ids' order.
    for n in range(10):
        (root / f"{n:03}.txt").unlink()
    await knowledge.index()

    found = await knowledge.search(
        "pet", limit=CANDIDATES, variant=Variant(Side.MEANING)
    )

    expected = [f"notes/{n:03}.txt:1" for n in range(10, 10 + CANDIDATES)]
    assert citations(found) == expected


@pytest.mark.parametrize("name", ["words", "meaning", "fused"])
@pytest.mark.parametrize("query", ["", "?!", " -- ... "])
async def test_a_query_with_no_word_finds_nothing(
    notes: Knowledge, name: str, query: str
) -> None:
    assert await notes.search(query, variant=Variant.parse(name)) == Found((), 0)


@pytest.fixture
async def with_diary(notes: Knowledge, tmp_path: Path) -> Knowledge:
    # Chunk 4 "the host" is nothing like "pet"; chunk 5 "my pet" is exactly it.
    diary = {"host.txt": "the host\n", "pet.txt": "my pet\n"}
    await notes.add(planted(tmp_path / "diary", diary), private=True)
    await notes.index()
    return notes


async def test_private_passages_show_by_default(with_diary: Knowledge) -> None:
    found = await with_diary.search("pet", limit=2)

    assert citations(found) == ["diary/pet.txt:1", CAT_AT]
    assert [p.private for p in found.passages] == [True, False]
    assert found.withheld == 0


@pytest.mark.parametrize(("limit", "withheld"), [(1, 1), (3, 1), (5, 2)])
async def test_withheld_counts_private_passages_among_the_best_limit(
    with_diary: Knowledge, limit: int, withheld: int
) -> None:
    # Fused for "pet": pet.txt, cat, server, sister, host.txt (the last three
    # at cosine 0, by chunk id); both .txt of the diary are private.
    found = await with_diary.search("pet", limit=limit, private=False)

    expected = [CAT_AT, SERVER_AT, SISTER_AT][:limit]
    assert citations(found) == expected
    assert found.withheld == withheld
    assert not any(p.private for p in found.passages)


async def test_only_private_passages_show_none_and_count_them(
    tmp_path: Path, db: Path
) -> None:
    knowledge = await opened(db)
    await knowledge.add(
        planted(tmp_path / "diary", {"pet.txt": "my pet\n"}), private=True
    )
    await knowledge.index()

    assert await knowledge.search("pet", private=False) == Found((), 1)


async def test_a_limit_below_one_is_refused(notes: Knowledge) -> None:
    with pytest.raises(ValueError, match=r"^limit=0 is below 1$"):
        await notes.search("cat", limit=0)


async def test_a_chunk_removed_behind_the_search_is_left_out(
    tmp_path: Path, db: Path
) -> None:
    shelf = wired(KnowledgeStore(db))
    source = await shelf.knowledge.add(planted(tmp_path / "notes", NOTES))
    await shelf.knowledge.index()
    stamps = await shelf.store.stamps(source.id)
    await shelf.store.remove_documents([stamps[PurePosixPath("cat.txt")].document])

    found = await shelf.knowledge.search("pet kestrel", variant=Variant(Side.MEANING))

    assert len(shelf.vectors) == 3
    assert citations(found) == [SISTER_AT, SERVER_AT]


LINKED = {
    "cat.txt": "my cat is miso\n",
    "guide.md": "my sister lives in pune city, see [next](middle.md)\n",
    "middle.md": "[on](server.md)\n",
    "server.md": "the database server host kestrel\n",
}
GUIDE, MIDDLE = "notes/guide.md:1", "notes/middle.md:1"
SERVER_MD = "notes/server.md:1"


@pytest.fixture
async def linked(tmp_path: Path, db: Path) -> Knowledge:
    knowledge = await opened(db)
    await knowledge.add(planted(tmp_path / "notes", LINKED))
    await knowledge.index()
    return knowledge


def scored(found: Found) -> dict[str, float]:
    return {passage.citation(): passage.score for passage in found.passages}


@pytest.mark.parametrize(
    ("name", "ranked"),
    [
        ("fused", [GUIDE, CAT_AT, MIDDLE, SERVER_MD]),
        ("fused+hop1x0.5", [GUIDE, MIDDLE, CAT_AT, SERVER_MD]),
        ("fused+hop2x0.5", [GUIDE, MIDDLE, SERVER_MD, CAT_AT]),
    ],
)
async def test_hops_lift_a_passage_only_a_link_relates_to_the_query(
    linked: Knowledge, name: str, ranked: list[str]
) -> None:
    found = await linked.search("pune", variant=Variant.parse(name))

    assert citations(found) == ranked


async def test_a_hop_adds_the_weighted_seed_score_to_the_linked_chunks_own(
    linked: Knowledge,
) -> None:
    alone = scored(await linked.search("pune kestrel"))
    hops = scored(
        await linked.search("pune kestrel", variant=Variant.parse("fused+hop2x0.5"))
    )

    middle = alone[MIDDLE] + 0.5 * alone[GUIDE]
    assert hops[MIDDLE] == pytest.approx(middle)
    assert hops[SERVER_MD] == pytest.approx(alone[SERVER_MD] + 0.5 * middle)
    assert alone[SERVER_MD] > 0
    assert hops[CAT_AT] == alone[CAT_AT]


async def test_a_linked_chunk_beyond_the_candidates_is_scored_by_its_own_meaning(
    tmp_path: Path, db: Path
) -> None:
    decoys = {f"cat{n:02}.txt": NOTES["cat.txt"] for n in range(CANDIDATES)}
    guide = "my sister lives in pune city, see [it](server.md)\n"
    files = {**decoys, "guide.md": guide, "server.md": NOTES["server.txt"]}
    knowledge = await opened(db)
    await knowledge.add(planted(tmp_path / "notes", files))
    await knowledge.index()
    meaning = Variant(Side.MEANING)

    alone = scored(await knowledge.search("pune kestrel", variant=meaning))
    hop = Variant(Side.MEANING, hops=1, link_weight=0.5)
    hops = scored(await knowledge.search("pune kestrel", variant=hop))

    assert SERVER_MD not in alone
    own = cosine((0.0, 0.0, 1.0, 0.0, 0.1), SERVER)
    assert hops[SERVER_MD] == pytest.approx(own + 0.5 * alone[GUIDE])
