import sqlite3
from pathlib import Path, PurePosixPath

import pytest

from synthia.knowledge.model import (
    Chunk,
    ChunkId,
    Document,
    DocumentKind,
    Draft,
    Link,
    LinkKind,
    Source,
    SourceId,
    SourceKind,
)
from synthia.knowledge.store import (
    KNOWLEDGE_FILE,
    SCHEMA_VERSION,
    KnowledgeStore,
    LinkTarget,
    Reading,
    Replaced,
    Row,
    Stamp,
)
from synthia.retrieval.fusion import WordScores

VAULT = Path("D:/vault")
REPO = Path("D:/repo")


def draft(ordinal: int, body: str, *links: Link, header: str = "") -> Draft:
    return Draft(ordinal, None, ordinal + 1, ordinal + 1, header, body, links)


def reading(
    source: SourceId,
    path: str,
    *drafts: Draft,
    sha256: str = "aa",
    recipe: str = "r1",
) -> Reading:
    kind = DocumentKind.PYTHON if path.endswith(".py") else DocumentKind.MARKDOWN
    return Reading(
        source,
        PurePosixPath(path),
        kind,
        10,
        20,
        sha256,
        recipe,
        PurePosixPath(path).stem,
        "",
        drafts,
    )


def wikilink(key: str, anchor: str | None = None) -> Link:
    return Link(LinkKind.WIKILINK, key, key, anchor)


def imports(key: str) -> Link:
    return Link(LinkKind.IMPORT, key, key)


async def chunk_ids(store: KnowledgeStore) -> list[ChunkId]:
    return [chunk for chunk, _ in await store.unembedded("none", 1000)]


async def titles_linked(store: KnowledgeStore) -> list[str]:
    chunks = await chunk_ids(store)
    linked = await store.linked(chunks)
    return [target.title for chunk in chunks for target in linked.get(chunk, ())]


@pytest.fixture
def store(tmp_path: Path) -> KnowledgeStore:
    return KnowledgeStore(tmp_path / KNOWLEDGE_FILE)


@pytest.fixture
async def vault(store: KnowledgeStore) -> SourceId:
    source = await store.put_source(
        VAULT, "vault", SourceKind.VAULT, private=True, excludes=()
    )
    return source.id


async def test_a_document_comes_back_with_its_chunks_source_and_stamp(
    store: KnowledgeStore,
) -> None:
    source = await store.put_source(
        VAULT,
        "vault",
        SourceKind.VAULT,
        private=True,
        excludes=(PurePosixPath("Private"),),
    )
    page = Draft(0, 3, 4, 6, "Notes > Fusion", "one\ntwo\nthree", (), cut=True)
    replaced = await store.replace_document(
        reading(source.id, "notes/Fusion.md", page, draft(1, "plain"))
    )
    first, second = await chunk_ids(store)

    (row,) = await store.rows([first])

    assert replaced == Replaced(replaced.document, (), 2)
    assert row == Row(
        Chunk(
            first,
            replaced.document,
            0,
            3,
            4,
            6,
            "Notes > Fusion",
            "one\ntwo\nthree",
            cut=True,
        ),
        Document(
            replaced.document,
            source.id,
            PurePosixPath("notes/Fusion.md"),
            DocumentKind.MARKDOWN,
            10,
            20,
            "aa",
            "r1",
            "Fusion",
        ),
        "vault",
        private=True,
    )
    assert await store.unembedded("mini", 10) == [
        (first, "Notes > Fusion\none\ntwo\nthree"),
        (second, "plain"),
    ]
    assert await store.sources() == [
        Source(
            source.id,
            "vault",
            VAULT,
            SourceKind.VAULT,
            private=True,
            excludes=(PurePosixPath("Private"),),
        )
    ]
    assert await store.stamps(source.id) == {
        PurePosixPath("notes/Fusion.md"): Stamp(replaced.document, 10, 20, "aa", "r1")
    }


async def test_rows_come_back_in_the_order_asked_without_the_missing(
    store: KnowledgeStore, vault: SourceId
) -> None:
    await store.replace_document(reading(vault, "a.md", draft(0, "x"), draft(1, "y")))
    first, second = await chunk_ids(store)

    rows = await store.rows([second, ChunkId(999), first])

    assert [row.chunk.body for row in rows] == ["y", "x"]


async def test_the_same_bytes_and_recipe_again_write_nothing(
    store: KnowledgeStore, vault: SourceId
) -> None:
    first = await store.replace_document(reading(vault, "a.md", draft(0, "x")))
    before = await chunk_ids(store)

    again = await store.replace_document(reading(vault, "a.md", draft(0, "changed")))

    assert again == Replaced(first.document, (), 0)
    assert await chunk_ids(store) == before
    assert [row.chunk.body for row in await store.rows(before)] == ["x"]


async def test_a_new_recipe_replaces_the_chunks_of_unchanged_bytes(
    store: KnowledgeStore, vault: SourceId
) -> None:
    first = await store.replace_document(reading(vault, "a.md", draft(0, "x")))
    (old,) = await chunk_ids(store)
    await store.put_vectors("mini", [(old, b"\x01")])

    again = await store.replace_document(
        reading(vault, "a.md", draft(0, "x"), draft(1, "y"), recipe="r2")
    )

    new = await chunk_ids(store)
    assert again == Replaced(first.document, (old,), 2)
    assert old not in new
    assert len(new) == 2
    assert await store.vectors("mini") == []
    assert (await store.stamps(vault))[PurePosixPath("a.md")].recipe == "r2"


async def test_a_replaced_documents_old_words_are_no_longer_found(
    store: KnowledgeStore, vault: SourceId
) -> None:
    await store.replace_document(reading(vault, "a.md", draft(0, "zebra")))

    await store.replace_document(
        reading(vault, "a.md", draft(0, "giraffe"), sha256="bb")
    )

    (giraffe,) = await chunk_ids(store)
    assert await store.word_scores("zebra", limit=10) == WordScores({}, 1)
    assert list((await store.word_scores("giraffe", limit=10)).scores) == [giraffe]


async def test_a_replace_that_fails_leaves_the_old_document_whole(
    store: KnowledgeStore, vault: SourceId
) -> None:
    await store.replace_document(reading(vault, "a.md", draft(0, "zebra")))
    (old,) = await chunk_ids(store)
    await store.put_vectors("mini", [(old, b"\x01")])

    with pytest.raises(sqlite3.IntegrityError):
        await store.replace_document(
            reading(vault, "a.md", draft(0, "giraffe"), draft(0, "lion"), sha256="bb")
        )

    assert await chunk_ids(store) == [old]
    assert (await store.stamps(vault))[PurePosixPath("a.md")].sha256 == "aa"
    assert await store.vectors("mini") == [(old, b"\x01")]
    assert list((await store.word_scores("zebra", limit=10)).scores) == [old]
    assert await store.word_scores("giraffe", limit=10) == WordScores({}, 1)


async def test_a_removed_chunks_id_is_never_given_again(
    store: KnowledgeStore, vault: SourceId
) -> None:
    first = await store.replace_document(reading(vault, "a.md", draft(0, "x")))
    (old,) = await chunk_ids(store)

    assert await store.remove_documents([first.document]) == (old,)
    await store.replace_document(reading(vault, "b.md", draft(0, "y")))

    (new,) = await chunk_ids(store)
    assert new > old


async def test_removing_a_source_takes_everything_under_it(
    store: KnowledgeStore, vault: SourceId
) -> None:
    repo = await store.put_source(
        REPO, "repo", SourceKind.FILES, private=False, excludes=()
    )
    await store.replace_document(
        reading(vault, "a.md", draft(0, "zebra", wikilink("b")))
    )
    await store.replace_document(reading(vault, "b.md", draft(0, "giraffe")))
    await store.replace_document(reading(repo.id, "c.md", draft(0, "giraffe")))
    vault_a, vault_b, repo_c = await chunk_ids(store)
    await store.put_vectors("mini", [(vault_a, b"\x01"), (repo_c, b"\x02")])

    removed = await store.remove_source(vault)

    assert removed == (vault_a, vault_b)
    assert await store.sources() == [repo]
    assert await store.stamps(vault) == {}
    assert await store.rows([vault_a, vault_b]) == []
    assert await store.vectors("mini") == [(repo_c, b"\x02")]
    assert await store.linked([vault_a]) == {}
    assert await store.word_scores("zebra", limit=10) == WordScores({}, 1)
    assert set((await store.word_scores("giraffe", limit=10)).scores) == {repo_c}


async def test_removing_documents_leaves_the_others(
    store: KnowledgeStore, vault: SourceId
) -> None:
    gone = await store.replace_document(
        reading(vault, "a.md", draft(0, "x"), draft(1, "y"))
    )
    await store.replace_document(reading(vault, "b.md", draft(0, "z")))
    first, second, kept = await chunk_ids(store)

    assert await store.remove_documents([gone.document]) == (first, second)
    assert await chunk_ids(store) == [kept]
    assert list(await store.stamps(vault)) == [PurePosixPath("b.md")]


async def test_adding_a_root_again_updates_it_in_place(store: KnowledgeStore) -> None:
    first = await store.put_source(
        VAULT, "vault", SourceKind.FILES, private=False, excludes=()
    )

    again = await store.put_source(
        VAULT,
        "other name",
        SourceKind.VAULT,
        private=True,
        excludes=(PurePosixPath("Private"),),
    )

    assert again == Source(
        first.id,
        "vault",
        VAULT,
        SourceKind.VAULT,
        private=True,
        excludes=(PurePosixPath("Private"),),
    )
    assert await store.sources() == [again]


async def test_two_roots_cannot_share_a_name(store: KnowledgeStore) -> None:
    await store.put_source(VAULT, "notes", SourceKind.FILES, private=False, excludes=())

    with pytest.raises(sqlite3.IntegrityError):
        await store.put_source(
            REPO, "notes", SourceKind.FILES, private=False, excludes=()
        )

    assert [source.root for source in await store.sources()] == [VAULT]


async def test_a_touch_records_the_new_stat(
    store: KnowledgeStore, vault: SourceId
) -> None:
    replaced = await store.replace_document(reading(vault, "a.md", draft(0, "x")))

    await store.touch(replaced.document, 99, 21)

    assert await store.stamps(vault) == {
        PurePosixPath("a.md"): Stamp(replaced.document, 99, 21, "aa", "r1")
    }


async def test_vectors_are_kept_only_for_chunks_still_there(
    store: KnowledgeStore, vault: SourceId
) -> None:
    gone = await store.replace_document(reading(vault, "a.md", draft(0, "x")))
    await store.replace_document(reading(vault, "b.md", draft(0, "y")))
    removed, kept = await chunk_ids(store)
    await store.remove_documents([gone.document])

    stored = await store.put_vectors("mini", [(removed, b"\x01"), (kept, b"\x02")])

    assert stored == [kept]
    assert await store.vectors("mini") == [(kept, b"\x02")]


async def test_a_vector_from_another_model_is_replaced(
    store: KnowledgeStore, vault: SourceId
) -> None:
    await store.replace_document(reading(vault, "a.md", draft(0, "x")))
    (chunk,) = await chunk_ids(store)
    await store.put_vectors("mini", [(chunk, b"\x01")])

    waiting = await store.unembedded("bge", 10)
    await store.put_vectors("bge", [(chunk, b"\x02")])

    assert waiting == [(chunk, "x")]
    assert await store.unembedded("bge", 10) == []
    assert await store.vectors("mini") == []
    assert await store.vectors("bge") == [(chunk, b"\x02")]


async def test_unembedded_gives_at_most_the_limit_oldest_first(
    store: KnowledgeStore, vault: SourceId
) -> None:
    await store.replace_document(
        reading(vault, "a.md", draft(0, "x"), draft(1, "y"), draft(2, "z"))
    )

    assert [text for _, text in await store.unembedded("mini", 2)] == ["x", "y"]


async def test_words_rank_the_chunk_holding_them_most_first(
    store: KnowledgeStore, vault: SourceId
) -> None:
    await store.replace_document(
        reading(
            vault,
            "a.md",
            draft(0, "fusion fusion fusion of scores"),
            draft(1, "a note on fusion and many other things besides"),
            draft(2, "nothing here"),
        )
    )
    dense, sparse, _ = await chunk_ids(store)

    found = await store.word_scores("fusion", limit=10)
    best = await store.word_scores("fusion", limit=1)

    assert set(found.scores) == {dense, sparse}
    assert found.scores[dense] > found.scores[sparse] > 0
    assert found.indexed == 3
    assert list(best.scores) == [dense]


async def test_a_header_word_finds_its_chunk(
    store: KnowledgeStore, vault: SourceId
) -> None:
    await store.replace_document(
        reading(vault, "a.md", draft(0, "body", header="Decision"), draft(1, "other"))
    )
    first, _ = await chunk_ids(store)

    assert set((await store.word_scores("decision", limit=10)).scores) == {first}


async def test_words_within_score_only_those_chunks(
    store: KnowledgeStore, vault: SourceId
) -> None:
    await store.replace_document(
        reading(
            vault, "a.md", draft(0, "fusion"), draft(1, "fusion"), draft(2, "fusion")
        )
    )
    first, _, third = await chunk_ids(store)

    found = await store.word_scores("fusion", limit=10, within=[first, third])
    none = await store.word_scores("fusion", limit=10, within=[])

    assert set(found.scores) == {first, third}
    assert found.indexed == 3
    assert none == WordScores({}, 3)


async def test_a_query_with_no_words_finds_nothing(
    store: KnowledgeStore, vault: SourceId
) -> None:
    await store.replace_document(reading(vault, "a.md", draft(0, "x"), draft(1, "y")))

    assert await store.word_scores("?! ...", limit=10) == WordScores({}, 2)


@pytest.mark.parametrize(
    ("query", "finds"),
    [
        ('"', False),
        ("NEAR(a b) OR *", False),
        ("a AND (", False),
        ("-x", False),
        ("^start", False),
        ("col:value", False),
        ("\U0001f600 ok", True),
        ("\ud800", False),
        ("ok\ud800", True),
    ],
)
async def test_what_a_person_types_is_never_search_syntax(
    store: KnowledgeStore, vault: SourceId, query: str, finds: bool
) -> None:
    await store.replace_document(reading(vault, "a.md", draft(0, "ok then")))
    (chunk,) = await chunk_ids(store)

    found = await store.word_scores(query, limit=10)

    assert set(found.scores) == ({chunk} if finds else set())


async def test_optimizing_keeps_every_word_findable(
    store: KnowledgeStore, vault: SourceId
) -> None:
    await store.replace_document(reading(vault, "a.md", draft(0, "fusion")))
    before = await store.word_scores("fusion", limit=10)

    await store.optimize()

    assert await store.word_scores("fusion", limit=10) == before


async def test_a_wikilink_reaches_the_note_by_its_casefolded_stem(
    store: KnowledgeStore, vault: SourceId
) -> None:
    target = await store.replace_document(
        reading(vault, "notes/Fusion Notes.md", draft(0, "t"))
    )
    await store.replace_document(
        reading(vault, "a.md", draft(0, "see", wikilink("fusion notes", "Decision")))
    )
    _, source = await chunk_ids(store)

    assert await store.linked([source]) == {
        source: (LinkTarget(target.document, "Fusion Notes", "Decision"),)
    }


async def test_a_markdown_link_reaches_the_note_by_its_path(
    store: KnowledgeStore, vault: SourceId
) -> None:
    await store.replace_document(reading(vault, "adr/0012.md", draft(0, "t")))
    meant = await store.replace_document(
        reading(vault, "elsewhere/0012.md", draft(0, "t"))
    )
    link = Link(LinkKind.MARKDOWN, "../elsewhere/0012.md", "elsewhere/0012.md")
    await store.replace_document(reading(vault, "a.md", draft(0, "see", link)))
    *_, source = await chunk_ids(store)

    assert await store.linked([source]) == {
        source: (LinkTarget(meant.document, "0012", None),)
    }


@pytest.mark.parametrize(
    ("key", "reaches"),
    [
        ("synthia.retrieval.fusion", ["fusion"]),
        ("fusion", ["fusion"]),
        ("retrieval.fusion", ["fusion"]),
        ("synthia.retrieval", ["__init__"]),
        ("usion", []),
        ("a_b", []),
        ("notes", []),
    ],
)
async def test_an_import_reaches_the_module_whole_or_by_dotted_suffix(
    store: KnowledgeStore, vault: SourceId, key: str, reaches: list[str]
) -> None:
    await store.replace_document(
        reading(vault, "src/synthia/retrieval/fusion.py", draft(0, "t"))
    )
    await store.replace_document(
        reading(vault, "src/synthia/retrieval/__init__.py", draft(0, "t"))
    )
    await store.replace_document(reading(vault, "pkg/axb.py", draft(0, "t")))
    await store.replace_document(reading(vault, "notes.md", draft(0, "t")))
    await store.replace_document(
        reading(vault, "importer.py", draft(0, "x", imports(key)))
    )

    assert await titles_linked(store) == reaches


async def test_a_package_at_the_root_has_no_module_name(
    store: KnowledgeStore, vault: SourceId
) -> None:
    await store.replace_document(reading(vault, "src/__init__.py", draft(0, "t")))
    await store.replace_document(reading(vault, "a.py", draft(0, "x", imports("src"))))

    assert await titles_linked(store) == []


@pytest.mark.parametrize(
    ("path", "key", "reaches"),
    [
        ("src.py", "src", ["src"]),
        ("src/src/x.py", "src.x", ["x"]),
        ("__init__.py", "__init__", []),
        ("foo.bar.py", "foo.bar", []),
    ],
)
async def test_a_module_is_named_as_python_would_import_it(
    store: KnowledgeStore, vault: SourceId, path: str, key: str, reaches: list[str]
) -> None:
    await store.replace_document(reading(vault, path, draft(0, "t")))
    await store.replace_document(reading(vault, "a.py", draft(0, "x", imports(key))))

    assert await titles_linked(store) == reaches


async def test_an_exact_module_wins_over_a_dotted_suffix(
    store: KnowledgeStore, vault: SourceId
) -> None:
    await store.replace_document(reading(vault, "x/a/b.py", draft(0, "t")))
    await store.replace_document(reading(vault, "src/a/b.py", draft(0, "t")))
    await store.replace_document(reading(vault, "i.py", draft(0, "x", imports("a.b"))))
    *_, source = await chunk_ids(store)
    exact = (await store.stamps(vault))[PurePosixPath("src/a/b.py")].document

    assert [t.document for t in (await store.linked([source]))[source]] == [exact]


async def test_a_wikilink_prefers_a_note_to_another_file_of_the_same_stem(
    store: KnowledgeStore, vault: SourceId
) -> None:
    await store.replace_document(reading(vault, "fusion.py", draft(0, "t")))
    await store.replace_document(reading(vault, "notes/fusion.md", draft(0, "t")))
    await store.replace_document(
        reading(vault, "x.md", draft(0, "see", wikilink("fusion")))
    )
    *_, source = await chunk_ids(store)
    note = (await store.stamps(vault))[PurePosixPath("notes/fusion.md")].document

    assert [t.document for t in (await store.linked([source]))[source]] == [note]


async def test_the_shorter_path_wins_when_two_notes_share_a_stem(
    store: KnowledgeStore, vault: SourceId
) -> None:
    await store.replace_document(reading(vault, "a/deep/fusion.md", draft(0, "t")))
    await store.replace_document(reading(vault, "b/fusion.md", draft(0, "t")))
    await store.replace_document(reading(vault, "a/fusion.md", draft(0, "t")))
    await store.replace_document(
        reading(vault, "x.md", draft(0, "see", wikilink("fusion")))
    )
    *_, source = await chunk_ids(store)
    winner = (await store.stamps(vault))[PurePosixPath("a/fusion.md")].document

    assert [t.document for t in (await store.linked([source]))[source]] == [winner]


async def test_a_note_added_later_resolves_an_earlier_link_untouched(
    store: KnowledgeStore, vault: SourceId
) -> None:
    await store.replace_document(
        reading(vault, "a.md", draft(0, "see", wikilink("later")))
    )
    (source,) = await chunk_ids(store)
    before = await store.linked([source])

    await store.replace_document(reading(vault, "Later.md", draft(0, "t")))

    assert before == {}
    assert await titles_linked(store) == ["Later"]
    assert (await chunk_ids(store))[0] == source


async def test_links_never_cross_sources(
    store: KnowledgeStore, vault: SourceId
) -> None:
    repo = await store.put_source(
        REPO, "repo", SourceKind.FILES, private=False, excludes=()
    )
    await store.replace_document(reading(repo.id, "fusion.md", draft(0, "t")))
    await store.replace_document(
        reading(vault, "a.md", draft(0, "see", wikilink("fusion")))
    )

    assert await titles_linked(store) == []


async def test_a_chunks_links_come_back_in_the_order_written(
    store: KnowledgeStore, vault: SourceId
) -> None:
    await store.replace_document(reading(vault, "one.md", draft(0, "t")))
    await store.replace_document(reading(vault, "two.md", draft(0, "t")))
    await store.replace_document(
        reading(
            vault,
            "a.md",
            draft(0, "x", wikilink("two"), wikilink("missing"), wikilink("one")),
            draft(1, "no links"),
        )
    )

    assert await titles_linked(store) == ["two", "one"]


async def test_another_store_on_the_same_file_sees_what_was_written(
    tmp_path: Path,
) -> None:
    path = tmp_path / "home" / KNOWLEDGE_FILE
    writer = KnowledgeStore(path)
    source = await writer.put_source(
        VAULT, "vault", SourceKind.VAULT, private=True, excludes=()
    )
    await writer.replace_document(reading(source.id, "a.md", draft(0, "blue door")))

    found = await KnowledgeStore(path).word_scores("door", limit=10)

    assert len(found.scores) == 1


async def test_a_file_from_a_newer_synthia_is_refused(tmp_path: Path) -> None:
    path = tmp_path / KNOWLEDGE_FILE
    KnowledgeStore(path)
    with sqlite3.connect(path) as connection:
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    connection.close()

    with pytest.raises(sqlite3.DatabaseError, match="newer than 1"):
        KnowledgeStore(path)
