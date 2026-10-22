"""The toy corpus of tests/knowledge/corpus.py, asked three questions.

- "kestrel" (named, gold server.txt): words find it first, as the one file
  holding the word. By meaning the query is the common axis alone, so the
  cosines are sister 0.3 / sqrt(5.09) = 0.133, cat 0.2 / sqrt(4.04) = 0.0995
  and server 0.2 / sqrt(9.04) = 0.0665: server comes third. Fused adds the
  weight times kestrel's BM25 over the scale, 0.5108 / (ln 7 * 2.2) = 0.1193,
  to server only: 0.0904, 0.0725 and 0.0785 at 0.2 (the default), 0.05 and
  0.1 stay below cat, 0.1142 at 0.4 passes cat but not sister, so it comes
  second.
- "pet" (paraphrase, gold cat.txt): no file holds the word, so words find
  nothing; by meaning cat is 2 / sqrt(4.04) = 0.995 and the others 0.
- "osprey" (none): words find nothing; by meaning it is "kestrel" again, so
  its top score is sister's 0.133.
"""

import json
import math
import shutil
import subprocess
from pathlib import Path, PurePosixPath

import pytest
import rag_eval
from rag_eval import (
    DEEP_RERANKED,
    MAX_EXPANSIONS_BYTES,
    MAX_SET_BYTES,
    PARTS,
    RERANKED,
    VARIANTS,
    DeepRun,
    Measured,
    Parts,
    SetError,
    check_gold,
    chunkings,
    deep_runs,
    evaluate,
    main,
    parse_expansions,
    parse_set,
    percentile,
    read_expansions,
    rows,
)

from synthia.knowledge.chunk import Chunking
from synthia.knowledge.deep import Expansion
from synthia.knowledge.evaluate import (
    Answerable,
    Gold,
    RankMetrics,
    Report,
    Unanswerable,
)
from synthia.knowledge.index import IndexReport
from synthia.knowledge.retrieve import Merge, Variant
from synthia.models.catalogue import DEFAULT_EMBEDDER, RERANKERS, Embedder
from synthia.retrieval.embed import TextEmbedder
from tests.knowledge.corpus import NOTES, planted
from tests.retrieval.toy import Judge, Network, embedder, reranker

COMMIT = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret
QUESTIONS: list[dict[str, object]] = [
    {
        "id": "named-01",
        "category": "named",
        "question": "kestrel",
        "gold": [{"path": "server.txt", "start": 1, "end": 1}],
    },
    {
        "id": "paraphrase-01",
        "category": "paraphrase",
        "question": "pet",
        "gold": [{"path": "cat.txt", "start": 1, "end": 1}],
    },
    {"id": "none-01", "category": "none", "question": "osprey", "gold": []},
]
SECOND = 1 / math.log2(3)
"""nDCG@10 of one gold span found at rank 2."""
# recall@1, recall@5, mrr, nDCG@10 of "kestrel" per variant.
NAMED = {
    "words": (1.0, 1.0, 1.0, 1.0),
    "meaning": (0.0, 1.0, 1 / 3, 0.5),
    "fused": (0.0, 1.0, 1 / 3, 0.5),
    "fused:0.05": (0.0, 1.0, 1 / 3, 0.5),
    "fused:0.1": (0.0, 1.0, 1 / 3, 0.5),
    "fused:0.4": (0.0, 1.0, 0.5, SECOND),
    "fused+hop1x0.1": (0.0, 1.0, 1 / 3, 0.5),
    "fused+hop1x0.3": (0.0, 1.0, 1 / 3, 0.5),
    "fused+hop2x0.3": (0.0, 1.0, 1 / 3, 0.5),
}
KEYS = [
    "chunking",
    "variant",
    "category",
    "questions",
    "answerable",
    "recall@1",
    "recall@5",
    "recall@10",
    "mrr",
    "ndcg@10",
    "top_score",
    "ms_median",
    "ms_p95",
    "index_s",
    "chunks",
]
TIMES = ("ms_median", "ms_p95", "index_s")


def encoded(questions: list[dict[str, object]], commit: str = COMMIT) -> bytes:
    return json.dumps({"commit": commit, "questions": questions}).encode()


def toy() -> TextEmbedder:
    return embedder(Network())


def headless() -> list[Chunking]:
    return [c for c in chunkings(toy().spec) if not c.header]


def test_chunkings_are_default_windows_then_structure_without_and_with_header() -> None:
    names = [c.recipe() for c in chunkings(toy().spec)]
    assert names == [
        "lines/32/40/r1",
        "structure/32/40/r1",
        "structure+header/32/40/r1",
    ]


def test_variants_are_the_nine_named() -> None:
    names = [v.name() for v in VARIANTS]
    assert names == [
        "words",
        "meaning",
        "fused",
        "fused:0.05",
        "fused:0.1",
        "fused:0.4",
        "fused+hop1x0.1",
        "fused+hop1x0.3",
        "fused+hop2x0.3",
    ]


async def test_evaluate_scores_every_variant_by_hand(tmp_path: Path) -> None:
    corpus = planted(tmp_path / "corpus", NOTES)
    questions = parse_set(encoded(QUESTIONS)).questions
    measured = [
        m async for m in evaluate(corpus, questions, toy(), headless(), VARIANTS)
    ]
    assert len(measured) == 2 * len(VARIANTS)
    for m in measured:
        named = m.report.categories["named"].ranks
        paraphrase = m.report.categories["paraphrase"].ranks
        assert named is not None
        assert paraphrase is not None
        got = (named.recall[1], named.recall[5], named.reciprocal_rank, named.ndcg)
        assert got == pytest.approx(NAMED[m.variant.name()])
        found = 0.0 if m.variant.name() == "words" else 1.0
        assert paraphrase.reciprocal_rank == found
        assert m.report.categories["none"].ranks is None
        assert (m.chunks, len(m.ms)) == (3, 3)


def test_rows_round_to_three_places_and_null_unanswerable() -> None:
    third = 1 / 3
    report = Report.of(
        [
            (
                "named",
                Answerable(
                    RankMetrics(dict.fromkeys((1, 5, 10), third), third, third), 0.12345
                ),
            ),
            ("none", Unanswerable(0.5)),
        ]
    )
    chunking = Chunking(32, header=False)
    measured = Measured(
        chunking,
        Variant.parse("words"),
        report,
        (3.0, 1.0, 2.0),
        1.23456,
        7,
        IndexReport(chunks=7, embedded=7),
    )
    printed = rows(measured)
    assert [list(row) for row in printed] == [KEYS] * 3
    assert [row["category"] for row in printed] == ["named", "none", "overall"]
    named, none, overall = printed
    assert (named["recall@1"], named["mrr"], named["top_score"]) == (
        0.333,
        0.333,
        0.123,
    )
    assert (none["recall@1"], none["ndcg@10"], none["answerable"]) == (None, None, 0)
    assert (overall["questions"], overall["top_score"]) == (2, 0.312)
    assert (overall["ms_median"], overall["ms_p95"], overall["index_s"]) == (
        2.0,
        3.0,
        1.235,
    )
    assert json.loads(json.dumps(none))["mrr"] is None


@pytest.mark.parametrize(
    ("values", "share", "expected"),
    [
        ((5.0,), 0.95, 5.0),
        (tuple(map(float, range(20, 0, -1))), 0.95, 19.0),
        ((1.0, 2.0), 0.5, 1.0),
    ],
)
def test_percentile_is_the_nearest_rank(
    values: tuple[float, ...], share: float, expected: float
) -> None:
    assert percentile(values, share) == expected


@pytest.mark.parametrize(
    ("values", "share"), [((), 0.95), ((1.0,), 0.0), ((1.0,), 1.5)]
)
def test_percentile_refuses_no_values_or_a_share_outside(
    values: tuple[float, ...], share: float
) -> None:
    with pytest.raises(ValueError, match=r"need values"):
        percentile(values, share)


def test_parse_set_reads_questions_into_records() -> None:
    parsed = parse_set(encoded(QUESTIONS))
    assert parsed.commit == COMMIT
    first = parsed.questions[0]
    assert (first.id, first.category, first.text) == ("named-01", "named", "kestrel")
    assert first.gold == (Gold(PurePosixPath("server.txt"), 1, 1),)


def changed(index: int, **fields: object) -> list[dict[str, object]]:
    questions = [dict(q) for q in QUESTIONS]
    questions[index] = {**questions[index], **fields}
    return questions


def gold(
    path: str = "server.txt", start: object = 1, end: object = 1
) -> list[dict[str, object]]:
    return [{"path": path, "start": start, "end": end}]


@pytest.mark.parametrize(
    ("questions", "named"),
    [
        (changed(0, category="trivia"), "'named-01'"),
        (changed(2, gold=gold()), "'none-01' is 'none' yet has gold"),
        (changed(0, gold=[]), "'named-01' is 'named' and has no gold"),
        (changed(0, gold=gold(start=0)), "'named-01': gold in server.txt"),
        (changed(0, gold=gold(start=3, end=2)), "'named-01': gold in server.txt"),
        (changed(0, gold=gold(path="/etc/passwd")), "'named-01': gold path"),
        (changed(0, gold=gold(path="../outside.py")), "'named-01': gold path"),
        (changed(0, gold=gold(path="src\\a.py")), "'named-01': gold path"),
        (changed(0, gold=gold(start="1")), "'named-01'"),
        (changed(1, page=2), "'paraphrase-01'"),
        (changed(1, id="named-01"), "used more than once"),
        ([{"category": "named", "question": "q", "gold": gold()}], "#1"),
        ([], "no question"),
    ],
    ids=[
        "unknown_category",
        "gold_on_none",
        "answerable_without_gold",
        "start_below_one",
        "end_before_start",
        "absolute_path",
        "climbing_path",
        "backslash_path",
        "line_as_text",
        "unknown_field",
        "repeated_id",
        "missing_id",
        "empty",
    ],
)
def test_parse_set_refuses_naming_the_question(
    questions: list[dict[str, object]], named: str
) -> None:
    with pytest.raises(SetError, match=named):
        parse_set(encoded(questions))


@pytest.mark.parametrize(
    ("data", "named"),
    [
        (encoded(QUESTIONS, commit="HEAD"), "not a commit"),
        (encoded(QUESTIONS, commit="--output=x"), "not a commit"),
        (b"[]", "not a commit"),
        (b" " * (MAX_SET_BYTES + 1), "MAX_SET_BYTES"),
    ],
    ids=["ref_name", "option", "not_an_object", "too_large"],
)
def test_parse_set_refuses_a_bad_set(data: bytes, named: str) -> None:
    with pytest.raises(SetError, match=named):
        parse_set(data)


def test_check_gold_refuses_a_path_not_in_the_corpus(tmp_path: Path) -> None:
    corpus = planted(tmp_path / "corpus", {"cat.txt": "my cat is miso\n"})
    questions = parse_set(encoded(QUESTIONS)).questions
    with pytest.raises(SetError, match=r"'named-01': server.txt is not in the corpus"):
        check_gold(corpus, questions)


def committed(repository: Path) -> str:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not on the PATH")
    fixed = ["-c", "core.autocrlf=false", "-c", "commit.gpgsign=false"]
    author = ["-c", "user.name=eval", "-c", "user.email=eval@example.invalid"]
    for arguments in (
        ["init", "--quiet"],
        ["add", "."],
        [*author, "commit", "--quiet", "--message", "corpus"],
    ):
        subprocess.run(  # noqa: S603 - a resolved git with fixed arguments
            [git, "-C", str(repository), *fixed, *arguments], check=True, timeout=60
        )
    head = subprocess.run(  # noqa: S603 - a resolved git with fixed arguments
        [git, "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return head.stdout.strip()


def unspent(row: dict[str, object]) -> dict[str, object]:
    for key in TIMES:
        value = row[key]
        assert isinstance(value, float)
        assert value >= 0
    return {key: value for key, value in row.items() if key not in TIMES}


def expected(
    variant: str, category: str, values: tuple[object, ...]
) -> dict[str, object]:
    fixed = {"chunking": "structure/32/40/r1", "variant": variant, "category": category}
    return {
        **fixed,
        **dict(zip([k for k in KEYS if k not in TIMES][3:], values, strict=True)),
    }


async def test_main_archives_the_commit_and_prints_the_chosen_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repository = planted(tmp_path / "repository", NOTES)
    commit = committed(repository)
    question_set = tmp_path / "set.json"
    question_set.write_bytes(encoded(QUESTIONS, commit=commit))
    per_question = tmp_path / "per_question.jsonl"
    argv = ["--set", str(question_set), "--repository", str(repository)]
    argv += ["--per-question", str(per_question)]
    argv += [
        "--chunking",
        "structure/32/40/r1",
        "--variant",
        "words",
        "--variant",
        "fused:0.4",
    ]
    assert await main(argv, load=toy, load_rerankers=dict) == 0
    header, index, *printed = [
        json.loads(line) for line in capsys.readouterr().out.splitlines()
    ]
    assert index["chunking"] == "structure/32/40/r1"
    assert {
        k: index["index"][k]
        for k in ("added", "unreadable", "held_back", "chunks", "embedded")
    } == {"added": 3, "unreadable": 0, "held_back": 0, "chunks": 3, "embedded": 3}
    assert {
        k: header[k] for k in ("commit", "questions", "corpus_files", "embedder")
    } == {
        "commit": commit,
        "questions": {"named": 1, "paraphrase": 1, "none": 1},
        "corpus_files": 3,
        "embedder": "tiny-embedder",
    }
    assert set(header) >= {"python", "numpy", "onnxruntime"}
    assert header["rerankers"] == []
    assert [unspent(row) for row in printed] == [
        expected("words", "named", (1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 0.511, 3)),
        expected("words", "paraphrase", (1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3)),
        expected("words", "none", (1, 0, None, None, None, None, None, 0.0, 3)),
        expected("words", "overall", (3, 2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.17, 3)),
        expected("fused:0.4", "named", (1, 1, 0.0, 1.0, 1.0, 0.5, 0.631, 0.133, 3)),
        expected("fused:0.4", "paraphrase", (1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 0.995, 3)),
        expected("fused:0.4", "none", (1, 0, None, None, None, None, None, 0.133, 3)),
        expected("fused:0.4", "overall", (3, 2, 0.5, 1.0, 1.0, 0.75, 0.815, 0.42, 3)),
    ]
    each = [json.loads(line) for line in per_question.read_text("utf-8").splitlines()]
    assert [(e["variant"], e["id"], e["ndcg@10"], e["mrr"]) for e in each] == [
        (variant, question["id"], ndcg, mrr)
        for variant, question, ndcg, mrr in zip(
            ["words"] * 3 + ["fused:0.4"] * 3,
            QUESTIONS * 2,
            [1.0, 0.0, None, 0.631, 1.0, None],
            [1.0, 0.0, None, 0.5, 1.0, None],
            strict=True,
        )
    ]


async def test_main_refuses_an_unknown_chunking(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    question_set = tmp_path / "set.json"
    question_set.write_bytes(encoded(QUESTIONS))
    argv = ["--set", str(question_set), "--chunking", "sentences/32/40/r1"]
    with pytest.raises(SystemExit) as stopped:
        await main(argv, load=toy, load_rerankers=dict)
    assert stopped.value.code == 2
    assert "unknown chunking ['sentences/32/40/r1']" in capsys.readouterr().err


def test_installed_embedder_loads_the_default_from_the_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[Path, Embedder]] = []
    model = object()

    def load(directory: Path, spec: Embedder) -> object:
        calls.append((directory, spec))
        return model

    monkeypatch.setattr(rag_eval, "default_home", lambda: tmp_path)
    monkeypatch.setattr(TextEmbedder, "load", load)
    assert rag_eval.installed_embedder() is model
    ((directory, spec),) = calls
    assert spec is DEFAULT_EMBEDDER
    assert directory.is_relative_to(tmp_path)


async def test_main_refuses_a_reranked_variant_whose_reranker_is_not_installed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    question_set = tmp_path / "set.json"
    question_set.write_bytes(encoded(QUESTIONS))
    name = RERANKED[0].name()
    argv = ["--set", str(question_set), "--variant", name]
    with pytest.raises(SystemExit) as stopped:
        await main(argv, load=toy, load_rerankers=dict)
    assert stopped.value.code == 2
    assert f"variant [{name!r}] needs a reranker" in capsys.readouterr().err


def test_every_reranked_variant_names_a_catalogued_reranker() -> None:
    ids = {spec.id for spec in RERANKERS}
    assert all(v.rerank is not None and v.rerank.model in ids for v in RERANKED)


async def test_evaluate_scores_a_reranked_variant_with_the_reranker_given(
    tmp_path: Path,
) -> None:
    corpus = planted(tmp_path / "corpus", NOTES)
    questions = parse_set(encoded(QUESTIONS)).questions
    judge = reranker(Judge("kestrel"))
    variant = Variant.parse("fused+rerank:ms-marco-minilm-l6-int8@10")

    [measured] = [
        m
        async for m in evaluate(
            corpus,
            questions,
            toy(),
            headless()[:1],
            [variant],
            rerankers={"ms-marco-minilm-l6-int8": judge},
        )
    ]

    named = measured.report.categories["named"].ranks
    assert named is not None
    assert (named.recall[1], named.ndcg) == (1.0, 1.0)


# "kestrel" asked again as "database server" finds server first by meaning;
# its passage is not drafted, so by the passage alone it is the plain search.
EXPANSIONS: dict[str, object] = {
    "model": "local-test",
    "questions": {
        "named-01": {
            "question": "kestrel",
            "queries": ["database server"],
            "passage": None,
            "seconds": 2.0,
        },
        "paraphrase-01": {
            "question": "pet",
            "queries": [],
            "passage": "my cat",
            "seconds": 4.0,
        },
        "none-01": {
            "question": "osprey",
            "queries": ["bird"],
            "passage": "an osprey",
            "seconds": 3.0,
        },
    },
}


def expansions_file(tmp_path: Path, data: dict[str, object] = EXPANSIONS) -> Path:
    path = tmp_path / "expansions.json"
    path.write_bytes(json.dumps(data).encode())
    return path


def test_read_expansions_reads_each_questions_expansion(tmp_path: Path) -> None:
    questions = parse_set(encoded(QUESTIONS)).questions

    read = read_expansions(expansions_file(tmp_path), questions)

    assert read.model == "local-test"
    assert read.by_question["named-01"] == Expansion(("database server",), None)
    assert read.by_question["none-01"] == Expansion(("bird",), "an osprey")
    assert read.seconds == {"named-01": 2.0, "paraphrase-01": 4.0, "none-01": 3.0}
    assert read.texts == {
        "named-01": "kestrel",
        "paraphrase-01": "pet",
        "none-01": "osprey",
    }


def test_read_expansions_refuses_an_expansion_of_another_text(tmp_path: Path) -> None:
    questions = parse_set(encoded(QUESTIONS)).questions
    data = json.loads(json.dumps(EXPANSIONS))
    data["questions"]["paraphrase-01"]["question"] = "dog"

    with pytest.raises(
        SetError, match=r"expanded another text of the questions \['paraphrase-01'\]"
    ):
        read_expansions(expansions_file(tmp_path, data), questions)


def expanded_as(question_id: str, /, **fields: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "question": question_id,
        "queries": [],
        "passage": None,
        "seconds": 1.0,
    }
    entry.update(fields)
    return {"model": "m", "questions": {question_id: entry}}


@pytest.mark.parametrize(
    ("data", "named"),
    [
        (b" " * (MAX_EXPANSIONS_BYTES + 1), "MAX_EXPANSIONS_BYTES"),
        (b"[]", "not a model and expansions"),
        (json.dumps({"model": "", "questions": {}}).encode(), "not a model"),
        (json.dumps(expanded_as("q", seconds=-1.0)).encode(), "not a model"),
        (json.dumps(expanded_as("q", extra=1)).encode(), "not a model"),
        (json.dumps(expanded_as("q", question="")).encode(), "not a model"),
        (json.dumps(expanded_as("q", queries=[" a"])).encode(), "question 'q': ' a'"),
        (json.dumps(expanded_as("q", passage="")).encode(), "question 'q': '' is"),
    ],
    ids=[
        "too_large",
        "not_an_object",
        "no_model",
        "negative_seconds",
        "unknown_field",
        "blank_question",
        "unclean_query",
        "blank_passage",
    ],
)
def test_parse_expansions_refuses_what_deep_search_does_not_make(
    data: bytes, named: str
) -> None:
    with pytest.raises(SetError, match=named):
        parse_expansions(data)


def test_read_expansions_refuses_a_question_not_expanded(tmp_path: Path) -> None:
    questions = parse_set(encoded(QUESTIONS)).questions
    path = expansions_file(tmp_path, expanded_as("named-01"))

    with pytest.raises(
        SetError, match=r"does not expand the questions \['paraphrase-01', 'none-01'\]"
    ):
        read_expansions(path, questions)


@pytest.mark.parametrize(
    ("parts", "also"),
    [
        ("rewrites", ("bird",)),
        ("hyde", ("an osprey",)),
        ("both", ("bird", "an osprey")),
    ],
)
def test_a_deep_run_searches_the_parts_it_names(
    parts: Parts, also: tuple[str, ...]
) -> None:
    run = DeepRun(parts, Merge.RRF, Variant(), {"q": Expansion(("bird",), "an osprey")})

    assert run.also("q") == also
    assert run.label() == f"{parts}/rrf"


def test_a_deep_run_without_a_passage_adds_none() -> None:
    run = DeepRun("hyde", Merge.BEST, Variant(), {"q": Expansion(("bird",), None)})

    assert run.also("q") == ()


def test_deep_runs_are_every_part_and_merge_then_both_reranked() -> None:
    plain = deep_runs({}, [])
    reranked = deep_runs({}, ["ms-marco-minilm-l6-int8"])

    assert [(r.label(), r.variant.name()) for r in plain] == [
        (f"{parts}/{merge}", "fused")
        for parts in ("rewrites", "hyde", "both")
        for merge in ("best", "rrf")
    ]
    assert reranked[:6] == plain
    assert [(r.label(), r.variant) for r in reranked[6:]] == [
        ("both/best", DEEP_RERANKED),
        ("both/rrf", DEEP_RERANKED),
    ]


async def test_evaluate_scores_deep_runs_on_the_first_chunking_only(
    tmp_path: Path,
) -> None:
    corpus = planted(tmp_path / "corpus", NOTES)
    questions = parse_set(encoded(QUESTIONS)).questions
    expansions = read_expansions(expansions_file(tmp_path), questions)

    measured = [
        m
        async for m in evaluate(
            corpus,
            questions,
            toy(),
            headless(),
            [Variant()],
            deep=deep_runs(expansions.by_question, []),
        )
    ]

    labels = [(m.chunking.recipe(), m.deep) for m in measured]
    first, second = (c.recipe() for c in headless())
    assert labels == [
        (first, None),
        *[(first, f"{p}/{m}") for p in PARTS for m in ("best", "rrf")],
        (second, None),
    ]
    named = {m.deep: m.report.categories["named"].ranks for m in measured[:7]}
    recall = {deep: ranks.recall[1] for deep, ranks in named.items() if ranks}
    assert recall == {
        None: 0.0,
        "rewrites/best": 1.0,
        "rewrites/rrf": 1.0,
        "hyde/best": 0.0,
        "hyde/rrf": 0.0,
        "both/best": 1.0,
        "both/rrf": 1.0,
    }


async def test_main_prints_deep_rows_marked_and_the_expansions_in_the_header(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repository = planted(tmp_path / "repository", NOTES)
    commit = committed(repository)
    question_set = tmp_path / "set.json"
    question_set.write_bytes(encoded(QUESTIONS, commit=commit))
    per_question = tmp_path / "per_question.jsonl"
    argv = ["--set", str(question_set), "--repository", str(repository)]
    argv += ["--chunking", "lines/32/40/r1", "--variant", "words"]
    argv += ["--expansions", str(expansions_file(tmp_path))]
    argv += ["--per-question", str(per_question)]

    assert await main(argv, load=toy, load_rerankers=dict) == 0

    header, _, *printed = [
        json.loads(line) for line in capsys.readouterr().out.splitlines()
    ]
    assert header["expansions"] == {
        "model": "local-test",
        "s_median": 3.0,
        "s_p95": 4.0,
    }
    marked = [(row["variant"], row.get("deep")) for row in printed]
    assert marked == [("words", None)] * 4 + [
        ("fused", f"{p}/{m}") for p in PARTS for m in ("best", "rrf") for _ in range(4)
    ]
    assert list(printed[4])[-1] == "deep"
    each = [json.loads(line) for line in per_question.read_text("utf-8").splitlines()]
    assert [e.get("deep") for e in each[:4]] == [None, None, None, "rewrites/best"]


async def test_main_refuses_expansions_without_the_default_chunking(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    question_set = tmp_path / "set.json"
    question_set.write_bytes(encoded(QUESTIONS))
    argv = ["--set", str(question_set), "--chunking", "structure/32/40/r1"]
    argv += ["--expansions", str(expansions_file(tmp_path))]

    with pytest.raises(SystemExit) as stopped:
        await main(argv, load=toy, load_rerankers=dict)

    assert stopped.value.code == 2
    assert "scores the default chunking lines/32/40/r1" in capsys.readouterr().err
