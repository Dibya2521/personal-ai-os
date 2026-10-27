import hashlib
import io
import json
from pathlib import Path, PurePosixPath

import pytest
from answer_eval import (
    MAX_CALIBRATION_BYTES,
    MAX_VERDICTS_BYTES,
    Agreement,
    Asked,
    Cited,
    Judged,
    Labelled,
    Outcome,
    agreement,
    calibrated,
    calibration_line,
    held_verdicts,
    judge_all,
    kappa,
    main,
    names,
    outcome,
    overlaps_gold,
    parse_calibration,
    parse_verdicts,
    passages_sha256,
    rows,
    shown,
    to_judge,
    written_verdicts,
)
from chat_answers import CUT_OFF, Answered, Search, written
from hypothesis import given
from hypothesis import strategies as st
from judge import NO_PASSAGES, Verdict, prompt_sha256
from rag_eval import Category, Question, SetError

from synthia.knowledge.evaluate import Gold
from tests.benchmarks.test_chat_answers import PROVENANCE, Service
from tests.benchmarks.test_rag_eval import QUESTIONS, encoded
from tests.knowledge.test_deep import INFO, Local
from tests.knowledge.test_evaluate import passage

COMMIT = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret
A = PurePosixPath("a.py")
SUPPORTED = Verdict(supported=True, unsupported=[])
UNSUPPORTED = Verdict(supported=False, unsupported=["a claim"])
LABELLED_PASSAGES = ("[corpus/a.py:1-5] x",)
PROMPT = prompt_sha256()


def said(verdict: Verdict) -> str:
    return verdict.model_dump_json()


def answered(  # noqa: PLR0913 - one answer's whole record
    category: Category,
    answer: str,
    *,
    sources: list[str],
    searches: list[Search],
    seconds: float,
    question: str = "q",
    error: str | None = None,
) -> Answered:
    return {
        "question": question,
        "category": category,
        "answer": answer,
        "searches": searches,
        "sources": sources,
        "seconds": seconds,
        "prompt_tokens": None,
        "completion_tokens": None,
        "route": "local",
        "model": "local",
        "error": error,
    }


def ran(result: str = "[corpus/a.py:1-5] x", *, ok: bool = True) -> Search:
    return {"query": "q", "result": result, "ok": ok}


def judged(
    verdict: Verdict | None,
    answer: str = "a",
    passages: tuple[str, ...] = LABELLED_PASSAGES,
) -> Judged:
    error = None if verdict is not None else "no valid Verdict"
    return Judged("q", answer, passages_sha256(passages), verdict, error, 1.0)


@pytest.mark.parametrize(
    ("citation", "expected"),
    [
        ("corpus/a.py:3", Cited(A, None, 3, 3)),
        ("corpus/a.py:3-9", Cited(A, None, 3, 9)),
        ("corpus/docs/b.md:p.2:4-9", Cited(PurePosixPath("docs/b.md"), 2, 4, 9)),
        ("corpus/c:d.py:1-2", Cited(PurePosixPath("c:d.py"), None, 1, 2)),
        ("a.py:3", None),
        ("corpus/a.py", None),
        ("corpus/a.py:3-", None),
    ],
    ids=["line", "range", "page", "colon_in_path", "no_source", "no_lines", "open"],
)
def test_a_citation_names_its_path_page_and_lines(
    citation: str, expected: Cited | None
) -> None:
    assert Cited.parse(citation) == expected


@pytest.mark.parametrize(
    ("path", "page", "start", "end"),
    [
        (A, None, 4, 4),
        (A, None, 3, 9),
        (PurePosixPath("docs/b c.md"), 2, 5, 5),
        (PurePosixPath("docs/b.md"), 2, 1, 7),
    ],
    ids=["line", "range", "page_line", "page_range"],
)
def test_cited_parses_what_a_passage_cites(
    path: PurePosixPath, page: int | None, start: int, end: int
) -> None:
    citation = passage(path, start, end, page).citation()

    assert Cited.parse(citation) == Cited(path, page, start, end)


SEGMENT = st.text(st.characters(exclude_characters=":/\n"), min_size=1)


@given(
    segments=st.lists(SEGMENT, min_size=1, max_size=4),
    page=st.none() | st.integers(min_value=1, max_value=10_000),
    start=st.integers(min_value=1, max_value=1_000_000),
    length=st.integers(min_value=0, max_value=1_000),
)
def test_cited_parses_every_citation_a_passage_writes(
    segments: list[str], page: int | None, start: int, length: int
) -> None:
    path = PurePosixPath(*segments)
    end = start + length

    assert Cited.parse(passage(path, start, end, page).citation()) == Cited(
        path, page, start, end
    )


@pytest.mark.parametrize(
    ("citation", "expected"),
    [
        ("corpus/a.py:1-5", True),
        ("corpus/a.py:8-30", True),
        ("corpus/a.py:9", False),
        ("corpus/a.py:1-4", False),
        ("corpus/b.py:5", False),
        ("corpus/a.py:p.1:5", False),
        ("not a citation", False),
    ],
)
def test_a_citation_overlaps_gold_by_the_search_eval_rule(
    citation: str, *, expected: bool
) -> None:
    assert overlaps_gold(citation, (Gold(A, 5, 8),)) is expected


@pytest.mark.parametrize(
    ("answer", "citation", "expected"),
    [
        ("It is [corpus/a.py:1].", "corpus/a.py:1", True),
        ("(corpus/a.py:1-20)", "corpus/a.py:1-20", True),
        ("see corpus/a.py:1-20", "corpus/a.py:1", False),
        ("see corpus/a.py:1-20", "corpus/a.py:1-2", False),
        ("see corpus/a.py:12", "corpus/a.py:1", False),
        ("see mycorpus/a.py:1", "corpus/a.py:1", False),
        ("see /x/corpus/a.py:1", "corpus/a.py:1", False),
        ("see a.py lines 1 to 5", "corpus/a.py:1-5", False),
    ],
    ids=["bracketed", "parens", "range", "longer", "digit", "word", "path", "prose"],
)
def test_an_answer_names_a_citation_only_as_written(
    answer: str, citation: str, *, expected: bool
) -> None:
    assert names(answer, citation) is expected


def test_shown_is_the_text_of_every_search_that_ran() -> None:
    record = answered(
        "named",
        "a",
        sources=[],
        searches=[ran("one"), ran("error: bad query", ok=False), ran("two")],
        seconds=1.0,
    )

    assert shown(record) == ("one", "two")


def outcomes() -> list[Outcome]:
    gold_a = (Gold(A, 5, 8),)
    asked = [
        (
            Question("n1", "named", "q", gold_a),
            answered(
                "named",
                "It is [corpus/a.py:1-5].",
                sources=["corpus/a.py:1-5", "corpus/z.py:1"],
                searches=[ran()],
                seconds=10.0,
            ),
            judged(SUPPORTED),
        ),
        (
            Question("n2", "named", "q", (Gold(PurePosixPath("b.py"), 1, 1),)),
            answered(
                "named",
                "See corpus/b.py:1.",
                sources=["corpus/b.py:3"],
                searches=[ran()],
                seconds=20.0,
            ),
            judged(UNSUPPORTED),
        ),
        (
            Question("p1", "paraphrase", "q", (Gold(PurePosixPath("c.md"), 10, 12),)),
            answered("paraphrase", "A cat.", sources=[], searches=[], seconds=30.0),
            judged(None),
        ),
        (
            Question("x1", "none", "q", ()),
            answered(
                "none",
                "Nothing found.",
                sources=[],
                searches=[ran("error", ok=False)],
                seconds=40.0,
            ),
            judged(SUPPORTED),
        ),
    ]
    return [outcome(question, record, verdict) for question, record, verdict in asked]


def test_an_answer_citing_gold_it_was_never_shown_is_not_cited() -> None:
    assert outcomes()[1] == Outcome(
        "named",
        searched=True,
        shown=False,
        cited=False,
        supported=False,
        seconds=20.0,
        chat_error=False,
    )


def test_an_unanswerable_question_has_no_recall() -> None:
    assert outcomes()[3] == Outcome(
        "none",
        searched=True,
        shown=None,
        cited=None,
        supported=True,
        seconds=40.0,
        chat_error=False,
    )


def test_an_answer_whose_turn_ended_in_an_error_is_a_chat_error_not_a_judge_one() -> (
    None
):
    record = answered(
        "named",
        "It is [corpus/a.py:1-5",
        sources=[],
        searches=[ran()],
        seconds=7.0,
        error=CUT_OFF,
    )

    found = outcome(Question("n3", "named", "q", (Gold(A, 5, 8),)), record, None)

    assert to_judge(record) is None
    assert found == Outcome(
        "named",
        searched=True,
        shown=None,
        cited=None,
        supported=None,
        seconds=7.0,
        chat_error=True,
    )
    named, overall = rows([found, *outcomes()[:2]])
    assert (named["searched"], named["shown_recall"], named["cited_recall"]) == (
        1.0,
        0.5,
        0.5,
    )
    assert (named["supported"], named["judge_errors"], named["chat_errors"]) == (
        0.5,
        0,
        1,
    )
    assert overall["chat_errors"] == 1


def test_an_answer_naming_only_a_shown_citation_off_gold_is_not_cited() -> None:
    record = answered(
        "named",
        "It is in [corpus/b.py:1].",
        sources=["corpus/a.py:1-5", "corpus/b.py:1"],
        searches=[ran()],
        seconds=3.0,
    )

    found = outcome(Question("n4", "named", "q", (Gold(A, 5, 8),)), record, None)

    assert (found.shown, found.cited) == (True, False)


def test_rows_are_per_category_then_overall_by_hand() -> None:
    assert rows(outcomes()) == [
        {
            "category": "named",
            "n": 2,
            "searched": 1.0,
            "shown_recall": 0.5,
            "cited_recall": 0.5,
            "supported": 0.5,
            "judge_errors": 0,
            "chat_errors": 0,
            "s_median": 15.0,
            "s_p95": 20.0,
        },
        {
            "category": "paraphrase",
            "n": 1,
            "searched": 0.0,
            "shown_recall": 0.0,
            "cited_recall": 0.0,
            "supported": None,
            "judge_errors": 1,
            "chat_errors": 0,
            "s_median": 30.0,
            "s_p95": 30.0,
        },
        {
            "category": "none",
            "n": 1,
            "searched": 1.0,
            "shown_recall": None,
            "cited_recall": None,
            "supported": 1.0,
            "judge_errors": 0,
            "chat_errors": 0,
            "s_median": 40.0,
            "s_p95": 40.0,
        },
        {
            "category": "overall",
            "n": 4,
            "searched": 0.75,
            "shown_recall": 0.333,
            "cited_recall": 0.333,
            "supported": 0.667,
            "judge_errors": 1,
            "chat_errors": 0,
            "s_median": 25.0,
            "s_p95": 40.0,
        },
    ]


def test_every_row_carries_the_judge_agreement_when_calibrated() -> None:
    calibration = agreement([(True, True), (False, True)])

    for row in rows(outcomes(), calibration):
        assert (row.get("judge_agreement"), row.get("judge_kappa")) == (0.5, 0.0)


T, F = True, False


@pytest.mark.parametrize(
    ("pairs", "expected"),
    [
        ([(T, T)] * 3 + [(F, F), (F, T), (T, F)], 0.25),
        ([(T, T), (F, F)], 1.0),
        ([(T, F), (F, T)], -1.0),
        ([(T, F)] * 3, 0.0),
        ([(T, T)] * 4, None),
        ([(F, F)] * 2, None),
        ([], None),
    ],
    ids=[
        "mixed",
        "perfect",
        "opposite",
        "judge_one_class",
        "all_yes",
        "all_no",
        "none",
    ],
)
def test_kappa_by_hand_and_undefined_without_dividing_by_zero(
    pairs: list[tuple[bool, bool]], expected: float | None
) -> None:
    assert kappa(pairs) == expected


def test_agreement_leaves_errors_out_and_counts_catches_and_false_alarms() -> None:
    found = agreement([(T, T), (F, T), (F, F), (T, F), (None, F), (F, F)])

    assert found == Agreement(
        n=5,
        errors=1,
        agreed=3,
        caught=2,
        labelled_unsupported=3,
        false_alarms=1,
        labelled_supported=2,
        kappa=kappa([(T, T), (F, T), (F, F), (T, F), (F, F)]),
    )
    assert found.share == 0.6


def test_agreement_on_nothing_judged_has_no_share() -> None:
    found = agreement([(None, T)])

    assert (found.n, found.errors, found.share, found.kappa) == (0, 1, None, None)


def labelled(
    item: str, answer: str, *, supported: bool, origin: str = "natural"
) -> dict[str, object]:
    return {
        "id": item,
        "question": "q",
        "passages": ["[corpus/a.py:1-5] x"],
        "answer": answer,
        "supported": supported,
        "unsupported": [] if supported else ["a claim"],
        "labelled_by": "Claude",
        "origin": origin,
    }


def calibration(*items: dict[str, object]) -> bytes:
    return json.dumps({"items": list(items)}).encode()


def test_parse_calibration_reads_each_labelled_answer() -> None:
    (item,) = parse_calibration(calibration(labelled("c1", "b", supported=False)))

    assert item == Labelled(
        "c1",
        "q",
        ("[corpus/a.py:1-5] x",),
        "b",
        supported=False,
        unsupported=("a claim",),
        labelled_by="Claude",
        origin="natural",
    )


@pytest.mark.parametrize(
    ("data", "named"),
    [
        (calibration(), "has no item"),
        (
            calibration(
                labelled("c1", "a", supported=T), labelled("c1", "b", supported=T)
            ),
            r"ids \['c1'\] are used more than once",
        ),
        (
            calibration({**labelled("c1", "a", supported=T), "unsupported": ["x"]}),
            r"items \['c1'\] are labelled supported with",
        ),
        (
            calibration({**labelled("c1", "a", supported=F), "unsupported": []}),
            r"items \['c1'\] are labelled",
        ),
        (calibration({**labelled("c1", "a", supported=T), "origin": "made"}), "not a"),
        (
            calibration({**labelled("c1", "a", supported=F), "unsupported": [""]}),
            r"items\.0\.unsupported\.0",
        ),
        (b"[]", "not a list of labelled answers"),
        (b" " * (MAX_CALIBRATION_BYTES + 1), "MAX_CALIBRATION_BYTES"),
    ],
    ids=[
        "empty",
        "repeated",
        "true_with_claims",
        "false_without",
        "origin",
        "empty_claim",
        "list",
        "big",
    ],
)
def test_parse_calibration_refuses_a_bad_set(data: bytes, named: str) -> None:
    with pytest.raises(SetError, match=named):
        parse_calibration(data)


def test_calibrated_is_overall_then_each_origin() -> None:
    items = parse_calibration(
        calibration(
            labelled("c1", "a", supported=T),
            labelled("c2", "b", supported=F, origin="perturbed"),
            labelled("c3", "c", supported=F),
        )
    )
    verdicts = {
        j.key: j
        for j in (judged(SUPPORTED, "a"), judged(SUPPORTED, "b"), judged(None, "c"))
    }

    found = calibrated(items, verdicts)

    assert [(group, a.n, a.errors, a.agreed) for group, a in found] == [
        ("overall", 2, 1, 1),
        ("natural", 1, 1, 1),
        ("perturbed", 1, 0, 0),
    ]
    assert calibration_line(*found[2]) == {
        "calibration": "perturbed",
        "n": 1,
        "judge_errors": 0,
        "agreement": 0.0,
        "unsupported_caught": 0,
        "unsupported_labelled": 1,
        "false_alarms": 0,
        "supported_labelled": 0,
        "kappa": 0.0,
    }
    assert calibrated((), verdicts) == []


def test_judged_holds_a_verdict_or_an_error() -> None:
    for verdict, error in [(SUPPORTED, "x"), (None, None)]:
        with pytest.raises(ValueError, match="exactly one is set"):
            Judged("q", "a", passages_sha256(()), verdict, error, 1.0)


def test_passages_sha256_is_of_the_passages_as_a_json_list() -> None:
    assert passages_sha256(["a"]) == hashlib.sha256(b'["a"]').hexdigest()
    assert passages_sha256(("a", "b")) == passages_sha256(["a", "b"])
    assert passages_sha256(["a", "b"]) != passages_sha256(["ab"])
    assert passages_sha256(()) != passages_sha256(("",))
    assert Asked("q", "a", ("a", "b")).key != Asked("q", "a", ("ab",)).key


async def test_the_same_answer_shown_other_passages_is_judged_again(
    tmp_path: Path,
) -> None:
    out = tmp_path / "verdicts.json"
    model = Local(said(UNSUPPORTED))
    before = Asked("q", "a", ("[corpus/a.py:1-5] one",))
    after = Asked("q", "a", ("[corpus/a.py:1-5] two",))
    verdicts = {before.key: Judged(*before.key, SUPPORTED, None, 1.0)}
    now = iter([0.0, 2.0])

    await judge_all(
        model,
        [before, after],
        out,
        verdicts,
        progress=io.StringIO(),
        clock=lambda: next(now),
    )

    (request,) = model.requests
    assert "Passages:\n[corpus/a.py:1-5] two\n" in request.messages[-1].text
    found = held_verdicts(out, INFO.id, PROMPT)
    assert found[before.key].verdict == SUPPORTED
    assert found[after.key] == Judged(*after.key, UNSUPPORTED, None, 2.0)


async def test_judge_all_records_a_judge_that_never_replied_as_an_error(
    tmp_path: Path,
) -> None:
    out = tmp_path / "new" / "verdicts.json"
    model = Local(said(UNSUPPORTED), "no", "still no", "never")
    asked = [
        Asked("q", "held", ()),
        Asked("q", "made up", ("[corpus/a.py:1-5] x",)),
        Asked("q", "garbled", ()),
    ]
    held = judged(SUPPORTED, "held", ())
    verdicts = {held.key: held}
    progress = io.StringIO()
    now = iter([0.0, 0.5, 1.0, 3.25])

    await judge_all(
        model, asked, out, verdicts, progress=progress, clock=lambda: next(now)
    )

    assert b"\r" not in out.read_bytes()
    found = held_verdicts(out, INFO.id, PROMPT)
    assert found[asked[0].key] == held
    assert found[asked[1].key] == Judged(*asked[1].key, UNSUPPORTED, None, 0.5)
    garbled = found[asked[2].key]
    assert (garbled.verdict, garbled.seconds) == (None, 2.25)
    assert garbled.error is not None
    assert garbled.error.startswith("no valid Verdict after 2 repairs")
    assert progress.getvalue().splitlines() == [
        "2/3 0.5 s, unsupported",
        "3/3 2.2 s, error",
    ]
    assert NO_PASSAGES in model.requests[-1].messages[5].text
    assert not (tmp_path / "new" / "verdicts.json.partial").exists()


def verdicts_data(*verdicts: dict[str, object], prompt: str = PROMPT) -> bytes:
    data = {"model": "m", "judge_prompt_sha256": prompt, "verdicts": list(verdicts)}
    return json.dumps(data).encode()


def one_verdict(**fields: object) -> dict[str, object]:
    return {
        "question": "q",
        "answer": "a",
        "passages_sha256": passages_sha256(()),
        "verdict": SUPPORTED.model_dump(),
        "error": None,
        "seconds": 1.0,
        **fields,
    }


async def test_judge_all_judges_again_an_answer_the_judge_gave_no_verdict_for(
    tmp_path: Path,
) -> None:
    out = tmp_path / "verdicts.json"
    item = Asked("q", "a", LABELLED_PASSAGES)
    failed = judged(None)
    model = Local(said(SUPPORTED))
    now = iter([0.0, 1.0])

    await judge_all(
        model,
        [item],
        out,
        {failed.key: failed},
        progress=io.StringIO(),
        clock=lambda: next(now),
    )

    assert len(model.requests) == 1
    found = held_verdicts(out, INFO.id, PROMPT)
    assert found == {item.key: Judged(*item.key, SUPPORTED, None, 1.0)}


@pytest.mark.parametrize(
    ("data", "named"),
    [
        (verdicts_data({}), "not a model, a prompt digest and a list"),
        (verdicts_data(one_verdict(verdict=None)), "exactly one is set"),
        (verdicts_data(one_verdict(passages_sha256="ab")), "passages_sha256"),
        (verdicts_data(prompt="ab"), "judge_prompt_sha256"),
        (b" " * (MAX_VERDICTS_BYTES + 1), "MAX_VERDICTS_BYTES"),
    ],
    ids=["not_a_verdict", "neither", "not_a_digest", "not_a_prompt", "too_large"],
)
def test_parse_verdicts_refuses_what_is_not_verdicts(data: bytes, named: str) -> None:
    with pytest.raises(SetError, match=named):
        parse_verdicts(data)


def test_held_verdicts_refuses_another_judges_verdicts(tmp_path: Path) -> None:
    out = tmp_path / "verdicts.json"
    written_verdicts(out, "other", PROMPT, {})

    with pytest.raises(SetError, match=r"holds other's verdicts, not local's"):
        held_verdicts(out, INFO.id, PROMPT)


def test_held_verdicts_refuses_verdicts_under_another_judge_prompt(
    tmp_path: Path,
) -> None:
    out = tmp_path / "verdicts.json"
    older = prompt_sha256(instructions="Say whether the answer is supported.")
    written_verdicts(out, INFO.id, older, {})

    with pytest.raises(
        SetError, match=f"holds verdicts under judge prompt {older}, not {PROMPT}$"
    ):
        held_verdicts(out, INFO.id, PROMPT)


async def test_main_resume_refuses_verdicts_under_another_judge_prompt(
    tmp_path: Path,
) -> None:
    answers = tmp_path / "answers.json"
    answers_file(answers)
    question_set = tmp_path / "set.json"
    question_set.write_bytes(encoded(QUESTIONS, commit=COMMIT))
    out = tmp_path / "verdicts.json"
    written_verdicts(out, INFO.id, "0" * 64, {})
    model = Local()
    argv = ["--answers", str(answers), "--verdicts", str(out), "--resume"]

    with pytest.raises(SetError, match=r"under judge prompt 0{64}, not"):
        await main(
            [*argv, "--set", str(question_set)], service=lambda _: Service(model)
        )

    assert model.requests == []


def answers_file(path: Path, paraphrase_error: str | None = None) -> None:
    named = answered(
        "named",
        "kestrel [corpus/server.txt:1]",
        sources=["corpus/server.txt:1", "corpus/cat.txt:1"],
        searches=[ran("[corpus/server.txt:1] kestrel")],
        seconds=2.0,
        question="kestrel",
    )
    paraphrase = answered(
        "paraphrase",
        "a cat",
        sources=[],
        searches=[],
        seconds=4.0,
        question="pet",
        error=paraphrase_error,
    )
    unanswerable = answered(
        "none",
        "nothing",
        sources=[],
        searches=[ran("error", ok=False)],
        seconds=6.0,
        question="osprey",
    )
    written(
        path,
        PROVENANCE,
        {"named-01": named, "paraphrase-01": paraphrase, "none-01": unanswerable},
    )


async def test_main_judges_once_then_scores_from_the_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    answers = tmp_path / "answers.json"
    answers_file(answers)
    question_set = tmp_path / "set.json"
    question_set.write_bytes(encoded(QUESTIONS, commit=COMMIT))
    labels = tmp_path / "labels.json"
    natural = {
        **labelled("c1", "kestrel [corpus/server.txt:1]", supported=T),
        "question": "kestrel",
        "passages": ["[corpus/server.txt:1] kestrel"],
    }
    perturbed = labelled("c2", "a dog", supported=F, origin="perturbed")
    labels.write_bytes(calibration(natural, perturbed))
    out = tmp_path / "verdicts.json"
    model = Local(
        said(SUPPORTED),
        said(Verdict(supported=False, unsupported=["a cat"])),
        said(SUPPORTED),
        said(SUPPORTED),
    )
    service = Service(model)
    argv = ["--answers", str(answers), "--verdicts", str(out)]
    argv += ["--set", str(question_set), "--calibration", str(labels)]

    assert await main(argv, service=lambda _: service) == 0
    first = capsys.readouterr().out
    assert await main([*argv, "--resume"], service=lambda _: service) == 0

    assert capsys.readouterr().out == first
    assert service.events == ["start", "stop"]
    assert len(model.requests) == 4
    assert NO_PASSAGES in model.requests[2].messages[-1].text
    printed = [json.loads(line) for line in first.splitlines()]
    assert [
        (r["category"], r["supported"], r["cited_recall"]) for r in printed[:4]
    ] == [
        ("named", 1.0, 1.0),
        ("paraphrase", 0.0, 0.0),
        ("none", 1.0, None),
        ("overall", 0.667, 0.5),
    ]
    assert printed[3] == {
        "category": "overall",
        "n": 3,
        "searched": 0.667,
        "shown_recall": 0.5,
        "cited_recall": 0.5,
        "supported": 0.667,
        "judge_errors": 0,
        "chat_errors": 0,
        "s_median": 4.0,
        "s_p95": 6.0,
        "judge_agreement": 0.5,
        "judge_kappa": 0.0,
    }
    assert [
        (c["calibration"], c["n"], c["agreement"], c["kappa"]) for c in printed[4:]
    ] == [
        ("overall", 2, 0.5, 0.0),
        ("natural", 1, 1.0, None),
        ("perturbed", 1, 0.0, 0.0),
    ]


async def test_main_never_sends_an_answer_whose_turn_ended_in_an_error_to_the_judge(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    answers = tmp_path / "answers.json"
    answers_file(answers, paraphrase_error=CUT_OFF)
    question_set = tmp_path / "set.json"
    question_set.write_bytes(encoded(QUESTIONS, commit=COMMIT))
    model = Local(said(SUPPORTED), said(SUPPORTED))
    argv = ["--answers", str(answers), "--verdicts", str(tmp_path / "v.json")]

    assert (
        await main(
            [*argv, "--set", str(question_set)], service=lambda _: Service(model)
        )
        == 0
    )

    assert [r.messages[-1].text.rsplit("Answer:\n", 1)[1] for r in model.requests] == [
        "kestrel [corpus/server.txt:1]",
        "nothing",
    ]
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    paraphrase, overall = printed[1], printed[3]
    assert (
        paraphrase["supported"],
        paraphrase["judge_errors"],
        paraphrase["chat_errors"],
    ) == (None, 0, 1)
    assert (overall["supported"], overall["chat_errors"]) == (1.0, 1)


async def test_main_resume_judges_again_what_the_judge_gave_no_verdict_for(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    answers = tmp_path / "answers.json"
    answers_file(answers)
    question_set = tmp_path / "set.json"
    question_set.write_bytes(encoded(QUESTIONS, commit=COMMIT))
    first = Local(said(SUPPORTED), "no", "still no", "never", said(SUPPORTED))
    again = Local(said(UNSUPPORTED))
    argv = ["--answers", str(answers), "--verdicts", str(tmp_path / "v.json")]
    argv += ["--set", str(question_set)]

    assert await main(argv, service=lambda _: Service(first)) == 0
    before = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert await main([*argv, "--resume"], service=lambda _: Service(again)) == 0
    after = [json.loads(line) for line in capsys.readouterr().out.splitlines()]

    assert (before[1]["judge_errors"], before[1]["supported"]) == (1, None)
    assert (after[1]["judge_errors"], after[1]["supported"]) == (0, 0.0)
    (request,) = again.requests
    assert request.messages[-1].text.endswith("Answer:\na cat")


async def test_main_reports_a_missing_calibration_set_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    answers = tmp_path / "answers.json"
    answers_file(answers)
    question_set = tmp_path / "set.json"
    question_set.write_bytes(encoded(QUESTIONS, commit=COMMIT))
    missing = tmp_path / "labels.json"
    argv = ["--answers", str(answers), "--verdicts", str(tmp_path / "v.json")]
    argv += ["--set", str(question_set), "--calibration", str(missing)]

    with pytest.raises(SystemExit) as stopped:
        await main(argv, service=lambda _: Service(Local()))

    assert stopped.value.code == 2
    error = capsys.readouterr().err
    assert f"{missing} does not exist: the labelled set is built by hand" in error
    assert "from an answers run" in error


async def test_main_without_calibration_prints_only_the_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    answers = tmp_path / "answers.json"
    answers_file(answers)
    question_set = tmp_path / "set.json"
    question_set.write_bytes(encoded(QUESTIONS, commit=COMMIT))
    service = Service(Local(said(SUPPORTED), said(SUPPORTED), said(SUPPORTED)))
    argv = ["--answers", str(answers), "--verdicts", str(tmp_path / "v.json")]

    assert (
        await main([*argv, "--set", str(question_set)], service=lambda _: service) == 0
    )

    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [r["category"] for r in printed] == [
        "named",
        "paraphrase",
        "none",
        "overall",
    ]
    assert "judge_kappa" not in printed[0]


async def test_main_refuses_to_overwrite_without_resume(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "verdicts.json"
    written_verdicts(out, INFO.id, PROMPT, {})

    with pytest.raises(SystemExit) as stopped:
        await main(["--answers", str(tmp_path / "a.json"), "--verdicts", str(out)])

    assert stopped.value.code == 2
    assert "pass --resume to keep it" in capsys.readouterr().err


def no_service(_: int | None) -> Service:
    pytest.fail("--no-judge opened the judge's service")


async def test_main_without_a_judge_scores_from_the_answers_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    answers = tmp_path / "answers.json"
    answers_file(answers, paraphrase_error=CUT_OFF)
    question_set = tmp_path / "set.json"
    question_set.write_bytes(encoded(QUESTIONS, commit=COMMIT))
    argv = ["--answers", str(answers), "--no-judge", "--set", str(question_set)]

    assert await main(argv, service=no_service) == 0

    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [(r["category"], r["supported"], r["judge_errors"]) for r in printed] == [
        ("named", None, None),
        ("paraphrase", None, None),
        ("none", None, None),
        ("overall", None, None),
    ]
    assert printed[3] == {
        "category": "overall",
        "n": 3,
        "searched": 0.667,
        "shown_recall": 1.0,
        "cited_recall": 1.0,
        "supported": None,
        "judge_errors": None,
        "chat_errors": 1,
        "s_median": 4.0,
        "s_p95": 6.0,
    }
    assert sorted(p.name for p in tmp_path.iterdir()) == ["answers.json", "set.json"]


@pytest.mark.parametrize(
    ("extra", "named"),
    [
        (["--calibration"], "--calibration"),
        (["--resume"], "--resume"),
        (["--threads", "0"], "--threads"),
        (["--calibration", "x.json", "--resume"], "--calibration, --resume"),
    ],
    ids=["calibration", "resume", "threads", "two"],
)
async def test_main_without_a_judge_refuses_what_only_the_judge_uses(
    extra: list[str], named: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as stopped:
        await main(["--answers", "a.json", "--no-judge", *extra], service=no_service)

    assert stopped.value.code == 2
    assert f"--no-judge starts no judge, so {named} has no use" in (
        capsys.readouterr().err
    )


@pytest.mark.parametrize(
    "judge",
    [[], ["--verdicts", "v.json", "--no-judge"]],
    ids=["neither", "both"],
)
async def test_main_takes_exactly_one_of_verdicts_and_no_judge(
    judge: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as stopped:
        await main(["--answers", "a.json", *judge], service=no_service)

    assert stopped.value.code == 2
    assert "--verdicts" in capsys.readouterr().err
