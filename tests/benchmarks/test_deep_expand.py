import io
import json
from pathlib import Path

import pytest
from deep_expand import Expanded, expand_all, held, main, written
from rag_eval import SetError, parse_expansions, parse_set

from tests.benchmarks.test_rag_eval import QUESTIONS, encoded
from tests.knowledge.test_deep import INFO, Local, reply


async def test_expand_all_writes_every_question_after_each_one(tmp_path: Path) -> None:
    out = tmp_path / "expansions.json"
    questions = parse_set(encoded(QUESTIONS)).questions
    model = Local(
        reply(["database server"], "The host is kestrel."),
        reply([], ""),
        reply(["bird", "BIRD"], "Un café, s'il vous plaît."),
    )
    progress = io.StringIO()
    now = iter([0.0, 1.25, 5.0, 7.5, 10.0, 13.0])

    await expand_all(
        model, questions, out, {}, progress=progress, clock=lambda: next(now)
    )

    raw = out.read_bytes()
    assert b"\r" not in raw
    assert raw.isascii()
    read = parse_expansions(raw)
    assert read.model == INFO.id
    assert read.seconds == {"named-01": 1.25, "paraphrase-01": 2.5, "none-01": 3.0}
    assert read.by_question["none-01"].queries == ("bird",)
    assert read.by_question["paraphrase-01"].passage is None
    assert progress.getvalue().splitlines() == [
        "1/3 named-01 1.2 s, 1 queries",
        "2/3 paraphrase-01 2.5 s, 0 queries",
        "3/3 none-01 3.0 s, 1 queries",
    ]
    assert not (tmp_path / "expansions.json.partial").exists()


async def test_expand_all_skips_what_is_held_and_held_reads_it_back(
    tmp_path: Path,
) -> None:
    out = tmp_path / "expansions.json"
    questions = parse_set(encoded(QUESTIONS)).questions
    first: dict[str, Expanded] = {
        "named-01": {
            "question": "kestrel",
            "queries": ["database server"],
            "passage": None,
            "seconds": 2.0,
        }
    }
    written(out, INFO.id, first)
    model = Local(reply(["cat"]), reply(["bird"]))
    now = iter([0.0, 1.0, 1.0, 3.0])

    expanded = held(out, INFO.id)
    await expand_all(
        model, questions, out, expanded, progress=io.StringIO(), clock=lambda: next(now)
    )

    assert [r.messages[-1].text for r in model.requests] == ["pet", "osprey"]
    assert (
        json.loads(out.read_text("utf-8"))["questions"]["named-01"] == first["named-01"]
    )
    assert list(held(out, INFO.id)) == ["named-01", "paraphrase-01", "none-01"]


async def test_expand_all_expands_again_a_question_whose_text_changed(
    tmp_path: Path,
) -> None:
    out = tmp_path / "new" / "expansions.json"
    questions = parse_set(encoded(QUESTIONS)).questions
    stale: dict[str, Expanded] = {
        "named-01": {
            "question": "falcon",
            "queries": ["old"],
            "passage": None,
            "seconds": 2.0,
        }
    }
    model = Local(reply(["host"]), reply(["cat"]), reply(["bird"]))

    await expand_all(
        model, questions, out, stale, progress=io.StringIO(), clock=lambda: 0.0
    )

    assert [r.messages[-1].text for r in model.requests] == ["kestrel", "pet", "osprey"]
    again = json.loads(out.read_text("utf-8"))["questions"]["named-01"]
    assert (again["question"], again["queries"]) == ("kestrel", ["host"])


def test_held_refuses_another_models_expansions(tmp_path: Path) -> None:
    out = tmp_path / "expansions.json"
    written(out, "other", {})

    with pytest.raises(SetError, match=r"holds other's expansions, not local's"):
        held(out, INFO.id)


async def test_main_refuses_to_overwrite_without_resume(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "expansions.json"
    written(out, INFO.id, {})

    with pytest.raises(SystemExit) as stopped:
        await main(["--out", str(out)])

    assert stopped.value.code == 2
    assert "pass --resume to keep it" in capsys.readouterr().err
