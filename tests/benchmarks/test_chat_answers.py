import hashlib
import io
import json
import re
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath

import httpx
import pytest
from chat_answers import (
    CUT_OFF,
    MAX_ANSWERS_BYTES,
    Answered,
    Chat,
    Provenance,
    Setup,
    answer,
    answer_all,
    chat_over,
    held,
    main,
    no_remote_gateway,
    parse_answers,
    read_answers,
    setup_of,
    written,
)
from pydantic import SecretStr
from rag_eval import Question, SetError, parse_set

from synthia.agent.tools import Toolbox
from synthia.gateway.assemble import GATEWAY_DB, build_gateway
from synthia.gateway.errors import BadRequestError
from synthia.gateway.protocol import LocalChatModel
from synthia.gateway.types import ChatChunk, ChatRequest, FinishReason, Usage
from synthia.kernel.config import Settings
from synthia.kernel.supervisor import ServiceEvent
from synthia.knowledge.chunk import Chunking
from synthia.knowledge.evaluate import Gold
from synthia.knowledge.tool import knowledge_tool
from synthia.persona.library import PersonaLibrary
from synthia.server.host import PERSONAS_DIR, publish_route
from tests.agent.scripted import Scripted, calls, says
from tests.benchmarks.test_rag_eval import QUESTIONS, committed, encoded, toy
from tests.knowledge.corpus import NOTES, planted
from tests.knowledge.test_tool import KESTREL, PUNE, knowing, stays_local
from tests.server.test_daemon import Local

COMMIT = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret
SETUP = Setup("SYNTHIA", "a" * 64, "toy", "lines/256/40/r1", "b" * 64)
PROVENANCE = Provenance("local", COMMIT, SETUP)
NAMED = Question(
    "named-01",
    "named",
    "what is kestrel?",
    (Gold(PurePosixPath("kestrel.md"), 5, 5),),
)
PARAPHRASE = Question("paraphrase-01", "paraphrase", "where does my sister live?", ())
SEARCH = '{"query": "kestrel"}'


def counted(chunks: list[ChatChunk], prompt: int, completion: int) -> list[ChatChunk]:
    return [*chunks[:-1], ChatChunk(usage=Usage(prompt, completion)), chunks[-1]]


def ticks(*times: float) -> Callable[[], float]:
    left = iter(times)
    return lambda: next(left)


async def chat(tmp_path: Path, model: Scripted, clock: Callable[[], float]) -> Chat:
    knowledge = await knowing(tmp_path, private=False)
    tool = knowledge_tool(knowledge, can_stay_local=stays_local, model=Scripted())
    return Chat(model, PersonaLibrary(), "synthia", Toolbox([tool]), clock)


async def test_answer_all_records_the_answer_searches_and_sources(
    tmp_path: Path,
) -> None:
    out = tmp_path / "answers.json"
    model = Scripted(
        counted(calls(("search_knowledge", SEARCH)), 100, 5),
        counted(
            [
                ChatChunk(text="Kestrel is the database host "),
                ChatChunk(text=f"[{KESTREL}]."),
                ChatChunk(finish_reason=FinishReason.STOP),
            ],
            900,
            20,
        ),
        says("She lives in Pune."),
    )
    progress = io.StringIO()
    run = await chat(tmp_path, model, ticks(99.0, 0.0, 1.5, 99.0, 10.0, 12.25))

    await answer_all(run, (NAMED, PARAPHRASE), out, PROVENANCE, {}, progress=progress)

    raw = out.read_bytes()
    assert b"\r" not in raw
    assert raw.isascii()
    answers = parse_answers(raw)
    assert answers.provenance == PROVENANCE
    named = answers.by_question["named-01"]
    (search,) = named["searches"]
    assert (search["query"], search["ok"]) == ("kestrel", True)
    assert search["result"].startswith(f"[{KESTREL}] Kestrel > Hosts\n")
    assert f"[{PUNE}]" in search["result"]
    assert named == {
        "question": "what is kestrel?",
        "category": "named",
        "answer": f"Kestrel is the database host [{KESTREL}].",
        "searches": [search],
        "sources": [KESTREL, PUNE],
        "seconds": 1.5,
        "prompt_tokens": 1000,
        "completion_tokens": 25,
        "route": "direct",
        "model": "scripted",
        "error": None,
    }
    paraphrase = answers.by_question["paraphrase-01"]
    assert (paraphrase["searches"], paraphrase["sources"]) == ([], [])
    assert (paraphrase["prompt_tokens"], paraphrase["seconds"]) == (None, 2.25)
    assert progress.getvalue().splitlines() == [
        "1/2 named-01 1.5 s, 1 searches",
        "2/2 paraphrase-01 2.2 s, 0 searches",
    ]
    assert not (tmp_path / "answers.json.partial").exists()


async def test_each_question_gets_a_fresh_session(tmp_path: Path) -> None:
    model = Scripted(says("first"), says("second"))
    run = await chat(tmp_path, model, ticks(0.0, 0.0, 1.0, 2.0, 2.0, 3.0))

    await answer(run, NAMED)
    await answer(run, PARAPHRASE)

    assert [m.text for m in model.requests[1].messages[1:]] == [PARAPHRASE.text]


async def test_answer_all_skips_what_is_held_and_asks_a_changed_text_again(
    tmp_path: Path,
) -> None:
    out = tmp_path / "new" / "answers.json"
    model = Scripted(says("She lives in Pune."))
    run = await chat(tmp_path, model, ticks(0.0, 0.0, 4.0))
    kept: Answered = {
        "question": NAMED.text,
        "category": "named",
        "answer": "kestrel",
        "searches": [],
        "sources": [],
        "seconds": 3.0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "route": "local",
        "model": "local",
        "error": None,
    }
    stale: Answered = {**kept, "question": "where did my sister live?"}
    written(tmp_path / "held.json", PROVENANCE, {"named-01": kept})
    answered = held(tmp_path / "held.json", PROVENANCE)
    answered["paraphrase-01"] = stale

    await answer_all(
        run,
        (NAMED, PARAPHRASE),
        out,
        PROVENANCE,
        answered,
        progress=io.StringIO(),
    )

    assert [r.messages[-1].text for r in model.requests] == [PARAPHRASE.text]
    again = parse_answers(out.read_bytes()).by_question
    assert again["named-01"] == kept
    assert again["paraphrase-01"]["question"] == PARAPHRASE.text


async def test_a_failed_search_keeps_its_arguments_and_other_tools_are_not_searches(
    tmp_path: Path,
) -> None:
    model = Scripted(
        calls(("search_knowledge", '{"q": 1}'), ("delete_files", "{}")),
        says("I could not search."),
    )
    run = await chat(tmp_path, model, ticks(0.0, 0.0, 1.0))

    done = await answer(run, NAMED)

    (search,) = done["searches"]
    assert (search["query"], search["ok"]) == ('{"q": 1}', False)
    assert done["sources"] == []


async def test_a_cut_off_turn_is_kept_with_its_error_and_what_it_got_to(
    tmp_path: Path,
) -> None:
    model = Scripted(calls(("search_knowledge", SEARCH)), [ChatChunk(text="half")])
    run = await chat(tmp_path, model, ticks(5.0, 6.0, 8.5))

    done = await answer(run, NAMED)

    (search,) = done["searches"]
    assert (search["query"], search["ok"]) == ("kestrel", True)
    assert done == {
        "question": NAMED.text,
        "category": "named",
        "answer": "half",
        "searches": [search],
        "sources": [],
        "seconds": 3.5,
        "prompt_tokens": None,
        "completion_tokens": None,
        "route": "",
        "model": "",
        "error": CUT_OFF,
    }


async def test_answer_all_goes_on_after_a_cut_off_turn_and_resume_asks_it_again(
    tmp_path: Path,
) -> None:
    out = tmp_path / "answers.json"
    model = Scripted(
        [ChatChunk(text="half")], says("She lives in Pune."), says("Kestrel.")
    )
    run = await chat(
        tmp_path, model, ticks(0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0)
    )
    progress = io.StringIO()

    await answer_all(run, (NAMED, PARAPHRASE), out, PROVENANCE, {}, progress=progress)
    first = parse_answers(out.read_bytes()).by_question
    await answer_all(
        run,
        (NAMED, PARAPHRASE),
        out,
        PROVENANCE,
        held(out, PROVENANCE),
        progress=progress,
    )

    assert (first["named-01"]["answer"], first["named-01"]["error"]) == (
        "half",
        CUT_OFF,
    )
    assert first["paraphrase-01"]["error"] is None
    assert [r.messages[-1].text for r in model.requests] == [
        NAMED.text,
        PARAPHRASE.text,
        NAMED.text,
    ]
    again = parse_answers(out.read_bytes()).by_question
    assert (again["named-01"]["answer"], again["named-01"]["error"]) == (
        "Kestrel.",
        None,
    )
    assert again["paraphrase-01"] == first["paraphrase-01"]
    assert progress.getvalue().splitlines() == [
        f"1/2 named-01 2.0 s, 0 searches, error: {CUT_OFF}",
        "2/2 paraphrase-01 3.0 s, 0 searches",
        "1/2 named-01 4.0 s, 0 searches",
    ]


async def test_a_gateway_error_stops_the_run_and_keeps_the_answers_before_it(
    tmp_path: Path,
) -> None:
    out = tmp_path / "answers.json"

    def refused(request: ChatRequest) -> list[ChatChunk]:
        del request
        message = "no model is loaded"
        raise BadRequestError(message)

    model = Scripted(says("Kestrel."), refused)
    run = await chat(tmp_path, model, ticks(0.0, 0.0, 1.0, 0.0, 0.0))

    with pytest.raises(BadRequestError, match="no model is loaded"):
        await answer_all(
            run,
            (NAMED, PARAPHRASE),
            out,
            PROVENANCE,
            {},
            progress=io.StringIO(),
        )

    assert list(parse_answers(out.read_bytes()).by_question) == ["named-01"]


async def test_the_chat_has_no_remote_model_even_when_a_key_is_set(
    tmp_path: Path,
) -> None:
    key = SecretStr("sk-or-x")
    own = Settings(home=tmp_path / "own", openrouter_api_key=key)
    control = Settings(home=tmp_path / "control", openrouter_api_key=key)
    scratch = tmp_path / "scratch"
    model = Local(calls(("search_knowledge", SEARCH)), says(f"Kestrel [{KESTREL}]."))
    knowledge = await knowing(tmp_path, private=False)
    async with httpx.AsyncClient() as client:
        assert build_gateway(control, client, publish_route, model).router.remote
        gateway = no_remote_gateway(own, scratch, client, model)
        run = chat_over(gateway, knowledge, own)

        done = await answer(run, NAMED)

    assert gateway.router.remote is None
    assert gateway.health is None
    assert own.openrouter_api_key is not None
    assert all(r.use_remote is False for r in model.requests)
    assert (done["route"], done["sources"]) == ("local", [KESTREL, PUNE])
    assert (scratch / GATEWAY_DB).is_file()
    assert not (tmp_path / "own" / GATEWAY_DB).exists()


def answers_data(questions: dict[str, object], **setup: str) -> bytes:
    held_setup = {**asdict(SETUP), **setup}
    data = {"model": "m", "commit": "c", "setup": held_setup, "questions": questions}
    return json.dumps(data).encode()


@pytest.mark.parametrize(
    ("data", "named"),
    [
        (b"[]", "not a model, a commit, a setup and answers"),
        (answers_data({"q": {}}), r"questions\.q\.question"),
        (answers_data({}, tools_sha256="ab"), r"setup\.tools_sha256"),
        (b" " * (MAX_ANSWERS_BYTES + 1), "MAX_ANSWERS_BYTES"),
    ],
    ids=["not_an_object", "not_an_answer", "not_a_digest", "too_large"],
)
def test_parse_answers_refuses_what_is_not_answers(data: bytes, named: str) -> None:
    with pytest.raises(SetError, match=named):
        parse_answers(data)


@pytest.mark.parametrize(("model", "commit"), [("other", COMMIT), ("local", "fedcba9")])
def test_held_refuses_answers_by_another_model_or_about_another_commit(
    tmp_path: Path, model: str, commit: str
) -> None:
    out = tmp_path / "answers.json"
    written(out, Provenance(model, commit, SETUP), {})

    with pytest.raises(SetError, match=rf"holds {model}'s answers about {commit}"):
        held(out, PROVENANCE)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("persona", "Ada"),
        ("system_prompt_sha256", "c" * 64),
        ("embedder", "other-embedder"),
        ("chunking", "structure+header/256/40/r1"),
        ("tools_sha256", "d" * 64),
    ],
)
def test_held_refuses_answers_of_another_setup_naming_what_differs(
    tmp_path: Path, field: str, value: str
) -> None:
    out = tmp_path / "answers.json"
    written(out, replace(PROVENANCE, setup=replace(SETUP, **{field: value})), {})
    named = f"another setup: {field} {value!r}, not {getattr(SETUP, field)!r}"

    with pytest.raises(SetError, match=f"{re.escape(named)}$"):
        held(out, PROVENANCE)


def answers_of(texts: dict[str, str]) -> dict[str, Answered]:
    return {
        question: {
            "question": text,
            "category": "named",
            "answer": "a",
            "searches": [],
            "sources": [],
            "seconds": 1.0,
            "prompt_tokens": None,
            "completion_tokens": None,
            "route": "local",
            "model": "local",
            "error": None,
        }
        for question, text in texts.items()
    }


def test_parse_answers_refuses_an_empty_error() -> None:
    entry = {**answers_of({"named-01": "kestrel"})["named-01"], "error": ""}

    with pytest.raises(SetError, match=r"questions\.named-01\.error"):
        parse_answers(answers_data({"named-01": entry}))


@pytest.mark.parametrize(
    ("commit", "texts", "named"),
    [
        ("fedcba9", {"named-01": "kestrel"}, "about commit fedcba9, not"),
        (COMMIT, {}, r"does not answer the questions \['named-01'"),
        (COMMIT, {"named-01": "falcon"}, r"another text of the questions \['named-01'"),
    ],
    ids=["commit", "missing", "changed"],
)
def test_read_answers_refuses_answers_not_to_these_questions(
    tmp_path: Path, commit: str, texts: dict[str, str], named: str
) -> None:
    out = tmp_path / "answers.json"
    full = {"paraphrase-01": "pet", "none-01": "osprey", **texts}
    written(out, Provenance("local", commit, SETUP), answers_of(full))
    questions = parse_set(encoded(QUESTIONS)).questions

    with pytest.raises(SetError, match=named):
        read_answers(out, COMMIT, questions)


class Service:
    """The local model's service, with a scripted model that is always ready."""

    def __init__(self, model: LocalChatModel) -> None:
        self.local = model
        self.events: list[str] = []

    def model(self, client: httpx.AsyncClient) -> LocalChatModel:
        del client
        return self.local

    def state(self) -> str:
        return "ready"

    def start(self, observe: Callable[[ServiceEvent], None] | None = None) -> None:
        del observe
        self.events.append("start")

    def stop(self) -> None:
        self.events.append("stop")


async def setup_of_run(folder: Path, settings: Settings) -> Setup:
    """Return the setup the builder computes for a run on ``settings`` and toy()."""
    library = PersonaLibrary(settings.home / PERSONAS_DIR)
    knowledge = await knowing(folder, private=False)
    tool = knowledge_tool(knowledge, can_stay_local=stays_local, model=Scripted())
    chat_of_run = Chat(Scripted(), library, settings.persona, Toolbox([tool]))
    spec = toy().spec
    setup = setup_of(chat_of_run, spec.id, Chunking.for_embedder(spec))
    prompt = library.get(settings.persona).system_prompt().encode()
    assert setup.system_prompt_sha256 == hashlib.sha256(prompt).hexdigest()
    return setup


async def test_main_answers_every_question_of_the_commit_and_resumes(
    tmp_path: Path,
) -> None:
    repository = planted(tmp_path / "repository", NOTES)
    commit = committed(repository)
    question_set = tmp_path / "set.json"
    question_set.write_bytes(encoded(QUESTIONS, commit=commit))
    out = tmp_path / "answers.json"
    service = Service(
        Local(
            calls(("search_knowledge", SEARCH)),
            says("Kestrel is a host [corpus/server.txt:1]."),
            says("Miso."),
            says("I do not know."),
        )
    )

    def settings() -> Settings:
        return Settings(home=tmp_path / "own", openrouter_api_key=SecretStr("k"))

    argv = ["--out", str(out), "--set", str(question_set)]
    argv += ["--repository", str(repository)]
    assert await main(argv, service=lambda _: service, load=toy, settings=settings) == 0
    assert (
        await main(
            [*argv, "--resume", "--threads", "2"],
            service=lambda _: service,
            load=toy,
            settings=settings,
        )
        == 0
    )

    answers = read_answers(out, commit, parse_set(question_set.read_bytes()).questions)
    named = answers.by_question["named-01"]
    expected = await setup_of_run(tmp_path / "expected", settings())
    assert answers.provenance.setup == expected
    assert answers.provenance.model == "scripted"
    assert named["route"] == "local"
    assert "corpus/server.txt:1" in named["sources"]
    assert [a["answer"] for a in answers.by_question.values()] == [
        "Kestrel is a host [corpus/server.txt:1].",
        "Miso.",
        "I do not know.",
    ]
    assert service.events == ["start", "stop", "start", "stop"]
    assert not (tmp_path / "own").exists()


async def test_main_refuses_to_overwrite_without_resume(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "answers.json"
    written(out, PROVENANCE, {})

    with pytest.raises(SystemExit) as stopped:
        await main(["--out", str(out)])

    assert stopped.value.code == 2
    assert "pass --resume to keep it" in capsys.readouterr().err
