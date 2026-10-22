"""Ask every question of the question set through the chat, and keep each answer.

    uv run python benchmarks/chat_answers.py --out PATH [--threads N] [--resume]

Each question of ``benchmarks/data/knowledge_set.json``, in order, is asked
once in a fresh :class:`synthia.server.session.ChatSession`, the object the
chat runs, with the chat's default persona and reasoning, no memory, and one
tool: ``search_knowledge`` over the files of this repository at the set's
commit (``git archive``), indexed with the installed default embedding model
and searched as the chat searches. The session talks to the gateway's
router with the installed local model behind it. No remote model exists in
this run: the settings are copied without the OpenRouter key, so the router
is built without one, and a chat starts with remote off. The router's usage
log goes to a temporary home, not the person's.

The file written is the input of ``benchmarks/answer_eval.py``::

    {
        "model": "<local model id>",
        "commit": "<the set's commit>",
        "setup": {
            "persona": "<the persona's name>",
            "system_prompt_sha256": "<SHA-256 of its system prompt, hex>",
            "embedder": "<the embedding model's id>",
            "chunking": "<the index's chunking recipe>",
            "tools_sha256": "<SHA-256 of the tools' specs, hex>",
        },
        "questions": {
            "<question id>": {
                "question": "<the text asked>",
                "category": "named",
                "answer": "<the text the person is shown>",
                "searches": [{"query": "...", "result": "...", "ok": true}],
                "sources": ["<citation>", ...],
                "seconds": 41.2,
                "prompt_tokens": 5120 or null,
                "completion_tokens": 180 or null,
                "route": "local",
                "model": "<the model that answered>",
                "error": null or "<why the turn gave no full answer>",
            }
        },
    }

``setup`` is what shaped the answers besides the model: the persona that
spoke and the SHA-256 of its system prompt's UTF-8 text, the index's
embedding model and chunking recipe (``Chunking.recipe``, as
``benchmarks/rag_eval.py`` names chunkings), and the SHA-256 of the tools'
specs (each one's name, description and parameters, here
``search_knowledge`` alone) as one JSON list with sorted keys.
``result`` is a search's text exactly as the model was shown it, and
``sources`` every citation the turn's searches showed, each once. A turn
that ends without its report, as one whose reply stream was cut off does,
is kept with its ``error`` set, the text and the searches it got to, no
sources, no tokens, empty ``route`` and ``model``, and the seconds by this
script's clock; the run goes on to the next question. A gateway error, such
as no local model or a dead server, would fail every question, so it stops
the run. The file is written again after every question, so a crash loses
one question, and ``--resume`` skips the questions it already holds with
the same text and no error (one whose text changed, or that ended in an
error, is asked again); a file from another model, commit or setup is
refused, naming what differs. A progress line per question goes to
stderr. A question costs one local generation or more, tens of seconds to
minutes on a CPU. It reaches no outside service and spends no budget.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import tempfile
import time
from contextlib import aclosing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, TypedDict

import httpx
from fact_extraction import TIMEOUT, installed, ready
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from rag_eval import (
    KNOWLEDGE_SET,
    REPOSITORY,
    Category,
    SetError,
    archived,
    check_gold,
    installed_embedder,
    read_set,
)

from synthia.agent.loop import ToolFinished
from synthia.agent.tools import Toolbox
from synthia.gateway.assemble import build_gateway
from synthia.gateway.types import ChatChunk
from synthia.kernel.config import load_settings
from synthia.knowledge.chunk import Chunking
from synthia.knowledge.service import Knowledge
from synthia.knowledge.tool import SEARCH_KNOWLEDGE, knowledge_tool
from synthia.persona.library import PersonaLibrary
from synthia.server.host import CURRENT_ROUTES, PERSONAS_DIR, publish_route
from synthia.server.session import ChatSession, LastRoute, TurnReport

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from typing import TextIO

    from rag_eval import Question, QuestionSet

    from synthia.gateway.assemble import Gateway
    from synthia.gateway.protocol import ChatModel, LocalChatModel
    from synthia.kernel.config import Settings
    from synthia.retrieval.embed import TextEmbedder
    from synthia.server.daemon import LocalModels

DESCRIPTION: Final = (
    "Ask every question of the set through the chat, on the local model."
)
DIGITS: Final = 3
MAX_ANSWERS_BYTES: Final = 50_000_000
"""A search shows at most six passages of 2,400 characters; a few hundred
questions with a few searches each, escaped, stay well below this."""
KNOWLEDGE_DB: Final = "knowledge.db"
SCRATCH_HOME: Final = "home"
CUT_OFF: Final = "the turn ended without its report: the reply stopped before its end"
_SHA256_HEX: Final = r"^[0-9a-f]{64}$"

type ServiceOpener = Callable[[int | None], LocalModels]
"""Returns the local model's service, on the given number of threads."""


class Search(TypedDict):
    """One ``search_knowledge`` call: what was asked, what was shown, and if it ran."""

    query: str
    result: str
    ok: bool


class Answered(TypedDict):
    """One question asked through the chat, as the answers file holds it.

    ``error`` is None for a turn that finished, else why it did not.
    """

    question: str
    category: Category
    answer: str
    searches: list[Search]
    sources: list[str]
    seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    route: str
    model: str
    error: str | None


class _SearchIn(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    query: str
    result: str
    ok: bool


class _AnsweredIn(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    question: str = Field(min_length=1)
    category: Category
    answer: str
    searches: list[_SearchIn]
    sources: list[str]
    seconds: float = Field(ge=0, allow_inf_nan=False)
    prompt_tokens: int | None = Field(ge=0)
    completion_tokens: int | None = Field(ge=0)
    route: str
    model: str
    error: Annotated[str, Field(min_length=1)] | None


class _SetupIn(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    persona: str = Field(min_length=1)
    system_prompt_sha256: str = Field(pattern=_SHA256_HEX)
    embedder: str = Field(min_length=1)
    chunking: str = Field(min_length=1)
    tools_sha256: str = Field(pattern=_SHA256_HEX)


class _AnswersIn(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    model: str = Field(min_length=1)
    commit: str = Field(min_length=1)
    setup: _SetupIn
    questions: dict[str, _AnsweredIn]


class _QueryIn(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str


@dataclass(frozen=True, slots=True)
class Setup:
    """What shaped the chat's answers besides the model, as :func:`setup_of` reads it.

    ``system_prompt_sha256`` is the SHA-256 of the persona's system prompt
    as UTF-8; ``tools_sha256`` of every tool's name, description and
    parameters as one JSON list with sorted keys.
    """

    persona: str
    system_prompt_sha256: str
    embedder: str
    chunking: str
    tools_sha256: str


@dataclass(frozen=True, slots=True)
class Provenance:
    """What made a file of answers: the local model, the set's commit, and the setup."""

    model: str
    commit: str
    setup: Setup


@dataclass(frozen=True, slots=True)
class Answers:
    """The local model's answer to each question by id, and what made them."""

    provenance: Provenance
    by_question: Mapping[str, Answered]


def _answered(given: _AnsweredIn) -> Answered:
    return {
        "question": given.question,
        "category": given.category,
        "answer": given.answer,
        "searches": [
            {"query": s.query, "result": s.result, "ok": s.ok} for s in given.searches
        ],
        "sources": list(given.sources),
        "seconds": given.seconds,
        "prompt_tokens": given.prompt_tokens,
        "completion_tokens": given.completion_tokens,
        "route": given.route,
        "model": given.model,
        "error": given.error,
    }


def parse_answers(data: bytes) -> Answers:
    """Return the answers ``data`` holds, as this script writes them.

    Raises:
        SetError: If ``data`` is larger than :data:`MAX_ANSWERS_BYTES` or is
            not a model, a commit, a setup and answers by question id.
    """
    if len(data) > MAX_ANSWERS_BYTES:
        message = (
            f"the answers are {len(data)} bytes, over MAX_ANSWERS_BYTES="
            f"{MAX_ANSWERS_BYTES}"
        )
        raise SetError(message)
    try:
        given = _AnswersIn.model_validate_json(data)
    except ValidationError as error:
        message = (
            f"the answers are not a model, a commit, a setup and answers by id: {error}"
        )
        raise SetError(message) from error
    s = given.setup
    setup = Setup(
        s.persona, s.system_prompt_sha256, s.embedder, s.chunking, s.tools_sha256
    )
    by_question = {q: _answered(a) for q, a in given.questions.items()}
    return Answers(Provenance(given.model, given.commit, setup), by_question)


def read_answers(path: Path, commit: str, questions: Iterable[Question]) -> Answers:
    """Return the answers in the file ``path``, one for each of ``questions``.

    Blocking.

    Its setup is read but not compared with anything: the answers are
    scored as they were made.

    Raises:
        SetError: As :func:`parse_answers` does, or if the file is about
            another commit than ``commit``, misses a question, or answered
            another text of one.
    """
    with path.open("rb") as file:
        answers = parse_answers(file.read(MAX_ANSWERS_BYTES + 1))
    held_commit = answers.provenance.commit
    if held_commit != commit:
        message = f"{path.name} answers about commit {held_commit}, not {commit}"
        raise SetError(message)
    asked = list(questions)
    missing = [q.id for q in asked if q.id not in answers.by_question]
    if missing:
        message = f"{path.name} does not answer the questions {missing}"
        raise SetError(message)
    changed = [q.id for q in asked if answers.by_question[q.id]["question"] != q.text]
    if changed:
        message = f"{path.name} answered another text of the questions {changed}"
        raise SetError(message)
    return answers


def held(path: Path, provenance: Provenance) -> dict[str, Answered]:
    """Return the answers already in ``path`` by question id; blocking.

    Raises:
        SetError: If the file is not answers, or was not made as
            ``provenance`` says: another model, commit or setup, named.
    """
    answers = parse_answers(path.read_bytes())
    found, model, commit = answers.provenance, provenance.model, provenance.commit
    if (found.model, found.commit) != (model, commit):
        message = (
            f"{path.name} holds {found.model}'s answers about {found.commit},"
            f" not {model}'s about {commit}"
        )
        raise SetError(message)
    had, wanted = asdict(found.setup), asdict(provenance.setup)
    differs = [f"{k} {had[k]!r}, not {v!r}" for k, v in wanted.items() if had[k] != v]
    if differs:
        message = f"{path.name} holds answers of another setup: {'; '.join(differs)}"
        raise SetError(message)
    return dict(answers.by_question)


def written(path: Path, provenance: Provenance, answered: dict[str, Answered]) -> None:
    """Replace ``path`` with the answers, so a crash never leaves half a file."""
    text = json.dumps(
        {
            "model": provenance.model,
            "commit": provenance.commit,
            "setup": asdict(provenance.setup),
            "questions": answered,
        },
        ensure_ascii=True,
        allow_nan=False,
        indent=2,
    )
    partial = path.with_name(f"{path.name}.partial")
    partial.write_text(f"{text}\n", encoding="utf-8", newline="")
    partial.replace(path)


@dataclass(frozen=True, slots=True)
class Chat:
    """What each question's fresh session is made of."""

    model: ChatModel
    library: PersonaLibrary
    persona: str
    tools: Toolbox
    clock: Callable[[], float] = time.perf_counter


def no_remote_gateway(
    settings: Settings, home: Path, client: httpx.AsyncClient, local: LocalChatModel
) -> Gateway:
    """Return the chat's gateway, with ``local`` and no remote model behind its router.

    The settings are copied without the OpenRouter key, which is the only
    thing that builds a remote model, and with ``home`` as the home, where
    the router's usage log is written.
    """
    kept = settings.model_copy(update={"openrouter_api_key": None, "home": home})
    return build_gateway(kept, client, publish_route, local)


def chat_over(gateway: Gateway, knowledge: Knowledge, settings: Settings) -> Chat:
    """Return a chat whose one tool searches ``knowledge``, wired as the daemon does.

    It speaks as the settings' persona, from the personas in their home.
    """
    router = gateway.router

    def can_stay_local() -> bool:
        return router.can_stay_local

    tool = knowledge_tool(knowledge, can_stay_local=can_stay_local, model=gateway.model)
    library = PersonaLibrary(settings.home / PERSONAS_DIR)
    return Chat(gateway.model, library, settings.persona, Toolbox([tool]))


def setup_of(chat: Chat, embedder: str, chunking: Chunking) -> Setup:
    """Return ``chat``'s setup over an index by the ``embedder`` id and ``chunking``.

    Raises:
        PersonaError: If the chat's persona is not in its library.
    """
    persona = chat.library.get(chat.persona)
    prompt = hashlib.sha256(persona.system_prompt().encode()).hexdigest()
    specs = [
        {"name": s.name, "description": s.description, "parameters": s.parameters}
        for s in chat.tools.specs()
    ]
    encoded = json.dumps(specs, sort_keys=True, ensure_ascii=True).encode("ascii")
    tools = hashlib.sha256(encoded).hexdigest()
    return Setup(persona.name, prompt, embedder, chunking.recipe(), tools)


def _query(arguments: str) -> str:
    """Return the query a call asked for, or its arguments if they are not one."""
    try:
        return _QueryIn.model_validate_json(arguments).query
    except ValidationError:
        return arguments


async def _turn(
    chat: Chat, question: Question
) -> tuple[str, list[Search], TurnReport | None]:
    """Return the text, the searches and the report of one turn in a fresh session."""
    routes = LastRoute()
    session = ChatSession(
        chat.model, chat.library, chat.persona, routes, chat.clock, tools=chat.tools
    )
    texts: list[str] = []
    searches: list[Search] = []
    report: TurnReport | None = None
    routed = CURRENT_ROUTES.set(routes)
    try:
        async with aclosing(session.turn(question.text)) as items:
            async for item in items:
                match item:
                    case ChatChunk(text=text):
                        texts.append(text)
                    case ToolFinished(call=call, result=result, ok=ok) if (
                        call.name == SEARCH_KNOWLEDGE
                    ):
                        searches.append(
                            {
                                "query": _query(call.arguments),
                                "result": result,
                                "ok": ok,
                            }
                        )
                    case TurnReport():
                        report = item
                    case _:
                        pass
    finally:
        CURRENT_ROUTES.reset(routed)
        await session.close()
    return "".join(texts), searches, report


async def answer(chat: Chat, question: Question) -> Answered:
    """Ask ``question`` in a fresh session of ``chat`` and return what it did.

    A turn that ends without its report, as one whose reply stream was cut
    off does, is returned with :data:`CUT_OFF` as its error, the text and the
    searches it got to, and the seconds by ``chat.clock``.

    Raises:
        GatewayError: If the answer failed.
    """
    started = chat.clock()
    text, searches, report = await _turn(chat, question)
    if report is None:
        return {
            "question": question.text,
            "category": question.category,
            "answer": text,
            "searches": searches,
            "sources": [],
            "seconds": round(chat.clock() - started, DIGITS),
            "prompt_tokens": None,
            "completion_tokens": None,
            "route": "",
            "model": "",
            "error": CUT_OFF,
        }
    return {
        "question": question.text,
        "category": question.category,
        "answer": text,
        "searches": searches,
        "sources": list(report.sources),
        "seconds": round(report.seconds, DIGITS),
        "prompt_tokens": report.prompt_tokens,
        "completion_tokens": report.completion_tokens,
        "route": report.route,
        "model": report.model,
        "error": None,
    }


async def answer_all(  # noqa: PLR0913 - the run's whole configuration
    chat: Chat,
    questions: Sequence[Question],
    out: Path,
    provenance: Provenance,
    answered: dict[str, Answered],
    *,
    progress: TextIO = sys.stderr,
) -> None:
    """Ask each question not answered yet, writing ``out`` after each.

    A held answer is kept when it has the question's text and no error.
    ``provenance`` is written with the answers.

    Raises:
        GatewayError: If an answer failed; the questions before it are kept.
    """
    await asyncio.to_thread(out.parent.mkdir, parents=True, exist_ok=True)
    for n, question in enumerate(questions, 1):
        kept = answered.get(question.id)
        if (
            kept is not None
            and kept["question"] == question.text
            and kept["error"] is None
        ):
            continue
        done = await answer(chat, question)
        answered[question.id] = done
        await asyncio.to_thread(written, out, provenance, answered)
        ended = "" if done["error"] is None else f", error: {done['error']}"
        progress.write(
            f"{n}/{len(questions)} {question.id} {done['seconds']:.1f} s,"
            f" {len(done['searches'])} searches{ended}\n"
        )
        progress.flush()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--set", type=Path, default=KNOWLEDGE_SET)
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep the questions the out file already answers",
    )
    return parser


@dataclass(frozen=True, slots=True)
class Index:
    """The set's commit, indexed: the index, its embedding model's id, its chunking."""

    knowledge: Knowledge
    embedder: str
    chunking: Chunking


async def indexed(
    repository: Path, question_set: QuestionSet, embedder: TextEmbedder, folder: Path
) -> Index:
    """Return an index, in ``folder``, of ``repository`` at the set's commit.

    Indexed as the chat indexes, with the default chunking for ``embedder``.

    Raises:
        SetError: If a gold span names a file the commit does not hold.
    """
    corpus = await asyncio.to_thread(archived, repository, question_set.commit, folder)
    await asyncio.to_thread(check_gold, corpus, question_set.questions)
    chunking = Chunking.for_embedder(embedder.spec)
    knowledge = await Knowledge.open(folder / KNOWLEDGE_DB, embedder, chunking=chunking)
    await knowledge.add(corpus)
    await knowledge.index()
    return Index(knowledge, embedder.spec.id, chunking)


async def _asked(  # noqa: PLR0913 - the run's whole configuration
    local_service: LocalModels,
    settings: Settings,
    index: Index,
    question_set: QuestionSet,
    *,
    out: Path,
    home: Path,
    resume: bool,
) -> None:
    """Start the local model and ask every question not held in ``out``.

    ``home`` takes the router's usage log.
    """
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        local = local_service.model(client)
        gateway = no_remote_gateway(settings, home, client, local)
        chat = chat_over(gateway, index.knowledge, settings)
        setup = setup_of(chat, index.embedder, index.chunking)
        provenance = Provenance(local.info.id, question_set.commit, setup)
        answered = await asyncio.to_thread(held, out, provenance) if resume else {}
        local_service.start()
        try:
            await ready(local)
            await answer_all(chat, question_set.questions, out, provenance, answered)
        finally:
            local_service.stop()


async def main(
    argv: Sequence[str] | None = None,
    *,
    service: ServiceOpener = installed,
    load: Callable[[], TextEmbedder] = installed_embedder,
    settings: Callable[[], Settings] = load_settings,
) -> int:
    """Write the chat's answer to every question, one question at a time.

    ``service`` opens the local model's service, ``load`` the embedder that
    indexes the corpus, and ``settings`` reads the chat's settings.
    """
    parser = _parser()
    arguments = parser.parse_args(argv)
    out: Path = arguments.out
    exists = await asyncio.to_thread(out.exists)
    if exists and not arguments.resume:
        parser.error(f"{out} exists: pass --resume to keep it, or remove it")
    question_set = await asyncio.to_thread(read_set, arguments.set)
    configured = settings()
    local_service = service(arguments.threads)
    embedder = await asyncio.to_thread(load)
    with tempfile.TemporaryDirectory() as scratch:
        folder = Path(scratch)
        index = await indexed(arguments.repository, question_set, embedder, folder)
        await _asked(
            local_service,
            configured,
            index,
            question_set,
            out=out,
            home=folder / SCRATCH_HOME,
            resume=exists,
        )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
