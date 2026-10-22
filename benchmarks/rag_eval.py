"""Score how well knowledge search finds the lines that answer a question.

    uv run python benchmarks/rag_eval.py [--chunking NAME]... [--variant NAME]...

The question set (``benchmarks/data/knowledge_set.json``) names a commit of
this repository and asks questions about the files at that commit, each with
the lines that answer it (gold spans) or none for a question nothing answers.
The files are taken with ``git archive <commit>`` and indexed as one source,
so a gold path, relative to the repository root, is a passage's path.

Each chunking builds its own cold index in a fresh home, embedded with the
installed default model: the default chunking (windows of 40 lines without a
header, at the embedder's token limit), then sections by structure without a
header and with one. Each variant then searches that index for every
question, 10 passages deep: words, meaning, and the fused score at the
default word weight 0.2 and at 0.05, 0.1 (the earlier default) and 0.4,
then the default following links from the best 5 passages: one hop at link
weight 0.1 and 0.3, two hops at 0.3. With the ms-marco-MiniLM-L6-v2
rerankers installed, also the default reranked: the first 10 and 20 by the
8-bit file, the first 10 by the full-precision one, and one hop at 0.3 then
the first 10 by the 8-bit file.

With ``--expansions PATH``, the local model's expansion of every question
(written by ``benchmarks/deep_expand.py``) is searched beside it too, as deep
search does, on the default chunking after its variants and with the
default variant: the rewrites alone, the drafted passage alone, and both,
each merged by best score and by reciprocal rank; then both reranked, the
first 10 by the 8-bit file, when it is installed. These rows carry a
``deep`` field naming what was searched and how it merged, as ``both/best``.
A question the file does not expand is an error.

Output is JSON lines: first a header (commit, questions per category, files
in the corpus, embedder, rerankers, python, numpy and onnxruntime versions,
and the expansions' model and seconds when given), then per chunking,
variant and category, and over all questions, the rank metrics of
:mod:`synthia.knowledge.evaluate` (null for a group with no answerable
question), the mean top score, ms per search (query embedding included) as
median and 95th percentile, the seconds the index took and its chunk count.
Needs the default embedding model (``synthia models install``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import platform
import shutil
import statistics
import subprocess
import tarfile
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final, Literal, NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from synthia.kernel.config import default_home
from synthia.knowledge.chunk import Chunking, Splitter
from synthia.knowledge.deep import Expansion
from synthia.knowledge.evaluate import Answerable, Gold, Report, scored
from synthia.knowledge.retrieve import DEFAULT_VARIANT, Merge, Variant
from synthia.knowledge.service import Knowledge
from synthia.models.catalogue import DEFAULT_EMBEDDER, RERANKERS
from synthia.models.install import BYTES_PER_GB, Installer
from synthia.retrieval.embed import TextEmbedder
from synthia.retrieval.rerank import CrossEncoder

if TYPE_CHECKING:
    from collections.abc import (
        AsyncIterator,
        Callable,
        Collection,
        Iterable,
        Mapping,
        Sequence,
    )

    from synthia.knowledge.evaluate import Means, Scored
    from synthia.knowledge.index import IndexReport
    from synthia.models.catalogue import Embedder

KNOWLEDGE_SET: Final = Path(__file__).parent / "data" / "knowledge_set.json"
REPOSITORY: Final = Path(__file__).resolve().parents[1]
MAX_SET_BYTES: Final = 1_000_000
"""A question set is a few hundred questions; anything larger is not one."""
SEARCH_LIMIT: Final = 10
GIT_TIMEOUT_S: Final = 120
PERCENTILE: Final = 0.95
DIGITS: Final = 3
OVERALL: Final = "overall"
VARIANTS: Final = tuple(
    Variant.parse(name)
    for name in (
        "words",
        "meaning",
        "fused",
        "fused:0.05",
        "fused:0.1",
        "fused:0.4",
        "fused+hop1x0.1",
        "fused+hop1x0.3",
        "fused+hop2x0.3",
    )
)
RERANKED: Final = tuple(
    Variant.parse(name)
    for name in (
        "fused+rerank:ms-marco-minilm-l6-int8@10",
        "fused+rerank:ms-marco-minilm-l6-int8@20",
        "fused+rerank:ms-marco-minilm-l6@10",
        "fused+hop1x0.3+rerank:ms-marco-minilm-l6-int8@10",
    )
)
"""Scored only when their reranker is installed, which the header names."""
DEEP_RERANKED: Final = RERANKED[0]
MAX_EXPANSIONS_BYTES: Final = 4_000_000
"""About 2 KB a question at most (3 queries of 200 characters, a passage of
1,200, escaped), for a set of a few hundred questions."""

type Category = Literal["named", "paraphrase", "code", "multi", "none"]
UNANSWERABLE: Final[Category] = "none"
type Parts = Literal["rewrites", "hyde", "both"]
PARTS: Final[tuple[Parts, ...]] = ("rewrites", "hyde", "both")


class SetError(ValueError):
    """A question set, or its expansions, not in the expected format."""


@dataclass(frozen=True, slots=True)
class Question:
    """One question, its category, and the lines that answer it."""

    id: str
    category: Category
    text: str
    gold: tuple[Gold, ...]


@dataclass(frozen=True, slots=True)
class QuestionSet:
    """The commit the questions are about, and the questions."""

    commit: str
    questions: tuple[Question, ...]


class _GoldIn(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    path: str
    start: int
    end: int


class _QuestionIn(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    id: str
    category: Category
    question: str
    gold: list[_GoldIn]


class _SetIn(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    commit: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    questions: list[dict[str, JsonValue]]


def _gold(label: str, given: _GoldIn) -> Gold:
    path = PurePosixPath(given.path)
    if "\\" in given.path or path.is_absolute() or ".." in path.parts or not path.parts:
        message = f"question {label}: gold path {given.path!r} is not a relative path"
        raise SetError(message)
    try:
        return Gold(path, given.start, given.end)
    except ValueError as error:
        message = f"question {label}: gold in {given.path}: {error}"
        raise SetError(message) from error


def _question(index: int, item: dict[str, JsonValue]) -> Question:
    named = item.get("id")
    label = repr(named) if isinstance(named, str) else f"#{index + 1}"
    try:
        given = _QuestionIn.model_validate(item)
    except ValidationError as error:
        message = f"question {label}: {error}"
        raise SetError(message) from error
    if given.category == UNANSWERABLE and given.gold:
        message = f"question {label} is {UNANSWERABLE!r} yet has gold spans"
        raise SetError(message)
    if given.category != UNANSWERABLE and not given.gold:
        message = f"question {label} is {given.category!r} and has no gold span"
        raise SetError(message)
    gold = tuple(_gold(label, span) for span in given.gold)
    return Question(given.id, given.category, given.question, gold)


def parse_set(data: bytes) -> QuestionSet:
    """Return the question set ``data`` holds, as JSON.

    Raises:
        SetError: If ``data`` is larger than :data:`MAX_SET_BYTES`, is not a
            commit and questions in the expected format, or a question's
            category, gold spans or id are wrong; the message names the
            question.
    """
    if len(data) > MAX_SET_BYTES:
        message = f"the set is {len(data)} bytes, over MAX_SET_BYTES={MAX_SET_BYTES}"
        raise SetError(message)
    try:
        given = _SetIn.model_validate_json(data)
    except ValidationError as error:
        message = f"the set is not a commit and a list of questions: {error}"
        raise SetError(message) from error
    questions = tuple(_question(n, item) for n, item in enumerate(given.questions))
    if not questions:
        message = "the set has no question"
        raise SetError(message)
    repeated = [name for name, n in Counter(q.id for q in questions).items() if n > 1]
    if repeated:
        message = f"question ids {repeated} are used more than once"
        raise SetError(message)
    return QuestionSet(given.commit, questions)


def read_set(path: Path) -> QuestionSet:
    """Return the question set in the file ``path``; blocking.

    Raises:
        SetError: As :func:`parse_set` does.
    """
    with path.open("rb") as file:
        return parse_set(file.read(MAX_SET_BYTES + 1))


class _ExpandedIn(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    question: str = Field(min_length=1)
    queries: list[str]
    passage: str | None
    seconds: float = Field(ge=0, allow_inf_nan=False)


class _ExpansionsIn(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    model: str = Field(min_length=1)
    questions: dict[str, _ExpandedIn]


@dataclass(frozen=True, slots=True)
class Expansions:
    """The local model's expansion of each question by id, its seconds and its text."""

    model: str
    by_question: Mapping[str, Expansion]
    seconds: Mapping[str, float]
    texts: Mapping[str, str]


def parse_expansions(data: bytes) -> Expansions:
    """Return the expansions ``data`` holds, as ``benchmarks/deep_expand.py`` writes.

    Raises:
        SetError: If ``data`` is larger than :data:`MAX_EXPANSIONS_BYTES`, is
            not a model and expansions by question id, or an expansion is not
            one deep search makes (see :class:`Expansion`); the message names
            the question.
    """
    if len(data) > MAX_EXPANSIONS_BYTES:
        message = (
            f"the expansions are {len(data)} bytes,"
            f" over MAX_EXPANSIONS_BYTES={MAX_EXPANSIONS_BYTES}"
        )
        raise SetError(message)
    try:
        given = _ExpansionsIn.model_validate_json(data)
    except ValidationError as error:
        message = f"the expansions are not a model and expansions by id: {error}"
        raise SetError(message) from error
    by_question: dict[str, Expansion] = {}
    for question, expanded in given.questions.items():
        try:
            by_question[question] = Expansion(tuple(expanded.queries), expanded.passage)
        except ValueError as error:
            message = f"question {question!r}: {error}"
            raise SetError(message) from error
    seconds = {question: e.seconds for question, e in given.questions.items()}
    texts = {question: e.question for question, e in given.questions.items()}
    return Expansions(given.model, by_question, seconds, texts)


def read_expansions(path: Path, questions: Iterable[Question]) -> Expansions:
    """Return the expansions in the file ``path``, one for each of ``questions``.

    Blocking.

    Raises:
        SetError: As :func:`parse_expansions` does, or naming a question the
            file does not expand, or expanded with another text.
    """
    with path.open("rb") as file:
        expansions = parse_expansions(file.read(MAX_EXPANSIONS_BYTES + 1))
    asked = list(questions)
    missing = [q.id for q in asked if q.id not in expansions.by_question]
    if missing:
        message = f"{path.name} does not expand the questions {missing}"
        raise SetError(message)
    changed = [q.id for q in asked if expansions.texts[q.id] != q.text]
    if changed:
        message = f"{path.name} expanded another text of the questions {changed}"
        raise SetError(message)
    return expansions


@dataclass(frozen=True, slots=True)
class DeepRun:
    """A deep search to score: what is searched beside each question, merged how."""

    parts: Parts
    merge: Merge
    variant: Variant
    expansions: Mapping[str, Expansion]

    def label(self) -> str:
        """Return the run's name in its rows, as ``both/best``."""
        return f"{self.parts}/{self.merge.name.lower()}"

    def also(self, question: str) -> tuple[str, ...]:
        """Return what is searched beside the question with the id ``question``."""
        expansion = self.expansions[question]
        match self.parts:
            case "rewrites":
                return expansion.queries
            case "hyde":
                return () if expansion.passage is None else (expansion.passage,)
            case "both":
                return expansion.also()


def deep_runs(
    expansions: Mapping[str, Expansion], rerankers: Collection[str]
) -> list[DeepRun]:
    """Return the deep searches scored, the reranked ones if their reranker is loaded.

    Every part with every merge at the default variant, then both parts
    reranked by :data:`DEEP_RERANKED`.
    """
    runs = [
        DeepRun(parts, merge, DEFAULT_VARIANT, expansions)
        for parts in PARTS
        for merge in Merge
    ]
    if DEEP_RERANKED.rerank is not None and DEEP_RERANKED.rerank.model in rerankers:
        runs += [DeepRun("both", merge, DEEP_RERANKED, expansions) for merge in Merge]
    return runs


def chunkings(spec: Embedder) -> tuple[Chunking, ...]:
    """Return the chunkings scored at ``spec``, the default first.

    Then sections by structure without a header, then with one, which was
    the default until windows measured better.
    """
    default = Chunking.for_embedder(spec)
    structure = replace(default, splitter=Splitter.STRUCTURE, header=False)
    return (default, structure, replace(structure, header=True))


@dataclass(frozen=True, slots=True)
class Measured:
    """One chunking and variant over every question.

    ``ms`` holds each search's milliseconds, in question order.
    """

    chunking: Chunking
    variant: Variant
    report: Report
    ms: tuple[float, ...]
    index_seconds: float
    chunks: int
    indexed: IndexReport
    by_question: tuple[tuple[str, Scored], ...] = ()
    """Each question's id and result, in question order."""
    deep: str | None = None
    """The deep run's label, None for a search of the question alone."""


def percentile(values: Sequence[float], share: float) -> float:
    """Return the nearest-rank percentile of ``values``, ``share`` 0.95 for p95.

    That is ``sorted(values)[ceil(share * n) - 1]``: always one of the values,
    and defined for a single one, which ``statistics.quantiles`` is not on
    Python 3.12.

    Raises:
        ValueError: If ``values`` is empty or ``share`` is not in (0, 1].
    """
    if not values or not 0 < share <= 1:
        message = f"{len(values)} values, share={share}: need values and 0 < share <= 1"
        raise ValueError(message)
    return sorted(values)[math.ceil(share * len(values)) - 1]


async def _searched(
    knowledge: Knowledge,
    questions: Sequence[Question],
    variant: Variant,
    deep: DeepRun | None,
) -> tuple[tuple[Scored, ...], tuple[float, ...]]:
    results: list[Scored] = []
    ms: list[float] = []
    for question in questions:
        started = time.perf_counter()
        found = await knowledge.search(
            question.text,
            limit=SEARCH_LIMIT,
            variant=variant,
            also=() if deep is None else deep.also(question.id),
            merge=Merge.BEST if deep is None else deep.merge,
        )
        ms.append((time.perf_counter() - started) * 1000)
        results.append(scored(found.passages, question.gold))
    return tuple(results), tuple(ms)


async def evaluate(  # noqa: PLR0913 - the eval's whole configuration
    corpus: Path,
    questions: Sequence[Question],
    embedder: TextEmbedder,
    chunkings: Iterable[Chunking],
    variants: Sequence[Variant],
    *,
    rerankers: Mapping[str, CrossEncoder] | None = None,
    deep: Sequence[DeepRun] = (),
) -> AsyncIterator[Measured]:
    """Yield, per chunking then variant, how its search scores on ``questions``.

    Each chunking indexes ``corpus`` as one source in a fresh, temporary
    home, timed from the first file to the last vector. The ``deep`` runs
    are scored on the first chunking only, after its variants.
    """
    for n, chunking in enumerate(chunkings):
        searches: list[tuple[Variant, DeepRun | None]] = [(v, None) for v in variants]
        if n == 0:
            searches += [(run.variant, run) for run in deep]
        with tempfile.TemporaryDirectory() as home:
            knowledge = await Knowledge.open(
                Path(home) / "knowledge.db",
                embedder,
                chunking=chunking,
                rerankers=rerankers,
            )
            await knowledge.add(corpus)
            started = time.perf_counter()
            indexed = await knowledge.index()
            seconds = time.perf_counter() - started
            for variant, run in searches:
                results, ms = await _searched(knowledge, questions, variant, run)
                report = Report.of(
                    (q.category, r) for q, r in zip(questions, results, strict=True)
                )
                yield Measured(
                    chunking,
                    variant,
                    report,
                    ms,
                    seconds,
                    indexed.chunks,
                    indexed,
                    tuple(zip((q.id for q in questions), results, strict=True)),
                    None if run is None else run.label(),
                )


Row = TypedDict(
    "Row",
    {
        "chunking": str,
        "variant": str,
        "category": str,
        "questions": int,
        "answerable": int,
        "recall@1": float | None,
        "recall@5": float | None,
        "recall@10": float | None,
        "mrr": float | None,
        "ndcg@10": float | None,
        "top_score": float,
        "ms_median": float,
        "ms_p95": float,
        "index_s": float,
        "chunks": int,
        "deep": NotRequired[str],
    },
)


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, DIGITS)


def _row(measured: Measured, category: str, means: Means) -> Row:
    ranks = means.ranks
    row: Row = {
        "chunking": measured.chunking.recipe(),
        "variant": measured.variant.name(),
        "category": category,
        "questions": means.questions,
        "answerable": means.answerable,
        "recall@1": _rounded(None if ranks is None else ranks.recall[1]),
        "recall@5": _rounded(None if ranks is None else ranks.recall[5]),
        "recall@10": _rounded(None if ranks is None else ranks.recall[10]),
        "mrr": _rounded(None if ranks is None else ranks.reciprocal_rank),
        "ndcg@10": _rounded(None if ranks is None else ranks.ndcg),
        "top_score": round(means.top_score, DIGITS),
        "ms_median": round(statistics.median(measured.ms), DIGITS),
        "ms_p95": round(percentile(measured.ms, PERCENTILE), DIGITS),
        "index_s": round(measured.index_seconds, DIGITS),
        "chunks": measured.chunks,
    }
    if measured.deep is not None:
        row["deep"] = measured.deep
    return row


def index_line(measured: Measured) -> str:
    """Return the JSON line of what indexing for ``measured``'s chunking did.

    A file held back, unreadable or not embedded scores as a miss in every
    row after it, so the counts are printed beside the rows they explain.
    """
    return json.dumps(
        {"chunking": measured.chunking.recipe(), "index": asdict(measured.indexed)},
        allow_nan=False,
    )


def question_lines(measured: Measured) -> list[str]:
    """Return one JSON line per question: its nDCG@10, reciprocal rank and top score.

    Unanswerable questions have null rank metrics. These lines let a reader
    resample the questions (a bootstrap) to see how much a mean difference
    between two variants depends on which questions were asked.
    """
    lines: list[str] = []
    deep = {} if measured.deep is None else {"deep": measured.deep}
    for question, result in measured.by_question:
        ranks = result.ranks if isinstance(result, Answerable) else None
        lines.append(
            json.dumps(
                {
                    "chunking": measured.chunking.recipe(),
                    "variant": measured.variant.name(),
                    "id": question,
                    "ndcg@10": _rounded(None if ranks is None else ranks.ndcg),
                    "mrr": _rounded(None if ranks is None else ranks.reciprocal_rank),
                    "top_score": round(result.top_score, DIGITS),
                    **deep,
                },
                allow_nan=False,
            )
        )
    return lines


def rows(measured: Measured) -> list[Row]:
    """Return one row per category, in the order first asked, then the overall row."""
    report = measured.report
    by_category = [_row(measured, c, m) for c, m in report.categories.items()]
    return [*by_category, _row(measured, OVERALL, report.overall)]


def archived(repository: Path, commit: str, into: Path) -> Path:
    """Return a folder in ``into`` holding the files of ``repository`` at ``commit``.

    Blocking.

    Raises:
        FileNotFoundError: If git is not on the PATH.
        subprocess.CalledProcessError: If git cannot archive ``commit``.
    """
    git = shutil.which("git")
    if git is None:
        message = "git is not on the PATH"
        raise FileNotFoundError(message)
    tar = into / "corpus.tar"
    subprocess.run(  # noqa: S603 - a resolved git, and a commit parsed as hex
        [
            git,
            "-C",
            str(repository),
            "archive",
            "--format=tar",
            f"--output={tar}",
            commit,
        ],
        check=True,
        capture_output=True,
        timeout=GIT_TIMEOUT_S,
    )
    corpus = into / "corpus"
    with tarfile.open(tar) as archive:
        archive.extractall(corpus, filter="data")
    tar.unlink()
    return corpus


def check_gold(corpus: Path, questions: Iterable[Question]) -> None:
    """Refuse a question whose gold names a file the corpus does not hold.

    Raises:
        SetError: Naming the question and the path.
    """
    for question in questions:
        for gold in question.gold:
            if not (corpus / gold.path).is_file():
                message = f"question {question.id!r}: {gold.path} is not in the corpus"
                raise SetError(message)


def installed_embedder() -> TextEmbedder:
    """Return the installed default embedding model; blocking."""
    models = Installer(default_home(), BYTES_PER_GB)
    return TextEmbedder.load(models.path_of(DEFAULT_EMBEDDER), DEFAULT_EMBEDDER)


def installed_rerankers() -> dict[str, CrossEncoder]:
    """Return every installed reranker, loaded, by catalogue id; blocking."""
    models = Installer(default_home(), BYTES_PER_GB)
    return {
        spec.id: CrossEncoder.load(models.path_of(spec), spec)
        for spec in RERANKERS
        if models.installed(spec)
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score knowledge search on the question set's commit."
    )
    parser.add_argument("--set", type=Path, default=KNOWLEDGE_SET)
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument(
        "--per-question",
        type=Path,
        help="also write each question's scores, one JSON line each, to this file",
    )
    parser.add_argument(
        "--chunking",
        action="append",
        default=[],
        help="score only this chunking, by recipe; repeatable",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        choices=[variant.name() for variant in (*VARIANTS, *RERANKED)],
        help="score only this variant; repeatable",
    )
    parser.add_argument(
        "--expansions",
        type=Path,
        help="also score deep search with the expansions in this file,"
        " as benchmarks/deep_expand.py writes them",
    )
    return parser


def _chosen[T](
    every: Sequence[T], names: Sequence[str], name: Callable[[T], str]
) -> list[T]:
    """Return the items of ``every`` named in ``names``, or all when it is empty."""
    return [item for item in every if not names or name(item) in names]


def _header(
    question_set: QuestionSet,
    corpus: Path,
    spec: Embedder,
    rerankers: Iterable[str],
    expansions: Expansions | None,
) -> str:
    files = sum(1 for path in corpus.rglob("*") if path.is_file())
    header: dict[str, object] = {
        "commit": question_set.commit,
        "questions": Counter(q.category for q in question_set.questions),
        "corpus_files": files,
        "embedder": spec.id,
        "rerankers": sorted(rerankers),
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "onnxruntime": version("onnxruntime"),
    }
    if expansions is not None:
        seconds = [expansions.seconds[q.id] for q in question_set.questions]
        header["expansions"] = {
            "model": expansions.model,
            "s_median": round(statistics.median(seconds), DIGITS),
            "s_p95": round(percentile(seconds, PERCENTILE), DIGITS),
        }
    return json.dumps(header)


async def _printed(measurements: AsyncIterator[Measured]) -> list[str]:
    """Print each chunking's index line and rows; return the per-question lines."""
    shown: set[str] = set()
    per_question: list[str] = []
    async for measured in measurements:
        if measured.chunking.recipe() not in shown:
            shown.add(measured.chunking.recipe())
            print(index_line(measured), flush=True)
        for row in rows(measured):
            print(json.dumps(row, allow_nan=False), flush=True)
        per_question += question_lines(measured)
    return per_question


def _chosen_chunkings(
    parser: argparse.ArgumentParser, known: Sequence[Chunking], names: Sequence[str]
) -> list[Chunking]:
    unknown = set(names) - {c.recipe() for c in known}
    if unknown:
        parser.error(
            f"unknown chunking {sorted(unknown)}: one of {[c.recipe() for c in known]}"
        )
    return _chosen(known, names, Chunking.recipe)


def _chosen_variants(
    parser: argparse.ArgumentParser, names: Sequence[str], rerankers: Collection[str]
) -> list[Variant]:
    usable = [
        v for v in RERANKED if v.rerank is not None and v.rerank.model in rerankers
    ]
    missing = set(names) - {v.name() for v in (*VARIANTS, *usable)}
    if missing:
        parser.error(
            f"variant {sorted(missing)} needs a reranker that is not installed"
        )
    return _chosen((*VARIANTS, *usable), names, Variant.name)


async def main(
    argv: Sequence[str] | None = None,
    *,
    load: Callable[[], TextEmbedder] = installed_embedder,
    load_rerankers: Callable[[], dict[str, CrossEncoder]] = installed_rerankers,
) -> int:
    """Print the header line, then one JSON line per chunking, variant and category.

    ``load`` returns the embedder to index and search with, ``load_rerankers``
    the rerankers by catalogue id; a reranked variant is scored only when its
    reranker is loaded.
    """
    parser = _parser()
    arguments = parser.parse_args(argv)
    question_set = await asyncio.to_thread(read_set, arguments.set)
    embedder = await asyncio.to_thread(load)
    known = chunkings(embedder.spec)
    chosen = _chosen_chunkings(parser, known, arguments.chunking)
    rerankers = await asyncio.to_thread(load_rerankers)
    variants = _chosen_variants(parser, arguments.variant, rerankers)
    expansions: Expansions | None = None
    if arguments.expansions is not None:
        if known[0] not in chosen:
            parser.error(
                f"--expansions scores the default chunking {known[0].recipe()},"
                " which --chunking leaves out"
            )
        expansions = await asyncio.to_thread(
            read_expansions, arguments.expansions, question_set.questions
        )
    deep = [] if expansions is None else deep_runs(expansions.by_question, rerankers)
    with tempfile.TemporaryDirectory() as scratch:
        corpus = await asyncio.to_thread(
            archived, arguments.repository, question_set.commit, Path(scratch)
        )
        await asyncio.to_thread(check_gold, corpus, question_set.questions)
        header = await asyncio.to_thread(
            _header, question_set, corpus, embedder.spec, rerankers, expansions
        )
        print(header, flush=True)
        per_question = await _printed(
            evaluate(
                corpus,
                question_set.questions,
                embedder,
                chosen,
                variants,
                rerankers=rerankers,
                deep=deep,
            )
        )
    if arguments.per_question is not None:
        text = "".join(f"{line}\n" for line in per_question)
        await asyncio.to_thread(arguments.per_question.write_text, text, "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
