"""Score how well knowledge search finds the lines that answer a question.

    uv run python benchmarks/rag_eval.py [--chunking NAME]... [--variant NAME]...

The question set (``benchmarks/data/knowledge_set.json``) names a commit of
this repository and asks questions about the files at that commit, each with
the lines that answer it (gold spans) or none for a question nothing answers.
The files are taken with ``git archive <commit>`` and indexed as one source,
so a gold path, relative to the repository root, is a passage's path.

Each chunking builds its own cold index in a fresh home, embedded with the
installed default model: the default chunking (structure with a header, at
the embedder's token limit), structure without a header, and windows of 40
lines without a header, the baseline. Each variant then searches that index
for every question, 10 passages deep: words, meaning, and the fused score at
the default word weight and at 0.05, 0.2 and 0.4.

Output is JSON lines: first a header (commit, questions per category, files
in the corpus, embedder, python, numpy and onnxruntime versions), then per
chunking, variant and category, and over all questions, the rank metrics of
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
from typing import TYPE_CHECKING, Final, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from synthia.kernel.config import default_home
from synthia.knowledge.chunk import Chunking, Splitter
from synthia.knowledge.evaluate import Answerable, Gold, Report, scored
from synthia.knowledge.retrieve import Variant
from synthia.knowledge.service import Knowledge
from synthia.models.catalogue import DEFAULT_EMBEDDER
from synthia.models.install import BYTES_PER_GB, Installer
from synthia.retrieval.embed import TextEmbedder

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterable, Sequence

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
    for name in ("words", "meaning", "fused", "fused:0.05", "fused:0.2", "fused:0.4")
)

type Category = Literal["named", "paraphrase", "code", "multi", "none"]
UNANSWERABLE: Final[Category] = "none"


class SetError(ValueError):
    """A question set that is not in the expected format."""


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


def chunkings(spec: Embedder) -> tuple[Chunking, ...]:
    """Return the chunkings scored at ``spec``: the default first, the baseline last."""
    default = Chunking.for_embedder(spec)
    return (
        default,
        replace(default, header=False),
        replace(default, splitter=Splitter.LINES, header=False),
    )


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
    knowledge: Knowledge, questions: Sequence[Question], variant: Variant
) -> tuple[tuple[Scored, ...], tuple[float, ...]]:
    results: list[Scored] = []
    ms: list[float] = []
    for question in questions:
        started = time.perf_counter()
        found = await knowledge.search(
            question.text, limit=SEARCH_LIMIT, variant=variant
        )
        ms.append((time.perf_counter() - started) * 1000)
        results.append(scored(found.passages, question.gold))
    return tuple(results), tuple(ms)


async def evaluate(
    corpus: Path,
    questions: Sequence[Question],
    embedder: TextEmbedder,
    chunkings: Iterable[Chunking],
    variants: Sequence[Variant],
) -> AsyncIterator[Measured]:
    """Yield, per chunking then variant, how its search scores on ``questions``.

    Each chunking indexes ``corpus`` as one source in a fresh, temporary
    home, timed from the first file to the last vector.
    """
    for chunking in chunkings:
        with tempfile.TemporaryDirectory() as home:
            knowledge = await Knowledge.open(
                Path(home) / "knowledge.db", embedder, chunking=chunking
            )
            await knowledge.add(corpus)
            started = time.perf_counter()
            indexed = await knowledge.index()
            seconds = time.perf_counter() - started
            for variant in variants:
                results, ms = await _searched(knowledge, questions, variant)
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
    },
)


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, DIGITS)


def _row(measured: Measured, category: str, means: Means) -> Row:
    ranks = means.ranks
    return {
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
        choices=[variant.name() for variant in VARIANTS],
        help="score only this variant; repeatable",
    )
    return parser


def _chosen[T](
    every: Sequence[T], names: Sequence[str], name: Callable[[T], str]
) -> list[T]:
    """Return the items of ``every`` named in ``names``, or all when it is empty."""
    return [item for item in every if not names or name(item) in names]


def _header(question_set: QuestionSet, corpus: Path, spec: Embedder) -> str:
    files = sum(1 for path in corpus.rglob("*") if path.is_file())
    return json.dumps(
        {
            "commit": question_set.commit,
            "questions": Counter(q.category for q in question_set.questions),
            "corpus_files": files,
            "embedder": spec.id,
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "onnxruntime": version("onnxruntime"),
        }
    )


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


async def main(
    argv: Sequence[str] | None = None,
    *,
    load: Callable[[], TextEmbedder] = installed_embedder,
) -> int:
    """Print the header line, then one JSON line per chunking, variant and category.

    ``load`` returns the embedder to index and search with.
    """
    parser = _parser()
    arguments = parser.parse_args(argv)
    question_set = await asyncio.to_thread(read_set, arguments.set)
    embedder = await asyncio.to_thread(load)
    known = chunkings(embedder.spec)
    unknown = set(arguments.chunking) - {c.recipe() for c in known}
    if unknown:
        parser.error(
            f"unknown chunking {sorted(unknown)}: one of {[c.recipe() for c in known]}"
        )
    chosen = _chosen(known, arguments.chunking, Chunking.recipe)
    variants = _chosen(VARIANTS, arguments.variant, Variant.name)
    with tempfile.TemporaryDirectory() as scratch:
        corpus = await asyncio.to_thread(
            archived, arguments.repository, question_set.commit, Path(scratch)
        )
        await asyncio.to_thread(check_gold, corpus, question_set.questions)
        header = await asyncio.to_thread(_header, question_set, corpus, embedder.spec)
        print(header, flush=True)
        per_question = await _printed(
            evaluate(corpus, question_set.questions, embedder, chosen, variants)
        )
    if arguments.per_question is not None:
        text = "".join(f"{line}\n" for line in per_question)
        await asyncio.to_thread(arguments.per_question.write_text, text, "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
