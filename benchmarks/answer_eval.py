"""Score the chat's answers: did it search, find, cite, and keep to what it was shown.

    uv run python benchmarks/answer_eval.py --answers PATH --verdicts PATH
        [--calibration [PATH]] [--threads N] [--resume]

``--answers`` is the file ``benchmarks/chat_answers.py`` wrote for the
question set (``benchmarks/data/knowledge_set.json``); it must answer every
question, with the same text, about the set's commit. Each answer not yet
judged is judged by the installed local model (``benchmarks/judge.py``)
against the exact search results the chat's model was shown, and the
verdict goes to ``--verdicts``::

    {
        "model": "<judge model id>",
        "judge_prompt_sha256": "<SHA-256 of what the judge is asked with, hex>",
        "verdicts": [
            {
                "question": "...",
                "answer": "...",
                "passages_sha256": "<SHA-256 of the passages, hex>",
                "verdict": {"supported": false, "unsupported": ["..."]} or null,
                "error": null or "<why the judge gave no valid verdict>",
                "seconds": 3.2,
            }
        ],
    }

A verdict is keyed by the question's text, the answer's text, and the
SHA-256 of the passages it was judged against, taken over the passages as
a JSON list, so ``["a", "b"]`` and ``["ab"]`` differ. A changed answer, or
the same answer shown other passages, is judged again; an unchanged one
never is. The file is written after every verdict; ``--resume`` keeps it.
A judge that never replies with a valid verdict, after its repairs, is
recorded as an error and counted, never guessed, and is judged again on
``--resume``. ``judge_prompt_sha256`` is ``judge.prompt_sha256()``: the
judge's instructions, worked examples and reply schema, so ``--resume``
refuses a file written under another judge prompt, as it refuses one by
another model. An answer whose turn ended in an error (``error`` set in
the answers file) is not judged. With everything already judged, no model
is started.

With ``--calibration`` (by default ``benchmarks/data/faithfulness_set.json``),
the judge also judges the labelled answers of that set, so its verdicts can
be set against a person's labels.

Output is JSON lines: one row per category, in the order first asked, then
over all questions:

- ``searched``: the share of questions with at least one search.
- ``shown_recall``: the share of answerable questions where a citation the
  searches showed overlaps a gold span: the same file and page, and a line
  in common, as :mod:`synthia.knowledge.evaluate` counts.
- ``cited_recall``: the share where the answer names such a citation, as
  shown, character for character.
- ``supported``: the share of judged answers the judge found supported,
  with ``judge_errors`` the answers it could not judge and ``chat_errors``
  the answers whose turn ended in an error, which are not judged.
- ``s_median`` and ``s_p95``: the chat's seconds per question.

Searched counts what each answer's entry recorded, an errored one
included: its searches that finished. Shown and cited are null for an
errored one, as supported is: its turn never reported the citations it
showed, so it counts only in ``chat_errors``, never as a miss. A ``none``
question has no gold span, so its recall is null. With
``--calibration``, every row also carries the judge's overall agreement and
Cohen's kappa, and one line follows per group (overall, then each origin):
agreement, unsupported answers caught, false alarms and kappa. It reaches
no outside service and spends no budget.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import (
    TYPE_CHECKING,
    Annotated,
    Final,
    Literal,
    NotRequired,
    Self,
    TypedDict,
)

import httpx
from chat_answers import ServiceOpener, read_answers
from fact_extraction import TIMEOUT, installed, ready
from judge import Verdict, judge, prompt_sha256
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from rag_eval import (
    DIGITS,
    KNOWLEDGE_SET,
    OVERALL,
    PERCENTILE,
    SetError,
    percentile,
    read_set,
)

from synthia.gateway.structured import StructuredOutputError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from typing import TextIO

    from chat_answers import Answered
    from rag_eval import Category, Question

    from synthia.gateway.protocol import ChatModel
    from synthia.knowledge.evaluate import Gold

DESCRIPTION: Final = "Judge and score the chat's answers to the question set."
FAITHFULNESS_SET: Final = Path(__file__).parent / "data" / "faithfulness_set.json"
MAX_VERDICTS_BYTES: Final = 20_000_000
"""A verdict holds a question, an answer and at most 8 claims of 300
characters; a few hundred of them stay well below this."""
MAX_CALIBRATION_BYTES: Final = 10_000_000
"""A labelled answer holds the searches it was shown, at most about 15,000
characters each; a set of a hundred stays below this."""
_CITATION: Final = re.compile(
    r"(?P<source>[^/]+)/(?P<path>.+?):(?:p\.(?P<page>\d+):)?"
    r"(?P<start>\d+)(?:-(?P<end>\d+))?"
)

_SHA256_HEX: Final = r"[0-9a-f]{64}"

type Origin = Literal["natural", "perturbed"]
type Key = tuple[str, str, str]
"""A verdict's question text, answer text, and passages' SHA-256."""


def passages_sha256(passages: Sequence[str]) -> str:
    """Return the SHA-256, in hex, of ``passages`` written as a JSON list.

    As a list, ``["a", "b"]`` and ``["ab"]`` give different digests.
    """
    encoded = json.dumps(list(passages), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class Cited:
    """The lines a citation names: a path in its source, a page, and a line range."""

    path: PurePosixPath
    page: int | None
    start: int
    end: int

    @classmethod
    def parse(cls, citation: str) -> Self | None:
        """Return the lines ``citation`` names, as ``source/path:start-end``.

        A single line is ``source/path:N`` and a page comes first, as in
        ``source/path:p.2:4-9``. None when ``citation`` is not one.
        """
        matched = _CITATION.fullmatch(citation)
        if matched is None:
            return None
        start = int(matched["start"])
        end = start if matched["end"] is None else int(matched["end"])
        page = None if matched["page"] is None else int(matched["page"])
        return cls(PurePosixPath(matched["path"]), page, start, end)


def overlaps_gold(citation: str, gold: Iterable[Gold]) -> bool:
    """Return whether ``citation`` shares a line with one of the ``gold`` spans.

    The eval indexes one source, so the source's name is not compared.
    """
    cited = Cited.parse(citation)
    if cited is None:
        return False
    return any(g.overlaps(cited.path, cited.page, cited.start, cited.end) for g in gold)


def names(answer: str, citation: str) -> bool:
    """Return whether ``answer`` holds ``citation`` as written, not inside a longer one.

    ``corpus/a.py:1`` is not named by ``corpus/a.py:12`` or ``corpus/a.py:1-20``.
    """
    pattern = rf"(?<![\w/.-]){re.escape(citation)}(?![\d-])"
    return re.search(pattern, answer) is not None


@dataclass(frozen=True, slots=True)
class Judged:
    """The judge's verdict on one answer, or why it gave none.

    ``passages_sha256`` is :func:`passages_sha256` of the passages it was
    judged against.

    Raises:
        ValueError: If both or neither of ``verdict`` and ``error`` are set.
    """

    question: str
    answer: str
    passages_sha256: str
    verdict: Verdict | None
    error: str | None
    seconds: float

    def __post_init__(self) -> None:
        if (self.verdict is None) == (self.error is None):
            message = (
                f"verdict={self.verdict!r}, error={self.error!r}: exactly one is set"
            )
            raise ValueError(message)

    @property
    def key(self) -> Key:
        """Return the key the verdict is kept under, as :data:`Key`."""
        return self.question, self.answer, self.passages_sha256


class _JudgedIn(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str
    answer: str
    passages_sha256: str = Field(pattern=f"^{_SHA256_HEX}$")
    verdict: Verdict | None
    error: str | None
    seconds: float = Field(ge=0, allow_inf_nan=False)


class _VerdictsIn(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(min_length=1)
    judge_prompt_sha256: str = Field(pattern=f"^{_SHA256_HEX}$")
    verdicts: list[_JudgedIn]


@dataclass(frozen=True, slots=True)
class Verdicts:
    """A verdicts file: the judge's model, its prompt's digest, the verdicts by key."""

    model: str
    judge_prompt_sha256: str
    by_key: dict[Key, Judged]


def parse_verdicts(data: bytes) -> Verdicts:
    """Return the verdicts ``data`` holds, with the judge that gave them.

    Raises:
        SetError: If ``data`` is larger than :data:`MAX_VERDICTS_BYTES`, is
            not a model, a prompt digest and a list of verdicts, or a verdict
            has both or neither of a verdict and an error.
    """
    if len(data) > MAX_VERDICTS_BYTES:
        message = (
            f"the verdicts are {len(data)} bytes,"
            f" over MAX_VERDICTS_BYTES={MAX_VERDICTS_BYTES}"
        )
        raise SetError(message)
    try:
        given = _VerdictsIn.model_validate_json(data)
        judged = [
            Judged(
                v.question, v.answer, v.passages_sha256, v.verdict, v.error, v.seconds
            )
            for v in given.verdicts
        ]
    except (ValidationError, ValueError) as error:
        message = (
            "the verdicts are not a model, a prompt digest and a list of verdicts:"
            f" {error}"
        )
        raise SetError(message) from error
    by_key = {j.key: j for j in judged}
    return Verdicts(given.model, given.judge_prompt_sha256, by_key)


def held_verdicts(path: Path, model: str, prompt: str) -> dict[Key, Judged]:
    """Return the verdicts already in ``path``; blocking.

    ``prompt`` is the digest of the judge's prompt, as ``judge.prompt_sha256``.

    Raises:
        SetError: If the file is not verdicts, or ``model`` did not write it
            under ``prompt``.
    """
    found = parse_verdicts(path.read_bytes())
    if found.model != model:
        message = f"{path.name} holds {found.model}'s verdicts, not {model}'s"
        raise SetError(message)
    if found.judge_prompt_sha256 != prompt:
        message = (
            f"{path.name} holds verdicts under judge prompt"
            f" {found.judge_prompt_sha256}, not {prompt}"
        )
        raise SetError(message)
    return found.by_key


def written_verdicts(
    path: Path, model: str, prompt: str, judged: Mapping[Key, Judged]
) -> None:
    """Replace ``path`` with the verdicts, so a crash never leaves half a file.

    ``prompt`` is the digest of the judge's prompt, as ``judge.prompt_sha256``.
    """
    verdicts = [
        {
            "question": j.question,
            "answer": j.answer,
            "passages_sha256": j.passages_sha256,
            "verdict": None if j.verdict is None else j.verdict.model_dump(),
            "error": j.error,
            "seconds": j.seconds,
        }
        for j in judged.values()
    ]
    text = json.dumps(
        {"model": model, "judge_prompt_sha256": prompt, "verdicts": verdicts},
        ensure_ascii=True,
        allow_nan=False,
        indent=2,
    )
    partial = path.with_name(f"{path.name}.partial")
    partial.write_text(f"{text}\n", encoding="utf-8", newline="")
    partial.replace(path)


@dataclass(frozen=True, slots=True)
class Asked:
    """An answer to judge: its question, its text, and the searches it was shown."""

    question: str
    answer: str
    passages: tuple[str, ...]

    @property
    def key(self) -> Key:
        """Return the key its verdict is kept under, as :data:`Key`."""
        return self.question, self.answer, passages_sha256(self.passages)


def shown(answered: Answered) -> tuple[str, ...]:
    """Return the result of each search that ran, as the model was shown it.

    A search that failed showed an error, not passages.
    """
    return tuple(s["result"] for s in answered["searches"] if s["ok"])


def to_judge(answered: Answered) -> Asked | None:
    """Return the answer to judge, or None if its turn ended in an error.

    A cut-off answer is part of a reply, so judging it would score text the
    chat never finished.
    """
    if answered["error"] is not None:
        return None
    return Asked(answered["question"], answered["answer"], shown(answered))


def unsettled(judged: Mapping[Key, Judged], key: Key) -> bool:
    """Return whether the answer under ``key`` has no verdict yet, or only an error."""
    held = judged.get(key)
    return held is None or held.error is not None


async def judge_all(  # noqa: PLR0913 - the run's whole configuration
    model: ChatModel,
    asked: Sequence[Asked],
    out: Path,
    judged: dict[Key, Judged],
    *,
    progress: TextIO = sys.stderr,
    clock: Callable[[], float] = time.perf_counter,
) -> None:
    """Judge each answer with no verdict yet, writing ``out`` after each.

    One the judge gave no valid verdict for before is judged again.

    Raises:
        GatewayError: If the model failed other than by never replying with
            a valid verdict.
    """
    prompt = prompt_sha256()
    await asyncio.to_thread(out.parent.mkdir, parents=True, exist_ok=True)
    for n, item in enumerate(asked, 1):
        key = item.key
        if not unsettled(judged, key):
            continue
        started = clock()
        try:
            verdict = await judge(model, item.question, item.answer, item.passages)
        except StructuredOutputError as failed:
            verdict, error, said = None, str(failed), "error"
        else:
            error, said = None, "supported" if verdict.supported else "unsupported"
        seconds = round(clock() - started, DIGITS)
        judged[key] = Judged(*key, verdict, error, seconds)
        await asyncio.to_thread(written_verdicts, out, model.info.id, prompt, judged)
        progress.write(f"{n}/{len(asked)} {seconds:.1f} s, {said}\n")
        progress.flush()


@dataclass(frozen=True, slots=True)
class Outcome:
    """What one answer did; ``shown`` and ``cited`` are None for an unanswerable one.

    ``chat_error`` marks an answer whose turn ended in an error, which is not
    judged and whose sources were never reported: ``shown``, ``cited`` and
    ``supported`` are None for it, so it leaves every rate but
    ``searched``. ``supported`` is also None when the judge gave no valid
    verdict.
    """

    category: Category
    searched: bool
    shown: bool | None
    cited: bool | None
    supported: bool | None
    seconds: float
    chat_error: bool


def outcome(question: Question, answered: Answered, judged: Judged | None) -> Outcome:
    """Return what the answer to ``question`` did, by the judge's verdict.

    ``judged`` is None for an answer whose turn ended in an error, whose
    recall is not scored: such a turn reports no sources, so it would count
    as a miss even when its searches showed gold.
    """
    found = [s for s in answered["sources"] if overlaps_gold(s, question.gold)]
    chat_error = answered["error"] is not None
    scored = bool(question.gold) and not chat_error
    verdict = None if judged is None else judged.verdict
    return Outcome(
        question.category,
        searched=bool(answered["searches"]),
        shown=bool(found) if scored else None,
        cited=any(names(answered["answer"], s) for s in found) if scored else None,
        supported=None if verdict is None else verdict.supported,
        seconds=answered["seconds"],
        chat_error=chat_error,
    )


def rate(flags: Iterable[bool | None]) -> float | None:
    """Return the share of true among the known ``flags``, or None if none is known."""
    known = [flag for flag in flags if flag is not None]
    return round(sum(known) / len(known), DIGITS) if known else None


class Row(TypedDict):
    """One category's line of the table, or the overall one."""

    category: str
    n: int
    searched: float | None
    shown_recall: float | None
    cited_recall: float | None
    supported: float | None
    judge_errors: int
    chat_errors: int
    s_median: float
    s_p95: float
    judge_agreement: NotRequired[float | None]
    judge_kappa: NotRequired[float | None]


@dataclass(frozen=True, slots=True)
class Agreement:
    """How the judge's verdicts compare with the labels on one group of answers.

    ``n`` counts the answers judged; ``errors`` those the judge could not
    judge, left out of everything else. ``caught`` of the
    ``labelled_unsupported`` answers the judge also found unsupported;
    ``false_alarms`` of the ``labelled_supported`` it found unsupported.
    """

    n: int
    errors: int
    agreed: int
    caught: int
    labelled_unsupported: int
    false_alarms: int
    labelled_supported: int
    kappa: float | None

    @property
    def share(self) -> float | None:
        """Return the share of judged answers where the judge and the label agree."""
        return round(self.agreed / self.n, DIGITS) if self.n else None


def kappa(pairs: Sequence[tuple[bool, bool]]) -> float | None:
    """Return Cohen's kappa of the judge against the labels, as ``(judge, label)``.

    The agreement beyond what the two shares would give by chance, over the
    most there could be beyond chance. None when there is no pair, or when
    both always say the same one thing, which leaves nothing beyond chance:
    kappa is then 0 / 0.
    """
    n = len(pairs)
    agreed = sum(judged == label for judged, label in pairs)
    judge_yes = sum(judged for judged, _ in pairs)
    label_yes = sum(label for _, label in pairs)
    # Counted over n * n, so a chance level of exactly 1 is an exact integer.
    chance = judge_yes * label_yes + (n - judge_yes) * (n - label_yes)
    if n * n == chance:
        return None
    return round((agreed * n - chance) / (n * n - chance), DIGITS)


def agreement(pairs: Iterable[tuple[bool | None, bool]]) -> Agreement:
    """Return how the judge's verdicts, as ``(judge, label)``, match the labels.

    A judge of None is an error.
    """
    given = list(pairs)
    judged = [(j, label) for j, label in given if j is not None]
    unsupported = [j for j, label in judged if not label]
    supported = [j for j, label in judged if label]
    return Agreement(
        n=len(judged),
        errors=len(given) - len(judged),
        agreed=sum(j == label for j, label in judged),
        caught=sum(not j for j in unsupported),
        labelled_unsupported=len(unsupported),
        false_alarms=sum(not j for j in supported),
        labelled_supported=len(supported),
        kappa=kappa(judged),
    )


def _row(category: str, outcomes: Sequence[Outcome], judged: Agreement | None) -> Row:
    seconds = [o.seconds for o in outcomes]
    row: Row = {
        "category": category,
        "n": len(outcomes),
        "searched": rate(o.searched for o in outcomes),
        "shown_recall": rate(o.shown for o in outcomes),
        "cited_recall": rate(o.cited for o in outcomes),
        "supported": rate(o.supported for o in outcomes),
        "judge_errors": sum(o.supported is None and not o.chat_error for o in outcomes),
        "chat_errors": sum(o.chat_error for o in outcomes),
        "s_median": round(statistics.median(seconds), DIGITS),
        "s_p95": round(percentile(seconds, PERCENTILE), DIGITS),
    }
    if judged is not None:
        row["judge_agreement"] = judged.share
        row["judge_kappa"] = judged.kappa
    return row


def rows(outcomes: Sequence[Outcome], judged: Agreement | None = None) -> list[Row]:
    """Return one row per category, in the order first asked, then the overall row.

    ``judged`` is the judge's agreement with the labels, carried by every row.
    """
    by_category: defaultdict[str, list[Outcome]] = defaultdict(list)
    for o in outcomes:
        by_category[o.category].append(o)
    return [
        *(_row(c, group, judged) for c, group in by_category.items()),
        _row(OVERALL, outcomes, judged),
    ]


@dataclass(frozen=True, slots=True)
class Labelled:
    """An answer a person labelled: supported or not, and the claims that are not."""

    id: str
    question: str
    passages: tuple[str, ...]
    answer: str
    supported: bool
    unsupported: tuple[str, ...]
    labelled_by: str
    origin: Origin

    def asked(self) -> Asked:
        """Return the answer as the judge is asked it."""
        return Asked(self.question, self.answer, self.passages)


class _LabelledIn(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    passages: list[str]
    answer: str
    supported: bool
    unsupported: list[Annotated[str, Field(min_length=1)]]
    labelled_by: str = Field(min_length=1)
    origin: Origin


class _CalibrationIn(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    items: list[_LabelledIn]


def parse_calibration(data: bytes) -> tuple[Labelled, ...]:
    """Return the labelled answers ``data`` holds, as ``{"items": [...]}``.

    Raises:
        SetError: If ``data`` is larger than :data:`MAX_CALIBRATION_BYTES`,
            is not a list of labelled answers (an empty claim included),
            holds none, repeats an id, or
            labels an answer supported while listing unsupported claims, or
            unsupported while listing none.
    """
    if len(data) > MAX_CALIBRATION_BYTES:
        message = (
            f"the calibration set is {len(data)} bytes,"
            f" over MAX_CALIBRATION_BYTES={MAX_CALIBRATION_BYTES}"
        )
        raise SetError(message)
    try:
        given = _CalibrationIn.model_validate_json(data)
    except ValidationError as error:
        message = f"the calibration set is not a list of labelled answers: {error}"
        raise SetError(message) from error
    if not given.items:
        message = "the calibration set has no item"
        raise SetError(message)
    repeated = [i for i, n in Counter(item.id for item in given.items).items() if n > 1]
    if repeated:
        message = f"calibration ids {repeated} are used more than once"
        raise SetError(message)
    mislabelled = [i.id for i in given.items if i.supported == bool(i.unsupported)]
    if mislabelled:
        message = (
            f"calibration items {mislabelled} are labelled supported with"
            " unsupported claims, or unsupported with none"
        )
        raise SetError(message)
    return tuple(
        Labelled(
            i.id,
            i.question,
            tuple(i.passages),
            i.answer,
            i.supported,
            tuple(i.unsupported),
            i.labelled_by,
            i.origin,
        )
        for i in given.items
    )


def read_calibration(path: Path) -> tuple[Labelled, ...]:
    """Return the labelled answers in the file ``path``; blocking.

    Raises:
        SetError: As :func:`parse_calibration` does.
    """
    with path.open("rb") as file:
        return parse_calibration(file.read(MAX_CALIBRATION_BYTES + 1))


def calibrated(
    labelled: Sequence[Labelled], judged: dict[Key, Judged]
) -> list[tuple[str, Agreement]]:
    """Return the judge's agreement over all ``labelled``, then per origin, if any."""
    if not labelled:
        return []
    by_origin: defaultdict[str, list[tuple[bool | None, bool]]] = defaultdict(list)
    for item in labelled:
        verdict = judged[item.asked().key].verdict
        pair = (None if verdict is None else verdict.supported, item.supported)
        by_origin[item.origin].append(pair)
    every = [pair for pairs in by_origin.values() for pair in pairs]
    return [
        (OVERALL, agreement(every)),
        *((origin, agreement(pairs)) for origin, pairs in by_origin.items()),
    ]


def calibration_line(group: str, judged: Agreement) -> dict[str, object]:
    """Return the JSON line of the judge's agreement on ``group``."""
    return {
        "calibration": group,
        "n": judged.n,
        "judge_errors": judged.errors,
        "agreement": judged.share,
        "unsupported_caught": judged.caught,
        "unsupported_labelled": judged.labelled_unsupported,
        "false_alarms": judged.false_alarms,
        "supported_labelled": judged.labelled_supported,
        "kappa": judged.kappa,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--verdicts", type=Path, required=True)
    parser.add_argument("--set", type=Path, default=KNOWLEDGE_SET)
    parser.add_argument(
        "--calibration",
        type=Path,
        nargs="?",
        const=FAITHFULNESS_SET,
        default=None,
        help="also judge this labelled set and print the judge's agreement",
    )
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep the verdicts the verdicts file already holds",
    )
    return parser


async def _judged(
    service: ServiceOpener,
    threads: int | None,
    asked: Sequence[Asked],
    out: Path,
    *,
    resume: bool,
) -> dict[Key, Judged]:
    """Return a verdict for each of ``asked``, starting the model only if one is new."""
    local_service = service(threads)
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        model = local_service.model(client)
        judged: dict[Key, Judged] = {}
        if resume:
            judged = await asyncio.to_thread(
                held_verdicts, out, model.info.id, prompt_sha256()
            )
        if not any(unsettled(judged, item.key) for item in asked):
            return judged
        local_service.start()
        try:
            await ready(model)
            await judge_all(model, asked, out, judged)
        finally:
            local_service.stop()
    return judged


async def _labelled(
    parser: argparse.ArgumentParser, path: Path | None
) -> tuple[Labelled, ...]:
    """Return the labelled answers at ``path``, or none without a path.

    A missing file exits through ``parser`` with how the set is made.
    """
    if path is None:
        return ()
    try:
        return await asyncio.to_thread(read_calibration, path)
    except FileNotFoundError:
        parser.error(
            f"{path} does not exist: the labelled set is built by hand from an"
            " answers run (benchmarks/chat_answers.py)"
        )


async def main(
    argv: Sequence[str] | None = None, *, service: ServiceOpener = installed
) -> int:
    """Judge every answer not judged yet, then print the table.

    ``service`` opens the local model's service that judges.
    """
    parser = _parser()
    arguments = parser.parse_args(argv)
    out: Path = arguments.verdicts
    exists = await asyncio.to_thread(out.exists)
    if exists and not arguments.resume:
        parser.error(f"{out} exists: pass --resume to keep it, or remove it")
    question_set = await asyncio.to_thread(read_set, arguments.set)
    questions = question_set.questions
    answers = await asyncio.to_thread(
        read_answers, arguments.answers, question_set.commit, questions
    )
    labelled = await _labelled(parser, arguments.calibration)
    answered = [answers.by_question[q.id] for q in questions]
    chats = [to_judge(a) for a in answered]
    asked = [item for item in chats if item is not None]
    asked += [i.asked() for i in labelled]
    judged = await _judged(service, arguments.threads, asked, out, resume=exists)
    agreements = calibrated(labelled, judged)
    overall = agreements[0][1] if agreements else None
    outcomes = [
        outcome(q, a, None if item is None else judged[item.key])
        for q, a, item in zip(questions, answered, chats, strict=True)
    ]
    for row in rows(outcomes, overall):
        print(json.dumps(row, allow_nan=False), flush=True)
    for group, judged_group in agreements:
        print(json.dumps(calibration_line(group, judged_group), allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
