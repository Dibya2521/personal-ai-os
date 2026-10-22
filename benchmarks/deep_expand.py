"""Expand every question of the question set with the local model, as deep search does.

    uv run python benchmarks/deep_expand.py --out PATH [--threads N] [--resume]

Each question of ``benchmarks/data/knowledge_set.json``, in order, goes
through :func:`synthia.knowledge.deep.expand` on the installed local model,
the code deep search runs. The file written is the input of
``benchmarks/rag_eval.py --expansions``::

    {
        "model": "<model id>",
        "questions": {
            "<question id>": {
                "question": "<the text expanded>",
                "queries": [...],
                "passage": "..." or null,
                "seconds": 12.3,
            }
        },
    }

It is written again after every question, so a crash loses one question,
and ``--resume`` skips the questions it already holds with the same text
(one whose text changed is expanded again). A progress line per
question goes to stderr. Each question costs a local generation, tens of
seconds on a CPU. It reaches no outside service and spends no budget.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final, TextIO, TypedDict

import httpx
from fact_extraction import TIMEOUT, installed, ready
from rag_eval import KNOWLEDGE_SET, SetError, parse_expansions, read_set

from synthia.knowledge.deep import expand

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from rag_eval import Question

    from synthia.gateway.protocol import LocalChatModel

DESCRIPTION: Final = "Expand every question of the set with the local model."
DIGITS: Final = 3


class Expanded(TypedDict):
    """One question's text, queries, passage and seconds, as the file holds it."""

    question: str
    queries: list[str]
    passage: str | None
    seconds: float


def held(path: Path, model: str) -> dict[str, Expanded]:
    """Return the expansions already in ``path`` by question id; blocking.

    Raises:
        SetError: If the file is not expansions, or ``model`` did not write it.
    """
    expansions = parse_expansions(path.read_bytes())
    if expansions.model != model:
        message = f"{path.name} holds {expansions.model}'s expansions, not {model}'s"
        raise SetError(message)
    return {
        question: {
            "question": expansions.texts[question],
            "queries": list(expansion.queries),
            "passage": expansion.passage,
            "seconds": expansions.seconds[question],
        }
        for question, expansion in expansions.by_question.items()
    }


def written(path: Path, model: str, expanded: dict[str, Expanded]) -> None:
    """Replace ``path`` with the expansions, so a crash never leaves half a file."""
    text = json.dumps(
        {"model": model, "questions": expanded},
        ensure_ascii=True,
        allow_nan=False,
        indent=2,
    )
    partial = path.with_name(f"{path.name}.partial")
    partial.write_text(f"{text}\n", encoding="utf-8", newline="")
    partial.replace(path)


async def expand_all(  # noqa: PLR0913 - the run's whole configuration
    model: LocalChatModel,
    questions: Sequence[Question],
    out: Path,
    expanded: dict[str, Expanded],
    *,
    progress: TextIO = sys.stderr,
    clock: Callable[[], float] = time.perf_counter,
) -> None:
    """Expand each question not held with its text, writing ``out`` after each."""
    await asyncio.to_thread(out.parent.mkdir, parents=True, exist_ok=True)
    for n, question in enumerate(questions, 1):
        held_already = expanded.get(question.id)
        if held_already is not None and held_already["question"] == question.text:
            continue
        started = clock()
        expansion = await expand(model, question.text)
        seconds = round(clock() - started, DIGITS)
        expanded[question.id] = {
            "question": question.text,
            "queries": list(expansion.queries),
            "passage": expansion.passage,
            "seconds": seconds,
        }
        await asyncio.to_thread(written, out, model.info.id, expanded)
        progress.write(
            f"{n}/{len(questions)} {question.id} {seconds:.1f} s,"
            f" {len(expansion.queries)} queries\n"
        )
        progress.flush()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--set", type=Path, default=KNOWLEDGE_SET)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep the questions the out file already expands",
    )
    return parser


async def main(argv: Sequence[str] | None = None) -> int:
    """Write the expansion of every question, one question at a time."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    out: Path = arguments.out
    exists = await asyncio.to_thread(out.exists)
    if exists and not arguments.resume:
        parser.error(f"{out} exists: pass --resume to keep it, or remove it")
    question_set = await asyncio.to_thread(read_set, arguments.set)
    service = installed(arguments.threads)
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        model = service.model(client)
        expanded = await asyncio.to_thread(held, out, model.info.id) if exists else {}
        service.start()
        try:
            await ready(model)
            await expand_all(model, question_set.questions, out, expanded)
        finally:
            service.stop()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
