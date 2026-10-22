"""Deep search: the local model asks the question again before it is searched.

A question put in other words than the file that answers it shares few of
its words, and the embedder alone does not bridge the vocabulary: on the 12
paraphrase questions of ``benchmarks/data/knowledge_set.json`` no ranking
passed an nDCG@10 of 0.227 (meaning alone, windows of 40 lines), and words
scored 0. So the local model writes up to :data:`MAX_QUERIES` search queries
in the words the files likely use (names, identifiers) and a short passage
drafted as if quoted from a file that answers the question (hypothetical
document embeddings, HyDE). The question, each query and the passage are
searched alike and their rankings merged by
:class:`synthia.knowledge.retrieve.Merge`: by default each chunk keeps its
best score over them.

It costs a generation on the local model, tens of seconds on a CPU, so it
runs only when the person asks for it and is never the default. The model is
handed in directly, never the router, and the request does not allow the
remote model: the question never leaves the machine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Final, Self

from pydantic import BaseModel, ConfigDict, Field

from synthia.gateway.structured import generate
from synthia.gateway.types import ChatRequest, Message, Reasoning
from synthia.knowledge.retrieve import Merge
from synthia.knowledge.service import SEARCH_LIMIT

if TYPE_CHECKING:
    from collections.abc import Callable

    from synthia.gateway.protocol import LocalChatModel
    from synthia.knowledge.retrieve import Found, Variant
    from synthia.knowledge.service import Knowledge

MAX_QUERIES: Final = 3
MAX_QUERY_CHARS: Final = 200
MAX_PASSAGE_CHARS: Final = 1200
INSTRUCTIONS: Final = (
    "You help search the person's own files: notes, documents and code. Given "
    "a question, write up to three short search queries that ask it again in "
    "the words the files are likely to use, such as names, identifiers and "
    "technical terms, each different from the question and from the others. "
    "Then write a passage of three to six sentences, as if quoted from the "
    "person's notes or code, that answers the question. Guess the details if "
    "you must: the passage is only used to search."
)
# An empty example teaches the model to answer empty; this one is full, and
# about nothing the person's files are likely to hold.
EXAMPLE: Final = (
    Message.user("How are expired login sessions cleaned up?"),
    Message.assistant(
        '{"queries": ["purge expired sessions job", "session expires_at cleanup", '
        '"delete stale login sessions"], "passage": "A nightly job, purge_sessions, '
        "deletes every session whose expires_at is in the past. It runs at 03:00 "
        "from the scheduler in jobs/cleanup.py. Each request also checks "
        "expires_at, so an expired session is refused before the purge removes "
        'it."}'
    ),
)


class Rewrites(BaseModel):
    """The model's reply: the question asked again, and a passage answering it."""

    model_config = ConfigDict(extra="forbid")

    queries: list[Annotated[str, Field(max_length=MAX_QUERY_CHARS)]] = Field(
        max_length=MAX_QUERIES
    )
    passage: str = Field(max_length=MAX_PASSAGE_CHARS)


@dataclass(frozen=True, slots=True)
class Expansion:
    """What deep search looks for beside the question: queries and a passage.

    Raises:
        ValueError: If there are more than :data:`MAX_QUERIES` queries, a query
            or the passage is empty or has space around it, or two queries
            differ only in case.
    """

    queries: tuple[str, ...]
    passage: str | None

    def __post_init__(self) -> None:
        if len(self.queries) > MAX_QUERIES:
            message = f"{len(self.queries)} queries, over MAX_QUERIES={MAX_QUERIES}"
            raise ValueError(message)
        for text in self.also():
            if not text or text != text.strip():
                message = f"{text!r} is empty or has space around it"
                raise ValueError(message)
        if len({query.casefold() for query in self.queries}) < len(self.queries):
            message = f"queries={self.queries!r} repeat one another"
            raise ValueError(message)

    @classmethod
    def of(cls, question: str, reply: Rewrites) -> Self:
        """Return ``reply`` cleaned for searching beside ``question``.

        Each text is stripped. A blank query, one that repeats an earlier one
        or the question in any case, and a blank passage are left out.
        """
        seen = {question.strip().casefold()}
        queries: list[str] = []
        for query in reply.queries:
            text = query.strip()
            if text and text.casefold() not in seen:
                seen.add(text.casefold())
                queries.append(text)
        return cls(tuple(queries), reply.passage.strip() or None)

    def also(self) -> tuple[str, ...]:
        """Return what is searched beside the question: queries, then the passage."""
        return self.queries if self.passage is None else (*self.queries, self.passage)


def expansion_request(question: str) -> ChatRequest:
    """Return the request that asks the local model to expand ``question``."""
    return ChatRequest(
        (Message.system(INSTRUCTIONS), *EXAMPLE, Message.user(question)),
        temperature=0.0,
        reasoning=Reasoning.OFF,
        use_remote=False,
    )


async def expand(model: LocalChatModel, question: str) -> Expansion:
    """Return the queries and the passage ``model`` writes for ``question``.

    Raises:
        StructuredOutputError: If the model's replies never match
            :class:`Rewrites`, each shown back with its errors to repair.
        GatewayError: Whatever the model raised.
    """
    reply = await generate(model, expansion_request(question), Rewrites)
    return Expansion.of(question, reply)


@dataclass(frozen=True, slots=True)
class Deep:
    """A deep search's passages, what the model added, and where the time went."""

    found: Found
    expansion: Expansion
    expand_s: float
    search_s: float


async def deep_search(  # noqa: PLR0913 - a search's whole request, and its clock
    knowledge: Knowledge,
    model: LocalChatModel,
    question: str,
    *,
    limit: int = SEARCH_LIMIT,
    variant: Variant | None = None,
    private: bool = True,
    merge: Merge = Merge.BEST,
    clock: Callable[[], float] = time.perf_counter,
) -> Deep:
    """Search ``knowledge`` for ``question`` and for what ``model`` expands it to.

    ``limit``, ``variant`` and ``private`` are as :meth:`Knowledge.search`
    takes them; ``merge`` joins the rankings.

    Raises:
        StructuredOutputError: As :func:`expand` does.
        GatewayError: Whatever the model raised.
        ValueError: If ``limit`` is below 1.
        VariantError: If ``variant`` reranks with a reranker not loaded.
    """
    started = clock()
    expansion = await expand(model, question)
    expanded = clock()
    found = await knowledge.search(
        question,
        limit=limit,
        variant=variant,
        private=private,
        also=expansion.also(),
        merge=merge,
    )
    return Deep(found, expansion, expanded - started, clock() - expanded)
