"""Facts SYNTHIA learns about the person, written down by the local model.

After a turn, the model is shown what the person said, what SYNTHIA
answered, and the five held facts closest to it, numbered, and returns the
lasting facts the exchange adds, each with the number of a shown fact it
changes, if any. A changed fact is kept with the day it stopped being true
and the fact that replaced it. Whether two statements contradict cannot be
read from their vectors ("Ananya lives in Pune" and "Ananya lives in Mumbai"
measured 0.859 with MiniLM, closer than most rewordings of one fact), so
the model judges it, seeing only the facts close enough to matter.

A new fact whose vector is at least 0.90 similar to a held one is the same
fact again and is not kept. On a set of 12 rewordings and 12 related but
different facts, 0.90 merged none of the different ones and caught 5 of the
rewordings; at 0.85, 2 different facts were merged, and a wrong merge loses
a true fact where a missed duplicate only shows twice.

The request never allows the remote model: facts about the person are
written on this machine.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from synthia.gateway.errors import GatewayError
from synthia.gateway.structured import StructuredOutputError, generate
from synthia.gateway.types import ChatRequest, Message, Reasoning
from synthia.memory.store import FactKind
from synthia.memory.vectors import VectorSet

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime

    from synthia.gateway.protocol import ChatModel
    from synthia.memory.embed import TextEmbedder
    from synthia.memory.store import Fact, MemoryStore, Remembered
    from synthia.memory.vectors import Vectors

logger = logging.getLogger(__name__)

SHOWN: Final = 5
BATCH: Final = 8
TRIES: Final = 3
MAX_NEW: Final = 5
MAX_FACT_CHARS: Final = 200
MAX_ANSWER_CHARS: Final = 1000
DUPLICATE: Final = 0.90
INSTRUCTIONS: Final = (
    "You keep a list of lasting facts about the person SYNTHIA talks with. "
    "From the exchange, write each new fact or preference the person states "
    "about themselves, their life, work, people, plans or likes, as one short "
    'sentence in the third person ("The person\'s cat is called Miso."). '
    "Leave out questions, general knowledge, passing moods, and anything only "
    "SYNTHIA said. Do not repeat a known fact. If a new fact changes a known "
    'fact, give that known fact\'s number in "replaces". If the exchange adds '
    "nothing lasting, give an empty list."
)


class NewFact(BaseModel):
    """One fact the model found in an exchange."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=MAX_FACT_CHARS)
    kind: FactKind
    replaces: int | None = None


class Learned(BaseModel):
    """Every fact the model found in an exchange."""

    model_config = ConfigDict(extra="forbid")

    facts: list[NewFact] = Field(max_length=MAX_NEW)


def shown_exchange(question: str, answer: str, known: Sequence[str] = ()) -> str:
    """Return an exchange as the model is shown it, after the known facts."""
    numbered = "\n".join(f"{n}. {text}" for n, text in enumerate(known, 1))
    return (
        f"Known facts:\n{numbered or '(none)'}\n\n"
        f"The person said:\n{question}\n\n"
        f"SYNTHIA answered:\n{answer[:MAX_ANSWER_CHARS]}"
    )


# Without worked examples the model gave an empty list for 11 of 15 exchanges
# that state facts; with these two, for none, and none for 9 with no facts.
EXAMPLES: Final = (
    Message.user(
        shown_exchange(
            "My brother Arjun is a doctor in Delhi, and I'd rather you call me Sam.",
            "Nice to meet you, Sam. A doctor in Delhi must be busy.",
        )
    ),
    Message.assistant(
        '{"facts": ['
        '{"text": "The person\'s brother is called Arjun.", "kind": "fact", '
        '"replaces": null}, '
        '{"text": "Arjun is a doctor in Delhi.", "kind": "fact", "replaces": null}, '
        '{"text": "The person wants to be called Sam.", "kind": "preference", '
        '"replaces": null}]}'
    ),
    Message.user(
        shown_exchange(
            "How long should I boil an egg?", "About nine minutes for a hard yolk."
        )
    ),
    Message.assistant('{"facts": []}'),
)


def learning_request(question: str, answer: str, known: Sequence[Fact]) -> ChatRequest:
    """Return the request that asks what an exchange teaches about the person."""
    exchange = shown_exchange(question, answer, [fact.text for fact in known])
    return ChatRequest(
        (Message.system(INSTRUCTIONS), *EXAMPLES, Message.user(exchange)),
        temperature=0.0,
        reasoning=Reasoning.OFF,
        use_remote=False,
    )


class FactKeeper:
    """Learns facts from finished turns and keeps them with their vectors."""

    def __init__(
        self, store: MemoryStore, embedder: TextEmbedder, model: ChatModel
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._model = model
        self._held = VectorSet(embedder.spec.dimensions)

    async def load(self) -> None:
        """Hold the vectors of the facts still held true, embedding any without one.

        A fact has no vector from this embedder after a change of embedding
        model, or when its vector could not be written.
        """
        model = self._embedder.spec.id
        held = {fact.id: fact for fact in await self._store.facts()}
        for kept in await self._store.fact_vectors(model):
            vector = np.frombuffer(kept.vector, np.float32)
            self._held.add(kept.fact, held[kept.fact].since, vector)
        missing = [fact for fact in held.values() if fact.id not in self._held]
        if not missing:
            return
        vectors = await self._embedder.embed([fact.text for fact in missing])
        pairs = list(zip(missing, vectors, strict=True))
        await self._store.put_fact_vectors(
            model, [(fact.id, vector.tobytes()) for fact, vector in pairs]
        )
        for fact, vector in pairs:
            self._held.add(fact.id, fact.since, vector)

    async def learn(
        self, turn: int, at: datetime, question: str, answer: str
    ) -> list[int]:
        """Learn from ``turn`` and return the ids of the facts kept.

        Raises:
            GatewayError: If the local model failed or gave no valid answer.
        """
        held = {fact.id: fact for fact in await self._store.facts()}
        for gone in [fact for fact in self._held if fact not in held]:
            self._held.remove(gone)
        known = await self._closest(question, held)
        request = learning_request(question, answer, known)
        learned = (await generate(self._model, request, Learned)).facts
        if not learned:
            return []
        vectors = await self._embedder.embed([new.text for new in learned])
        kept: list[int] = []
        for new, vector in zip(learned, vectors, strict=True):
            replaced = _shown(known, new.replaces)
            same = self._repeated(vector, besides=replaced)
            if same is not None:
                if replaced is not None:
                    await self._end(replaced, at, same)
                continue
            fact = await self._store.add_fact(turn, new.text, new.kind, at)
            if fact is None:
                break
            await self._store.put_fact_vectors(
                self._embedder.spec.id, [(fact, vector.tobytes())]
            )
            self._held.add(fact, at, vector)
            if replaced is not None:
                await self._end(replaced, at, fact)
            kept.append(fact)
        return kept

    async def _closest(self, question: str, held: Mapping[int, Fact]) -> list[Fact]:
        (vector,) = await self._embedder.embed([question])
        near = self._held.similarities(vector).nearest(SHOWN)
        return [held[fact] for fact in sorted(near, key=lambda fact: -near[fact])]

    def _repeated(self, vector: Vectors, *, besides: Fact | None) -> int | None:
        """Return a held fact ``vector`` says again, other than ``besides``."""
        skip = None if besides is None else besides.id
        near = self._held.similarities(vector).nearest(2)
        return next(
            (
                fact
                for fact, cosine in near.items()
                if fact != skip and cosine >= DUPLICATE
            ),
            None,
        )

    async def _end(self, fact: Fact, at: datetime, replaced_by: int) -> None:
        await self._store.end_fact(fact.id, at, replaced_by)
        self._held.remove(fact.id)


class FactLearning:
    """Learns from the turns waiting in the store, one at a time, in the background.

    The waiting turns are a table, not a queue in memory, so a stop or a
    crash loses none: whatever is left is learned from after the next start,
    a turn cut off part way included, so its first facts may be found twice.
    A turn is dropped only when the model's answer is never valid, or after
    :data:`TRIES` failures of another kind.
    """

    def __init__(self, keeper: FactKeeper, store: MemoryStore) -> None:
        self._keeper = keeper
        self._store = store
        self._waiting = asyncio.Event()
        self._waiting.set()

    def wake(self) -> None:
        """Say that a turn is waiting to be learned from."""
        self._waiting.set()

    async def run(self) -> None:
        """Learn from each waiting turn, oldest first, until cancelled."""
        while True:
            await self._waiting.wait()
            self._waiting.clear()
            try:
                await self._drain()
            except Exception:
                logger.exception("facts were not learned; trying at the next turn")

    async def _drain(self) -> None:
        while waiting := await self._store.to_learn(BATCH):
            for turn in waiting:
                if not await self._learn(turn):
                    return

    async def _learn(self, turn: Remembered) -> bool:
        """Learn from ``turn``; False when it should be tried again later."""
        try:
            await self._keeper.learn(turn.id, turn.at, turn.question, turn.answer)
        except StructuredOutputError:
            logger.warning("no facts from turn %d: the answer was not valid", turn.id)
        except GatewayError:
            logger.warning("facts wait for the local model", exc_info=True)
            return False
        except Exception:
            if await self._store.tried(turn.id, most=TRIES):
                logger.exception("no facts from turn %d after %d tries", turn.id, TRIES)
                return True
            logger.exception("facts from turn %d wait for another try", turn.id)
            return False
        await self._store.learned(turn.id)
        return True


def _shown(known: Sequence[Fact], number: int | None) -> Fact | None:
    """Return the shown fact numbered ``number``, counting from 1, if there is one."""
    if number is None or not 1 <= number <= len(known):
        return None
    return known[number - 1]
