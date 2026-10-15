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
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from synthia.gateway.errors import GatewayError
from synthia.gateway.structured import StructuredOutputError, generate
from synthia.gateway.types import ChatRequest, Message, Reasoning
from synthia.memory.store import FactKind
from synthia.retrieval.vectors import VectorSet

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from synthia.gateway.protocol import ChatModel
    from synthia.memory.store import Fact, MemoryStore, Remembered
    from synthia.retrieval.embed import TextEmbedder
    from synthia.retrieval.vectors import Vectors

logger = logging.getLogger(__name__)

SHOWN: Final = 5
BATCH: Final = 8
TRIES: Final = 3
MAX_NEW: Final = 5
MAX_FACT_CHARS: Final = 200
MAX_ANSWER_CHARS: Final = 1000
DUPLICATE: Final = 0.90
MERGE_FLOOR: Final = 0.80
IDLE_S: Final = 600.0
CONSOLIDATE_EVERY_S: Final = 3600.0
CANCELLED: Final = "cancelled"
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


SAME_INSTRUCTIONS: Final = (
    "You decide whether two facts SYNTHIA keeps about the person say the same "
    "thing, so that keeping both would be a repeat. They are the same only if "
    "every detail agrees: who, what, where, when, how much, and whether it is "
    "true now. Wording does not matter."
)


def shown_pair(first: str, second: str) -> str:
    """Return two facts as the model is shown them to compare."""
    return f"1. {first}\n2. {second}"


SAME_EXAMPLES: Final = (
    Message.user(
        shown_pair(
            "The person has two children.", "The person is a parent of two kids."
        )
    ),
    Message.assistant('{"same": true}'),
    Message.user(
        shown_pair(
            "The person is learning Spanish.", "The person speaks fluent Spanish."
        )
    ),
    Message.assistant('{"same": false}'),
)


class Same(BaseModel):
    """Whether two facts are one."""

    model_config = ConfigDict(extra="forbid")

    same: bool


def same_request(first: str, second: str) -> ChatRequest:
    """Return the request that asks whether two facts say the same thing."""
    return ChatRequest(
        (
            Message.system(SAME_INSTRUCTIONS),
            *SAME_EXAMPLES,
            Message.user(shown_pair(first, second)),
        ),
        temperature=0.0,
        reasoning=Reasoning.OFF,
        use_remote=False,
    )


@dataclass(slots=True)
class Merged:
    """A merge pass so far: pairs asked about, facts merged, whether it stopped."""

    asked: int = 0
    merged: int = 0
    stopped: bool = False


@dataclass(frozen=True, slots=True)
class Consolidated:
    """One consolidation run: its merge pass, how long it took, and how it ended.

    ``stopped`` is set when a scheduled run gave way to a turn; ``error`` says
    why a run failed. Merges made before a failure stay made.
    """

    asked: int
    merged: int
    seconds: float
    scheduled: bool
    stopped: bool
    error: str | None = None


def _never() -> bool:
    return False


def _named(error: Exception) -> str:
    return str(error) or type(error).__name__


def _ignore(_: Consolidated) -> None:
    pass


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
        known = await self._closest(question, await self._reconciled())
        request = learning_request(question, answer, known)
        learned = (await generate(self._model, request, Learned)).facts
        if not learned:
            return []
        vectors = await self._embedder.embed([new.text for new in learned])
        # The person may have forgotten facts while the model was answering.
        await self._reconciled()
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

    async def _reconciled(self) -> dict[int, Fact]:
        """Return the facts held, after letting go of the vectors of any forgotten."""
        held = {fact.id: fact for fact in await self._store.facts()}
        for gone in [fact for fact in self._held if fact not in held]:
            self._held.remove(gone)
        return held

    async def _closest(self, question: str, held: Mapping[int, Fact]) -> list[Fact]:
        (vector,) = await self._embedder.embed([question])
        near = self._held.similarities(vector).nearest(SHOWN)
        closest = sorted(near, key=lambda fact: (-near[fact], fact))[:SHOWN]
        return [held[fact] for fact in closest]

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

    async def merge_repeats(
        self,
        at: datetime,
        tally: Merged | None = None,
        *,
        busy: Callable[[], bool] = _never,
    ) -> Merged:
        """End the older of each pair of held facts the local model calls one.

        Only pairs at least :data:`MERGE_FLOOR` similar are asked about, most
        similar first, each fact in one merge at most; a pair is one only if
        the model says so with the facts in either order. A pair it called
        two facts is remembered and never asked about again: each question
        takes seconds of the local model, and each repeat is another chance
        of a wrong yes, which would lose a true fact.

        The pass counts into ``tally`` as it goes, so a pass ended by an
        error still says what it did, and stops before its next question
        once ``busy`` says so.

        Raises:
            GatewayError: If the local model failed.
        """
        tally = Merged() if tally is None else tally
        held = await self._reconciled()
        apart = await self._store.apart()
        done = set[int]()
        for one, other in self._held.close_pairs(MERGE_FLOOR):
            pair = (min(one, other), max(one, other))
            if one in done or other in done or pair in apart:
                continue
            if busy():
                tally.stopped = True
                break
            first, second = held[one], held[other]
            tally.asked += 1
            same = await self._same(first.text, second.text)
            if same is None:
                continue
            if not same:
                await self._store.keep_apart(first, second)
                continue
            older, newer = sorted(
                (first, second), key=lambda fact: (fact.since, fact.id)
            )
            if await self._store.end_fact(older.id, at, newer.id):
                self._held.remove(older.id)
                tally.merged += 1
            done |= {first.id, second.id}
        return tally

    async def _same(self, first: str, second: str) -> bool | None:
        """Return whether the model calls the two one; None with no valid answer."""
        for one, other in ((first, second), (second, first)):
            try:
                if not (
                    await generate(self._model, same_request(one, other), Same)
                ).same:
                    return False
            except StructuredOutputError:
                logger.warning("no valid answer on whether two facts are one")
                return None
        return True


class FactLearning:
    """Learns from the turns waiting in the store, one at a time, in the background.

    The waiting turns are a table, not a queue in memory, so a stop or a
    crash loses none: whatever is left is learned from after the next start,
    a turn cut off part way included, so its first facts may be found twice.
    A turn is dropped only when the model's answer is never valid, or after
    :data:`TRIES` failures of another kind.

    It also consolidates the facts held: when no turn has come for
    ``idle_s``, at most once every ``every_s``, and only if a turn was
    remembered since the last run or that run failed. Learning and
    consolidating take turns under one lock: both ask the local model and
    change the same facts.
    """

    def __init__(  # noqa: PLR0913 - the schedule's parts are injected for tests
        self,
        keeper: FactKeeper,
        store: MemoryStore,
        *,
        report: Callable[[Consolidated], None] = _ignore,
        idle_s: float = IDLE_S,
        every_s: float = CONSOLIDATE_EVERY_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._keeper = keeper
        self._store = store
        self._report = report
        self._idle_s = idle_s
        self._every_s = every_s
        self._clock = clock
        self._lock = asyncio.Lock()
        self._waiting = asyncio.Event()
        self._waiting.set()
        self._last: float | None = None
        self._changed = True

    def wake(self) -> None:
        """Say that a turn is waiting to be learned from."""
        self._waiting.set()

    async def run(self) -> None:
        """Learn from each waiting turn, oldest first, and consolidate when idle.

        Runs until cancelled.
        """
        while True:
            try:
                await asyncio.wait_for(self._waiting.wait(), timeout=self._idle_s)
            except TimeoutError:
                if self._due():
                    await self.consolidate(scheduled=True)
                continue
            self._waiting.clear()
            try:
                async with self._lock:
                    self._changed = True
                    await self._drain()
            except Exception:
                logger.exception("facts were not learned; trying at the next turn")

    def _due(self) -> bool:
        now = self._clock()
        return self._changed and (
            self._last is None or now - self._last >= self._every_s
        )

    async def consolidate(self, *, scheduled: bool = False) -> Consolidated:
        """Merge the held facts the local model calls one, then tidy the indexes.

        Waits while a turn is learned from. A ``scheduled`` run gives way
        before its next question once a turn is waiting; the pairs it left
        are asked at the next run. Never fails: a failure is logged and
        named in :attr:`Consolidated.error`. A run that is cancelled is
        reported as such before the cancellation goes on.
        """
        async with self._lock:
            started = self._last = self._clock()
            tally = Merged()
            error: str | None = CANCELLED
            try:
                await self._keeper.merge_repeats(
                    datetime.now(UTC),
                    tally,
                    busy=self._waiting.is_set if scheduled else _never,
                )
                await self._store.optimize()
                error = None
            except GatewayError as failed:
                logger.warning("facts were not consolidated: %s", failed)
                error = _named(failed)
            except Exception as failed:
                logger.exception("facts were not consolidated")
                error = _named(failed)
            finally:
                if error is None:
                    self._changed = False
                done = Consolidated(
                    tally.asked,
                    tally.merged,
                    self._clock() - started,
                    scheduled,
                    tally.stopped,
                    error,
                )
                self._reported(done)
        return done

    def _reported(self, done: Consolidated) -> None:
        try:
            self._report(done)
        except Exception:
            logger.exception("a consolidation was not reported")

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
