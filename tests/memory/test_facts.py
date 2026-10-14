import asyncio
import contextlib
import json
import logging
import sqlite3
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from synthia.gateway.errors import GatewayError
from synthia.gateway.structured import DEFAULT_REPAIRS
from synthia.gateway.types import ChatChunk, ChatRequest, Reasoning, Role
from synthia.memory.facts import (
    EXAMPLES,
    INSTRUCTIONS,
    SAME_EXAMPLES,
    SAME_INSTRUCTIONS,
    TRIES,
    FactKeeper,
    FactLearning,
    Learned,
    Same,
)
from synthia.memory.store import FactKind, MemoryStore, Remembered, Turn
from tests.agent.scripted import Scripted, says
from tests.memory.test_hybrid import Network, embedder
from tests.memory.test_remembering import FailsAfterStart
from tests.models.fakes import EMBEDDER

MONDAY = datetime(2026, 9, 28, 9, 0, tzinfo=UTC)
LATER = MONDAY + timedelta(days=2)


def found(*facts: tuple[str, str] | tuple[str, str, int]) -> list[ChatChunk]:
    return says(
        json.dumps(
            {
                "facts": [
                    {
                        "text": f[0],
                        "kind": f[1],
                        "replaces": f[2] if len(f) > 2 else None,
                    }
                    for f in facts
                ]
            }
        )
    )


class Watched(MemoryStore):
    """A store that says when a waiting turn has been dealt with."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.changed = asyncio.Event()

    async def learned(self, turn: int) -> bool:
        done = await super().learned(turn)
        self.changed.set()
        return done

    async def tried(self, turn: int, *, most: int) -> bool:
        dropped = await super().tried(turn, most=most)
        self.changed.set()
        return dropped


class FailsOnce(Watched):
    """A store whose first look for waiting turns fails."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.failed = False

    async def to_learn(self, limit: int) -> list[Remembered]:
        if not self.failed:
            self.failed = True
            self.changed.set()
            message = "database is locked"
            raise sqlite3.OperationalError(message)
        return await super().to_learn(limit)


@pytest.fixture
def store(tmp_path: Path) -> Watched:
    return Watched(tmp_path / "memory.db")


async def a_turn(store: MemoryStore, question: str) -> int:
    chat = await store.begin_conversation("SYNTHIA", MONDAY)
    return await store.add_turn(
        chat, Turn(question, "Noted.", MONDAY, "SYNTHIA", "local", "m")
    )


async def keeper(store: MemoryStore, model: Scripted) -> FactKeeper:
    kept = FactKeeper(store, embedder(Network()), model)
    await kept.load()
    return kept


async def held(store: MemoryStore) -> list[tuple[str, FactKind]]:
    return [(fact.text, fact.kind) for fact in await store.facts()]


async def test_facts_found_in_an_exchange_are_kept_with_their_vectors(
    store: MemoryStore,
) -> None:
    turn = await a_turn(store, "my cat is called miso")
    model = Scripted(
        found(("my cat is called miso", "fact"), ("likes green tea", "preference"))
    )

    kept = await (await keeper(store, model)).learn(
        turn, MONDAY, "my cat is called miso", "Noted."
    )

    assert len(kept) == 2
    assert await held(store) == [
        ("likes green tea", FactKind.PREFERENCE),
        ("my cat is called miso", FactKind.FACT),
    ]
    assert [v.fact for v in await store.fact_vectors(EMBEDDER.id)] == kept


async def test_the_request_shows_the_exchange_and_never_leaves_the_machine(
    store: MemoryStore,
) -> None:
    turn = await a_turn(store, "hello")
    model = Scripted(found())

    await (await keeper(store, model)).learn(turn, MONDAY, "hello", "a" * 1500)

    (request,) = model.requests
    system, *examples, user = request.messages
    assert system.role is Role.SYSTEM
    assert tuple(examples) == EXAMPLES
    assert user.text == (
        "Known facts:\n(none)\n\nThe person said:\nhello\n\n"
        f"SYNTHIA answered:\n{'a' * 1000}"
    )
    assert (request.use_remote, request.reasoning, request.temperature) == (
        False,
        Reasoning.OFF,
        0.0,
    )
    assert request.response_schema is not None


def test_the_worked_examples_answer_in_the_form_asked_for() -> None:
    asked, answered = EXAMPLES[0::2], EXAMPLES[1::2]

    replies = [Learned.model_validate_json(m.text) for m in answered]

    assert '{"facts": []}' not in INSTRUCTIONS
    assert [m.role for m in asked] == [Role.USER, Role.USER]
    assert [len(r.facts) for r in replies] == [3, 0]
    assert [f.kind for f in replies[0].facts] == [
        FactKind.FACT,
        FactKind.FACT,
        FactKind.PREFERENCE,
    ]


async def test_a_fact_that_changes_a_shown_one_replaces_it(store: MemoryStore) -> None:
    first = await a_turn(store, "my sister lives in pune")
    second = await a_turn(store, "my sister lives in mumbai now")
    model = Scripted(
        found(("my sister lives in pune", "fact")),
        found(("my sister lives in mumbai", "fact", 1)),
    )
    facts = await keeper(store, model)
    (pune,) = await facts.learn(first, MONDAY, "my sister lives in pune", "Noted.")

    (mumbai,) = await facts.learn(
        second, LATER, "my sister lives in mumbai now", "Noted."
    )

    assert (
        "Known facts:\n1. my sister lives in pune\n"
        in model.requests[1].messages[-1].text
    )
    assert await held(store) == [("my sister lives in mumbai", FactKind.FACT)]
    old = next(f for f in await store.facts(held=False) if f.id == pune)
    assert (old.until, old.replaced_by) == (LATER, mumbai)
    assert [v.fact for v in await store.fact_vectors(EMBEDDER.id)] == [mumbai]


async def test_a_fact_said_again_is_not_kept_twice(store: MemoryStore) -> None:
    turn = await a_turn(store, "my cat is called miso")
    model = Scripted(
        found(("my cat is called miso", "fact")),
        found(("my cat is called miso", "fact")),
    )
    facts = await keeper(store, model)
    await facts.learn(turn, MONDAY, "my cat is called miso", "Noted.")

    again = await facts.learn(turn, LATER, "my cat is called miso", "Noted.")

    assert again == []
    assert await held(store) == [("my cat is called miso", FactKind.FACT)]


async def test_a_fact_said_twice_in_one_reply_is_kept_once(store: MemoryStore) -> None:
    turn = await a_turn(store, "my cat is called miso")
    model = Scripted(
        found(("my cat is called miso", "fact"), ("my cat is called miso", "fact"))
    )

    kept = await (await keeper(store, model)).learn(
        turn, MONDAY, "my cat is called miso", "Noted."
    )

    assert len(kept) == 1
    assert await held(store) == [("my cat is called miso", FactKind.FACT)]


async def test_a_replacement_is_kept_however_close_to_the_fact_it_changes(
    store: MemoryStore,
) -> None:
    turn = await a_turn(store, "my cat is called miso")
    model = Scripted(
        found(("my cat is called miso", "fact")),
        found(("my cat is called mochi", "fact", 1)),
    )
    facts = await keeper(store, model)
    await facts.learn(turn, MONDAY, "my cat is called miso", "Noted.")

    (mochi,) = await facts.learn(turn, LATER, "my cat is called mochi", "Noted.")

    assert await held(store) == [("my cat is called mochi", FactKind.FACT)]
    assert [v.fact for v in await store.fact_vectors(EMBEDDER.id)] == [mochi]


async def test_a_forgotten_fact_is_neither_shown_nor_in_the_way_of_learning_it_again(
    store: MemoryStore,
) -> None:
    turn = await a_turn(store, "my cat is called miso")
    model = Scripted(
        found(("my cat is called miso", "fact")),
        found(("my cat is called miso", "fact")),
    )
    facts = await keeper(store, model)
    (miso,) = await facts.learn(turn, MONDAY, "my cat is called miso", "Noted.")
    await store.forget_fact(miso)

    again = await facts.learn(turn, LATER, "my cat is called miso", "Noted.")

    assert model.requests[1].messages[-1].text.startswith("Known facts:\n(none)\n")
    assert len(again) == 1
    assert await held(store) == [("my cat is called miso", FactKind.FACT)]


async def test_a_replacement_that_repeats_another_fact_ends_the_old_one_by_it(
    store: MemoryStore,
) -> None:
    turn = await a_turn(store, "my sister lives in pune")
    model = Scripted(
        found(
            ("my sister lives in pune", "fact"), ("my sister lives in mumbai", "fact")
        ),
        found(("my sister lives in mumbai", "fact", 1)),
    )
    facts = await keeper(store, model)
    pune, mumbai = await facts.learn(turn, MONDAY, "my sister lives in pune", "Noted.")

    again = await facts.learn(turn, LATER, "my sister lives in pune", "Noted.")

    assert "Known facts:\n1. my sister lives in pune\n" in (
        model.requests[1].messages[-1].text
    )
    assert again == []
    assert await held(store) == [("my sister lives in mumbai", FactKind.FACT)]
    old = next(f for f in await store.facts(held=False) if f.id == pune)
    assert (old.until, old.replaced_by) == (LATER, mumbai)


class ForgetsWhileAnswering(Scripted):
    """A model during whose answer the person forgets ``fact``."""

    def __init__(self, store: MemoryStore, *replies: list[ChatChunk]) -> None:
        super().__init__(*replies)
        self.store = store
        self.fact: int | None = None

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        if self.fact is not None:
            await self.store.forget_fact(self.fact)
        async for chunk in super().stream(request):
            yield chunk


async def test_a_fact_forgotten_while_the_model_answers_is_not_repeated_by_its_answer(
    store: MemoryStore,
) -> None:
    turn = await a_turn(store, "my sister lives in pune")
    model = ForgetsWhileAnswering(
        store,
        found(
            ("my sister lives in pune", "fact"), ("my sister lives in mumbai", "fact")
        ),
        found(("my sister lives in mumbai", "fact", 1)),
    )
    facts = await keeper(store, model)
    pune, model.fact = await facts.learn(
        turn, MONDAY, "my sister lives in pune", "Noted."
    )

    (mumbai,) = await facts.learn(turn, LATER, "my sister lives in pune", "Noted.")

    assert await held(store) == [("my sister lives in mumbai", FactKind.FACT)]
    old = next(f for f in await store.facts(held=False) if f.id == pune)
    assert (old.until, old.replaced_by) == (LATER, mumbai)


def test_the_worked_examples_of_one_fact_answer_in_the_form_asked_for() -> None:
    replies = [m.text for m in SAME_EXAMPLES if m.role is Role.ASSISTANT]

    assert [Same.model_validate_json(r).same for r in replies] == [True, False]


async def held_facts(store: MemoryStore, *texts: str) -> list[int]:
    """Keep ``texts`` as facts a day apart, oldest first, without learning them."""
    turn = await a_turn(store, "about me")
    ids: list[int] = []
    for day, text in enumerate(texts):
        fact = await store.add_fact(turn, text, FactKind.FACT, MONDAY + timedelta(day))
        assert fact is not None
        ids.append(fact)
    return ids


SAME = says('{"same": true}')
NOT_SAME = says('{"same": false}')


async def test_two_facts_the_model_calls_one_in_both_orders_are_merged(
    store: MemoryStore,
) -> None:
    older, newer, _ = await held_facts(
        store, "my cat is called miso", "my cat miso", "likes green tea"
    )
    model = Scripted(SAME, SAME)
    facts = await keeper(store, model)

    merged = await facts.merge_repeats(LATER)

    assert (merged.asked, merged.merged) == (1, 1)
    assert sorted(m.messages[-1].text for m in model.requests) == [
        "1. my cat is called miso\n2. my cat miso",
        "1. my cat miso\n2. my cat is called miso",
    ]
    assert {f.id for f in await store.facts()} == {newer, 3}
    old = next(f for f in await store.facts(held=False) if f.id == older)
    assert (old.until, old.replaced_by) == (LATER, newer)


async def test_whether_two_facts_are_one_is_asked_on_this_machine_without_chance(
    store: MemoryStore,
) -> None:
    await held_facts(store, "likes green tea", "likes green tea")
    model = Scripted(SAME, SAME)

    await (await keeper(store, model)).merge_repeats(LATER)

    request = model.requests[0]
    assert request.messages[0].text == SAME_INSTRUCTIONS
    assert (request.use_remote, request.temperature) == (False, 0.0)
    assert request.reasoning is Reasoning.OFF


@pytest.mark.parametrize(
    ("replies", "asked"),
    [((NOT_SAME,), 1), ((SAME, NOT_SAME), 2)],
    ids=["no-at-once", "no-the-other-way"],
)
async def test_two_facts_are_kept_unless_the_model_says_one_both_ways(
    store: MemoryStore, replies: tuple[list[ChatChunk], ...], asked: int
) -> None:
    await held_facts(store, "likes green tea", "likes green tea")
    model = Scripted(*replies)

    merged = await (await keeper(store, model)).merge_repeats(LATER)

    assert (merged.asked, merged.merged) == (1, 0)
    assert len(model.requests) == asked
    assert len(await store.facts()) == 2


async def test_an_answer_that_is_never_valid_keeps_both_facts(
    store: MemoryStore, caplog: pytest.LogCaptureFixture
) -> None:
    await held_facts(store, "likes green tea", "likes green tea")
    model = Scripted(*[says("maybe")] * (1 + DEFAULT_REPAIRS))

    with caplog.at_level(logging.WARNING, "synthia.memory.facts"):
        merged = await (await keeper(store, model)).merge_repeats(LATER)

    assert (merged.asked, merged.merged) == (1, 0)
    assert "no valid answer on whether two facts are one" in caplog.text


async def test_each_fact_is_in_one_merge_at_most_per_pass(store: MemoryStore) -> None:
    await held_facts(store, "likes green tea", "likes green tea", "likes green tea")
    model = Scripted(SAME, SAME)

    merged = await (await keeper(store, model)).merge_repeats(LATER)

    assert (merged.asked, merged.merged) == (1, 1)
    assert len(await store.facts()) == 2


async def test_a_pair_called_two_facts_is_not_asked_about_again(
    store: MemoryStore,
) -> None:
    tea, also_tea = await held_facts(store, "likes green tea", "likes green tea")
    model = Scripted(SAME, NOT_SAME)
    facts = await keeper(store, model)

    first = await facts.merge_repeats(LATER)
    second = await facts.merge_repeats(LATER)

    assert (first.asked, second.asked) == (1, 0)
    assert len(model.requests) == 2
    assert await store.apart() == {(tea, also_tea)}


async def test_a_pair_with_no_valid_answer_is_asked_about_again(
    store: MemoryStore,
) -> None:
    await held_facts(store, "likes green tea", "likes green tea")
    model = Scripted(*[says("maybe")] * (1 + DEFAULT_REPAIRS), NOT_SAME)
    facts = await keeper(store, model)

    first = await facts.merge_repeats(LATER)
    second = await facts.merge_repeats(LATER)

    assert (first.asked, second.asked) == (1, 1)
    assert len(await store.apart()) == 1


async def test_a_fact_forgotten_while_its_pair_is_judged_is_not_kept_apart(
    store: MemoryStore,
) -> None:
    tea, also_tea = await held_facts(store, "likes green tea", "likes green tea")
    model = ForgetsWhileAnswering(store, NOT_SAME)
    model.fact = also_tea

    merged = await (await keeper(store, model)).merge_repeats(LATER)

    assert (merged.asked, merged.merged) == (1, 0)
    assert [f.id for f in await store.facts()] == [tea]
    assert await store.apart() == set()


class ReplacedWhileAnswering(ForgetsWhileAnswering):
    """A model during whose answer ``fact`` is forgotten and its id given again."""

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        if self.fact is not None:
            await self.store.forget_fact(self.fact)
            turn = await a_turn(self.store, "about me")
            await self.store.add_fact(turn, "likes coffee", FactKind.PREFERENCE, LATER)
            self.fact = None
        async for chunk in Scripted.stream(self, request):
            yield chunk


async def test_a_judgement_is_not_kept_for_a_fact_that_took_a_judged_id(
    store: MemoryStore,
) -> None:
    _, also_tea = await held_facts(store, "likes green tea", "likes green tea")
    model = ReplacedWhileAnswering(store, NOT_SAME)
    model.fact = also_tea

    await (await keeper(store, model)).merge_repeats(LATER)

    assert also_tea in {f.id for f in await store.facts()}
    assert await store.apart() == set()


async def test_a_model_that_fails_stops_the_pass(store: MemoryStore) -> None:
    await held_facts(store, "likes green tea", "likes green tea")

    def down(_: ChatRequest) -> list[ChatChunk]:
        message = "the local model is not running"
        raise GatewayError(message)

    with pytest.raises(GatewayError, match="the local model is not running"):
        await (await keeper(store, Scripted(down))).merge_repeats(LATER)

    assert len(await store.facts()) == 2


async def test_a_number_that_was_not_shown_replaces_nothing(store: MemoryStore) -> None:
    turn = await a_turn(store, "my sister lives in pune")
    model = Scripted(
        found(("my sister lives in pune", "fact")),
        found(("my cat is called miso", "fact", 7)),
    )
    facts = await keeper(store, model)
    await facts.learn(turn, MONDAY, "my sister lives in pune", "Noted.")

    await facts.learn(turn, LATER, "my sister lives in pune", "Noted.")

    assert await held(store) == [
        ("my cat is called miso", FactKind.FACT),
        ("my sister lives in pune", FactKind.FACT),
    ]


async def test_nothing_is_kept_from_a_turn_forgotten_meanwhile(
    store: MemoryStore,
) -> None:
    turn = await a_turn(store, "my cat is called miso")
    await store.forget_turn(turn)
    model = Scripted(found(("my cat is called miso", "fact")))

    kept = await (await keeper(store, model)).learn(
        turn, MONDAY, "my cat is called miso", "Noted."
    )

    assert kept == []
    assert await store.facts(held=False) == []


async def test_an_exchange_with_nothing_lasting_adds_nothing(
    store: MemoryStore,
) -> None:
    turn = await a_turn(store, "thanks")
    model = Scripted(found())

    assert await (await keeper(store, model)).learn(turn, MONDAY, "thanks", "Ok.") == []
    assert await store.facts() == []


async def test_facts_held_before_are_loaded_and_changed_ones_left_out(
    store: MemoryStore,
) -> None:
    first = await a_turn(store, "my sister lives in pune")
    earlier = Scripted(
        found(("my sister lives in pune", "fact")),
        found(("my sister lives in mumbai", "fact", 1)),
    )
    facts = await keeper(store, earlier)
    await facts.learn(first, MONDAY, "my sister lives in pune", "Noted.")
    await facts.learn(first, LATER, "my sister lives in pune", "Noted.")
    model = Scripted(
        found(("my sister lives in mumbai", "fact")),
        found(("my sister lives in pune", "fact")),
    )
    restarted = await keeper(store, model)

    assert await restarted.learn(first, LATER, "mumbai", "Noted.") == []
    assert len(await restarted.learn(first, LATER, "pune", "Noted.")) == 1


async def test_facts_without_a_vector_are_embedded_when_loaded(
    store: MemoryStore,
) -> None:
    turn = await a_turn(store, "my cat is called miso")
    fact = await store.add_fact(turn, "my cat is called miso", FactKind.FACT, MONDAY)
    model = Scripted(found(("my cat is called miso", "fact")))

    facts = await keeper(store, model)

    assert [v.fact for v in await store.fact_vectors(EMBEDDER.id)] == [fact]
    assert await facts.learn(turn, LATER, "my cat is called miso", "Noted.") == []


async def waiting_turn(store: MemoryStore, question: str) -> int:
    chat = await store.begin_conversation("SYNTHIA", MONDAY)
    return await store.add_turn(
        chat, Turn(question, "Noted.", MONDAY, "SYNTHIA", "local", "m"), to_learn=True
    )


async def until_learned(store: Watched) -> None:
    while True:
        store.changed.clear()
        if not await store.to_learn(1):
            return
        await asyncio.wait_for(store.changed.wait(), timeout=5)


@contextlib.asynccontextmanager
async def learning(
    store: MemoryStore, model: Scripted, network: Network | None = None
) -> AsyncGenerator[FactLearning]:
    kept = FactKeeper(store, embedder(network or Network()), model)
    await kept.load()
    learner = FactLearning(kept, store)
    task = asyncio.create_task(learner.run())
    try:
        yield learner
    finally:
        task.cancel()
        await asyncio.wait([task])


async def test_turns_left_waiting_are_learned_from_oldest_first_at_start(
    store: Watched,
) -> None:
    await waiting_turn(store, "my cat is called miso")
    await waiting_turn(store, "my sister lives in pune")
    model = Scripted(
        found(("my cat is called miso", "fact")),
        found(("my sister lives in pune", "fact")),
    )

    async with learning(store, model):
        await until_learned(store)

    assert [r.messages[-1].text.split("\n")[4] for r in model.requests] == [
        "my cat is called miso",
        "my sister lives in pune",
    ]
    assert {text for text, _ in await held(store)} == {
        "my cat is called miso",
        "my sister lives in pune",
    }


async def test_a_turn_remembered_later_wakes_the_learner(store: Watched) -> None:
    model = Scripted(found(("my cat is called miso", "fact")))

    async with learning(store, model) as learner:
        await waiting_turn(store, "my cat is called miso")
        learner.wake()
        await until_learned(store)

    assert await held(store) == [("my cat is called miso", FactKind.FACT)]


async def test_an_answer_that_is_never_valid_drops_its_turn_only(
    store: Watched, caplog: pytest.LogCaptureFixture
) -> None:
    await waiting_turn(store, "my cat is called miso")
    await waiting_turn(store, "my sister lives in pune")
    model = Scripted(
        *[says("not json")] * (DEFAULT_REPAIRS + 1),
        found(("my sister lives in pune", "fact")),
    )

    with caplog.at_level(logging.WARNING, "synthia.memory.facts"):
        async with learning(store, model):
            await until_learned(store)

    assert await held(store) == [("my sister lives in pune", FactKind.FACT)]
    assert "the answer was not valid" in caplog.text


async def test_a_model_that_does_not_answer_leaves_the_turn_waiting(
    store: Watched,
) -> None:
    turn = await waiting_turn(store, "my cat is called miso")
    asked = asyncio.Event()

    def down(_: ChatRequest) -> list[ChatChunk]:
        asked.set()
        message = "the local model is not running"
        raise GatewayError(message)

    model = Scripted(down, found(("my cat is called miso", "fact")))

    async with learning(store, model) as learner:
        await asyncio.wait_for(asked.wait(), timeout=5)
        assert [t.id for t in await store.to_learn(5)] == [turn]
        learner.wake()
        await until_learned(store)

    assert await held(store) == [("my cat is called miso", FactKind.FACT)]


async def test_a_turn_that_fails_otherwise_is_tried_again_then_dropped(
    store: Watched, caplog: pytest.LogCaptureFixture
) -> None:
    turn = await waiting_turn(store, "my cat is called miso")

    with caplog.at_level(logging.ERROR, "synthia.memory.facts"):
        async with learning(store, Scripted(), FailsAfterStart()) as learner:
            for _ in range(TRIES - 1):
                await asyncio.wait_for(store.changed.wait(), timeout=5)
                store.changed.clear()
                assert [t.id for t in await store.to_learn(5)] == [turn]
                learner.wake()
            await until_learned(store)

    assert await store.facts() == []
    assert caplog.text.count(f"facts from turn {turn} wait for another try") == 2
    assert f"no facts from turn {turn} after 3 tries" in caplog.text


async def test_a_store_that_fails_is_logged_and_the_learner_goes_on(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = FailsOnce(tmp_path / "memory.db")
    model = Scripted(found(("my cat is called miso", "fact")))

    with caplog.at_level(logging.ERROR, "synthia.memory.facts"):
        async with learning(store, model) as learner:
            await asyncio.wait_for(store.changed.wait(), timeout=5)
            await waiting_turn(store, "my cat is called miso")
            learner.wake()
            await until_learned(store)

    assert "facts were not learned" in caplog.text
    assert await held(store) == [("my cat is called miso", FactKind.FACT)]
