import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from synthia.gateway.types import ChatChunk, Reasoning, Role
from synthia.memory.facts import FactKeeper
from synthia.memory.store import FactKind, MemoryStore, Turn
from tests.agent.scripted import Scripted, says
from tests.memory.test_hybrid import Network, embedder
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


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory.db")


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
    system, user = request.messages
    assert system.role is Role.SYSTEM
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
        in model.requests[1].messages[1].text
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

    assert model.requests[1].messages[1].text.startswith("Known facts:\n(none)\n")
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
        model.requests[1].messages[1].text
    )
    assert again == []
    assert await held(store) == [("my sister lives in mumbai", FactKind.FACT)]
    old = next(f for f in await store.facts(held=False) if f.id == pune)
    assert (old.until, old.replaced_by) == (LATER, mumbai)


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
