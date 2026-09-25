import pytest
from hypothesis import given
from hypothesis import strategies as st

from synthia.gateway.reasoning import choose, limit, points, resolve, token_limit
from synthia.gateway.types import ChatRequest, Message, Reasoning

OFF, LOW, MEDIUM, HIGH = Reasoning.OFF, Reasoning.LOW, Reasoning.MEDIUM, Reasoning.HIGH

LOG = "\n".join(
    f"2026-09-25 12:{m:02d}:00 INFO worker {m} finished batch in 3 s" for m in range(40)
)
TRACEBACK = """Traceback (most recent call last):
  File "app.py", line 3, in <module>
    main()
KeyError: 'user'"""


def asked(text: str) -> Reasoning:
    return choose(ChatRequest((Message.user(text),)))


@pytest.mark.parametrize(
    ("text", "level"),
    [
        ("hi", OFF),
        ("thanks!", OFF),
        ("good morning SYNTHIA", OFF),
        ("how are you?", OFF),
        ("capital of France?", OFF),
        ("what time is it in Tokyo", OFF),
        ("Import duties in India?", OFF),
        ("who is the prime minister of India", OFF),
        ("tell me something nice about the ocean and the fish in it", LOW),
        ("17*23", LOW),
        ("what is 2 + 2", LOW),
        ("is 91 prime", MEDIUM),
        ("why is the sky blue?", MEDIUM),
        ("how does a hash map work", MEDIUM),
        ("compare SQLite and Postgres", MEDIUM),
        ("solve x^2 - 5x + 6 = 0", MEDIUM),
        ("```\nprint(1)\n```", MEDIUM),
        ("prove that the square root of 2 is irrational", HIGH),
        ("why does this fail?\n" + TRACEBACK, HIGH),
        ("explain step by step how TCP opens a connection", HIGH),
        ("what is a mutex? and when is a semaphore better?", LOW),
    ],
)
def test_the_rules_choose_the_expected_level(text: str, level: Reasoning) -> None:
    assert asked(text) is level


def test_a_long_pasted_log_with_a_trivial_question_stays_low() -> None:
    assert asked(LOG + "\nanything odd here?") is LOW


def test_a_one_word_maths_problem_is_not_off() -> None:
    assert asked("integrate") is MEDIUM
    assert asked("12/4") is LOW


def test_each_sign_counts_once_however_often_it_appears() -> None:
    assert points("why? why? why?") == points("why? why?") == 3


def test_several_listed_questions_count_like_several_question_marks() -> None:
    listed = "a few things:\n1. what is DNS?\n2. what is a CDN\n3. and TLS"
    single = "a few things: what is DNS?"

    assert points(listed) == points(single) + 1


def test_the_latest_user_message_decides_not_the_history() -> None:
    request = ChatRequest(
        (
            Message.user("prove the four colour theorem step by step"),
            Message.assistant("That is a long proof."),
            Message.user("thanks"),
        )
    )

    assert choose(request) is OFF


def test_a_request_with_no_user_message_does_not_think() -> None:
    assert choose(ChatRequest((Message.system("why? " * 100),))) is OFF


def test_resolve_replaces_auto_and_nothing_else() -> None:
    auto = ChatRequest((Message.user("why?"),), reasoning=Reasoning.AUTO)

    assert resolve(auto).reasoning is MEDIUM
    assert resolve(auto).messages == auto.messages
    for level in (None, OFF, LOW, MEDIUM, HIGH):
        request = ChatRequest(auto.messages, reasoning=level)
        assert resolve(request) is request


@pytest.mark.parametrize(
    ("level", "speed", "tokens"),
    [
        (LOW, 30.0, 150),
        (MEDIUM, 35.5, 710),
        (HIGH, None, 1500),
        (HIGH, 0.0, 1500),
        (LOW, 0.1, 1),
        (OFF, 30.0, None),
        (Reasoning.AUTO, 30.0, None),
        (None, 30.0, None),
    ],
)
def test_a_levels_allowance_becomes_tokens_at_the_measured_speed(
    level: Reasoning | None, speed: float | None, tokens: int | None
) -> None:
    assert token_limit(level, speed) == tokens


def test_limit_sets_tokens_only_for_a_level_with_an_allowance() -> None:
    high = ChatRequest((Message.user("x"),), reasoning=HIGH)
    off = ChatRequest((Message.user("x"),), reasoning=OFF)

    assert limit(high, 20.0).reasoning_tokens == 1200
    assert limit(off, 20.0) is off


@given(st.text())
def test_choose_never_returns_auto_and_never_fails(text: str) -> None:
    assert asked(text) is not Reasoning.AUTO


@given(st.text(alphabet="why? 12*3\n-`{;", max_size=200))
def test_adding_a_line_never_lowers_the_level(text: str) -> None:
    order = [OFF, LOW, MEDIUM, HIGH]

    assert order.index(asked(text + "\nwhy")) >= order.index(asked(text))
