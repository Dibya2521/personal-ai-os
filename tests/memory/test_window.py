import pytest

from synthia.gateway.types import (
    ChatRequest,
    ImagePart,
    Message,
    Reasoning,
    ToolCall,
    ToolSpec,
)
from synthia.memory.summary import SUMMARY_TOKENS, summarize, transcript
from synthia.memory.window import (
    IMAGE_TOKENS,
    MAX_CHARS_PER_TOKEN,
    MESSAGE_TOKENS,
    MIN_CHARS_PER_TOKEN,
    ContextWindow,
    message_chars,
    tools_chars,
)
from tests.agent.scripted import Scripted, says

PICTURE = ImagePart(b"\x89PNG", "image/png")


def test_a_message_counts_its_text_and_its_tool_calls() -> None:
    asked = Message.assistant("ok", ToolCall("c1", "read_file", '{"p": 1}'))

    assert message_chars(asked) == len("ok") + len("read_file") + len('{"p": 1}')


def test_tokens_add_each_messages_overhead_and_each_image() -> None:
    window = ContextWindow(1000, chars_per_token=4.0)
    messages = (Message.user("x" * 40, PICTURE), Message.assistant("y" * 4))

    assert window.tokens(messages) == 11 + 2 * MESSAGE_TOKENS + IMAGE_TOKENS
    assert window.tokens((), extra_chars=9) == 3


def test_tool_definitions_are_counted_with_their_schema() -> None:
    spec = ToolSpec("t", "does it", {"type": "object"})

    assert tools_chars((spec,)) == len("t") + len("does it") + len('{"type": "object"}')


def test_a_quarter_of_the_window_is_kept_for_the_turn() -> None:
    assert ContextWindow(32_768).budget == 24_576


def test_the_oldest_turns_leave_until_the_rest_fit() -> None:
    window = ContextWindow(400, chars_per_token=1.0)
    fixed = (Message.system("s" * 92),)  # 100 tokens of the 300 budget
    turn = [Message.user("u" * 42), Message.assistant("a" * 42)]  # 100 tokens

    assert window.overflow(fixed, [turn, turn]) == 0
    assert window.overflow(fixed, [turn, turn, turn]) == 1
    assert window.overflow(fixed, [turn, turn, turn], extra_chars=100) == 2


def test_when_even_the_question_does_not_fit_every_turn_leaves() -> None:
    window = ContextWindow(100, chars_per_token=1.0)
    turn = [Message.user("u")]

    assert window.overflow((Message.user("q" * 500),), [turn, turn]) == 2


def test_the_rate_moves_towards_what_the_model_reports() -> None:
    window = ContextWindow(1000)

    window.learn(4000, 1000)

    assert window.chars_per_token == pytest.approx(0.7 * 3.0 + 0.3 * 4.0)


def test_the_rate_stays_within_its_bounds() -> None:
    low = ContextWindow(1000, chars_per_token=MIN_CHARS_PER_TOKEN)
    high = ContextWindow(1000, chars_per_token=MAX_CHARS_PER_TOKEN)

    low.learn(100, 1000)
    high.learn(100_000, 1000)

    assert (low.chars_per_token, high.chars_per_token) == (
        MIN_CHARS_PER_TOKEN,
        MAX_CHARS_PER_TOKEN,
    )


@pytest.mark.parametrize(("chars", "tokens"), [(4000, None), (4000, 0), (0, 10)])
def test_a_report_with_nothing_to_learn_leaves_the_rate(
    chars: int, tokens: int | None
) -> None:
    window = ContextWindow(1000)

    window.learn(chars, tokens)

    assert window.chars_per_token == 3.0


def test_a_transcript_names_who_spoke_and_cuts_long_lines() -> None:
    messages = (
        Message.user("my  cat\nis Miso"),
        Message.assistant("", ToolCall("c", "note", "{}")),
        Message.tool_result("c", "noted " + "x" * 2000),
        Message.assistant("Lovely."),
    )

    lines = transcript(messages).splitlines()

    assert lines[0] == "person: my cat is Miso"
    assert lines[1].startswith("tool: noted x")
    assert len(lines[1]) == len("tool: ") + 1000
    assert lines[2] == "SYNTHIA: Lovely."


async def test_a_summary_is_asked_of_this_machine_only_without_thinking() -> None:
    model = Scripted(says("They have a cat named Miso."))

    summary = await summarize(model, "", (Message.user("my cat is Miso"),))

    (request,) = model.requests
    assert summary == "They have a cat named Miso."
    assert isinstance(request, ChatRequest)
    assert (request.use_remote, request.reasoning, request.max_tokens) == (
        False,
        Reasoning.OFF,
        SUMMARY_TOKENS,
    )
    assert "Summary so far:\n(nothing yet)" in request.messages[1].text
    assert "person: my cat is Miso" in request.messages[1].text
