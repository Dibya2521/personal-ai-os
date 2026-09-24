from collections.abc import AsyncGenerator

import pytest
from pydantic import BaseModel, ConfigDict

from synthia.gateway.structured import (
    StructuredOutputError,
    extract_json,
    generate,
    schema_of,
)
from synthia.gateway.types import (
    ChatChunk,
    ChatRequest,
    FinishReason,
    Message,
    ModelInfo,
    Role,
)

INFO = ModelInfo("json", 1000, vision=False, tools=False)
ASK = ChatRequest((Message.user("Extract the meeting."),), temperature=0.0)


class Meeting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    minutes: int


class Scripted:
    """Reply with the scripted texts in turn, recording every request."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.requests: list[ChatRequest] = []

    @property
    def info(self) -> ModelInfo:
        return INFO

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        self.requests.append(request)
        yield ChatChunk(text=self.replies.pop(0))
        yield ChatChunk(finish_reason=FinishReason.STOP)


GOOD = '{"title": "Standup", "minutes": 15}'


async def test_a_valid_reply_takes_one_request_that_carries_the_schema() -> None:
    model = Scripted(GOOD)

    meeting = await generate(model, ASK, Meeting)

    assert meeting == Meeting(title="Standup", minutes=15)
    assert len(model.requests) == 1
    assert model.requests[0].response_schema == schema_of(Meeting)
    assert model.requests[0].temperature == 0.0


async def test_a_reply_wrapped_in_a_code_fence_is_unwrapped() -> None:
    model = Scripted(f"```json\n{GOOD}\n```")

    assert (await generate(model, ASK, Meeting)).minutes == 15


async def test_a_wrong_reply_is_shown_back_with_its_errors_and_repaired() -> None:
    model = Scripted('{"title": "Standup", "minutes": "a quarter hour"}', GOOD)

    meeting = await generate(model, ASK, Meeting)

    assert meeting.minutes == 15
    repair = model.requests[1].messages
    assert repair[:-2] == ASK.messages
    assert repair[-2].role is Role.ASSISTANT
    assert repair[-2].text == '{"title": "Standup", "minutes": "a quarter hour"}'
    assert repair[-1].role is Role.USER
    assert "- minutes: Input should be a valid integer" in repair[-1].text


async def test_repairs_are_bounded_and_the_last_reply_is_kept() -> None:
    model = Scripted("no", "still no", "never")

    with pytest.raises(
        StructuredOutputError, match="no valid Meeting after 2 repairs"
    ) as caught:
        await generate(model, ASK, Meeting)

    assert len(model.requests) == 3
    assert caught.value.reply == "never"
    assert "Invalid JSON" in str(caught.value)


async def test_each_repair_keeps_every_earlier_mistake_in_order() -> None:
    model = Scripted("first", "second", GOOD)

    await generate(model, ASK, Meeting)

    texts = [m.text for m in model.requests[2].messages]
    assert texts[1] == "first"
    assert texts[3] == "second"
    assert len(texts) == 5


async def test_an_empty_reply_is_repaired_like_any_other() -> None:
    model = Scripted("", GOOD)

    assert (await generate(model, ASK, Meeting)).title == "Standup"


async def test_extra_fields_are_refused_when_the_model_forbids_them() -> None:
    model = Scripted('{"title": "Standup", "minutes": 15, "room": "B2"}', GOOD)

    await generate(model, ASK, Meeting)

    assert (
        "- room: Extra inputs are not permitted" in model.requests[1].messages[-1].text
    )


async def test_no_repairs_means_one_request() -> None:
    model = Scripted("no")

    with pytest.raises(StructuredOutputError, match="after 0 repairs"):
        await generate(model, ASK, Meeting, repairs=0)
    assert len(model.requests) == 1


async def test_negative_repairs_are_refused() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        await generate(Scripted(GOOD), ASK, Meeting, repairs=-1)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("  {}  \n", "{}"),
        ("```\n{}\n```", "{}"),
        ("```json\n{\n}\n```\n", "{\n}"),
        ("Here it is:\n```json\n{}\n```", "Here it is:\n```json\n{}\n```"),
        ("```json\n{}", "```json\n{}"),
    ],
)
def test_only_a_reply_that_is_wholly_one_fence_is_unwrapped(
    text: str, expected: str
) -> None:
    assert extract_json(text) == expected
