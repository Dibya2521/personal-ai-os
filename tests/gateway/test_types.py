import base64

import pytest

from synthia.gateway.types import (
    ChatRequest,
    ChatResponse,
    FinishReason,
    ImagePart,
    Message,
    Role,
    TextPart,
    ToolCall,
    Usage,
)

PNG = b"\x89PNG\r\n\x1a\n-not-really-an-image"


def test_constructors_build_the_expected_roles() -> None:
    call = ToolCall("c1", "clock", "{}")

    assert Message.system("be brief").role is Role.SYSTEM
    assert Message.user("hi").text == "hi"
    assert Message.assistant("", call).parts == ()
    assert Message.assistant("ok", call).tool_calls == (call,)
    assert Message.tool_result("c1", "12:00").tool_call_id == "c1"


def test_text_joins_text_parts_and_skips_images() -> None:
    image = ImagePart(PNG, "image/png")
    message = Message(Role.USER, (TextPart("a "), image, TextPart("b")))

    assert message.text == "a b"


@pytest.mark.parametrize(
    "build",
    [
        lambda: Message(Role.TOOL, (TextPart("x"),)),
        lambda: Message(Role.USER, (TextPart("x"),), tool_call_id="c1"),
        lambda: Message(Role.USER, (), (ToolCall("c", "n", "{}"),)),
        lambda: Message(Role.SYSTEM, (ImagePart(PNG, "image/png"),)),
        lambda: Message(Role.ASSISTANT, (ImagePart(PNG, "image/png"),)),
    ],
)
def test_a_message_that_does_not_fit_its_role_is_rejected(build: object) -> None:
    with pytest.raises(ValueError, match="only"):
        build()  # type: ignore[operator]


def test_image_data_url_round_trips() -> None:
    url = ImagePart(PNG, "image/jpeg").data_url()

    prefix, encoded = url.split(",", 1)
    assert prefix == "data:image/jpeg;base64"
    assert base64.b64decode(encoded) == PNG


def test_image_bytes_stay_out_of_the_repr() -> None:
    assert "PNG" not in repr(ImagePart(PNG, "image/png"))


@pytest.mark.parametrize(
    ("data", "media_type"),
    [(PNG, "image/tiff"), (PNG, "text/plain"), (b"", "image/png")],
)
def test_invalid_images_are_rejected(data: bytes, media_type: str) -> None:
    with pytest.raises(ValueError, match="image"):
        ImagePart(data, media_type)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"messages": ()}, "at least one message"),
        ({"temperature": -0.1}, "temperature"),
        ({"temperature": 2.01}, "temperature"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"reasoning_tokens": 0}, "reasoning_tokens"),
    ],
)
def test_request_validation(kwargs: dict[str, object], match: str) -> None:
    arguments: dict[str, object] = {"messages": (Message.user("hi"),)} | kwargs

    with pytest.raises(ValueError, match=match):
        ChatRequest(**arguments)  # type: ignore[arg-type]


def test_request_knows_when_it_carries_an_image() -> None:
    plain = ChatRequest((Message.user("hi"),))
    visual = ChatRequest((Message.user("what is this", ImagePart(PNG, "image/png")),))

    assert not plain.has_images
    assert visual.has_images


def test_unknown_finish_reasons_map_to_other() -> None:
    assert FinishReason.parse("stop") is FinishReason.STOP
    assert FinishReason.parse("function_call") is FinishReason.OTHER
    assert FinishReason.parse("") is FinishReason.OTHER


def test_usage_total_and_response_as_message() -> None:
    call = ToolCall("c1", "clock", "{}")
    response = ChatResponse("ok", (call,), FinishReason.TOOL_CALLS, Usage(3, 4), "m")

    assert response.usage is not None
    assert response.usage.total_tokens == 7
    assert response.as_message() == Message.assistant("ok", call)
