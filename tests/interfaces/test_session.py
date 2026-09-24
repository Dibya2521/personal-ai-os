from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from synthia.gateway.errors import ProviderError
from synthia.gateway.router import Route, RouteDecided, RouteReason
from synthia.gateway.types import (
    ChatChunk,
    ChatRequest,
    FinishReason,
    ImagePart,
    ModelInfo,
    Role,
    Usage,
)
from synthia.interfaces.session import (
    MAX_IMAGE_BYTES,
    ChatSession,
    ImageError,
    LastRoute,
    TurnReport,
    load_image,
)
from synthia.kernel.bus import Event
from synthia.persona.library import PersonaLibrary
from synthia.persona.model import PersonaError

INFO = ModelInfo("fake", 1000, vision=True, tools=False)


class Echo:
    """Answer 'you said: <text>', or fail when told to."""

    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []
        self.fail: ProviderError | None = None
        self.unfinished = False

    @property
    def info(self) -> ModelInfo:
        return INFO

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        self.requests.append(request)
        yield ChatChunk(text="you said: ", model="vendor/echo")
        if self.fail:
            raise self.fail
        yield ChatChunk(text=request.messages[-1].text)
        if not self.unfinished:
            yield ChatChunk(finish_reason=FinishReason.STOP, usage=Usage(40, 3))


class Ticks:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 0.75
        return self.now


@pytest.fixture
def library() -> PersonaLibrary:
    return PersonaLibrary()


def session(library: PersonaLibrary, model: Echo) -> ChatSession:
    return ChatSession(model, library, "synthia", clock=Ticks())


async def run(chat: ChatSession, text: str) -> tuple[str, list[TurnReport]]:
    text_out: list[str] = []
    reports: list[TurnReport] = []
    async for item in chat.turn(text):
        if isinstance(item, TurnReport):
            reports.append(item)
        else:
            text_out.append(item.text)
    return "".join(text_out), reports


async def test_a_turn_streams_the_answer_then_reports_it(
    library: PersonaLibrary,
) -> None:
    chat = session(library, Echo())

    text, reports = await run(chat, "hello")

    assert text == "you said: hello"
    assert reports == [TurnReport("direct", "vendor/echo", 40, 3, 0.75)]


async def test_every_request_starts_with_the_persona_and_carries_the_history(
    library: PersonaLibrary,
) -> None:
    model = Echo()
    chat = session(library, model)

    await run(chat, "one")
    await run(chat, "two")

    first, second = model.requests
    assert first.messages[0].role is Role.SYSTEM
    assert first.messages[0].text == library.get("synthia").system_prompt()
    assert [m.text for m in second.messages[1:]] == ["one", "you said: one", "two"]
    assert len(chat.history) == 4


async def test_a_failed_turn_leaves_no_trace_in_the_history(
    library: PersonaLibrary,
) -> None:
    model = Echo()
    chat = session(library, model)
    model.fail = ProviderError("down")

    with pytest.raises(ProviderError):
        await run(chat, "hello")

    assert chat.history == []


async def test_an_interrupted_turn_leaves_no_trace_in_the_history(
    library: PersonaLibrary,
) -> None:
    chat = session(library, Echo())

    turn = chat.turn("hello")
    await anext(turn)
    await turn.aclose()

    assert chat.history == []


async def test_an_answer_without_a_finish_is_shown_but_not_kept_or_reported(
    library: PersonaLibrary,
) -> None:
    model = Echo()
    model.unfinished = True
    chat = session(library, model)

    text, reports = await run(chat, "hello")

    assert text == "you said: hello"
    assert reports == []
    assert chat.history == []


async def test_the_report_names_the_route_the_router_announced(
    library: PersonaLibrary,
) -> None:
    routes = LastRoute()
    chat = ChatSession(Echo(), library, "synthia", routes, clock=Ticks())
    await routes(
        RouteDecided(route=Route.LOCAL, reason=RouteReason.BACKGROUND, model="qwen")
    )

    await routes(Event())  # other events are not routes and change nothing
    _, (report,) = await run(chat, "hello")

    assert (report.route, report.model) == (Route.LOCAL, "vendor/echo")


async def test_switching_and_adjusting_persona_change_the_next_prompt_only(
    library: PersonaLibrary,
) -> None:
    model = Echo()
    chat = session(library, model)
    await run(chat, "one")

    chat.switch("edith")
    chat.adjust({"wit": 1.0})
    await run(chat, "two")

    system = model.requests[1].messages[0].text
    assert system.startswith("You are EDITH, ")
    assert "Use dry wit freely" in system
    assert len(chat.history) == 4
    with pytest.raises(PersonaError):
        chat.switch("friday")
    with pytest.raises(PersonaError):
        chat.adjust({"charm": 0.5})


async def test_reset_forgets_the_conversation(library: PersonaLibrary) -> None:
    chat = session(library, Echo())
    await run(chat, "one")

    chat.reset()

    assert chat.history == []


def test_an_image_request_carries_the_picture(library: PersonaLibrary) -> None:
    picture = ImagePart(b"\x89PNG", "image/png")

    request = session(library, Echo()).request("what is it?", picture)

    assert request.has_images


def test_an_image_is_read_with_the_media_type_of_its_suffix(tmp_path: Path) -> None:
    path = tmp_path / "Cat.JPG"
    path.write_bytes(b"\xff\xd8\xff")

    image = load_image(path)

    assert (image.media_type, image.data) == ("image/jpeg", b"\xff\xd8\xff")


def test_images_that_cannot_be_sent_are_refused_before_any_request(
    tmp_path: Path,
) -> None:
    huge = tmp_path / "huge.png"
    with huge.open("wb") as file:
        file.truncate(MAX_IMAGE_BYTES + 1)

    with pytest.raises(ImageError, match="not a PNG, JPEG, WebP or GIF"):
        load_image(tmp_path / "notes.txt")
    with pytest.raises(ImageError, match="cannot read"):
        load_image(tmp_path / "absent.png")
    with pytest.raises(ImageError, match="the limit is 20 MB"):
        load_image(huge)
