import asyncio
import threading
from typing import Annotated, Literal

import pytest
from pydantic import Field

from synthia.agent.tools import (
    Effect,
    FunctionTool,
    InvalidArgumentsError,
    Reach,
    Toolbox,
    ToolFunction,
)


def convert(
    value: float,
    from_unit: Literal["celsius", "fahrenheit"],
    to_unit: Annotated[
        Literal["celsius", "fahrenheit"], Field(description="the unit wanted")
    ] = "celsius",
    places: int | None = None,
) -> str:
    """Convert a temperature between units.

    The model never sees this second paragraph.
    """
    return f"{value} {from_unit} {to_unit} {places}"


def local_read(function: ToolFunction, name: str | None = None) -> FunctionTool:
    return FunctionTool.of(function, reach=Reach.LOCAL, effect=Effect.READ, name=name)


def test_the_schema_comes_from_the_signature() -> None:
    spec = local_read(convert).spec

    assert spec.name == "convert"
    assert spec.description == "Convert a temperature between units."
    assert spec.parameters == {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "value": {"type": "number"},
            "from_unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            "to_unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "default": "celsius",
                "description": "the unit wanted",
            },
            "places": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
            },
        },
        "required": ["value", "from_unit"],
    }


def test_a_parameter_named_title_keeps_its_name() -> None:
    def note(title: str) -> str:
        """Write a note."""
        return title

    spec = local_read(note).spec

    assert spec.parameters["properties"] == {"title": {"type": "string"}}
    assert spec.parameters["required"] == ["title"]


async def test_valid_arguments_call_the_function_with_defaults() -> None:
    tool = local_read(convert)

    result = await tool.run('{"value": 100, "from_unit": "fahrenheit"}')

    assert result == "100.0 fahrenheit celsius None"


async def test_no_arguments_at_all_mean_an_empty_object() -> None:
    def now() -> str:
        """Return the time."""
        return "noon"

    assert await local_read(now).run("") == "noon"


@pytest.mark.parametrize(
    ("arguments", "explanation"),
    [
        (
            "{not json",
            (
                "invalid arguments for convert: arguments: Invalid JSON: "
                "key must be a string at line 1 column 2"
            ),
        ),
        (
            '{"value": 1}',
            "invalid arguments for convert: from_unit: Field required",
        ),
        (
            '{"value": "hot", "from_unit": "kelvin"}',
            (
                "invalid arguments for convert: value: Input should be a valid "
                "number, unable to parse string as a number; from_unit: Input "
                "should be 'celsius' or 'fahrenheit'"
            ),
        ),
        (
            '{"value": 1, "from_unit": "celsius", "colour": "red"}',
            "invalid arguments for convert: colour: Extra inputs are not permitted",
        ),
        (
            "[1, 2]",
            "invalid arguments for convert: arguments: Input should be an object",
        ),
    ],
)
async def test_bad_arguments_are_explained_and_never_reach_the_function(
    arguments: str, explanation: str
) -> None:
    calls: list[object] = []

    def recording(**kwargs: object) -> str:
        calls.append(kwargs)
        return ""

    tool = local_read(convert)
    tool = FunctionTool(tool.spec, tool.reach, tool.effect, recording, tool.parameters)

    with pytest.raises(InvalidArgumentsError) as raised:
        await tool.run(arguments)

    assert str(raised.value) == explanation
    assert calls == []


async def test_a_number_sent_as_text_is_accepted() -> None:
    assert await local_read(convert).run('{"value": "5", "from_unit": "celsius"}') == (
        "5.0 celsius celsius None"
    )


async def test_an_async_function_is_awaited() -> None:
    async def later(text: str) -> str:
        """Echo later."""
        await asyncio.sleep(0)
        return text.upper()

    assert await local_read(later).run('{"text": "hi"}') == "HI"


async def test_a_plain_function_runs_off_the_event_loop() -> None:
    loop_thread = threading.get_ident()

    def where() -> str:
        """Report the thread."""
        return str(threading.get_ident())

    assert await local_read(where).run("{}") != str(loop_thread)


async def test_hints_written_as_strings_are_resolved() -> None:
    # What every hint becomes under `from __future__ import annotations`.
    def repeat(
        text: "str",
        times: "Annotated[int, Field(ge=1, description='how often')]" = 2,
    ) -> "str":
        """Repeat text."""
        return text * times

    tool = local_read(repeat)

    assert tool.spec.parameters["properties"] == {
        "text": {"type": "string"},
        "times": {
            "type": "integer",
            "minimum": 1,
            "default": 2,
            "description": "how often",
        },
    }
    assert await tool.run('{"text": "ab"}') == "abab"


def test_a_hint_naming_nothing_is_refused() -> None:
    def ghost(x: "Nowhere") -> str:  # type: ignore[name-defined]  # noqa: F821
        """Ghost."""
        return str(x)  # pyright: ignore[reportUnknownArgumentType]

    with pytest.raises(
        ValueError, match="ghost: a type hint names something undefined"
    ):
        local_read(ghost)  # pyright: ignore[reportUnknownArgumentType]


def test_name_can_be_given() -> None:
    assert local_read(convert, name="temperature").spec.name == "temperature"


def test_a_function_without_a_docstring_is_refused() -> None:
    def bare(x: int) -> str:
        return str(x)

    with pytest.raises(ValueError, match="bare needs a docstring"):
        local_read(bare)


def test_a_parameter_without_a_type_hint_is_refused() -> None:
    def loose(x) -> str:  # noqa: ANN001  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
        """Loose."""
        return str(x)  # pyright: ignore[reportUnknownArgumentType]

    with pytest.raises(ValueError, match="parameter x needs a type hint"):
        local_read(loose)  # pyright: ignore[reportUnknownArgumentType]


@pytest.mark.parametrize("kind", ["positional", "args", "kwargs"])
def test_parameters_that_cannot_be_passed_by_name_are_refused(kind: str) -> None:
    def positional(x: int, /) -> str:
        """P."""
        return str(x)

    def args(*x: int) -> str:
        """A."""
        return str(x)

    def kwargs(**x: int) -> str:
        """K."""
        return str(x)

    function = {"positional": positional, "args": args, "kwargs": kwargs}[kind]

    with pytest.raises(ValueError, match="must be passable by name"):
        local_read(function)


def test_the_toolbox_keeps_order_and_finds_by_name() -> None:
    def a() -> str:
        """A."""
        return "a"

    def b() -> str:
        """B."""
        return "b"

    first, second = local_read(b), local_read(a)
    box = Toolbox([first, second])

    assert [s.name for s in box.specs()] == ["b", "a"]
    assert box.get("a") is second
    assert box.get("nobody") is None
    assert list(box) == [first, second]
    assert len(box) == 2


def test_the_toolbox_refuses_a_second_tool_with_the_same_name() -> None:
    box = Toolbox([local_read(convert)])

    with pytest.raises(ValueError, match="'convert' is already in the toolbox"):
        box.add(local_read(convert))


@pytest.mark.parametrize("name", ["", "has space", "x" * 65, "dot.ted", "ümlaut"])
def test_the_toolbox_refuses_names_servers_reject(name: str) -> None:
    with pytest.raises(ValueError, match="must be 1 to 64 letters"):
        Toolbox([local_read(convert, name=name)])


def test_a_64_character_name_is_accepted() -> None:
    assert len(Toolbox([local_read(convert, name="x" * 64)])) == 1
