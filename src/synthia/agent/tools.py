"""Tools the agent can call, built from typed Python functions.

A tool's parameters are its function's signature. One pydantic model built
from that signature gives both the JSON Schema the model is shown and the
check of the arguments it sends back, so the two can never disagree.
"""

from __future__ import annotations

import asyncio
import inspect
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from synthia.gateway.types import ToolSpec
from synthia.kernel.errors import SynthiaError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping

# What OpenAI-compatible servers accept as a function name.
TOOL_NAME: Final = re.compile(r"[A-Za-z0-9_-]{1,64}")
SCHEMA_MAPS: Final = frozenset({"properties", "$defs"})

type ToolFunction = Callable[..., str] | Callable[..., Awaitable[str]]


class Reach(StrEnum):
    """Where a tool's work happens."""

    LOCAL = "local"
    OUTSIDE = "outside"


class Effect(StrEnum):
    """Whether a tool only looks, or changes something."""

    READ = "read"
    CHANGE = "change"


class ToolError(SynthiaError):
    """A tool could not do what was asked; the message is for the model."""


class InvalidArgumentsError(ToolError):
    """The arguments the model sent do not fit the tool's parameters."""


class Tool(Protocol):
    """Something the agent can call."""

    @property
    def spec(self) -> ToolSpec:
        """Return the name, description and parameter schema shown to the model."""
        ...

    @property
    def reach(self) -> Reach:
        """Return where the tool's work happens."""
        ...

    @property
    def effect(self) -> Effect:
        """Return whether the tool changes anything."""
        ...

    async def run(self, arguments: str) -> str:
        """Run with ``arguments``, the JSON text the model produced.

        Raises:
            ToolError: If the tool cannot do it; the message goes to the model.
        """
        ...


@dataclass(frozen=True, slots=True)
class FunctionTool:
    """A :class:`Tool` that calls a typed Python function.

    Parameters may carry a description with
    ``Annotated[int, Field(description=...)]``. Arguments are checked in lax
    mode, so ``"5"`` for an integer is accepted, as a model often sends it.
    """

    spec: ToolSpec
    reach: Reach
    effect: Effect
    function: ToolFunction
    parameters: type[BaseModel]

    @classmethod
    def of(
        cls,
        function: ToolFunction,
        *,
        reach: Reach,
        effect: Effect,
        name: str | None = None,
    ) -> FunctionTool:
        """Return the tool that calls ``function``.

        The description is the first paragraph of its docstring.

        Raises:
            ValueError: If the function has no docstring, or a parameter that
                is positional-only, variadic, without a type hint, or hinted
                with a name its module does not define at run time.
        """
        name = function.__name__ if name is None else name
        doc = inspect.getdoc(function)
        if not doc:
            message = f"{name} needs a docstring: it is the description the model reads"
            raise ValueError(message)
        try:
            # Under ``from __future__ import annotations`` hints are strings;
            # pydantic needs the types they name.
            signature = inspect.signature(function, eval_str=True)
        except NameError as error:
            message = f"{name}: a type hint names something undefined: {error}"
            raise ValueError(message) from None
        model = create_model(
            name,
            # A parameter may be called model_anything; that is the tool's
            # business, not pydantic's.
            __config__=ConfigDict(extra="forbid", protected_namespaces=()),
            **_fields(name, signature),
        )
        spec = ToolSpec(
            name, doc.split("\n\n")[0], _untitled(model.model_json_schema())
        )
        return cls(spec, reach, effect, function, model)

    async def run(self, arguments: str) -> str:
        """Check ``arguments`` against the parameters, then call the function.

        Raises:
            InvalidArgumentsError: If they do not fit; the function is not called.
        """
        try:
            values = self.parameters.model_validate_json(arguments or "{}")
        except ValidationError as error:
            raise InvalidArgumentsError(_explain(self.spec.name, error)) from None
        kwargs = {key: getattr(values, key) for key in type(values).model_fields}
        if inspect.iscoroutinefunction(self.function):
            return await self.function(**kwargs)
        # A plain function may block (a file read); off the loop it cannot
        # stall other tools running at the same time.
        return await asyncio.to_thread(
            cast("Callable[..., str]", self.function), **kwargs
        )


class Toolbox:
    """The tools an agent may call, by name.

    Raises:
        ValueError: From :meth:`add`.
    """

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.add(tool)

    def add(self, tool: Tool) -> None:
        """Add ``tool``.

        Raises:
            ValueError: If its name is taken or is not 1 to 64 letters, digits,
                ``_`` or ``-``.
        """
        name = tool.spec.name
        if not TOOL_NAME.fullmatch(name):
            message = f"tool name {name!r} must be 1 to 64 letters, digits, _ or -"
            raise ValueError(message)
        if name in self._tools:
            message = f"a tool named {name!r} is already in the toolbox"
            raise ValueError(message)
        self._tools[name] = tool

    def get(self, name: str) -> Tool | None:
        """Return the tool called ``name``, if there is one."""
        return self._tools.get(name)

    def specs(self) -> tuple[ToolSpec, ...]:
        """Return every tool's spec, in the order they were added."""
        return tuple(t.spec for t in self._tools.values())

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)


def _fields(name: str, signature: inspect.Signature) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for parameter in signature.parameters.values():
        if parameter.kind not in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            message = f"{name}: parameter {parameter.name} must be passable by name"
            raise ValueError(message)
        if parameter.annotation is inspect.Parameter.empty:
            message = f"{name}: parameter {parameter.name} needs a type hint"
            raise ValueError(message)
        default = (
            ... if parameter.default is inspect.Parameter.empty else parameter.default
        )
        fields[parameter.name] = (parameter.annotation, default)
    return fields


def _untitled(schema: Mapping[str, object]) -> dict[str, object]:
    """Drop the titles pydantic derives from names; they only repeat the name.

    Under ``properties`` the keys are parameter names, so a parameter called
    ``title`` is kept.
    """
    clean: dict[str, object] = {}
    for key, value in schema.items():
        if key == "title":
            continue
        if key in SCHEMA_MAPS and isinstance(value, dict):
            entries = cast("dict[str, Mapping[str, object]]", value)
            clean[key] = {k: _untitled(v) for k, v in entries.items()}
        else:
            clean[key] = _nested(value)
    return clean


def _nested(value: object) -> object:
    if isinstance(value, dict):
        return _untitled(cast("dict[str, object]", value))
    if isinstance(value, list):
        return [_nested(v) for v in cast("list[object]", value)]
    return value


def _explain(name: str, error: ValidationError) -> str:
    problems = [
        f"{'.'.join(str(p) for p in e['loc']) or 'arguments'}: {e['msg']}"
        for e in error.errors(include_url=False)
    ]
    return f"invalid arguments for {name}: " + "; ".join(problems)
