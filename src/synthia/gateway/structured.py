"""Ask a model for JSON that matches a schema, and never trust that it did.

The schema travels as the request's ``response_schema``; llama.cpp enforces it
with a grammar, and some remote models honour it. The reply is validated here
either way. When it fails, the model is shown its reply and the exact errors
and asked again, at most ``repairs`` times. Each repair is a new request, so
it is routed and metered like any other.
"""

from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ValidationError

from synthia.gateway.errors import GatewayError
from synthia.gateway.protocol import collect
from synthia.gateway.types import Message

if TYPE_CHECKING:
    from synthia.gateway.protocol import ChatModel
    from synthia.gateway.types import ChatRequest

DEFAULT_REPAIRS: Final = 2
MAX_PROBLEMS: Final = 10

_FENCED = re.compile(r"\A\s*```(?:json)?[ \t]*\n(.*?)\n?```\s*\Z", re.DOTALL)


class StructuredOutputError(GatewayError):
    """The model gave no valid output within its repairs; ``reply`` is the last one."""

    def __init__(self, message: str, reply: str) -> None:
        super().__init__(message)
        self.reply = reply


def schema_of(kind: type[BaseModel]) -> dict[str, object]:
    """Return the JSON Schema a reply must match to become a ``kind``."""
    return kind.model_json_schema()


def extract_json(text: str) -> str:
    """Return the JSON in a reply, unwrapped if the whole reply is one code fence."""
    fenced = _FENCED.match(text)
    return fenced.group(1) if fenced else text.strip()


def _problems(error: ValidationError) -> str:
    lines = [
        f"- {'.'.join(map(str, e['loc'])) or 'reply'}: {e['msg']}"
        for e in error.errors()[:MAX_PROBLEMS]
    ]
    return "\n".join(lines)


def _repair_prompt(problems: str) -> str:
    return (
        "That reply did not match the required JSON schema:\n"
        f"{problems}\n"
        "Reply again with only the corrected JSON, and nothing else."
    )


async def generate[T: BaseModel](
    model: ChatModel,
    request: ChatRequest,
    kind: type[T],
    *,
    repairs: int = DEFAULT_REPAIRS,
) -> T:
    """Return the model's answer to ``request`` as a validated ``kind``.

    Raises:
        StructuredOutputError: If no reply validated after ``repairs`` repairs.
        GatewayError: Whatever the model raised.
        ValueError: If ``repairs`` is negative.
    """
    if repairs < 0:
        message = "repairs cannot be negative"
        raise ValueError(message)
    ask = dataclasses.replace(request, response_schema=schema_of(kind))
    reply, problems = "", ""
    for attempt in range(repairs + 1):
        if attempt:
            repair = (Message.assistant(reply), Message.user(_repair_prompt(problems)))
            ask = dataclasses.replace(ask, messages=(*ask.messages, *repair))
        reply = (await collect(model.stream(ask))).text
        try:
            return kind.model_validate_json(extract_json(reply))
        except ValidationError as error:
            problems = _problems(error)
    message = f"no valid {kind.__name__} after {repairs} repairs:\n{problems}"
    raise StructuredOutputError(message, reply)
