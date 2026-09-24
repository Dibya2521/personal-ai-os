"""A persona, its trait sliders, and the system prompt rendered from them.

Each trait is a number from 0 to 1. Rendering turns it into one of three
sentences (low, middle, high), so a prompt is a pure function of the persona:
the same sliders always give the same words, which keeps behaviour testable
and cacheable. The template is a :class:`string.Template` rather than
``str.format``, because a format string in a user's file could reach into
attributes of the values it is given.
"""

from __future__ import annotations

from string import Template
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from synthia.kernel.errors import SynthiaError

LOW_BELOW: Final = 1 / 3
HIGH_FROM: Final = 2 / 3

Level = Annotated[float, Field(ge=0.0, le=1.0)]

# For each trait: what it sounds like low, in the middle, and high.
PHRASES: Final[dict[str, tuple[str, str, str]]] = {
    "warmth": (
        "Keep a cool, matter-of-fact tone.",
        "Be friendly without fuss.",
        "Be warm: notice how the person is doing, and show that you care.",
    ),
    "formality": (
        "Speak casually, as a friend would.",
        "Speak plainly and politely.",
        "Speak formally and precisely.",
    ),
    "wit": (
        "Stay earnest; no jokes.",
        "Allow a light touch of humour when it fits.",
        "Use dry wit freely, never at the person's expense.",
    ),
    "vigilance": (
        "Take requests at face value unless something is clearly wrong.",
        "Point out risks when they matter.",
        (
            "Guard security and privacy actively: flag risks, confirm before "
            "anything irreversible, and never reveal a secret."
        ),
    ),
    "verbosity": (
        "Answer in as few words as will do.",
        "Answer fully, without padding.",
        "Explain thoroughly, with the reasoning and the details.",
    ),
}

DEFAULT_TEMPLATE: Final = """\
You are $name, $description

How you speak:
$traits

What you hold to:
$principles"""


class PersonaError(SynthiaError):
    """A persona is invalid, missing, or asked for a trait it does not have."""


class Traits(BaseModel):
    """The five sliders, each from 0 to 1."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    warmth: Level
    formality: Level
    wit: Level
    vigilance: Level
    verbosity: Level

    def adjusted(self, **values: float) -> Traits:
        """Return these traits with some sliders moved.

        Raises:
            PersonaError: If a name is not a trait or a value is outside 0 to 1.
        """
        unknown = sorted(set(values) - set(PHRASES))
        if unknown:
            message = (
                f"no such trait: {', '.join(unknown)}; traits are {', '.join(PHRASES)}"
            )
            raise PersonaError(message)
        try:
            return Traits.model_validate(self.model_dump() | values)
        except ValidationError as error:
            message = "a trait must be a number from 0 to 1"
            raise PersonaError(message) from error

    def sentences(self) -> tuple[str, ...]:
        """Return one sentence per trait, in a fixed order."""
        return tuple(_phrase(PHRASES[name], getattr(self, name)) for name in PHRASES)


def _phrase(options: tuple[str, str, str], level: float) -> str:
    low, middle, high = options
    if level < LOW_BELOW:
        return low
    if level < HIGH_FROM:
        return middle
    return high


class Persona(BaseModel):
    """Who SYNTHIA is for a conversation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    traits: Traits
    principles: tuple[str, ...] = ()
    template: str = DEFAULT_TEMPLATE

    def adjusted(self, **values: float) -> Persona:
        """Return this persona with some trait sliders moved.

        Raises:
            PersonaError: If a name is not a trait or a value is outside 0 to 1.
        """
        return self.model_copy(update={"traits": self.traits.adjusted(**values)})

    def system_prompt(self) -> str:
        """Render the system prompt.

        Raises:
            PersonaError: If the template names a placeholder that does not exist.
        """
        fields = {
            "name": self.name,
            "description": self.description,
            "traits": _bullets(self.traits.sentences()),
            "principles": _bullets(self.principles),
        }
        try:
            return Template(self.template).substitute(fields)
        except (KeyError, ValueError) as error:
            message = f"persona {self.name!r} has a bad template placeholder: {error}"
            raise PersonaError(message) from error


def _bullets(lines: tuple[str, ...]) -> str:
    return "\n".join(f"- {line}" for line in lines)
