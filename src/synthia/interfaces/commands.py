"""Parse one line typed into ``synthia chat`` into what it asks for.

Parsing is pure, so every command and every mistake is tested without a
terminal. A line that does not start with ``/`` is something to say.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

PREFIX: Final = "/"


@dataclass(frozen=True, slots=True)
class Say:
    """Send ``text`` to SYNTHIA."""

    text: str


@dataclass(frozen=True, slots=True)
class SwitchPersona:
    """Continue as the persona ``key``; an empty key lists the personas."""

    key: str


@dataclass(frozen=True, slots=True)
class AdjustPersona:
    """Move some trait sliders of the current persona."""

    values: dict[str, float]


@dataclass(frozen=True, slots=True)
class ShowImage:
    """Send the image at ``path`` with ``text``."""

    path: Path
    text: str


@dataclass(frozen=True, slots=True)
class ShowBudget:
    """Print today's remote budget."""


@dataclass(frozen=True, slots=True)
class ShowModel:
    """Print which models are behind the router and the last route taken."""


@dataclass(frozen=True, slots=True)
class Reset:
    """Forget the conversation so far."""


@dataclass(frozen=True, slots=True)
class Help:
    """List the commands."""


@dataclass(frozen=True, slots=True)
class Exit:
    """Leave the chat."""


@dataclass(frozen=True, slots=True)
class Invalid:
    """A command that could not be understood; ``reason`` says why."""

    reason: str


type Command = (
    Say
    | SwitchPersona
    | AdjustPersona
    | ShowImage
    | ShowBudget
    | ShowModel
    | Reset
    | Help
    | Exit
    | Invalid
)

DEFAULT_IMAGE_QUESTION: Final = "What is in this image?"

HELP: Final = """\
/persona [name]            switch persona, or list them
/persona set wit=0.3 ...   move sliders: warmth formality wit vigilance verbosity
/image <path> [question]   send an image; quote a path that has spaces
/budget                    today's remote requests
/model                     the models behind the router, and the last route
/reset                     forget the conversation
/help                      this list
/exit                      leave (Ctrl+D works too)"""

_SIMPLE: Final[dict[str, Command]] = {
    "budget": ShowBudget(),
    "model": ShowModel(),
    "reset": Reset(),
    "help": Help(),
    "exit": Exit(),
    "quit": Exit(),
}


def parse(line: str) -> Command:
    """Return what ``line`` asks for."""
    stripped = line.strip()
    if not stripped.startswith(PREFIX):
        return Say(stripped)
    name, _, rest = stripped[len(PREFIX) :].partition(" ")
    rest = rest.strip()
    if name in _SIMPLE:
        return _SIMPLE[name] if not rest else Invalid(f"/{name} takes no arguments")
    if name == "persona":
        return _persona(rest)
    if name == "image":
        return _image(rest)
    return Invalid(f"unknown command /{name}; /help lists them")


def _persona(rest: str) -> Command:
    head, _, tail = rest.partition(" ")
    if head != "set":
        if tail:
            return Invalid("a persona name has no spaces")
        return SwitchPersona(head)
    values: dict[str, float] = {}
    for pair in tail.split():
        trait, _, raw = pair.partition("=")
        try:
            values[trait] = float(raw)
        except ValueError:
            return Invalid(f"{pair!r} is not trait=number")
    if not values:
        return Invalid("/persona set needs at least one trait=number")
    return AdjustPersona(values)


def _image(rest: str) -> Command:
    if rest.startswith('"'):
        end = rest.find('"', 1)
        if end < 0:
            return Invalid("the image path has no closing quote")
        path, text = rest[1:end], rest[end + 1 :].strip()
    else:
        path, _, text = rest.partition(" ")
    if not path:
        return Invalid("/image needs a path")
    return ShowImage(Path(path), text.strip() or DEFAULT_IMAGE_QUESTION)
