"""Parse one line typed into ``synthia chat`` into what it asks for.

Parsing is pure, so every command and every mistake is tested without a
terminal. A line that does not start with ``/`` is something to say.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from synthia.gateway.types import Reasoning

if TYPE_CHECKING:
    from collections.abc import Callable

PREFIX: Final = "/"


@dataclass(frozen=True, slots=True)
class Say:
    """Send ``text`` to SYNTHIA."""

    text: str


@dataclass(frozen=True, slots=True)
class Plan:
    """Do ``task`` as a plan of steps: plan first, then each step, then answer."""

    task: str


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
class Think:
    """Think at ``level`` from the next turn on; ``None`` shows the level."""

    level: Reasoning | None


@dataclass(frozen=True, slots=True)
class UseRemote:
    """Allow, or stop, sending turns off the machine; ``None`` shows which."""

    on: bool | None


@dataclass(frozen=True, slots=True)
class KeepPrivate:
    """Stop, or resume, remembering turns; ``None`` shows which."""

    on: bool | None


@dataclass(frozen=True, slots=True)
class Forget:
    """Forget the last turn: out of the conversation and out of memory."""


@dataclass(frozen=True, slots=True)
class ShowBudget:
    """Print today's remote budget."""


@dataclass(frozen=True, slots=True)
class ShowModel:
    """Print which models are behind the router and the last route taken."""


@dataclass(frozen=True, slots=True)
class ShowTools:
    """Print the tools SYNTHIA may call, where each works and what it may change."""


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
    | Plan
    | SwitchPersona
    | AdjustPersona
    | ShowImage
    | Think
    | UseRemote
    | KeepPrivate
    | Forget
    | ShowBudget
    | ShowModel
    | ShowTools
    | Reset
    | Help
    | Exit
    | Invalid
)

DEFAULT_IMAGE_QUESTION: Final = "What is in this image?"

HELP: Final = """\
/plan <task>               plan the task in steps, do each step, then answer
/persona [name]            switch persona, or list them
/persona set wit=0.3 ...   move sliders: warmth formality wit vigilance verbosity
/image <path> [question]   send an image; quote a path that has spaces
/think [level]             how much to think: off low medium high auto, or show it
/remote [on|off]           send turns to the remote model, or keep them local
/private [on|off]          stop remembering this conversation, or start again
/forget                    forget the last turn, here and in memory
/budget                    today's remote requests
/model                     the models behind the router, and the last route
/tools                     the tools SYNTHIA may call, and which ask first
/reset                     forget the conversation
/help                      this list
/exit                      leave (Ctrl+D works too)"""

_SIMPLE: Final[dict[str, Command]] = {
    "budget": ShowBudget(),
    "model": ShowModel(),
    "tools": ShowTools(),
    "reset": Reset(),
    "forget": Forget(),
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
    if name in _WITH_ARGUMENTS:
        return _WITH_ARGUMENTS[name](rest)
    return Invalid(f"unknown command /{name}; /help lists them")


_SWITCH: Final = {"on": True, "off": False}


def _switch(rest: str) -> bool | Invalid | None:
    if not rest:
        return None
    if rest.lower() not in _SWITCH:
        return Invalid(f"{rest!r} is not on or off")
    return _SWITCH[rest.lower()]


def _remote(rest: str) -> Command:
    on = _switch(rest)
    return on if isinstance(on, Invalid) else UseRemote(on)


def _private(rest: str) -> Command:
    on = _switch(rest)
    return on if isinstance(on, Invalid) else KeepPrivate(on)


def _think(rest: str) -> Command:
    if not rest:
        return Think(None)
    try:
        return Think(Reasoning(rest.lower()))
    except ValueError:
        levels = ", ".join(level.value for level in Reasoning)
        return Invalid(f"{rest!r} is not a thinking level; use one of {levels}")


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


def _plan(rest: str) -> Command:
    return Plan(rest) if rest else Invalid("/plan needs a task")


_WITH_ARGUMENTS: Final[dict[str, Callable[[str], Command]]] = {
    "plan": _plan,
    "persona": _persona,
    "image": _image,
    "think": _think,
    "remote": _remote,
    "private": _private,
}
