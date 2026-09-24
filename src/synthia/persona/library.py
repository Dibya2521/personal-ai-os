"""Load the built-in and the user's personas, resolving blends.

Each persona is a TOML file named after the key used to pick it (``edith.toml``
is ``/persona edith``). A file gives either ``[traits]`` or ``[blend]``: a blend
names other personas with weights, its traits are their weighted mean, and its
principles are its own followed by theirs, without repeats. That is how the
default persona is defined, so moving one ingredient moves the default too.

Files in the user's directory replace built-ins with the same key, so a user
can redefine even the default without touching the package.
"""

from __future__ import annotations

import tomllib
from importlib.resources import files
from typing import TYPE_CHECKING, Annotated, Final, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from synthia.persona.model import (
    DEFAULT_TEMPLATE,
    PHRASES,
    Persona,
    PersonaError,
    Traits,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from importlib.resources.abc import Traversable
    from pathlib import Path

SUFFIX: Final = ".toml"

Weight = Annotated[float, Field(gt=0.0)]


class PersonaFile(BaseModel):
    """What one persona file may contain."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    traits: Traits | None = None
    blend: dict[str, Weight] | None = None
    principles: tuple[str, ...] = ()
    template: str = DEFAULT_TEMPLATE

    @model_validator(mode="after")
    def _traits_or_blend(self) -> Self:
        if (self.traits is None) == (self.blend is None):
            message = "give exactly one of [traits] or [blend]"
            raise ValueError(message)
        if self.blend is not None and not self.blend:
            message = "a blend needs at least one persona"
            raise ValueError(message)
        return self


def parse(text: str, source: str) -> PersonaFile:
    """Parse one persona file; ``source`` names it in errors.

    Raises:
        PersonaError: If the file is not valid TOML or not a valid persona.
    """
    try:
        return PersonaFile.model_validate(tomllib.loads(text))
    except tomllib.TOMLDecodeError as error:
        message = f"{source} is not valid TOML: {error}"
        raise PersonaError(message) from error
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(map(str, e['loc'])) or 'file'}: {e['msg']}"
            for e in error.errors()
        )
        message = f"{source} is not a valid persona: {problems}"
        raise PersonaError(message) from error


def _read(entries: Iterable[Traversable | Path]) -> dict[str, PersonaFile]:
    found: dict[str, PersonaFile] = {}
    for entry in entries:
        if entry.is_file() and entry.name.endswith(SUFFIX):
            key = entry.name.removesuffix(SUFFIX)
            found[key] = parse(entry.read_text(encoding="utf-8"), entry.name)
    return found


def builtin_files() -> dict[str, PersonaFile]:
    """Return the personas shipped with SYNTHIA, by key."""
    return _read(files("synthia.persona").joinpath("builtin").iterdir())


class PersonaLibrary:
    """Every persona available, built-in and the user's, fully resolved."""

    def __init__(self, user_dir: Path | None = None) -> None:
        """Load the built-ins, then the ``.toml`` files in ``user_dir`` over them.

        Raises:
            PersonaError: If a file is invalid, or a blend names a missing
                persona or includes itself.
        """
        found = builtin_files()
        if user_dir is not None and user_dir.is_dir():
            found |= _read(sorted(user_dir.iterdir()))
        self._personas = resolve(found)

    def names(self) -> tuple[str, ...]:
        """Return the keys of every persona, sorted."""
        return tuple(sorted(self._personas))

    def get(self, key: str) -> Persona:
        """Return the persona picked by ``key``.

        Raises:
            PersonaError: If there is none.
        """
        try:
            return self._personas[key]
        except KeyError:
            message = f"no persona {key!r}; there are {', '.join(self.names())}"
            raise PersonaError(message) from None


def resolve(found: Mapping[str, PersonaFile]) -> dict[str, Persona]:
    """Turn parsed files into personas, computing every blend.

    Raises:
        PersonaError: If a blend names a missing persona or includes itself.
    """
    resolved: dict[str, Persona] = {}
    for key in found:
        _resolve(key, found, resolved, ())
    return resolved


def _resolve(
    key: str,
    found: Mapping[str, PersonaFile],
    resolved: dict[str, Persona],
    chain: tuple[str, ...],
) -> Persona:
    if key in resolved:
        return resolved[key]
    if key in chain:
        message = f"persona blend loops: {' -> '.join((*chain, key))}"
        raise PersonaError(message)
    file = found[key]
    principles = list(file.principles)
    if file.traits is not None:
        traits = file.traits
    else:
        blend = file.blend or {}
        missing = sorted(set(blend) - set(found))
        if missing:
            message = (
                f"{key} blends a persona that does not exist: {', '.join(missing)}"
            )
            raise PersonaError(message)
        parts = [
            (_resolve(part, found, resolved, (*chain, key)), weight)
            for part, weight in blend.items()
        ]
        traits = _mean(parts)
        for part, _ in parts:
            principles += [p for p in part.principles if p not in principles]
    persona = Persona(
        name=file.name,
        description=file.description,
        traits=traits,
        principles=tuple(principles),
        template=file.template,
    )
    resolved[key] = persona
    return persona


def _mean(parts: list[tuple[Persona, float]]) -> Traits:
    total = sum(weight for _, weight in parts)
    levels: dict[str, float] = {}
    for trait in PHRASES:
        values = [float(getattr(p.traits, trait)) for p, _ in parts]
        mean = sum(v * w for v, (_, w) in zip(values, parts, strict=True)) / total
        # Rounding can put the mean an ulp outside its parts, and so past 1.0.
        levels[trait] = min(max(mean, min(values)), max(values))
    return Traits.model_validate(levels)
