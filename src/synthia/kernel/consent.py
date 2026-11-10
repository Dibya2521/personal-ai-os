"""The person's decisions about what may leave this machine, and for whom.

Each decision is a :class:`Consent`: a receiver (a remote provider, or another
agent that runs SYNTHIA as its MCP server), a scope
(the conversation's text, or private memory and knowledge passages), yes or
no, when, and whether it was answered in a chat or with ``synthia consent``.
They are kept in ``SYNTHIA_HOME/consent.json``, the newest decision for a
receiver and scope replacing the one before. No decision means no: the request
stays local.

A provider is its name and the address its requests go to, so a yes given to
one address does not follow the name to another. A host is the name it was
installed under; a provider and a host of one name are different receivers.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, Literal, final

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from synthia.kernel.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Callable

CONSENT_FILE: Final = "consent.json"
CONSENT_VERSION: Final = 1
MAX_CONSENT_BYTES: Final = 262_144
MAX_CONSENTS: Final = 1024
PROVIDER_NAME: Final = r"^[a-z0-9][a-z0-9-]{0,31}$"
MAX_URL_LENGTH: Final = 2048

logger = logging.getLogger(__name__)


class Scope(StrEnum):
    """What a consent lets leave the machine."""

    CHAT = "chat"
    PRIVATE = "private"


class ConsentError(ConfigError):
    """The consent file could not be read or written."""


class Provider(BaseModel):
    """A remote model provider: its name in ``providers.toml`` and its ``base_url``."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["provider"] = "provider"
    name: Annotated[str, Field(pattern=PROVIDER_NAME)]
    url: Annotated[str, Field(min_length=1, max_length=MAX_URL_LENGTH)]


class Host(BaseModel):
    """Another agent running SYNTHIA as its MCP server, by its installed name."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["host"] = "host"
    name: Annotated[str, Field(pattern=PROVIDER_NAME)]


type Receiver = Annotated[Provider | Host, Field(discriminator="kind")]


class Consent(BaseModel):
    """One decision: may ``scope`` go to ``receiver``."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    receiver: Receiver
    scope: Scope
    granted: bool
    decided_at: AwareDatetime
    by: Literal["chat", "cli"]


class _Book(BaseModel):
    # Strict, so a "true" or a 1 written by hand is damage, never a yes.
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[1]
    consents: tuple[Consent, ...] = Field(max_length=MAX_CONSENTS)

    @model_validator(mode="after")
    def _one_decision_each(self) -> _Book:
        keys = [(c.receiver, c.scope) for c in self.consents]
        if len(set(keys)) != len(keys):
            message = "two decisions for one receiver and scope"
            raise ValueError(message)
        return self


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


@final
class ConsentBook:
    """The decisions in ``home/consent.json``."""

    def __init__(self, home: Path, clock: Callable[[], dt.datetime] = _now) -> None:
        self.path = home / CONSENT_FILE
        self._clock = clock

    def consents(self) -> tuple[Consent, ...]:
        """Return every decision, oldest first.

        Raises:
            ConsentError: If the file cannot be read or is damaged.
        """
        try:
            with self.path.open("rb") as file:
                data = file.read(MAX_CONSENT_BYTES + 1)
        except FileNotFoundError:
            return ()
        except OSError as error:
            message = f"cannot read {self.path}: {error.strerror}"
            raise ConsentError(message) from None
        if len(data) > MAX_CONSENT_BYTES:
            message = f"{self.path} is larger than {MAX_CONSENT_BYTES} bytes"
            raise ConsentError(message)
        try:
            return _Book.model_validate_json(data).consents
        except ValidationError:
            message = (
                f"{self.path} is damaged, so nothing may leave this machine; "
                "delete it and decide again with synthia consent grant <provider>"
            )
            raise ConsentError(message) from None

    def decision(self, receiver: Provider | Host, scope: Scope) -> bool | None:
        """Return the recorded answer for ``scope`` to ``receiver``, None if none.

        Raises:
            ConsentError: If the file cannot be read or is damaged.
        """
        for consent in self.consents():
            if (consent.receiver, consent.scope) == (receiver, scope):
                return consent.granted
        return None

    def granted(self, receiver: Provider | Host, scope: Scope) -> bool:
        """Return whether ``scope`` may go to ``receiver``.

        Raises:
            ConsentError: If the file cannot be read or is damaged.
        """
        return self.decision(receiver, scope) is True

    def allows(self, receiver: Provider | Host, scope: Scope) -> bool:
        """Return whether ``scope`` may go to ``receiver``; a damaged file is no."""
        try:
            return self.granted(receiver, scope)
        except ConsentError as error:
            logger.warning("%s", error)
            return False

    def decide(
        self,
        receiver: Provider | Host,
        scope: Scope,
        *,
        granted: bool,
        by: Literal["chat", "cli"],
    ) -> Consent:
        """Record a decision, replacing the earlier one for ``receiver`` and ``scope``.

        Raises:
            ConsentError: If the file is damaged or cannot be written.
        """
        consent = Consent(
            receiver=receiver,
            scope=scope,
            granted=granted,
            decided_at=self._clock(),
            by=by,
        )
        kept = [
            c for c in self.consents() if (c.receiver, c.scope) != (receiver, scope)
        ]
        self._write((*kept, consent))
        return consent

    def revoke(
        self,
        name: str,
        scope: Scope,
        *,
        current: Provider | None = None,
        by: Literal["chat", "cli"],
    ) -> tuple[Consent, ...]:
        """Refuse ``scope`` to every address of provider ``name``, and ``current``.

        Raises:
            ConsentError: If the file is damaged or cannot be written.
        """
        earlier = self.consents()
        receivers: dict[Provider | Host, None] = dict.fromkeys(
            c.receiver
            for c in earlier
            if isinstance(c.receiver, Provider)
            and (c.receiver.name, c.scope) == (name, scope)
        )
        if current is not None:
            receivers[current] = None
        if not receivers:
            return ()
        when = self._clock()
        refused = tuple(
            Consent(receiver=r, scope=scope, granted=False, decided_at=when, by=by)
            for r in receivers
        )
        kept = [
            c for c in earlier if c.scope is not scope or c.receiver not in receivers
        ]
        self._write((*kept, *refused))
        return refused

    def _write(self, consents: tuple[Consent, ...]) -> None:
        try:
            book = _Book(version=CONSENT_VERSION, consents=consents)
        except ValidationError:
            message = (
                f"{self.path} holds {MAX_CONSENTS} decisions, the most it may; "
                "remove the ones no longer needed from it"
            )
            raise ConsentError(message) from None
        text = json.dumps(book.model_dump(mode="json"), indent=2)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="",
                dir=self.path.parent,
                prefix=self.path.name,
                suffix=".part",
                delete_on_close=False,
            ) as part:
                part.write(text + "\n")
                part.close()
                Path(part.name).replace(self.path)
        except OSError as error:
            message = f"cannot write {self.path}: {error.strerror}"
            raise ConsentError(message) from None
