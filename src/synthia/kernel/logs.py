"""Structured logging on the standard library, with secrets redacted.

Every library SYNTHIA uses already logs through :mod:`logging`, so one handler
on the root logger sees everything. Records are rendered as one JSON object per
line or as readable console lines, and carry the correlation id of the work
that produced them.

Redaction runs on the fully formatted text, after arguments, extra fields and
tracebacks have been rendered into it, because a secret can arrive through any
of them.
"""

from __future__ import annotations

import json
import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar, TextIO, override

from pydantic import SecretStr

from synthia.kernel.config import LogFormat, Settings

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

REDACTED = "**********"
HANDLER_NAME = "synthia"

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)

# Attributes every LogRecord has. Anything else on a record came from `extra=`.
_STANDARD_ATTRIBUTES = frozenset(
    vars(logging.LogRecord("", logging.INFO, "", 0, "", None, None))
) | {"message", "asctime", "taskName"}


def current_correlation_id() -> str | None:
    """Return the correlation id bound in the current context, if any."""
    return _correlation_id.get()


@contextmanager
def correlation(correlation_id: str) -> Generator[None]:
    """Tag every record logged inside the block, including in tasks it starts."""
    token = _correlation_id.set(correlation_id)
    try:
        yield
    finally:
        _correlation_id.reset(token)


class Redactor:
    """Replace every registered secret value in a text with a fixed mask."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        self._secrets: set[str] = set()
        self.add(secrets)

    def add(self, secrets: Iterable[str]) -> None:
        """Register more secret values.

        A value that is part of the mask itself, the empty string included, is
        ignored: the mask already shows it, and masking it could never finish.
        """
        self._secrets.update(secret for secret in secrets if secret not in REDACTED)

    def redact(self, text: str) -> str:
        """Return ``text`` with every registered secret masked."""
        # Longest first, so a secret containing another is masked whole.
        for secret in sorted(self._secrets, key=len, reverse=True):
            # The mask next to leftover text can rebuild the secret ("xx*" with
            # secret "x*"). Each pass removes a character the mask does not
            # contain, so the loop ends.
            while secret in text:
                text = text.replace(secret, REDACTED)
        return text


def _extra_fields(record: logging.LogRecord) -> dict[str, Any]:
    return {k: v for k, v in vars(record).items() if k not in _STANDARD_ATTRIBUTES}


class _RedactingFormatter(logging.Formatter):
    def __init__(self, redactor: Redactor) -> None:
        super().__init__()
        self._redactor = redactor

    @override
    def format(self, record: logging.LogRecord) -> str:
        return self._redactor.redact(self.render(record))

    def render(self, record: logging.LogRecord) -> str:
        """Return the record as text, before redaction."""
        raise NotImplementedError


class JsonFormatter(_RedactingFormatter):
    """Render each record as a single-line JSON object."""

    @override
    def render(self, record: logging.LogRecord) -> str:
        document: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "correlation_id": current_correlation_id(),
            **_extra_fields(record),
        }
        if record.exc_info:
            document["exc"] = self.formatException(record.exc_info)
        return json.dumps(document, default=str, ensure_ascii=False)


class ConsoleFormatter(_RedactingFormatter):
    """Render each record as one readable line, with the traceback after it."""

    TIME_FORMAT: ClassVar[str] = "%H:%M:%S"

    @override
    def render(self, record: logging.LogRecord) -> str:
        when = datetime.fromtimestamp(record.created).astimezone()
        parts = [
            when.strftime(self.TIME_FORMAT),
            f"{record.levelname:<7}",
            f"{record.name}:",
            record.getMessage(),
        ]
        if (correlation_id := current_correlation_id()) is not None:
            parts.append(f"[{correlation_id}]")
        parts.extend(f"{k}={v}" for k, v in _extra_fields(record).items())
        line = " ".join(parts)
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


def secret_values(settings: Settings) -> list[str]:
    """Return the value of every secret setting that is set."""
    return [
        value.get_secret_value()
        for value in vars(settings).values()
        if isinstance(value, SecretStr)
    ]


def configure_logging(settings: Settings, stream: TextIO | None = None) -> Redactor:
    """Install SYNTHIA's handler on the root logger, replacing an earlier one.

    Returns the redactor, so secrets obtained later (a token issued at run
    time) can be registered with it.
    """
    redactor = Redactor(secret_values(settings))
    formatter = (
        JsonFormatter(redactor)
        if settings.log_format is LogFormat.JSON
        else ConsoleFormatter(redactor)
    )
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.set_name(HANDLER_NAME)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    for existing in [h for h in root.handlers if h.get_name() == HANDLER_NAME]:
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(settings.log_level.value)
    return redactor
