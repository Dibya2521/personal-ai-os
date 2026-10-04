"""Where a running daemon is, and the token that lets a client talk to it.

The daemon writes ``SYNTHIA_HOME/daemon.json`` once it serves and removes it
when it stops: its port on 127.0.0.1, its process id, its version, when it
started and a random token. Whoever reads the token can run SYNTHIA's tools
as the person who owns it, so the file is made readable by its owner only
(on Windows the per-user data folder already allows only that person). A
file left by a daemon that died is recognised: no process with its id, or one
that started after the daemon wrote the file, which means the id was reused.
"""

from __future__ import annotations

import logging
import os
import secrets
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

import psutil
from pydantic import BaseModel, ConfigDict, Field, ValidationError

if TYPE_CHECKING:
    from collections.abc import Callable

DAEMON_FILE: Final = Path("daemon.json")
TOKEN_BYTES: Final = 32
SHARING_WAIT_S: Final = 1.0
_SHARING_POLL_S: Final = 0.01
_OWNER_ONLY: Final = 0o600

logger = logging.getLogger(__name__)


class DaemonInfo(BaseModel):
    """What a client needs to reach the daemon."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    port: int = Field(gt=0, lt=65536)
    pid: int = Field(gt=0)
    version: str
    started: datetime
    token: str = Field(min_length=1, repr=False)

    @property
    def base_url(self) -> str:
        """Return the daemon's HTTP root."""
        return f"http://127.0.0.1:{self.port}"

    @property
    def socket_url(self) -> str:
        """Return the daemon's WebSocket address."""
        return f"ws://127.0.0.1:{self.port}/ws"


def new_token() -> str:
    """Return a token no one can guess."""
    return secrets.token_urlsafe(TOKEN_BYTES)


def patiently[T](action: Callable[[], T]) -> T:
    """Return ``action()``, retrying for up to ``SHARING_WAIT_S`` while it is refused.

    On Windows a file another process has open cannot be replaced or removed,
    and one being replaced cannot be opened: a client reading ``daemon.json``
    while the daemon removes it made the removal fail 218 times in 300. Each
    refusal lasts only as long as the other side's read or write.

    Raises:
        PermissionError: If it is still refused when the time is up.
    """
    deadline = time.monotonic() + SHARING_WAIT_S
    while True:
        try:
            return action()
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(_SHARING_POLL_S)


def write_info(path: Path, info: DaemonInfo) -> None:
    """Write ``info`` to ``path`` as a whole, readable by its owner only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".tmp")
    partial.unlink(missing_ok=True)
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _OWNER_ONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
        file.write(info.model_dump_json() + "\n")
    patiently(lambda: partial.replace(path))


def read_info(path: Path) -> DaemonInfo | None:
    """Return the daemon described at ``path``, or None if there is none to read."""
    try:
        return DaemonInfo.model_validate_json(patiently(path.read_bytes))
    except FileNotFoundError:
        return None
    except (OSError, ValidationError) as error:
        logger.warning("ignored an unreadable %s: %s", path.name, error)
        return None


def is_running(info: DaemonInfo) -> bool:
    """Return whether the process that wrote ``info`` is still the one with its id."""
    try:
        process = psutil.Process(info.pid)
        return process.create_time() <= info.started.timestamp()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def remove_info(path: Path, pid: int) -> None:
    """Remove ``path`` if it still describes the daemon ``pid``."""
    info = read_info(path)
    if info is not None and info.pid == pid:
        patiently(lambda: path.unlink(missing_ok=True))


def started_now() -> datetime:
    """Return the time to record as the daemon's start, in UTC."""
    return datetime.now(UTC)
