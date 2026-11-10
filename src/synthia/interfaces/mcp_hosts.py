"""What each agent that runs SYNTHIA as its MCP server may see.

The person lists hosts in ``SYNTHIA_HOME/mcp_hosts.toml``, one table each: a
``[hosts.claude-code]`` table with ``collections = ["notes", "code"]`` (the
knowledge sources it may search, by name) and ``memory = true`` (earlier
conversations too; default false). A host with no table sees nothing.
Private passages, and turns of chats that held private text, also need the
host's private grant in ``consent.json``
(``synthia consent grant --host <name> --private``).
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    ValidationError,
)

from synthia.kernel.consent import PROVIDER_NAME, ConsentBook, Host, Scope
from synthia.kernel.errors import ConfigError
from synthia.knowledge.model import MAX_SOURCE_NAME, MAX_SOURCES

HOSTS_FILE: Final = Path("mcp_hosts.toml")
MAX_HOSTS_BYTES: Final = 65_536

logger = logging.getLogger(__name__)


class HostEntry(BaseModel):
    """What one host may see: knowledge sources by name, and whether memory."""

    # Strict per field, not per model: TOML's arrays are lists, which a strict
    # tuple refuses, while a "true" written as text must stay damage.
    model_config = ConfigDict(frozen=True, extra="forbid")

    collections: tuple[
        Annotated[StrictStr, Field(min_length=1, max_length=MAX_SOURCE_NAME)], ...
    ] = Field(default=(), max_length=MAX_SOURCES)
    memory: StrictBool = False


class _HostsFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    hosts: dict[Annotated[str, Field(pattern=PROVIDER_NAME)], HostEntry] = {}


def load_hosts(path: Path) -> dict[str, HostEntry]:
    """Return what each host in ``path`` may see; none if the file does not exist.

    Raises:
        ConfigError: If the file is too large, not TOML, or does not fit the shape.
    """
    try:
        with path.open("rb") as file:
            data = file.read(MAX_HOSTS_BYTES + 1)
    except FileNotFoundError:
        return {}
    except OSError as error:
        message = f"cannot read {path}: {error.strerror}"
        raise ConfigError(message) from None
    if len(data) > MAX_HOSTS_BYTES:
        message = f"{path} is larger than {MAX_HOSTS_BYTES} bytes"
        raise ConfigError(message)
    try:
        return _HostsFile.model_validate(tomllib.loads(data.decode())).hosts
    except (
        tomllib.TOMLDecodeError,
        ValidationError,
        UnicodeDecodeError,
        RecursionError,
    ) as error:
        message = f"{path} is not a valid list of MCP hosts: {error}"
        raise ConfigError(message) from None


@dataclass(frozen=True, slots=True)
class HostPolicy:
    """What ``host`` may see now: which sources, whether memory, whether private."""

    host: Host
    collections: frozenset[str]
    memory: bool
    private: bool


def policy_of(home: Path, host: Host) -> HostPolicy:
    """Return ``host``'s policy from the files under ``home``.

    A damaged hosts file shows the host nothing; a damaged consent file is no
    private grant. Both are logged.
    """
    try:
        entry = load_hosts(home / HOSTS_FILE).get(host.name, HostEntry())
    except ConfigError as error:
        logger.warning("%s", error)
        entry = HostEntry()
    return HostPolicy(
        host,
        frozenset(entry.collections),
        entry.memory,
        ConsentBook(home).allows(host, Scope.PRIVATE),
    )
