"""Typed settings, read from the environment.

Precedence, highest first: arguments to :class:`Settings`, environment
variables, the ``.env`` file, then the defaults declared here. Every variable is
prefixed ``SYNTHIA_`` and documented in ``.env.example``, which a test keeps in
step with this module.
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import platformdirs
from pydantic import Field, PrivateAttr, SecretStr, ValidationError, field_validator
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    NoDecode,
    SettingsConfigDict,
)

from synthia.kernel.errors import ConfigError
from synthia.kernel.keys import KeyStore, SystemCredentials

if TYPE_CHECKING:
    from collections.abc import Mapping

    from synthia.kernel.keys import Credentials

APP_NAME = "synthia"
ENV_PREFIX = "SYNTHIA_"
DEFAULT_ENV_FILE = Path(".env")
DEFAULT_DISK_BUDGET_GB = 10.0
DEFAULT_PERSONA = "synthia"
DEFAULT_LOCAL_MODEL = "qwen3.5-4b"
DEFAULT_REMOTE = "openrouter"
API_KEY_SUFFIX = "_API_KEY"  # pragma: allowlist secret


class Brain(StrEnum):
    """Where a chat's turns go when it starts; ``remote`` still asks consent first."""

    LOCAL = "local"
    REMOTE = "remote"


class LocalBackend(StrEnum):
    """What the local model runs on; ``auto`` picks the fastest that starts."""

    AUTO = "auto"
    CPU = "cpu"
    VULKAN = "vulkan"
    CUDA = "cuda"
    METAL = "metal"


class LogLevel(StrEnum):
    """Minimum severity that reaches the log."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class LogFormat(StrEnum):
    """How log records are rendered."""

    CONSOLE = "console"
    JSON = "json"


def default_home() -> Path:
    """Return the platform's per-user data directory for SYNTHIA."""
    return platformdirs.user_data_path(APP_NAME, appauthor=False)


class Settings(BaseSettings):
    """Every setting SYNTHIA reads, validated once at start-up.

    Secrets are :class:`~pydantic.SecretStr`, so they print as asterisks in a
    repr, a log line or a traceback, and code must ask for the value explicitly.
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file_encoding="utf-8",
        # A copied .env.example leaves variables empty; empty means "use the default".
        env_ignore_empty=True,
        extra="ignore",
        frozen=True,
    )

    home: Path = Field(default_factory=default_home)
    log_level: LogLevel = LogLevel.INFO
    log_format: LogFormat = LogFormat.CONSOLE
    disk_budget_gb: float = Field(default=DEFAULT_DISK_BUDGET_GB, gt=0)
    openrouter_api_key: SecretStr | None = None
    remote: str = Field(default=DEFAULT_REMOTE, min_length=1)
    brain: Brain = Brain.LOCAL
    persona: str = Field(default=DEFAULT_PERSONA, min_length=1)
    local_model: str = Field(default=DEFAULT_LOCAL_MODEL, min_length=1)
    local_backend: LocalBackend = LocalBackend.AUTO
    file_roots: Annotated[tuple[Path, ...], NoDecode] = ()

    _env_file_read: Path | None = PrivateAttr(default=None)
    _stored_keys: dict[str, SecretStr] = PrivateAttr(default_factory=dict)

    @field_validator("file_roots", mode="before")
    @classmethod
    def _split_paths(cls, value: object) -> object:
        """Split a path list as PATH is split: ``;`` on Windows, ``:`` elsewhere."""
        if isinstance(value, str):
            return tuple(Path(p) for p in value.split(os.pathsep) if p.strip())
        return value

    def api_keys(self) -> dict[str, SecretStr]:
        """Return every ``SYNTHIA_<NAME>_API_KEY`` set, by its upper-case name.

        A key in the system key store wins, so ``synthia keys set`` takes
        effect even when ``.env`` still holds an older key. Then the
        environment, then the ``.env`` file :func:`load_settings` was given.
        An empty value is not set.
        """
        return api_key_variables(self._env_file_read) | self._stored_keys


def load_settings(
    env_file: Path | None = DEFAULT_ENV_FILE,
    *,
    credentials: Credentials | None = None,
    stored_keys: bool = True,
) -> Settings:
    """Load and validate the settings, reading ``env_file`` if it exists.

    The keys in the system key store, or in ``credentials`` when given, are
    read here, once, unless ``stored_keys`` is false: a command that never
    sends a key (``synthia stop``, ``synthia keys``) must still work while the
    store is locked.

    Raises:
        ConfigError: If any value fails validation, or the key store holds
            keys it cannot give. The message names the variables and the
            problem, never the values, which may be secrets.
    """
    try:
        settings = Settings(_env_file=env_file)  # pyright: ignore[reportCallIssue]
    except ValidationError as error:
        problems = "; ".join(
            f"{ENV_PREFIX}{'.'.join(map(str, e['loc'])).upper()}: {e['msg']}"
            for e in error.errors(include_input=False)
        )
        message = f"invalid configuration: {problems}"
        raise ConfigError(message) from None
    settings._env_file_read = env_file  # noqa: SLF001 # pyright: ignore[reportPrivateUsage]
    if not stored_keys:
        return settings
    store = KeyStore(
        settings.home, SystemCredentials() if credentials is None else credentials
    )
    settings._stored_keys = store.load()  # noqa: SLF001 # pyright: ignore[reportPrivateUsage]
    return settings


def environment_without_own(extra: Mapping[str, str]) -> dict[str, str]:
    """Return the environment without any ``SYNTHIA_*`` variable, plus ``extra``.

    SYNTHIA's own settings, every provider key among them, never reach a
    program it starts.
    """
    prefix = ENV_PREFIX.upper()
    inherited = {
        name: value
        for name, value in os.environ.items()
        if not name.upper().startswith(prefix)
    }
    return inherited | dict(extra)


def _is_api_key(name: str) -> bool:
    lower = name.lower()
    return lower.startswith(ENV_PREFIX.lower()) and lower.endswith(
        API_KEY_SUFFIX.lower()
    )


def api_key_variables(env_file: Path | None) -> dict[str, SecretStr]:
    """Return every ``SYNTHIA_*_API_KEY`` that is set, by its upper-case name.

    The environment and ``env_file`` are parsed exactly as :class:`Settings`
    parses them, and the environment wins. An empty value is not set.
    """
    process = EnvSettingsSource(Settings, env_ignore_empty=True).env_vars
    dotenv = DotEnvSettingsSource(
        Settings, env_file=env_file, env_ignore_empty=True
    ).env_vars
    return {
        name.upper(): SecretStr(value)
        for name, value in {**dotenv, **process}.items()
        if _is_api_key(name) and value
    }


def unknown_variables(env_file: Path | None = DEFAULT_ENV_FILE) -> list[str]:
    """Return the ``SYNTHIA_`` variables that are set but are not settings.

    The environment and ``env_file`` are parsed exactly as :class:`Settings`
    parses them. A name with an empty value is reported too: given a value
    later, it would still do nothing. Any ``SYNTHIA_*_API_KEY`` counts as
    known: it is the key of a remote provider, named after the provider.
    """
    prefix = ENV_PREFIX.lower()
    known = {prefix + name for name in Settings.model_fields}
    process = EnvSettingsSource(Settings, env_ignore_empty=False).env_vars
    dotenv = DotEnvSettingsSource(
        Settings, env_file=env_file, env_ignore_empty=False
    ).env_vars
    found = process.keys() | dotenv.keys()
    return sorted(
        n.upper()
        for n in found
        if n.startswith(prefix) and n not in known and not _is_api_key(n)
    )
