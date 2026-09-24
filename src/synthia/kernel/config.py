"""Typed settings, read from the environment.

Precedence, highest first: arguments to :class:`Settings`, environment
variables, the ``.env`` file, then the defaults declared here. Every variable is
prefixed ``SYNTHIA_`` and documented in ``.env.example``, which a test keeps in
step with this module.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import platformdirs
from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from synthia.kernel.errors import ConfigError

APP_NAME = "synthia"
ENV_PREFIX = "SYNTHIA_"
DEFAULT_ENV_FILE = Path(".env")
DEFAULT_DISK_BUDGET_GB = 10.0
# OpenRouter's free-model limit per UTC day for accounts with under 10 credits.
DEFAULT_REMOTE_DAILY_CAP = 50
# OpenRouter's documented rate limit for free models.
DEFAULT_REMOTE_RPM = 20
# Requests kept back for what only the remote can serve, once chat has gone local.
DEFAULT_REMOTE_RESERVE = 10
DEFAULT_PERSONA = "synthia"
DEFAULT_OPENROUTER_MODEL = "openrouter/free"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LOCAL_MODEL = "qwen3.5-4b"
# The smallest window the remote may have, so going local never shrinks a request.
DEFAULT_LOCAL_CONTEXT = 32_768


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
    openrouter_model: str = Field(default=DEFAULT_OPENROUTER_MODEL, min_length=1)
    openrouter_base_url: str = Field(default=DEFAULT_OPENROUTER_BASE_URL, min_length=1)
    remote_daily_cap: int = Field(default=DEFAULT_REMOTE_DAILY_CAP, ge=0)
    remote_rpm: int = Field(default=DEFAULT_REMOTE_RPM, gt=0)
    remote_reserve: int = Field(default=DEFAULT_REMOTE_RESERVE, ge=0)
    persona: str = Field(default=DEFAULT_PERSONA, min_length=1)
    local_model: str = Field(default=DEFAULT_LOCAL_MODEL, min_length=1)
    local_backend: LocalBackend = LocalBackend.AUTO
    local_context: int = Field(default=DEFAULT_LOCAL_CONTEXT, gt=0)


def load_settings(env_file: Path | None = DEFAULT_ENV_FILE) -> Settings:
    """Load and validate the settings, reading ``env_file`` if it exists.

    Raises:
        ConfigError: If any value fails validation. The message names the
            variables and the problem, never the values, which may be secrets.
    """
    try:
        return Settings(_env_file=env_file)  # pyright: ignore[reportCallIssue]
    except ValidationError as error:
        problems = "; ".join(
            f"{ENV_PREFIX}{'.'.join(map(str, e['loc'])).upper()}: {e['msg']}"
            for e in error.errors(include_input=False)
        )
        message = f"invalid configuration: {problems}"
        raise ConfigError(message) from None
