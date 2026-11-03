"""The remote providers SYNTHIA can reach, each described by one record.

OpenRouter is built in. A person adds any other OpenAI-compatible provider as a
``[providers.<name>]`` table in ``SYNTHIA_HOME/providers.toml``, and
``SYNTHIA_REMOTE`` names the one in the remote slot. That provider is asked
only when remote is turned on in a chat (``/remote on`` or
``synthia chat --remote``); every other turn stays on the machine.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Final
from urllib.parse import urlsplit

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError

from synthia.gateway.clients import LOOPBACK_HOSTS
from synthia.gateway.openai_compat import Dialect, Endpoint
from synthia.gateway.types import ModelInfo
from synthia.kernel.config import API_KEY_SUFFIX, ENV_PREFIX
from synthia.kernel.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic import SecretStr

    from synthia.kernel.config import Settings

OPENROUTER: Final = "openrouter"
# OpenRouter's app attribution headers: who is calling, shown on its dashboards.
APP_URL: Final = "https://github.com/Dibya2521/personal-ai-os"
APP_TITLE: Final = "SYNTHIA"

PROVIDERS_FILE: Final = Path("providers.toml")
MAX_PROVIDERS_FILE_BYTES: Final = 64 * 1024
PROVIDER_NAME: Final = r"^[a-z0-9][a-z0-9-]{0,31}$"
PROVIDER_NAME_RULE: Final = (
    "a provider name is 1 to 32 lowercase letters, digits and hyphens, "
    "starting with a letter or a digit"
)
DEFAULT_REQUESTS_PER_MINUTE: Final = 20
# A ceiling on what one UTC day can cost on a paid key; raised per provider.
DEFAULT_DAILY_CAP: Final = 200
# The largest signed 32-bit integer: far above any real window, rate or cap,
# and small enough for every store and server that receives the number.
MAX_INT32: Final = 2**31 - 1
HTTPS: Final = "https"
HTTP: Final = "http"
SHOWN_NAME: Final = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
HIDDEN_NAME: Final = "<a name longer than 32 characters or with other characters>"
REMOTE_KEY_HINT: Final = (
    "store the remote provider's key with synthia keys set <name>, or set it "
    "in .env as SYNTHIA_<NAME>_API_KEY, NAME being "
    "SYNTHIA_REMOTE in capitals with - as _ (SYNTHIA_OPENROUTER_API_KEY for the "
    "default, openrouter; SYNTHIA_OPENAI_API_KEY for a provider named openai)"
)
_PYDANTIC_VALUE_ERROR: Final = "Value error, "
_TOML_POSITION: Final = re.compile(
    r"\((?P<where>at line \d+, column \d+|at end of document)\)$"
)


@dataclass(frozen=True, slots=True)
class RemoteProvider:
    """A remote provider: where it is, the model asked for, and its limits.

    ``name`` is the key the daily budget is counted under. ``daily_cap`` is
    used until the provider has reported the cap of the key in use, which
    only a provider with ``reports_daily_cap`` can do. ``dialect`` says how
    the provider takes what the OpenAI chat API has no field for.
    """

    name: str
    base_url: str
    model: ModelInfo
    requests_per_minute: int
    daily_cap: int
    headers: Mapping[str, str]
    dialect: Dialect
    reports_daily_cap: bool

    def endpoint(self, api_key: SecretStr) -> Endpoint:
        """Return the endpoint that sends ``api_key`` and the headers to this model."""
        return Endpoint(
            self.base_url,
            self.model.id,
            api_key,
            dict(self.headers),
            dialect=self.dialect,
        )


OPENROUTER_FREE: Final = RemoteProvider(
    name=OPENROUTER,
    base_url="https://openrouter.ai/api/v1",
    # openrouter/free sends each request to a free model picked at random, so a
    # request can rely only on the smallest window among them: 32,768 tokens in
    # OpenRouter's model list on 2026-09-23. It filters for one that takes the
    # request's images and tools.
    model=ModelInfo("openrouter/free", 32_768, vision=True, tools=True),
    # OpenRouter's documented rate limit for free models.
    requests_per_minute=20,
    # OpenRouter's free-model limit per UTC day for a key with under 10 credits
    # bought; 1000 once 10 are bought, which the key's own record reports.
    daily_cap=50,
    headers=MappingProxyType({"HTTP-Referer": APP_URL, "X-Title": APP_TITLE}),
    dialect=Dialect.OPENROUTER,
    # OpenRouter's GET /key returns the key's free-model count and limit.
    reports_daily_cap=True,
)


def _check_base_url(url: str) -> str:
    """Return ``url`` if the key may be sent to it.

    The messages never repeat the URL, which could carry a key by mistake.

    Raises:
        ValueError: If the URL has a space or a control character, is not
            https (http is allowed only to localhost, 127.0.0.1 and ::1), has
            no host, a bad port, a user name or password, a query or a
            fragment.
    """
    # urlsplit drops or keeps some of these where httpx refuses the URL, so
    # the two would disagree about the host.
    if any(c.isspace() or not c.isprintable() for c in url):
        message = "base_url must not contain spaces or control characters"
        raise ValueError(message)
    parts = urlsplit(url)
    host = parts.hostname
    if "@" in parts.netloc:
        message = (
            "base_url must not carry a user name or password; "
            "store the key with synthia keys set <name>, or in .env as "
            "SYNTHIA_<NAME>_API_KEY"
        )
        raise ValueError(message)
    if not host:
        message = "base_url needs a host, as in https://api.example.com/v1"
        raise ValueError(message)
    try:
        _ = parts.port
    except ValueError:
        message = "base_url has a port that is not a number from 0 to 65535"
        raise ValueError(message) from None
    if parts.query or parts.fragment:
        message = (
            "base_url must not have a query or a fragment; "
            "requests go to base_url + /chat/completions"
        )
        raise ValueError(message)
    if parts.scheme == HTTPS or (parts.scheme == HTTP and host in LOOPBACK_HOSTS):
        return url
    message = (
        "base_url must start with https://, or with http:// only for this "
        "machine (exactly localhost, 127.0.0.1 or [::1]), so the key never "
        "crosses a network or a proxy unencrypted"
    )
    raise ValueError(message)


class ProviderEntry(BaseModel):
    """One ``[providers.<name>]`` table of ``providers.toml``.

    Every number and switch must be written as its TOML type: a quoted
    ``"20"`` or a ``1`` for ``true`` is refused, not converted.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    base_url: Annotated[str, AfterValidator(_check_base_url)]
    model: str = Field(min_length=1)
    context_window: int = Field(gt=0, le=MAX_INT32)
    tools: bool = False
    vision: bool = False
    requests_per_minute: int = Field(
        default=DEFAULT_REQUESTS_PER_MINUTE, gt=0, le=MAX_INT32
    )
    daily_cap: int = Field(default=DEFAULT_DAILY_CAP, gt=0, le=MAX_INT32)
    dialect: Dialect = Field(default=Dialect.OPENAI, strict=False)
    reports_daily_cap: bool = False

    def provider(self, name: str) -> RemoteProvider:
        """Return the provider this table describes, named ``name``.

        It gets no extra headers: OpenRouter's attribution headers belong to
        the built-in provider alone.
        """
        return RemoteProvider(
            name=name,
            base_url=self.base_url,
            model=ModelInfo(
                self.model, self.context_window, vision=self.vision, tools=self.tools
            ),
            requests_per_minute=self.requests_per_minute,
            daily_cap=self.daily_cap,
            headers=MappingProxyType[str, str]({}),
            dialect=self.dialect,
            reports_daily_cap=self.reports_daily_cap,
        )


class _ProvidersFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    providers: dict[Annotated[str, Field(pattern=PROVIDER_NAME)], ProviderEntry] = {}


def load_providers(home: Path) -> dict[str, RemoteProvider]:
    """Return the built-in OpenRouter provider, then each in ``providers.toml``.

    Without the file there is only the built-in one. An error names the file,
    the table and the field, never a value from the file, which may hold a
    key pasted by mistake.

    Raises:
        ConfigError: If the file cannot be read, is larger than
            ``MAX_PROVIDERS_FILE_BYTES``, is not UTF-8 TOML, does not fit the
            shape of :class:`ProviderEntry`, or redefines ``openrouter``.
    """
    path = home / PROVIDERS_FILE
    text = _read(path)
    entries = {} if text is None else _parse(path, text)
    if OPENROUTER in entries:
        message = (
            f"{path}: [providers.{OPENROUTER}] is built in and cannot be "
            "redefined; give the table another name"
        )
        raise ConfigError(message)
    added = {name: entry.provider(name) for name, entry in entries.items()}
    return {OPENROUTER: OPENROUTER_FREE} | added


def _read(path: Path) -> str | None:
    try:
        with path.open("rb") as file:
            data = file.read(MAX_PROVIDERS_FILE_BYTES + 1)
    except FileNotFoundError:
        return None
    except OSError as error:
        message = f"{path} could not be read: {error.strerror}"
        raise ConfigError(message) from None
    if len(data) > MAX_PROVIDERS_FILE_BYTES:
        message = (
            f"{path} is larger than {MAX_PROVIDERS_FILE_BYTES:,} bytes, "
            "the most a provider list may be"
        )
        raise ConfigError(message)
    try:
        # utf-8-sig drops the byte order mark Windows Notepad writes, which
        # TOML would otherwise reject at line 1, column 1.
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        message = f"{path} is not UTF-8 text; save it as UTF-8"
        raise ConfigError(message) from None


def _parse(path: Path, text: str) -> dict[str, ProviderEntry]:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        found = _TOML_POSITION.search(str(error))
        where = "" if found is None else f" {found['where']}"
        message = f"{path} is not valid TOML{where}"
        raise ConfigError(message) from None
    # Deep nesting overflows tomllib's recursion; a number past Python's
    # integer digit limit raises a plain ValueError.
    except (RecursionError, ValueError):
        message = (
            f"{path} cannot be read: it nests too deeply or holds a number "
            "with too many digits"
        )
        raise ConfigError(message) from None
    try:
        return _ProvidersFile.model_validate(data).providers
    except ValidationError as error:
        problems = "; ".join(
            _problem(e["loc"], e["msg"].removeprefix(_PYDANTIC_VALUE_ERROR))
            for e in error.errors(include_input=False, include_url=False)
        )
        message = f"{path} is not a valid provider list: {problems}"
        raise ConfigError(message) from None


def _shown(name: int | str) -> str:
    """Return a name from the file as it may be shown, a key pasted there never."""
    text = str(name)
    return text if SHOWN_NAME.fullmatch(text) else HIDDEN_NAME


def _problem(loc: tuple[int | str, ...], message: str) -> str:
    match loc:
        case ("providers", name, "[key]"):
            return f"[providers.{_shown(name)}]: {PROVIDER_NAME_RULE}"
        case ("providers", name, *field) if field:
            where = ".".join(map(_shown, field))
            return f"[providers.{_shown(name)}] {where}: {message}"
        case _:
            return f"{'.'.join(map(_shown, loc)) or 'file'}: {message}"


def select_remote(settings: Settings) -> RemoteProvider:
    """Return the provider ``SYNTHIA_REMOTE`` names.

    Raises:
        ConfigError: If ``providers.toml`` is invalid, or no provider has
            that name; the message lists the names there are.
    """
    known = load_providers(settings.home)
    chosen = known.get(settings.remote)
    if chosen is None:
        message = (
            f"SYNTHIA_REMOTE={_shown(settings.remote)} is not a provider; "
            f"the providers are {', '.join(known)}. Add one as a "
            f"[providers.<name>] table in {settings.home / PROVIDERS_FILE}"
        )
        raise ConfigError(message)
    return chosen


def key_variable(name: str) -> str:
    """Return the variable that holds provider ``name``'s key.

    ``SYNTHIA_<NAME>_API_KEY``, NAME being ``name`` in capitals with ``-`` as
    ``_``. Provider names have no capitals and no ``_``, so two providers
    never share a variable.
    """
    return f"{ENV_PREFIX}{name.upper().replace('-', '_')}{API_KEY_SUFFIX}"


def remote_key(settings: Settings, provider: RemoteProvider) -> SecretStr | None:
    """Return ``provider``'s own key, or ``None`` when it is not set.

    Only the provider's own variable is read (:func:`key_variable`), so a key
    never reaches another provider's host when ``SYNTHIA_REMOTE`` changes.
    A key in the system key store wins over ``.env``.
    """
    key = settings.api_keys().get(key_variable(provider.name))
    if key is None and provider.name == OPENROUTER:
        return settings.openrouter_api_key
    return key
