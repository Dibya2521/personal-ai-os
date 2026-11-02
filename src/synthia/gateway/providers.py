"""The remote providers SYNTHIA can reach, each described by one record."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from synthia.gateway.openai_compat import Dialect, Endpoint
from synthia.gateway.types import ModelInfo

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic import SecretStr

OPENROUTER: Final = "openrouter"
# OpenRouter's app attribution headers: who is calling, shown on its dashboards.
APP_URL: Final = "https://github.com/Dibya2521/personal-ai-os"
APP_TITLE: Final = "SYNTHIA"


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
