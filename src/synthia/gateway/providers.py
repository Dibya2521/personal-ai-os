"""The remote providers SYNTHIA can reach: an endpoint, and what its model can do."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from synthia.gateway.openai_compat import Endpoint
from synthia.gateway.types import ModelInfo
from synthia.kernel.config import DEFAULT_OPENROUTER_BASE_URL, DEFAULT_OPENROUTER_MODEL

if TYPE_CHECKING:
    from pydantic import SecretStr

OPENROUTER: Final = "openrouter"
OPENROUTER_BASE_URL: Final = DEFAULT_OPENROUTER_BASE_URL
OPENROUTER_FREE: Final = DEFAULT_OPENROUTER_MODEL
# openrouter/free sends each request to a free model picked at random. The
# smallest context window among the free models it may pick is all a request
# can rely on: 32,768 tokens in OpenRouter's model list on 2026-09-23.
OPENROUTER_FREE_CONTEXT: Final = 32_768
# OpenRouter's app attribution headers: who is calling, shown on its dashboards.
APP_URL: Final = "https://github.com/Dibya2521/personal-ai-os"
APP_TITLE: Final = "SYNTHIA"


def openrouter_endpoint(
    api_key: SecretStr,
    model: str = OPENROUTER_FREE,
    base_url: str = OPENROUTER_BASE_URL,
) -> Endpoint:
    """Return the OpenRouter endpoint for ``model``, with the attribution headers."""
    headers = {"HTTP-Referer": APP_URL, "X-Title": APP_TITLE}
    return Endpoint(base_url, model, api_key, headers)


def openrouter_info(model: str = OPENROUTER_FREE) -> ModelInfo:
    """Return what ``model`` on OpenRouter can be assumed to do.

    ``openrouter/free`` filters for a free model that supports what a request
    needs, images and tools included. Any other model is assumed to take text
    only until it is described here, so nothing is sent that it may refuse.
    """
    if model == OPENROUTER_FREE:
        return ModelInfo(model, OPENROUTER_FREE_CONTEXT, vision=True, tools=True)
    return ModelInfo(model, OPENROUTER_FREE_CONTEXT, vision=False, tools=False)
