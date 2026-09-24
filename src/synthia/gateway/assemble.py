"""Build the gateway a conversation talks to, from the settings.

This is the one place that knows how the pieces fit: the OpenRouter adapter,
metered and guarded, behind a router whose budget, rate and circuit are the
same objects the guards use, beside the local model when one is installed,
with every finished call accounted for on top. Everything else receives the
finished model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from synthia.gateway.budget import BudgetLedger
from synthia.gateway.circuit import CircuitBreaker
from synthia.gateway.openai_compat import OpenAICompatibleModel
from synthia.gateway.providers import OPENROUTER, openrouter_endpoint, openrouter_info
from synthia.gateway.ratelimit import SlidingWindowLimiter
from synthia.gateway.router import RemoteHealth, Router, guard_remote
from synthia.gateway.usage import AccountingModel, UsageLog
from synthia.kernel.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    import httpx

    from synthia.gateway.protocol import ChatModel
    from synthia.kernel.bus import Event
    from synthia.kernel.config import Settings
    from synthia.models.local import LocalModel

GATEWAY_DB: Final = Path("db") / "gateway.db"


@dataclass(frozen=True, slots=True)
class Gateway:
    """The model to send requests to, and what is needed to report on it.

    ``model`` is the router with accounting on top; ``router`` is exposed for
    its decisions and capabilities.
    """

    model: ChatModel
    router: Router
    health: RemoteHealth
    usage: UsageLog


def build_gateway(
    settings: Settings,
    client: httpx.AsyncClient,
    publish: Callable[[Event], Awaitable[None]],
    local: LocalModel | None = None,
) -> Gateway:
    """Assemble the gateway; ``client`` carries every remote request.

    ``local`` is used while it is ready, as the routing rules decide.

    Raises:
        ConfigError: If no model can be reached with these settings.
    """
    key = settings.openrouter_api_key
    if key is None:
        message = "no model is available: set SYNTHIA_OPENROUTER_API_KEY in .env"
        raise ConfigError(message)
    caps = {OPENROUTER: settings.remote_daily_cap}
    health = RemoteHealth(
        OPENROUTER,
        BudgetLedger(settings.home / GATEWAY_DB, caps),
        SlidingWindowLimiter(settings.remote_rpm),
        CircuitBreaker(OPENROUTER),
        reserve=settings.remote_reserve,
    )
    endpoint = openrouter_endpoint(
        key, settings.openrouter_model, settings.openrouter_base_url
    )
    adapter = OpenAICompatibleModel(
        client=client,
        endpoint=endpoint,
        info=openrouter_info(settings.openrouter_model),
    )
    router = Router(
        guard_remote(adapter, health),
        health,
        local,
        publish=publish,
        local_ready=(lambda: False) if local is None else local.ready,
    )
    usage = UsageLog(settings.home / GATEWAY_DB)
    return Gateway(AccountingModel(router, usage, publish), router, health, usage)
