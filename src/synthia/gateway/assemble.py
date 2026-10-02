"""Build the gateway a conversation talks to, from the settings.

This is the one place that knows how the pieces fit: the OpenRouter adapter,
metered and guarded, behind a router whose budget, rate and circuit are the
same objects the guards use, beside the local model when one is installed,
with every finished call accounted for on top. Everything else receives the
finished model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final

from synthia.gateway.budget import BudgetLedger
from synthia.gateway.circuit import CircuitBreaker
from synthia.gateway.errors import GatewayError
from synthia.gateway.openai_compat import OpenAICompatibleModel
from synthia.gateway.providers import OPENROUTER_FREE
from synthia.gateway.ratelimit import SlidingWindowLimiter
from synthia.gateway.router import RemoteHealth, Router, guard_remote
from synthia.gateway.usage import AccountingModel, UsageLog, free_requests_today
from synthia.kernel.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine

    import httpx
    from pydantic import SecretStr

    from synthia.gateway.protocol import ChatModel
    from synthia.gateway.providers import RemoteProvider
    from synthia.kernel.bus import Event
    from synthia.kernel.config import Settings
    from synthia.models.local import LocalModel

GATEWAY_DB: Final = Path("db") / "gateway.db"

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Gateway:
    """The model to send requests to, and what is needed to report on it.

    ``model`` is the router with accounting on top; ``router`` is exposed for
    its decisions and capabilities. ``learn_daily_cap`` asks the remote for
    the key's daily cap and stores it; see :func:`learn_daily_cap`.
    """

    model: ChatModel
    router: Router
    health: RemoteHealth
    usage: UsageLog
    learn_daily_cap: Callable[[], Coroutine[None, None, int | None]]


async def learn_daily_cap(
    ledger: BudgetLedger,
    client: httpx.AsyncClient,
    api_key: SecretStr,
    remote: RemoteProvider,
) -> int | None:
    """Store the daily cap the remote reports for ``api_key``, and return it.

    The question is not a model request and costs no budget. If the remote
    cannot answer, the cap stays as it was, the last one reported or the
    provider's default, and ``None`` is returned.
    """
    try:
        count = await free_requests_today(client, api_key, remote.base_url)
    except GatewayError as error:
        logger.warning("could not learn the %s daily cap: %s", remote.name, error)
        return None
    await ledger.set_cap(remote.name, count.limit)
    return count.limit


def build_gateway(
    settings: Settings,
    client: httpx.AsyncClient,
    publish: Callable[[Event], Awaitable[None]],
    local: LocalModel | None = None,
    *,
    remote: RemoteProvider = OPENROUTER_FREE,
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
    ledger = BudgetLedger(settings.home / GATEWAY_DB, {remote.name: remote.daily_cap})
    health = RemoteHealth(
        remote.name,
        ledger,
        SlidingWindowLimiter(remote.requests_per_minute),
        CircuitBreaker(remote.name),
    )
    adapter = OpenAICompatibleModel(
        client=client, endpoint=remote.endpoint(key), info=remote.model
    )
    usage = UsageLog(settings.home / GATEWAY_DB)
    local_ids = frozenset[str]() if local is None else frozenset({local.info.id})

    async def remote_speed() -> float | None:
        return await usage.tokens_per_second(exclude=local_ids)

    router = Router(
        guard_remote(adapter, health),
        health,
        local,
        publish=publish,
        local_ready=(lambda: False) if local is None else local.ready,
        remote_speed=remote_speed,
    )
    return Gateway(
        AccountingModel(router, usage, publish),
        router,
        health,
        usage,
        partial(learn_daily_cap, ledger, client, key, remote),
    )
