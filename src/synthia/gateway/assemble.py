"""Build the gateway a conversation talks to, from the settings.

This is the one place that knows how the pieces fit: the local model when one
is installed, and, when its key is set, the OpenRouter adapter, metered and
guarded, behind a router whose budget, rate and circuit are the same objects
the guards use, with every finished call accounted for on top. Everything else
receives the finished model.
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
from synthia.gateway.router import Remote, RemoteHealth, Router, guard_remote
from synthia.gateway.usage import AccountingModel, UsageLog, free_requests_today
from synthia.kernel.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine

    import httpx
    from pydantic import SecretStr

    from synthia.gateway.protocol import ChatModel, LocalChatModel
    from synthia.gateway.providers import RemoteProvider
    from synthia.kernel.bus import Event
    from synthia.kernel.config import Settings

GATEWAY_DB: Final = Path("db") / "gateway.db"

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Gateway:
    """The model to send requests to, and what is needed to report on it.

    ``model`` is the router with accounting on top; ``router`` is exposed for
    its decisions, capabilities and remote guards. ``learn_daily_cap`` asks
    the remote for the key's daily cap and stores it (see
    :func:`learn_daily_cap`); without a remote it returns ``None``.
    """

    model: ChatModel
    router: Router
    usage: UsageLog
    learn_daily_cap: Callable[[], Coroutine[None, None, int | None]]

    @property
    def health(self) -> RemoteHealth | None:
        """Return the remote model's guards, or ``None`` without a remote."""
        remote = self.router.remote
        return None if remote is None else remote.health


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


NO_MODEL: Final = (
    "no model is available: install a local model with "
    "`synthia models install`, or set SYNTHIA_OPENROUTER_API_KEY in .env "
    "to ask a remote one"
)


def build_gateway(
    settings: Settings,
    client: httpx.AsyncClient,
    publish: Callable[[Event], Awaitable[None]],
    local: LocalChatModel | None = None,
    *,
    remote: RemoteProvider = OPENROUTER_FREE,
) -> Gateway:
    """Assemble the gateway; ``client`` carries every remote request.

    ``local`` serves every request that does not ask for the remote model.
    The remote model exists only when its key is set.

    Raises:
        ConfigError: If there is neither a local model nor a remote key.
    """
    key = settings.openrouter_api_key
    if key is None and local is None:
        raise ConfigError(NO_MODEL)
    usage = UsageLog(settings.home / GATEWAY_DB)
    guarded: Remote | None = None
    learn: Callable[[], Coroutine[None, None, int | None]] = _nothing_to_learn
    if key is not None:
        ledger = BudgetLedger(
            settings.home / GATEWAY_DB, {remote.name: remote.daily_cap}
        )
        health = RemoteHealth(
            remote.name,
            ledger,
            SlidingWindowLimiter(remote.requests_per_minute),
            CircuitBreaker(remote.name),
        )
        adapter = OpenAICompatibleModel(
            client=client, endpoint=remote.endpoint(key), info=remote.model
        )
        guarded = Remote(guard_remote(adapter, health), health)
        learn = partial(learn_daily_cap, ledger, client, key, remote)
    local_ids = frozenset[str]() if local is None else frozenset({local.info.id})

    async def remote_speed() -> float | None:
        return await usage.tokens_per_second(exclude=local_ids)

    router = Router(
        local,
        guarded,
        publish=publish,
        local_ready=(lambda: False) if local is None else local.ready,
        remote_speed=remote_speed,
    )
    return Gateway(AccountingModel(router, usage, publish), router, usage, learn)


async def _nothing_to_learn() -> int | None:
    return None
