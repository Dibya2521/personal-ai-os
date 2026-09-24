"""``synthia budget``: today's remote requests and tokens, and OpenRouter's count."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx

from synthia.gateway.assemble import GATEWAY_DB
from synthia.gateway.budget import BudgetLedger
from synthia.gateway.errors import GatewayError
from synthia.gateway.providers import OPENROUTER
from synthia.gateway.usage import UsageLog, free_requests_today

if TYPE_CHECKING:
    from synthia.gateway.budget import BudgetStatus
    from synthia.gateway.usage import ProviderCount
    from synthia.kernel.config import Settings


@dataclass(slots=True)
class Report:
    """Lines to print, and whether everything could be checked."""

    lines: list[str] = field(default_factory=list[str])
    complete: bool = True


async def budget_report(
    settings: Settings,
    *,
    check: bool,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Report:
    """Describe today's use; with ``check``, compare with OpenRouter's own count."""
    path = settings.home / GATEWAY_DB
    ledger = BudgetLedger(path, {OPENROUTER: settings.remote_daily_cap})
    status = await ledger.status(OPENROUTER)
    summary = (
        f"remote requests today ({status.day} UTC): {status.used} of "
        f"{status.cap} used, {status.remaining} left, "
        f"{settings.remote_reserve} kept in reserve"
    )
    report = Report([summary])
    totals = await UsageLog(path).today()
    if not totals:
        report.lines.append("no model calls finished today")
    for t in totals:
        report.lines.append(
            f"  {t.model}: requests {t.calls}, tokens {t.prompt_tokens} in "
            f"and {t.completion_tokens} out, {t.seconds:.1f} s"
        )
    if check:
        await _check(settings, status, report, transport)
    return report


async def _check(
    settings: Settings,
    status: BudgetStatus,
    report: Report,
    transport: httpx.AsyncBaseTransport | None,
) -> None:
    key = settings.openrouter_api_key
    if key is None:
        report.lines.append("set SYNTHIA_OPENROUTER_API_KEY to compare with OpenRouter")
        report.complete = False
        return
    async with httpx.AsyncClient(transport=transport) as client:
        try:
            count = await free_requests_today(client, key, settings.openrouter_base_url)
        except GatewayError as error:
            report.lines.append(f"could not ask OpenRouter: {error}")
            report.complete = False
            return
    report.lines += _compare(count, status)


def _compare(count: ProviderCount, status: BudgetStatus) -> list[str]:
    lines = [f"OpenRouter counts {count.used} of {count.limit} free requests today"]
    difference = count.used - status.used
    if difference == 0:
        lines.append("that matches this machine")
    elif difference > 0:
        lines.append(
            f"{difference} more than this machine counted: something else is using "
            "this key"
        )
    else:
        lines.append(
            f"{-difference} fewer than this machine counted: some requests never "
            "reached OpenRouter"
        )
    if count.limit != status.cap:
        lines.append(
            f"OpenRouter allows {count.limit} a day; set SYNTHIA_REMOTE_DAILY_CAP="
            f"{count.limit} to use them"
        )
    return lines
