import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from synthia.gateway.budget import BudgetExhaustedError, BudgetLedger

IST = timezone(timedelta(hours=5, minutes=30))


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def ledger_at(path: Path, now: datetime, cap: int = 3) -> tuple[BudgetLedger, Clock]:
    clock = Clock(now)
    return BudgetLedger(path, {"openrouter": cap}, clock=clock), clock


async def test_claims_count_down_the_budget(tmp_path: Path) -> None:
    ledger, _ = ledger_at(
        tmp_path / "db" / "gateway.db", datetime(2026, 9, 23, 10, tzinfo=UTC)
    )

    first = await ledger.claim("openrouter")
    second = await ledger.claim("openrouter")
    status = await ledger.status("openrouter")

    assert (first.used, first.remaining) == (1, 2)
    assert (second.used, second.remaining) == (2, 1)
    assert (status.day, status.used, status.cap) == ("2026-09-23", 2, 3)


async def test_the_claim_past_the_cap_is_refused_and_takes_nothing(
    tmp_path: Path,
) -> None:
    ledger, _ = ledger_at(
        tmp_path / "gateway.db", datetime(2026, 9, 23, 22, 15, tzinfo=UTC)
    )
    for _ in range(3):
        await ledger.claim("openrouter")

    with pytest.raises(BudgetExhaustedError) as caught:
        await ledger.claim("openrouter")

    assert caught.value.resets_at == datetime(2026, 9, 24, tzinfo=UTC)
    assert "resets at 2026-09-24 00:00 UTC" in str(caught.value)
    assert not caught.value.retryable
    assert (await ledger.status("openrouter")).used == 3


async def test_request_fifty_one_is_refused_with_the_real_cap(tmp_path: Path) -> None:
    ledger, _ = ledger_at(
        tmp_path / "gateway.db", datetime(2026, 9, 23, tzinfo=UTC), cap=50
    )
    for _ in range(50):
        await ledger.claim("openrouter")

    with pytest.raises(BudgetExhaustedError, match="all 50 of today's"):
        await ledger.claim("openrouter")


async def test_a_zero_cap_refuses_everything(tmp_path: Path) -> None:
    ledger, _ = ledger_at(
        tmp_path / "gateway.db", datetime(2026, 9, 23, tzinfo=UTC), cap=0
    )

    with pytest.raises(BudgetExhaustedError):
        await ledger.claim("openrouter")
    assert (await ledger.status("openrouter")).remaining == 0


async def test_the_count_resets_at_utc_midnight_not_local_midnight(
    tmp_path: Path,
) -> None:
    before = datetime(2026, 9, 24, 5, 29, tzinfo=IST)  # 23:59 UTC on the 23rd
    ledger, clock = ledger_at(tmp_path / "gateway.db", before)
    for _ in range(3):
        await ledger.claim("openrouter")

    clock.now = datetime(
        2026, 9, 24, 0, 30, tzinfo=IST
    )  # local midnight passed, UTC day did not
    with pytest.raises(BudgetExhaustedError):
        await ledger.claim("openrouter")

    clock.now = datetime(2026, 9, 24, 5, 31, tzinfo=IST)  # 00:01 UTC on the 24th
    fresh = await ledger.claim("openrouter")
    assert (fresh.day, fresh.used) == ("2026-09-24", 1)


async def test_a_clock_that_jumps_back_does_not_restore_budget(tmp_path: Path) -> None:
    ledger, clock = ledger_at(
        tmp_path / "gateway.db", datetime(2026, 9, 23, 12, tzinfo=UTC)
    )
    for _ in range(3):
        await ledger.claim("openrouter")

    clock.now = datetime(2026, 9, 23, 1, tzinfo=UTC)

    with pytest.raises(BudgetExhaustedError):
        await ledger.claim("openrouter")


async def test_the_count_survives_a_restart(tmp_path: Path) -> None:
    now = datetime(2026, 9, 23, tzinfo=UTC)
    first, _ = ledger_at(tmp_path / "gateway.db", now)
    await first.claim("openrouter")
    await first.claim("openrouter")

    reopened, _ = ledger_at(tmp_path / "gateway.db", now)

    assert (await reopened.status("openrouter")).used == 2


async def test_providers_have_separate_budgets(tmp_path: Path) -> None:
    ledger = BudgetLedger(tmp_path / "gateway.db", {"openrouter": 1, "other": 5})
    await ledger.claim("openrouter")

    assert (await ledger.claim("other")).used == 1
    with pytest.raises(KeyError):
        await ledger.claim("unconfigured")


def test_a_negative_cap_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="negative"):
        BudgetLedger(tmp_path / "gateway.db", {"openrouter": -1})


async def test_concurrent_claims_never_exceed_the_cap(tmp_path: Path) -> None:
    ledger, _ = ledger_at(
        tmp_path / "gateway.db", datetime(2026, 9, 23, tzinfo=UTC), cap=50
    )

    results = await asyncio.gather(
        *(ledger.claim("openrouter") for _ in range(100)), return_exceptions=True
    )

    granted = [r for r in results if not isinstance(r, BaseException)]
    refused = [r for r in results if isinstance(r, BudgetExhaustedError)]
    assert (len(granted), len(refused)) == (50, 50)
    assert sorted(g.used for g in granted) == list(range(1, 51))


def test_two_ledgers_on_one_file_share_the_cap_under_contention(
    tmp_path: Path,
) -> None:
    """Separate ledgers race as separate processes would: SQLite locks the file."""
    path = tmp_path / "gateway.db"
    now = datetime(2026, 9, 23, tzinfo=UTC)
    ledgers = [ledger_at(path, now, cap=120)[0] for _ in range(2)]

    def claim(index: int) -> bool:
        try:
            asyncio.run(ledgers[index % 2].claim("openrouter"))
        except BudgetExhaustedError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(claim, range(200)))

    assert outcomes.count(True) == 120
    assert asyncio.run(ledgers[0].status("openrouter")).used == 120
