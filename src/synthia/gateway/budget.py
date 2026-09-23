"""A persistent daily budget of remote model requests.

OpenRouter's free tier allows a fixed number of requests per UTC day, and a
request counts once it reaches the provider, whether it succeeds or not. So a
request is claimed from the budget *before* it is sent, and one that would go
over is refused on this machine instead of failing at the provider.

The ledger is a SQLite file, so the count survives restarts and is shared by
every process on the machine (the daemon and a CLI alike). A claim is a single
``BEGIN IMMEDIATE`` transaction: it takes the database's write lock before
reading, so two processes can never both see room for the last request.

Days are UTC days because that is when the provider's counter resets; a local
midnight would disagree with it for hours every day.
"""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from synthia.gateway.errors import GatewayError

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

BUSY_TIMEOUT_S = 5.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS remote_requests (
    provider TEXT NOT NULL,
    day TEXT NOT NULL,
    used INTEGER NOT NULL,
    PRIMARY KEY (provider, day)
)
"""


class BudgetExhaustedError(GatewayError):
    """Today's remote requests are used up. ``resets_at`` is the next UTC midnight."""

    def __init__(self, provider: str, cap: int, resets_at: datetime) -> None:
        super().__init__(
            f"all {cap} of today's {provider} requests are used; "
            f"the budget resets at {resets_at:%Y-%m-%d %H:%M} UTC"
        )
        self.provider = provider
        self.resets_at = resets_at


@dataclass(frozen=True, slots=True)
class BudgetStatus:
    """One provider's budget for the current UTC day."""

    provider: str
    day: str
    used: int
    cap: int

    @property
    def remaining(self) -> int:
        """Return the requests still available today."""
        return max(self.cap - self.used, 0)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class BudgetLedger:
    """Count remote requests per provider per UTC day, and refuse past the cap."""

    def __init__(
        self,
        path: Path,
        caps: dict[str, int],
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        """Open the ledger at ``path``, creating it if needed.

        ``caps`` maps each provider to its daily request cap. ``clock`` must
        return an aware datetime; it is injectable so tests can cross midnight.

        Raises:
            ValueError: If a cap is negative.
        """
        if any(cap < 0 for cap in caps.values()):
            message = "a daily cap cannot be negative"
            raise ValueError(message)
        self._path = path
        self._caps = dict(caps)
        self._clock = clock
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(_SCHEMA)
        finally:
            connection.close()

    async def claim(self, provider: str) -> BudgetStatus:
        """Take one request from today's budget and return what is left.

        Raises:
            BudgetExhaustedError: If today's cap is already reached. Nothing is
                taken in that case.
            KeyError: If ``provider`` has no cap configured.
        """
        return await asyncio.to_thread(self._claim, provider)

    async def status(self, provider: str) -> BudgetStatus:
        """Return today's budget for ``provider`` without taking anything.

        Raises:
            KeyError: If ``provider`` has no cap configured.
        """
        return await asyncio.to_thread(self._status, provider)

    def _today(self) -> tuple[str, datetime]:
        now = self._clock().astimezone(UTC)
        midnight = datetime(now.year, now.month, now.day, tzinfo=UTC)
        return now.date().isoformat(), midnight + timedelta(days=1)

    def _connect(self) -> sqlite3.Connection:
        # isolation_level=None stops the module opening transactions implicitly,
        # so the only transactions are the explicit ones below; a locked database
        # is waited on for up to the timeout rather than failing at once.
        return sqlite3.connect(self._path, timeout=BUSY_TIMEOUT_S, isolation_level=None)

    def _claim(self, provider: str) -> BudgetStatus:
        cap = self._caps[provider]
        day, resets_at = self._today()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            used = self._used(connection, provider, day)
            if used >= cap:
                connection.execute("ROLLBACK")
                raise BudgetExhaustedError(provider, cap, resets_at)
            connection.execute(
                "INSERT INTO remote_requests (provider, day, used) VALUES (?, ?, 1) "
                "ON CONFLICT (provider, day) DO UPDATE SET used = used + 1",
                (provider, day),
            )
            connection.execute("COMMIT")
        finally:
            connection.close()
        return BudgetStatus(provider, day, used + 1, cap)

    def _status(self, provider: str) -> BudgetStatus:
        cap = self._caps[provider]
        day, _ = self._today()
        connection = self._connect()
        try:
            return BudgetStatus(
                provider, day, self._used(connection, provider, day), cap
            )
        finally:
            connection.close()

    @staticmethod
    def _used(connection: sqlite3.Connection, provider: str, day: str) -> int:
        row = connection.execute(
            "SELECT used FROM remote_requests WHERE provider = ? AND day = ?",
            (provider, day),
        ).fetchone()
        return int(row[0]) if row else 0
