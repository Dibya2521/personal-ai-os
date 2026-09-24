from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr
from typer.testing import CliRunner

from synthia.gateway.assemble import GATEWAY_DB
from synthia.gateway.budget import BudgetLedger
from synthia.gateway.providers import OPENROUTER
from synthia.gateway.types import Usage
from synthia.gateway.usage import UsageLog
from synthia.interfaces.cli import app
from synthia.interfaces.usage_report import budget_report
from synthia.kernel.config import Settings

KEY = "sk-or-v1-report-test-key-000"  # pragma: allowlist secret


def provider_says(used: int, limit: int = 50) -> httpx.MockTransport:
    body = {
        "data": {
            "is_free_tier": True,
            "free_model_daily_requests": {
                "used": used,
                "limit": limit,
                "remaining": limit - used,
            },
        }
    }
    return httpx.MockTransport(lambda _: httpx.Response(200, json=body))


async def spend(home: Path, requests: int) -> None:
    ledger = BudgetLedger(home / GATEWAY_DB, {OPENROUTER: 50})
    for _ in range(requests):
        await ledger.claim(OPENROUTER)
    await UsageLog(home / GATEWAY_DB).record("z-ai/glm-5.2:free", Usage(800, 90), 4.07)


async def test_the_report_gives_requests_left_and_tokens_per_model(
    tmp_path: Path,
) -> None:
    await spend(tmp_path, 3)
    day = datetime.now(UTC).date().isoformat()

    report = await budget_report(Settings(home=tmp_path), check=False)

    assert report.lines == [
        f"remote requests today ({day} UTC): 3 of 50 used, 47 left, 10 kept in reserve",
        "  z-ai/glm-5.2:free: requests 1, tokens 800 in and 90 out, 4.1 s",
    ]
    assert report.complete


async def test_a_quiet_day_says_so(tmp_path: Path) -> None:
    report = await budget_report(Settings(home=tmp_path), check=False)

    assert report.lines[-1] == "no model calls finished today"


MATCHES = "that matches this machine"
MORE = "2 more than this machine counted: something else is using this key"
FEWER = "1 fewer than this machine counted: some requests never reached OpenRouter"
RAISE_CAP = (
    "OpenRouter allows 1000 a day; set SYNTHIA_REMOTE_DAILY_CAP=1000 to use them"
)


@pytest.mark.parametrize(
    ("provider_used", "limit", "expected"),
    [
        (3, 50, [MATCHES]),
        (5, 50, [MORE]),
        (2, 50, [FEWER]),
        (3, 1000, [MATCHES, RAISE_CAP]),
    ],
)
async def test_the_check_compares_with_openrouters_own_count(
    tmp_path: Path, provider_used: int, limit: int, expected: list[str]
) -> None:
    await spend(tmp_path, 3)
    settings = Settings(home=tmp_path, openrouter_api_key=SecretStr(KEY))

    report = await budget_report(
        settings, check=True, transport=provider_says(provider_used, limit)
    )

    assert (
        report.lines[2]
        == f"OpenRouter counts {provider_used} of {limit} free requests today"
    )
    assert report.lines[3:] == expected
    assert report.complete


async def test_a_check_without_a_key_or_an_answer_is_incomplete(tmp_path: Path) -> None:
    refused = httpx.MockTransport(
        lambda _: httpx.Response(401, json={"error": {"message": "User not found."}})
    )

    no_key = await budget_report(Settings(home=tmp_path), check=True)
    no_answer = await budget_report(
        Settings(home=tmp_path, openrouter_api_key=SecretStr(KEY)),
        check=True,
        transport=refused,
    )

    assert (
        no_key.lines[-1] == "set SYNTHIA_OPENROUTER_API_KEY to compare with OpenRouter"
    )
    assert no_answer.lines[-1] == "could not ask OpenRouter: HTTP 401: User not found."
    assert (no_key.complete, no_answer.complete) == (False, False)


def test_the_command_prints_the_report_and_exits_1_when_the_check_is_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SYNTHIA_HOME", str(tmp_path / "home"))
    cli = CliRunner()

    plain = cli.invoke(app, ["budget"])
    checked = cli.invoke(app, ["budget", "--check"])
    monkeypatch.setenv("SYNTHIA_REMOTE_DAILY_CAP", "-1")
    broken = cli.invoke(app, ["budget"])

    assert (plain.exit_code, checked.exit_code, broken.exit_code) == (0, 1, 2)
    assert "0 of 50 used" in plain.stdout
    assert "set SYNTHIA_OPENROUTER_API_KEY" in checked.stdout
    assert "SYNTHIA_REMOTE_DAILY_CAP" in broken.stderr
