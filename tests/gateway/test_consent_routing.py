from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from synthia.gateway.assemble import build_gateway
from synthia.gateway.errors import GatewayError, LocalUnavailableError
from synthia.gateway.protocol import collect
from synthia.gateway.router import Remote, Route, Router, RouteReason
from synthia.kernel.bus import Event
from synthia.kernel.config import Settings
from synthia.kernel.consent import ConsentBook, Scope
from tests.consenting import OPENROUTER
from tests.gateway.test_router import ASK, LOCAL, REMOTE, Fake, Setup


@pytest.fixture
def setup(tmp_path: Path) -> Setup:
    return Setup(tmp_path)


class Switch:
    def __init__(self) -> None:
        self.granted = False

    def __call__(self, scope: Scope) -> bool:
        return self.granted and scope is Scope.CHAT


async def test_no_consent_sends_nothing_out(setup: Setup) -> None:
    remote = Fake(REMOTE)
    router = Router(
        Fake(LOCAL), Remote(remote, setup.health, Switch(), OPENROUTER), setup.publish
    )

    answer = await collect(router.stream(ASK))

    assert answer.text == LOCAL.id
    assert remote.calls == 0
    assert setup.routes() == [(Route.LOCAL, RouteReason.NO_CONSENT, LOCAL.id)]


async def test_no_consent_and_no_local_fails_and_says_why(setup: Setup) -> None:
    remote = Fake(REMOTE)
    router = Router(
        None, Remote(remote, setup.health, Switch(), OPENROUTER), setup.publish
    )

    with pytest.raises(LocalUnavailableError, match="openrouter has no consent"):
        await collect(router.stream(ASK))
    assert remote.calls == 0


async def test_a_grant_and_a_revoke_take_effect_on_the_next_request(
    setup: Setup,
) -> None:
    remote, consent = Fake(REMOTE), Switch()
    router = Router(
        Fake(LOCAL), Remote(remote, setup.health, consent, OPENROUTER), setup.publish
    )

    consent.granted = True
    first = await collect(router.stream(ASK))
    consent.granted = False
    second = await collect(router.stream(ASK))

    assert (first.text, second.text) == (REMOTE.id, LOCAL.id)
    assert remote.calls == 1


async def _discard(_: Event) -> None:
    return None


async def test_the_assembled_gateway_asks_the_consent_file(tmp_path: Path) -> None:
    sent: list[httpx.Request] = []

    def refuse(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(401, json={"error": {"message": "no"}})

    settings = Settings(home=tmp_path, openrouter_api_key=SecretStr("sk-or-test"))
    async with httpx.AsyncClient(transport=httpx.MockTransport(refuse)) as client:
        gateway = build_gateway(settings, client, _discard)
        with pytest.raises(LocalUnavailableError, match="no consent"):
            await collect(gateway.model.stream(ASK))
        before = len(sent)
        ConsentBook(tmp_path).decide(OPENROUTER, Scope.CHAT, granted=True, by="cli")
        with pytest.raises(GatewayError):
            await collect(gateway.model.stream(ASK))

    assert before == 0
    assert len(sent) >= 1
