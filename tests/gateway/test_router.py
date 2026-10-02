from collections.abc import AsyncGenerator, Callable
from contextlib import suppress
from dataclasses import replace
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

import pytest

from synthia.gateway.budget import BudgetLedger
from synthia.gateway.circuit import CircuitBreaker, CircuitOpenError
from synthia.gateway.errors import (
    AuthError,
    GatewayError,
    LocalUnavailableError,
    ProviderError,
    RemoteUnavailableError,
)
from synthia.gateway.protocol import collect
from synthia.gateway.ratelimit import SlidingWindowLimiter
from synthia.gateway.retry import RetryPolicy
from synthia.gateway.router import (
    Remote,
    RemoteHealth,
    Route,
    RouteDecided,
    Router,
    RouteReason,
    guard_remote,
)
from synthia.gateway.types import (
    ChatChunk,
    ChatRequest,
    FinishReason,
    ImagePart,
    Message,
    ModelInfo,
    Reasoning,
    ToolSpec,
)
from synthia.kernel.bus import Event

AUTO = Reasoning.AUTO
PROVIDER = "openrouter"
REMOTE = ModelInfo("openrouter/free", 128_000, vision=False, tools=True)
LOCAL = ModelInfo("qwen3.5-4b", 32_000, vision=True, tools=True)
BLIND = ModelInfo("blind", 8000, vision=False, tools=False)
HELLO = ChatRequest((Message.user("hello"),))
PHOTO = ChatRequest(
    (Message.user("what is this?", ImagePart(b"\x89PNG", "image/png")),)
)
ASK = replace(HELLO, use_remote=True)
PHOTO_ASK = replace(PHOTO, use_remote=True)
WEATHER = ToolSpec("weather", "Look up the weather.", {"type": "object"})


class Fake:
    """Answer with the model's id, after the scripted errors."""

    def __init__(self, info: ModelInfo, *errors: GatewayError) -> None:
        self._info = info
        self.errors = list(errors)
        self.fail_after_first = False
        self.calls = 0
        self.requests: list[ChatRequest] = []

    @property
    def info(self) -> ModelInfo:
        return self._info

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        self.requests.append(request)
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        yield ChatChunk(text=self._info.id)
        if self.fail_after_first:
            message = "cut mid-answer"
            raise ProviderError(message)
        yield ChatChunk(finish_reason=FinishReason.STOP)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _ready() -> bool:
    return True


class Setup:
    def __init__(self, tmp_path: Path, *, cap: int = 50, reserve: int = 10) -> None:
        now = datetime(2026, 9, 24, 10, tzinfo=UTC)
        self.clock = Clock()
        self.ledger = BudgetLedger(
            tmp_path / "gateway.db", {PROVIDER: cap}, clock=lambda: now
        )
        self.limiter = SlidingWindowLimiter(2, clock=self.clock)
        self.breaker = CircuitBreaker(PROVIDER, failure_threshold=1, clock=self.clock)
        self.health = RemoteHealth(
            PROVIDER, self.ledger, self.limiter, self.breaker, reserve=reserve
        )
        self.events: list[Event] = []

    async def publish(self, event: Event) -> None:
        self.events.append(event)

    def router(
        self,
        remote: Fake | None,
        local: Fake | None = None,
        local_ready: Callable[[], bool] = _ready,
    ) -> Router:
        guarded = None if remote is None else Remote(remote, self.health)
        return Router(local, guarded, self.publish, local_ready)

    def routes(self) -> list[tuple[Route, RouteReason, str]]:
        return [
            (e.route, e.reason, e.model)
            for e in self.events
            if isinstance(e, RouteDecided)
        ]

    async def spend(self, count: int) -> None:
        for _ in range(count):
            await self.ledger.claim(PROVIDER)


@pytest.fixture
def setup(tmp_path: Path) -> Setup:
    return Setup(tmp_path)


async def test_a_request_that_does_not_ask_for_the_remote_goes_local(
    setup: Setup,
) -> None:
    remote = Fake(REMOTE)

    answer = await collect(setup.router(remote, Fake(LOCAL)).stream(HELLO))

    assert answer.text == LOCAL.id
    assert remote.calls == 0
    assert setup.routes() == [(Route.LOCAL, RouteReason.LOCAL_FIRST, LOCAL.id)]


async def test_a_local_model_alone_serves_without_any_remote(setup: Setup) -> None:
    router = setup.router(None, Fake(LOCAL))

    answer = await collect(router.stream(HELLO))

    assert answer.text == LOCAL.id
    assert router.remote is None


async def test_a_starting_local_model_is_waited_for_not_replaced_by_the_remote(
    setup: Setup,
) -> None:
    remote, local = Fake(REMOTE), Fake(LOCAL)

    answer = await collect(setup.router(remote, local, lambda: False).stream(HELLO))

    assert answer.text == LOCAL.id
    assert remote.calls == 0


async def test_without_a_local_model_a_local_request_fails_with_the_reason(
    setup: Setup,
) -> None:
    remote = Fake(REMOTE)

    with pytest.raises(LocalUnavailableError, match="no local model is installed"):
        await collect(setup.router(remote).stream(HELLO))
    assert remote.calls == 0
    assert setup.routes() == []


async def test_an_image_the_local_model_cannot_see_is_refused_not_sent_out(
    setup: Setup,
) -> None:
    remote = Fake(ModelInfo("seeing", 128_000, vision=True, tools=True))

    with pytest.raises(LocalUnavailableError, match="cannot take images"):
        await collect(setup.router(remote, Fake(BLIND)).stream(PHOTO))
    assert remote.calls == 0


async def test_tools_the_local_model_cannot_call_are_refused_not_sent_out(
    setup: Setup,
) -> None:
    request = ChatRequest((Message.user("weather?"),), tools=(WEATHER,))

    with pytest.raises(LocalUnavailableError, match="cannot take tools"):
        await collect(setup.router(Fake(REMOTE), Fake(BLIND)).stream(request))


LOCALS: dict[str, Callable[[], Fake | None]] = {
    "none": lambda: None,
    "seeing": lambda: Fake(LOCAL),
    "blind": lambda: Fake(BLIND),
    "failing": lambda: Fake(LOCAL, ProviderError("local down")),
}


@pytest.mark.parametrize(
    ("local", "ready", "background", "request_"),
    list(product(LOCALS, [True, False], [True, False], [HELLO, PHOTO])),
)
async def test_a_request_that_does_not_ask_never_reaches_the_remote(
    setup: Setup,
    local: str,
    ready: bool,
    background: bool,
    request_: ChatRequest,
) -> None:
    remote = Fake(ModelInfo("seeing", 128_000, vision=True, tools=True))
    await setup.spend(45)
    setup.breaker.record_failure()
    router = setup.router(remote, LOCALS[local](), lambda: ready)

    with suppress(GatewayError):
        await collect(router.stream(request_, background=background))

    assert remote.calls == 0
    assert all(route is Route.LOCAL for route, _, _ in setup.routes())


async def test_asking_for_a_remote_that_is_not_configured_fails_with_the_setting(
    setup: Setup,
) -> None:
    local = Fake(LOCAL)

    with pytest.raises(RemoteUnavailableError, match="SYNTHIA_OPENROUTER_API_KEY"):
        await collect(setup.router(None, local).stream(ASK))
    assert local.calls == 0


def test_a_router_needs_at_least_one_model() -> None:
    with pytest.raises(ValueError, match="a local model, a remote one, or both"):
        Router(None)


async def test_a_healthy_remote_answers_first_when_asked(setup: Setup) -> None:
    answer = await collect(setup.router(Fake(REMOTE), Fake(LOCAL)).stream(ASK))

    assert answer.text == REMOTE.id
    assert setup.routes() == [(Route.REMOTE, RouteReason.REMOTE_ASKED, REMOTE.id)]


async def test_a_background_job_goes_local_even_when_remote_is_asked(
    setup: Setup,
) -> None:
    router = setup.router(Fake(REMOTE), Fake(LOCAL))

    answer = await collect(router.stream(ASK, background=True))

    assert answer.text == LOCAL.id
    assert setup.routes() == [(Route.LOCAL, RouteReason.BACKGROUND, LOCAL.id)]


async def test_an_image_the_remote_cannot_see_goes_to_the_local_model(
    setup: Setup,
) -> None:
    router = setup.router(Fake(REMOTE), Fake(LOCAL))

    assert await router.decide(PHOTO_ASK) == (Route.LOCAL, RouteReason.CAPABILITY)


async def test_tools_the_remote_cannot_call_go_to_the_local_model(
    setup: Setup,
) -> None:
    plain = ModelInfo("plain", 8000, vision=False, tools=False)
    router = setup.router(Fake(plain), Fake(LOCAL))
    request = ChatRequest(
        (Message.user("weather?"),), tools=(WEATHER,), use_remote=True
    )

    assert await router.decide(request) == (Route.LOCAL, RouteReason.CAPABILITY)
    assert await router.decide(ASK) == (Route.REMOTE, RouteReason.REMOTE_ASKED)


async def test_an_open_circuit_goes_local_at_once(setup: Setup) -> None:
    setup.breaker.record_failure()
    router = setup.router(Fake(REMOTE), Fake(LOCAL))

    assert await router.decide(ASK) == (Route.LOCAL, RouteReason.CIRCUIT_OPEN)


async def test_the_reserve_is_where_chat_turns_local(setup: Setup) -> None:
    router = setup.router(Fake(REMOTE), Fake(LOCAL))

    await setup.spend(39)  # 11 left, one above the reserve of 10
    assert await router.decide(ASK) == (Route.REMOTE, RouteReason.REMOTE_ASKED)
    await setup.spend(1)  # 10 left: at the reserve
    assert await router.decide(ASK) == (Route.LOCAL, RouteReason.LOW_BUDGET)


async def test_a_long_rate_limit_wait_goes_local_and_a_short_one_waits(
    setup: Setup,
) -> None:
    router = setup.router(Fake(REMOTE), Fake(LOCAL))
    await setup.limiter.acquire()
    await setup.limiter.acquire()  # full: the next slot opens at t=60

    assert await router.decide(ASK) == (Route.LOCAL, RouteReason.RATE_LIMITED)
    setup.clock.now = 55.0  # the slot opens in 5 s, exactly the longest wait
    assert await router.decide(ASK) == (Route.REMOTE, RouteReason.REMOTE_ASKED)


async def test_without_a_local_model_the_remote_serves_down_to_zero(
    setup: Setup,
) -> None:
    router = setup.router(Fake(REMOTE))
    await setup.spend(45)

    answer = await collect(router.stream(ASK))

    assert answer.text == REMOTE.id
    assert setup.routes() == [(Route.REMOTE, RouteReason.NO_LOCAL, REMOTE.id)]


async def test_a_local_model_that_cannot_serve_leaves_it_to_the_remote(
    setup: Setup,
) -> None:
    router = setup.router(Fake(REMOTE), Fake(BLIND))

    assert await router.decide(ASK, background=True) == (
        Route.LOCAL,
        RouteReason.BACKGROUND,
    )
    assert await router.decide(PHOTO_ASK) == (Route.REMOTE, RouteReason.NO_LOCAL)


async def test_a_remote_that_fails_before_answering_falls_back_once(
    setup: Setup,
) -> None:
    remote, local = Fake(REMOTE, AuthError("bad key")), Fake(LOCAL)

    answer = await collect(setup.router(remote, local).stream(ASK))

    assert answer.text == LOCAL.id
    assert setup.routes() == [
        (Route.REMOTE, RouteReason.REMOTE_ASKED, REMOTE.id),
        (Route.LOCAL, RouteReason.FALLBACK, LOCAL.id),
    ]


async def test_a_remote_that_fails_mid_answer_is_not_switched(setup: Setup) -> None:
    remote, local = Fake(REMOTE), Fake(LOCAL)
    remote.fail_after_first = True

    with pytest.raises(ProviderError):
        await collect(setup.router(remote, local).stream(ASK))
    assert local.calls == 0


async def test_a_failed_remote_with_no_local_model_raises(setup: Setup) -> None:
    router = setup.router(Fake(REMOTE, AuthError("bad key")))

    with pytest.raises(AuthError):
        await collect(router.stream(ASK))


async def test_a_failed_remote_with_a_local_model_that_cannot_serve_raises(
    setup: Setup,
) -> None:
    seeing = ModelInfo("seeing", 128_000, vision=True, tools=True)
    router = setup.router(Fake(seeing, AuthError("bad key")), Fake(BLIND))

    with pytest.raises(AuthError):
        await collect(router.stream(PHOTO_ASK))


async def test_info_is_what_either_model_can_do(setup: Setup) -> None:
    assert setup.router(Fake(REMOTE), Fake(LOCAL)).info == ModelInfo(
        "auto", 32_000, vision=True, tools=True
    )
    assert setup.router(Fake(REMOTE)).info == ModelInfo(
        "auto", 128_000, vision=False, tools=True
    )
    assert setup.router(None, Fake(BLIND)).info == ModelInfo(
        "auto", 8000, vision=False, tools=False
    )


def test_a_negative_reserve_or_wait_is_refused(setup: Setup) -> None:
    for reserve, wait in [(-1, 5.0), (10, -0.1)]:
        with pytest.raises(ValueError, match="cannot be negative"):
            RemoteHealth(
                PROVIDER,
                setup.ledger,
                setup.limiter,
                setup.breaker,
                reserve=reserve,
                max_wait_s=wait,
            )


async def test_the_guarded_remote_counts_attempts_and_opens_the_circuit(
    tmp_path: Path,
) -> None:
    setup = Setup(tmp_path)
    breaker = CircuitBreaker(PROVIDER, failure_threshold=2, clock=setup.clock)
    health = RemoteHealth(PROVIDER, setup.ledger, setup.limiter, breaker, reserve=10)
    setup.clock.now = 1000.0  # past any earlier window
    flaky = Fake(REMOTE, ProviderError("down"), ProviderError("down"))
    remote = guard_remote(flaky, health, RetryPolicy(max_attempts=3, base_s=0.0))
    router = Router(Fake(LOCAL), Remote(remote, health), setup.publish)

    answer = await collect(router.stream(ASK))

    assert answer.text == LOCAL.id
    assert flaky.calls == 2  # the third attempt met an open circuit
    assert (await setup.ledger.status(PROVIDER)).used == 2
    assert await router.decide(ASK) == (Route.LOCAL, RouteReason.CIRCUIT_OPEN)
    with pytest.raises(CircuitOpenError):
        await collect(remote.stream(ASK))


async def test_a_local_model_that_is_not_ready_is_treated_as_absent_when_remote_is_asked(  # noqa: E501
    setup: Setup,
) -> None:
    router = setup.router(Fake(REMOTE), Fake(LOCAL), lambda: False)

    answer = await collect(router.stream(ASK, background=True))

    assert answer.text == REMOTE.id
    assert setup.routes() == [(Route.REMOTE, RouteReason.NO_LOCAL, REMOTE.id)]


async def test_a_failed_remote_does_not_fall_back_to_a_local_model_still_loading(
    setup: Setup,
) -> None:
    router = setup.router(
        Fake(REMOTE, AuthError("bad key")), Fake(LOCAL), lambda: False
    )

    with pytest.raises(AuthError):
        await collect(router.stream(ASK))


async def test_readiness_is_checked_for_every_request(setup: Setup) -> None:
    ready = [False]
    router = setup.router(Fake(REMOTE), Fake(LOCAL), lambda: ready[0])

    await collect(router.stream(ASK, background=True))
    ready[0] = True
    await collect(router.stream(ASK, background=True))

    assert [route for route, _, _ in setup.routes()] == [Route.REMOTE, Route.LOCAL]


async def test_auto_reasoning_is_resolved_before_any_model_sees_it(
    setup: Setup,
) -> None:
    remote, local = Fake(REMOTE, AuthError("bad key")), Fake(LOCAL)
    request = ChatRequest(
        (Message.user("why is the sky blue?"),), reasoning=AUTO, use_remote=True
    )

    await collect(setup.router(remote, local).stream(request))

    sent = [r.reasoning for r in remote.requests + local.requests]
    announced = [e.reasoning for e in setup.events if isinstance(e, RouteDecided)]
    assert sent == announced == [Reasoning.MEDIUM, Reasoning.MEDIUM]


async def test_a_local_request_has_auto_reasoning_resolved_too(setup: Setup) -> None:
    local = Fake(LOCAL)
    request = ChatRequest((Message.user("why is the sky blue?"),), reasoning=AUTO)

    await collect(setup.router(None, local).stream(request))

    assert [r.reasoning for r in local.requests] == [Reasoning.MEDIUM]
    assert [r.reasoning_tokens for r in local.requests] == [None]


async def test_only_a_remote_request_gets_a_token_limit(setup: Setup) -> None:
    async def speed() -> float:
        return 40.0

    remote, local = Fake(REMOTE, AuthError("bad key")), Fake(LOCAL)
    router = Router(
        local, Remote(remote, setup.health), setup.publish, remote_speed=speed
    )
    request = ChatRequest(
        (Message.user("hi there"),), reasoning=Reasoning.MEDIUM, use_remote=True
    )

    await collect(router.stream(request))
    await collect(router.stream(request, background=True))

    assert remote.requests[0].reasoning_tokens == 20 * 40
    assert [r.reasoning_tokens for r in local.requests] == [20 * 40, None]


@pytest.mark.parametrize("level", [None, Reasoning.OFF, Reasoning.HIGH])
async def test_a_level_other_than_auto_is_kept(
    setup: Setup, level: Reasoning | None
) -> None:
    remote = Fake(REMOTE)
    request = ChatRequest(
        (Message.user("why is the sky blue?"),), reasoning=level, use_remote=True
    )

    await collect(setup.router(remote).stream(request))

    assert [r.reasoning for r in remote.requests] == [level]
    assert [r.messages for r in remote.requests] == [request.messages]
