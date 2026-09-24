from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from synthia.gateway.budget import BudgetLedger
from synthia.gateway.circuit import CircuitBreaker, CircuitOpenError
from synthia.gateway.errors import AuthError, GatewayError, ProviderError
from synthia.gateway.protocol import collect
from synthia.gateway.ratelimit import SlidingWindowLimiter
from synthia.gateway.retry import RetryPolicy
from synthia.gateway.router import (
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
    ToolSpec,
)
from synthia.kernel.bus import Event

PROVIDER = "openrouter"
REMOTE = ModelInfo("openrouter/free", 128_000, vision=False, tools=True)
LOCAL = ModelInfo("qwen3.5-4b", 32_000, vision=True, tools=True)
HELLO = ChatRequest((Message.user("hello"),))
PHOTO = ChatRequest(
    (Message.user("what is this?", ImagePart(b"\x89PNG", "image/png")),)
)
WEATHER = ToolSpec("weather", "Look up the weather.", {"type": "object"})


class Fake:
    """Answer with the model's id, after the scripted errors."""

    def __init__(self, info: ModelInfo, *errors: GatewayError) -> None:
        self._info = info
        self.errors = list(errors)
        self.fail_after_first = False
        self.calls = 0

    @property
    def info(self) -> ModelInfo:
        return self._info

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        del request
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

    def router(self, remote: Fake, local: Fake | None = None) -> Router:
        return Router(remote, self.health, local, self.publish)

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


async def test_a_healthy_remote_with_budget_answers_first(setup: Setup) -> None:
    answer = await collect(setup.router(Fake(REMOTE), Fake(LOCAL)).stream(HELLO))

    assert answer.text == REMOTE.id
    assert setup.routes() == [(Route.REMOTE, RouteReason.PREFERRED, REMOTE.id)]


async def test_a_background_job_always_goes_local(setup: Setup) -> None:
    router = setup.router(Fake(REMOTE), Fake(LOCAL))

    answer = await collect(router.stream(HELLO, background=True))

    assert answer.text == LOCAL.id
    assert setup.routes() == [(Route.LOCAL, RouteReason.BACKGROUND, LOCAL.id)]


async def test_an_image_the_remote_cannot_see_goes_to_the_local_model(
    setup: Setup,
) -> None:
    router = setup.router(Fake(REMOTE), Fake(LOCAL))

    assert await router.decide(PHOTO) == (Route.LOCAL, RouteReason.CAPABILITY)


async def test_tools_the_remote_cannot_call_go_to_the_local_model(
    setup: Setup,
) -> None:
    plain = ModelInfo("plain", 8000, vision=False, tools=False)
    router = setup.router(Fake(plain), Fake(LOCAL))
    request = ChatRequest((Message.user("weather?"),), tools=(WEATHER,))

    assert await router.decide(request) == (Route.LOCAL, RouteReason.CAPABILITY)
    assert await router.decide(HELLO) == (Route.REMOTE, RouteReason.PREFERRED)


async def test_an_open_circuit_goes_local_at_once(setup: Setup) -> None:
    setup.breaker.record_failure()
    router = setup.router(Fake(REMOTE), Fake(LOCAL))

    assert await router.decide(HELLO) == (Route.LOCAL, RouteReason.CIRCUIT_OPEN)


async def test_the_reserve_is_where_chat_turns_local(setup: Setup) -> None:
    router = setup.router(Fake(REMOTE), Fake(LOCAL))

    await setup.spend(39)  # 11 left, one above the reserve of 10
    assert await router.decide(HELLO) == (Route.REMOTE, RouteReason.PREFERRED)
    await setup.spend(1)  # 10 left: at the reserve
    assert await router.decide(HELLO) == (Route.LOCAL, RouteReason.LOW_BUDGET)


async def test_a_long_rate_limit_wait_goes_local_and_a_short_one_waits(
    setup: Setup,
) -> None:
    router = setup.router(Fake(REMOTE), Fake(LOCAL))
    await setup.limiter.acquire()
    await setup.limiter.acquire()  # full: the next slot opens at t=60

    assert await router.decide(HELLO) == (Route.LOCAL, RouteReason.RATE_LIMITED)
    setup.clock.now = 55.0  # the slot opens in 5 s, exactly the longest wait
    assert await router.decide(HELLO) == (Route.REMOTE, RouteReason.PREFERRED)


async def test_without_a_local_model_the_remote_serves_down_to_zero(
    setup: Setup,
) -> None:
    router = setup.router(Fake(REMOTE))
    await setup.spend(45)

    answer = await collect(router.stream(HELLO))

    assert answer.text == REMOTE.id
    assert setup.routes() == [(Route.REMOTE, RouteReason.NO_LOCAL, REMOTE.id)]


async def test_a_local_model_that_cannot_serve_leaves_it_to_the_remote(
    setup: Setup,
) -> None:
    blind = ModelInfo("blind", 8000, vision=False, tools=False)
    router = setup.router(Fake(REMOTE), Fake(blind))

    assert await router.decide(HELLO, background=True) == (
        Route.LOCAL,
        RouteReason.BACKGROUND,
    )
    assert await router.decide(PHOTO) == (Route.REMOTE, RouteReason.NO_LOCAL)


async def test_a_remote_that_fails_before_answering_falls_back_once(
    setup: Setup,
) -> None:
    remote, local = Fake(REMOTE, AuthError("bad key")), Fake(LOCAL)

    answer = await collect(setup.router(remote, local).stream(HELLO))

    assert answer.text == LOCAL.id
    assert setup.routes() == [
        (Route.REMOTE, RouteReason.PREFERRED, REMOTE.id),
        (Route.LOCAL, RouteReason.FALLBACK, LOCAL.id),
    ]


async def test_a_remote_that_fails_mid_answer_is_not_switched(setup: Setup) -> None:
    remote, local = Fake(REMOTE), Fake(LOCAL)
    remote.fail_after_first = True

    with pytest.raises(ProviderError):
        await collect(setup.router(remote, local).stream(HELLO))
    assert local.calls == 0


async def test_a_failed_remote_with_no_local_model_raises(setup: Setup) -> None:
    router = setup.router(Fake(REMOTE, AuthError("bad key")))

    with pytest.raises(AuthError):
        await collect(router.stream(HELLO))


async def test_a_failed_remote_with_a_local_model_that_cannot_serve_raises(
    setup: Setup,
) -> None:
    seeing = ModelInfo("seeing", 128_000, vision=True, tools=True)
    blind = ModelInfo("blind", 8000, vision=False, tools=False)
    router = setup.router(Fake(seeing, AuthError("bad key")), Fake(blind))

    with pytest.raises(AuthError):
        await collect(router.stream(PHOTO))


async def test_info_is_what_either_model_can_do(setup: Setup) -> None:
    assert setup.router(Fake(REMOTE), Fake(LOCAL)).info == ModelInfo(
        "auto", 32_000, vision=True, tools=True
    )
    assert setup.router(Fake(REMOTE)).info == ModelInfo(
        "auto", 128_000, vision=False, tools=True
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
    router = Router(remote, health, Fake(LOCAL), setup.publish)

    answer = await collect(router.stream(HELLO))

    assert answer.text == LOCAL.id
    assert flaky.calls == 2  # the third attempt met an open circuit
    assert (await setup.ledger.status(PROVIDER)).used == 2
    assert await router.decide(HELLO) == (Route.LOCAL, RouteReason.CIRCUIT_OPEN)
    with pytest.raises(CircuitOpenError):
        await collect(remote.stream(HELLO))
