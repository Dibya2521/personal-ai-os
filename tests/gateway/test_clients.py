import httpx
import pytest

from synthia.gateway.clients import http_client

PROXY = "http://proxy.invalid:3128"


@pytest.fixture
def proxied(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.setenv(variable, PROXY)
    for variable in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(variable, raising=False)


def chosen(client: httpx.AsyncClient, url: str) -> httpx.AsyncBaseTransport:
    """Return the transport httpx picks for ``url``: a proxy's, or the direct one."""
    return client._transport_for_url(httpx.URL(url))  # noqa: SLF001 # pyright: ignore[reportPrivateUsage]


def direct(client: httpx.AsyncClient) -> httpx.AsyncBaseTransport:
    """Return the client's own transport, the one that uses no proxy."""
    return client._transport  # noqa: SLF001 # pyright: ignore[reportPrivateUsage]


@pytest.mark.usefixtures("proxied")
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/v1/chat/completions",
        "http://localhost:11434/v1/chat/completions",
        "http://[::1]:8080/health",
    ],
)
async def test_plain_http_to_this_machine_never_goes_through_a_proxy(url: str) -> None:
    async with http_client() as client:
        assert chosen(client, url) is direct(client)


@pytest.mark.usefixtures("proxied")
@pytest.mark.parametrize(
    "url",
    ["https://api.openai.com/v1/chat/completions", "http://127.9.9.9:8080/v1"],
)
async def test_everything_else_keeps_the_environments_proxy(url: str) -> None:
    async with http_client() as client:
        assert chosen(client, url) is not direct(client)


async def test_a_given_transport_carries_every_request() -> None:
    sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(str(request.url))
        return httpx.Response(200)

    async with http_client(transport=httpx.MockTransport(handler)) as client:
        await client.get("http://127.0.0.1:8080/health")
        await client.get("https://api.openai.com/v1/models")

    assert sent == ["http://127.0.0.1:8080/health", "https://api.openai.com/v1/models"]
