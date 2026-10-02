import asyncio
import json

import httpx
import pytest

from synthia import __version__
from synthia.agent.tools import Effect, Reach, ToolError
from synthia.tools.web import (
    MAX_FETCH_BYTES,
    MAX_TEXT_CHARS,
    fetch_tool,
    is_text,
    visible_text,
)

PAGE = (
    "<html><head><title>Kettles</title><style>p {color: red}</style></head>"
    "<body><h1>Best   kettle</h1><script>alert('x')</script>"
    "<p>Boils in <b>90 s</b> &amp; stays quiet.</p><ul><li>one</li><li>two</li></ul>"
    "</body></html>"
)


async def fetched(
    handler: object, url: str = "https://example.org/kettle", **options: float
) -> str:
    tool = fetch_tool(httpx.MockTransport(handler), **options)  # type: ignore[arg-type]
    return await tool.run(json.dumps({"url": url}))


def test_visible_text_keeps_what_a_reader_sees() -> None:
    assert (
        visible_text(PAGE)
        == "Kettles\nBest kettle\nBoils in 90 s & stays quiet.\none\ntwo"
    )


@pytest.mark.parametrize(
    ("content_type", "text"),
    [
        ("text/html; charset=utf-8", True),
        ("TEXT/PLAIN", True),
        ("application/json", True),
        ("application/ld+json", True),
        ("application/atom+xml", True),
        ("image/png", False),
        ("application/octet-stream", False),
        ("", False),
    ],
)
def test_only_text_types_are_read(content_type: str, text: bool) -> None:
    assert is_text(content_type) is text


async def test_a_page_comes_back_as_its_visible_text() -> None:
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, headers={"content-type": "text/html"}, text=PAGE)

    assert await fetched(handler) == (
        "https://example.org/kettle (200, text/html)\n\n"
        "Kettles\nBest kettle\nBoils in 90 s & stays quiet.\none\ntwo"
    )
    (request,) = sent
    assert request.headers["user-agent"] == f"SYNTHIA/{__version__}"
    assert "authorization" not in request.headers


async def test_json_is_returned_as_it_came() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"a": 1})

    assert await fetched(handler) == (
        'https://example.org/kettle (200, application/json)\n\n{"a":1}'
    )


async def test_redirects_are_followed_and_the_final_address_named() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(302, headers={"location": "https://example.org/new"})
        return httpx.Response(200, text="moved here")

    assert (await fetched(handler, "https://example.org/old")).startswith(
        "https://example.org/new (200, text/plain; charset=utf-8)\n\nmoved here"
    )


async def test_a_binary_answer_is_refused() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "image/png"}, content=b"\x89PNG"
        )

    with pytest.raises(
        ToolError, match=r"^https://example.org/kettle is image/png, not text$"
    ):
        await fetched(handler)


async def test_an_error_status_is_a_tool_error_with_the_page() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="no such kettle")

    with pytest.raises(
        ToolError, match=r"^HTTP 404 from https://example.org/kettle:\nno such kettle$"
    ):
        await fetched(handler)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.org/x",
        "example.org",
        "https://",
        "http://[::1",
    ],
)
async def test_only_http_and_https_addresses_are_fetched(url: str) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        pytest.fail("nothing may be sent")

    with pytest.raises(ToolError, match=r"only http and https|not a web address"):
        await fetched(handler, url)


async def test_a_long_answer_is_read_only_to_the_byte_limit_and_cut() -> None:
    body = b"a" * (MAX_FETCH_BYTES * 2)

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/plain"}, content=body)

    result = await fetched(handler)

    head, text = result.split("\n\n", 1)
    assert head == "https://example.org/kettle (200, text/plain)"
    assert (
        text == "a" * MAX_TEXT_CHARS + f"\n[cut: {MAX_FETCH_BYTES:,} characters in all]"
    )


async def test_a_slow_server_is_given_up_on() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        await asyncio.Event().wait()
        return httpx.Response(200)

    with pytest.raises(ToolError, match=r"^no complete answer within 0.05 s$"):
        await fetched(handler, timeout_s=0.05)


async def test_a_failed_connection_is_a_tool_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        reason = "refused"
        raise httpx.ConnectError(reason, request=request)

    with pytest.raises(
        ToolError,
        match=r"^could not fetch https://example.org/kettle: ConnectError: refused$",
    ):
        await fetched(handler)


async def test_no_cookie_is_kept_from_one_call_to_the_next() -> None:
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("cookie"))
        return httpx.Response(200, headers={"set-cookie": "id=1; Path=/"}, text="hi")

    tool = fetch_tool(httpx.MockTransport(handler))
    for _ in range(2):
        await tool.run('{"url": "https://example.org/"}')

    assert seen == [None, None]


def test_fetching_reaches_outside_and_changes_nothing() -> None:
    tool = fetch_tool()

    assert (tool.spec.name, tool.reach, tool.effect) == (
        "fetch_url",
        Reach.OUTSIDE,
        Effect.READ,
    )
