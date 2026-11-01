import asyncio
import json

import httpx
import pytest

from synthia import __version__
from synthia.agent.tools import Effect, Reach, ToolError
from synthia.tools.web import (
    MAX_FETCH_BYTES,
    MAX_REDIRECTS,
    MAX_TEXT_CHARS,
    Resolver,
    fetch_tool,
    host_addresses,
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
    handler: object,
    url: str = "https://example.org/kettle",
    resolve: Resolver = host_addresses,
    **options: float,
) -> str:
    tool = fetch_tool(httpx.MockTransport(handler), resolve=resolve, **options)  # type: ignore[arg-type]
    return await tool.run(json.dumps({"url": url}))


def resolving_to(*addresses: str) -> Resolver:
    async def resolve(host: str) -> list[str]:
        del host
        return list(addresses)

    return resolve


def redirecting_to(location: str, sent: list[str]) -> object:
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(str(request.url))
        if request.url.host == "example.org":
            return httpx.Response(302, headers={"location": location})
        return httpx.Response(200, text="inner secret")

    return handler


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


@pytest.mark.parametrize(
    "location",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1:11434/api/tags",
        "http://[::1]/",
        "http://10.0.0.5/admin",
        "http://192.168.1.1/",
        "http://[::ffff:127.0.0.1]/",
        "http://168.63.129.16/metadata/",
    ],
)
async def test_a_redirect_to_a_local_or_private_address_is_refused(
    location: str,
) -> None:
    sent: list[str] = []

    with pytest.raises(ToolError, match=r"^refused a redirect to .+ local or private"):
        await fetched(redirecting_to(location, sent))

    assert sent == ["https://example.org/kettle"]


async def test_a_redirect_to_a_name_that_resolves_locally_is_refused() -> None:
    sent: list[str] = []

    with pytest.raises(ToolError, match=r"localtest\.me is a local .+\(127\.0\.0\.1\)"):
        await fetched(
            redirecting_to("http://localtest.me/", sent),
            resolve=resolving_to("93.184.215.14", "127.0.0.1"),
        )

    assert sent == ["https://example.org/kettle"]


async def test_a_redirect_to_a_public_host_is_followed() -> None:
    sent: list[str] = []

    result = await fetched(
        redirecting_to("https://www.example.com/", sent),
        resolve=resolving_to("93.184.215.14"),
    )

    assert result.startswith("https://www.example.com/ (200")
    assert sent == ["https://example.org/kettle", "https://www.example.com/"]


async def test_a_redirect_within_an_approved_local_host_is_followed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(302, headers={"location": "/login"})
        return httpx.Response(200, text="dev server")

    result = await fetched(handler, "http://127.0.0.1:8000/")

    assert result.startswith("http://127.0.0.1:8000/login (200")


async def test_a_redirect_to_another_port_of_an_approved_local_host_is_refused() -> (
    None
):
    sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1:11434/"})

    with pytest.raises(
        ToolError, match=r"^refused a redirect to http://127\.0\.0\.1:11434"
    ):
        await fetched(handler, "http://127.0.0.1:8000/")

    assert sent == ["http://127.0.0.1:8000/"]


async def test_a_redirect_host_that_cannot_be_resolved_is_a_tool_error() -> None:
    async def fail(host: str) -> list[str]:
        raise OSError(host)

    with pytest.raises(ToolError, match=r"^could not resolve nowhere\.test"):
        await fetched(redirecting_to("http://nowhere.test/", []), resolve=fail)


async def test_a_redirect_to_another_scheme_is_refused() -> None:
    with pytest.raises(ToolError, match=r"^only http and https"):
        await fetched(redirecting_to("ftp://example.com/file", []))


async def test_too_many_redirects_is_a_tool_error() -> None:
    sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(str(request.url))
        return httpx.Response(302, headers={"location": f"/{len(sent)}"})

    with pytest.raises(ToolError, match=rf"^more than {MAX_REDIRECTS} redirects"):
        await fetched(handler)

    assert len(sent) == MAX_REDIRECTS + 1


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


@pytest.mark.parametrize(
    ("size", "note"),
    [
        (MAX_TEXT_CHARS, None),
        (MAX_TEXT_CHARS + 1, "20,001 characters in all"),
        (MAX_FETCH_BYTES, "2,000,000 characters in all"),
        (
            MAX_FETCH_BYTES + 1,
            "2,000,000 characters in the first 2,000,000 bytes, and the page is longer",
        ),
        (
            MAX_FETCH_BYTES * 2,
            "2,000,000 characters in the first 2,000,000 bytes, and the page is longer",
        ),
    ],
    ids=["text-limit", "text-limit+1", "byte-limit", "byte-limit+1", "twice"],
)
async def test_a_long_answer_is_cut_and_says_how_much_was_read(
    size: int, note: str | None
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/plain"}, content=b"a" * size
        )

    result = await fetched(handler)

    head, text = result.split("\n\n", 1)
    assert head == "https://example.org/kettle (200, text/plain)"
    if note is None:
        assert text == "a" * size
    else:
        assert text == "a" * MAX_TEXT_CHARS + f"\n[cut: {note}]"


async def test_a_long_page_with_little_visible_text_still_says_it_was_cut() -> None:
    page = b"<p>kettle</p><script>" + b"x" * MAX_FETCH_BYTES + b"</script>"

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, content=page)

    assert await fetched(handler) == (
        "https://example.org/kettle (200, text/html)\n\n"
        "kettle\n[cut: 6 characters in the first 2,000,000 bytes, "
        "and the page is longer]"
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
