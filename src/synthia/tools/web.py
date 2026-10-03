"""Fetch a web page as text, on command: OUTSIDE, so every call asks first.

Each call uses a new HTTP client, so no cookie or connection is kept between
calls or shared with model traffic, and it carries no key. Only http and
https are fetched, only text comes back (HTML is reduced to its visible
text), and both the bytes read and the time taken are capped.
"""

from __future__ import annotations

import asyncio
from html.parser import HTMLParser
from typing import Annotated, Final, override

import httpx
from pydantic import Field

from synthia import __version__
from synthia.agent.tools import Effect, FunctionTool, Reach, ToolError

FETCH_TIMEOUT_S: Final = 20.0
MAX_FETCH_BYTES: Final = 2_000_000
MAX_TEXT_CHARS: Final = 20_000
MAX_ERROR_CHARS: Final = 2000
MAX_URL_CHARS: Final = 2000
MAX_REDIRECTS: Final = 5
SCHEMES: Final = frozenset({"http", "https"})
_TEXT_TYPES: Final = frozenset(
    {"application/json", "application/xml", "application/javascript"}
)
_HIDDEN: Final = frozenset({"script", "style", "noscript", "template", "svg"})
_BLOCKS: Final = frozenset(
    {
        "p",
        "div",
        "br",
        "li",
        "tr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "section",
        "article",
        "header",
        "footer",
        "pre",
        "blockquote",
        "table",
        "ul",
        "ol",
        "title",
    }
)


class _VisibleText(HTMLParser):
    """Collects the text a reader would see, a line per block."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._hidden = 0
        self._parts: list[str] = []

    @override
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _HIDDEN:
            self._hidden += 1
        elif tag in _BLOCKS:
            self._parts.append("\n")

    @override
    def handle_endtag(self, tag: str) -> None:
        if tag in _HIDDEN:
            self._hidden = max(0, self._hidden - 1)
        elif tag in _BLOCKS:
            self._parts.append("\n")

    @override
    def handle_data(self, data: str) -> None:
        if not self._hidden:
            self._parts.append(data)

    def text(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self._parts).splitlines())
        return "\n".join(line for line in lines if line)


def visible_text(html: str) -> str:
    """Return the visible text of an HTML page, without scripts or styles."""
    parser = _VisibleText()
    parser.feed(html)
    parser.close()
    return parser.text()


def is_text(content_type: str) -> bool:
    """Return whether a Content-Type names text a model can read."""
    media = content_type.split(";", 1)[0].strip().lower()
    return (
        media.startswith("text/")
        or media in _TEXT_TYPES
        or media.endswith(("+json", "+xml"))
    )


def fetch_tool(
    transport: httpx.AsyncBaseTransport | None = None,
    *,
    timeout_s: float = FETCH_TIMEOUT_S,
) -> FunctionTool:
    """Return the tool that fetches a page; ``transport`` is for tests."""

    async def fetch_url(
        url: Annotated[
            str,
            Field(
                min_length=1,
                max_length=MAX_URL_CHARS,
                description="an http or https address",
            ),
        ],
    ) -> str:
        """Fetch a web page and return its text, HTML reduced to what a reader sees."""
        address = _checked(url)
        async with httpx.AsyncClient(
            transport=transport,
            follow_redirects=True,
            max_redirects=MAX_REDIRECTS,
            timeout=timeout_s,
            headers={"User-Agent": f"SYNTHIA/{__version__}"},
        ) as client:
            try:
                async with asyncio.timeout(timeout_s):
                    body, response = await _read(client, address)
            except TimeoutError as error:
                message = f"no complete answer within {timeout_s:g} s"
                raise ToolError(message) from error
            except httpx.HTTPError as error:
                message = f"could not fetch {address}: {type(error).__name__}: {error}"
                raise ToolError(message) from error
        return _describe(response, body)

    return FunctionTool.of(fetch_url, reach=Reach.OUTSIDE, effect=Effect.READ)


def _checked(url: str) -> httpx.URL:
    try:
        address = httpx.URL(url.strip())
    except httpx.InvalidURL as error:
        message = f"not a web address: {error}"
        raise ToolError(message) from error
    if address.scheme not in SCHEMES or not address.host:
        message = "only http and https addresses with a host can be fetched"
        raise ToolError(message)
    return address


async def _read(
    client: httpx.AsyncClient, address: httpx.URL
) -> tuple[bytes, httpx.Response]:
    async with client.stream("GET", address) as response:
        content_type = response.headers.get("content-type", "")
        if response.is_success and not is_text(content_type):
            message = (
                f"{response.url} is {content_type or 'of no stated type'}, not text"
            )
            raise ToolError(message)
        body = bytearray()
        async for chunk in response.aiter_bytes():
            body.extend(chunk)
            # Read past the limit, so a page of exactly the limit counts as whole.
            if len(body) > MAX_FETCH_BYTES:
                break
        return bytes(body), response


def _describe(response: httpx.Response, body: bytes) -> str:
    url, status = response.url, response.status_code
    content_type = response.headers.get("content-type", "")
    whole = len(body) <= MAX_FETCH_BYTES
    # httpx gives the declared charset, or UTF-8 when it is missing or unknown.
    text = body[:MAX_FETCH_BYTES].decode(response.encoding or "utf-8", "replace")
    if "html" in content_type.lower():
        text = visible_text(text)
    if status >= httpx.codes.BAD_REQUEST:
        message = f"HTTP {status} from {url}:\n{text[:MAX_ERROR_CHARS]}"
        raise ToolError(message)
    head = f"{url} ({status}, {content_type or 'no type'})\n\n"
    if whole and len(text) <= MAX_TEXT_CHARS:
        return head + text
    if whole:
        note = f"{len(text):,} characters in all"
    else:
        note = (
            f"{len(text):,} characters in the first {MAX_FETCH_BYTES:,} bytes, "
            "and the page is longer"
        )
    return f"{head}{text[:MAX_TEXT_CHARS]}\n[cut: {note}]"
