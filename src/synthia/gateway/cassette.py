"""Record HTTP exchanges once, replay them forever.

Tests and evaluations must not spend the remote request budget or depend on the
network, yet they should exercise real provider responses. A cassette is an
``httpx`` transport that, when recording, forwards each request to a real
transport and writes the exchange to a JSON file, and, when replaying, answers
from that file and never touches the network.

What a cassette stores is deliberately narrow, because cassettes are committed:

- **no request headers**, so an ``Authorization`` header can never be written;
- the request body as a hash of its canonical JSON (plus a short summary), so
  key order does not affect matching;
- response headers from an allowlist only, so cookies and tracking ids are
  dropped;
- response bodies with every known secret replaced.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast, override

import httpx

from synthia.gateway.errors import GatewayError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

FORMAT_VERSION = 1
REDACTED = "**********"
SUMMARY_LENGTH = 120
KEPT_RESPONSE_HEADERS = frozenset({"content-type", "retry-after"})
KEPT_RESPONSE_HEADER_PREFIXES = ("x-ratelimit-",)

type JSON = dict[str, Any]


class CassetteError(GatewayError):
    """A cassette file is unreadable or has no recording for a request."""


class Mode(StrEnum):
    """Whether a cassette answers from its file or from the network."""

    REPLAY = "replay"
    RECORD = "record"


@dataclass(frozen=True, slots=True)
class _Key:
    method: str
    url: str
    body_sha256: str


def _canonical_body(content: bytes) -> bytes:
    """Return a JSON body in a canonical form, or any other body unchanged."""
    try:
        return json.dumps(
            json.loads(content), sort_keys=True, separators=(",", ":")
        ).encode()
    except (ValueError, UnicodeDecodeError):
        return content


def _key(request: httpx.Request) -> _Key:
    digest = hashlib.sha256(_canonical_body(request.content)).hexdigest()
    return _Key(request.method, str(request.url), digest)


def _summary(content: bytes) -> str:
    """Return a short, human-readable hint of what was asked, for reviewing diffs."""
    try:
        body = cast("JSON", json.loads(content))
        messages = cast("list[JSON]", body.get("messages") or [])
        last = messages[-1].get("content") if messages else None
        text = last if isinstance(last, str) else "(structured content)"
        hint = f"{body.get('model', '?')}: {text}"
    except (ValueError, UnicodeDecodeError, AttributeError, IndexError, TypeError):
        hint = f"{len(content)} bytes"
    return hint[:SUMMARY_LENGTH]


def _kept_headers(headers: httpx.Headers) -> dict[str, str]:
    return {
        name: value
        for name, value in sorted((n.lower(), v) for n, v in headers.items())
        if name in KEPT_RESPONSE_HEADERS
        or name.startswith(KEPT_RESPONSE_HEADER_PREFIXES)
    }


def _encode_body(body: bytes) -> JSON:
    try:
        return {"text": body.decode("utf-8")}
    except UnicodeDecodeError:
        return {"base64": base64.b64encode(body).decode("ascii")}


def _decode_body(stored: JSON) -> bytes:
    if "text" in stored:
        return str(stored["text"]).encode("utf-8")
    return base64.b64decode(str(stored["base64"]))


class CassetteTransport(httpx.AsyncBaseTransport):
    """An ``httpx`` transport backed by a cassette file.

    Use :meth:`replaying` in tests. Use :meth:`recording` once, against the real
    service, to create or refresh the file, then commit it.
    """

    def __init__(
        self,
        path: Path,
        mode: Mode,
        inner: httpx.AsyncBaseTransport | None = None,
        secrets: Iterable[str] = (),
    ) -> None:
        self.path = path
        self.mode = mode
        self._inner = inner
        self._secrets = sorted((s for s in secrets if s), key=len, reverse=True)
        self._recorded: list[JSON] = []
        self._queues: dict[_Key, deque[JSON]] = defaultdict(deque)
        if mode is Mode.REPLAY:
            self._load()

    @classmethod
    def replaying(cls, path: Path) -> CassetteTransport:
        """Answer every request from ``path``, never from the network.

        Raises:
            CassetteError: If the file is missing, unreadable or of another version.
        """
        return cls(path, Mode.REPLAY)

    @classmethod
    def recording(
        cls,
        path: Path,
        inner: httpx.AsyncBaseTransport | None = None,
        secrets: Iterable[str] = (),
    ) -> CassetteTransport:
        """Send requests through ``inner`` and write the exchanges to ``path`` on close.

        ``secrets`` are replaced in every recorded body.
        """
        return cls(path, Mode.RECORD, inner or httpx.AsyncHTTPTransport(), secrets)

    @property
    def unplayed(self) -> int:
        """Return how many recorded exchanges have not been replayed yet."""
        return sum(len(queue) for queue in self._queues.values())

    @override
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        if self._inner is None:
            return self._replay(request)
        return await self._record(request, self._inner)

    @override
    async def aclose(self) -> None:
        if self._inner is None:
            return
        await self._inner.aclose()
        document = {"version": FORMAT_VERSION, "interactions": self._recorded}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(document, file, indent=2, sort_keys=True, ensure_ascii=False)
            file.write("\n")

    def _replay(self, request: httpx.Request) -> httpx.Response:
        key = _key(request)
        queue = self._queues.get(key)
        if not queue:
            message = (
                f"no recording in {self.path.name} for {key.method} {key.url} "
                f"(body sha256 {key.body_sha256[:12]}); re-record this cassette"
            )
            raise CassetteError(message)
        response = queue.popleft()
        return httpx.Response(
            int(response["status"]),
            headers=cast("dict[str, str]", response["headers"]),
            content=_decode_body(cast("JSON", response["body"])),
            request=request,
        )

    async def _record(
        self, request: httpx.Request, inner: httpx.AsyncBaseTransport
    ) -> httpx.Response:
        live = await inner.handle_async_request(request)
        # aread() undoes any content-encoding, and the encoding header is not
        # kept, so replay never tries to decompress an already plain body.
        body = self._scrub(await live.aread())
        await live.aclose()
        key = _key(request)
        response: JSON = {
            "status": live.status_code,
            "headers": _kept_headers(live.headers),
            "body": _encode_body(body),
        }
        self._recorded.append(
            {
                "request": {
                    "method": key.method,
                    "url": key.url,
                    "body_sha256": key.body_sha256,
                    "summary": self._scrub(_summary(request.content).encode()).decode(),
                },
                "response": response,
            }
        )
        return httpx.Response(
            live.status_code, headers=response["headers"], content=body, request=request
        )

    def _scrub(self, body: bytes) -> bytes:
        for secret in self._secrets:
            body = body.replace(secret.encode(), REDACTED.encode())
        return body

    def _load(self) -> None:
        try:
            document = cast("JSON", json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as error:
            message = f"cannot read cassette {self.path.name}: {type(error).__name__}"
            raise CassetteError(message) from error
        if document.get("version") != FORMAT_VERSION:
            message = (
                f"cassette {self.path.name} is not format version {FORMAT_VERSION}"
            )
            raise CassetteError(message)
        for interaction in cast("list[JSON]", document.get("interactions", [])):
            recorded = cast("JSON", interaction["request"])
            key = _Key(
                str(recorded["method"]),
                str(recorded["url"]),
                str(recorded["body_sha256"]),
            )
            self._queues[key].append(cast("JSON", interaction["response"]))
