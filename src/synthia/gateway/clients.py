"""The one way SYNTHIA makes an HTTP client for its models and its own servers.

httpx sends requests through the proxy named by ``HTTP_PROXY``, ``HTTPS_PROXY``
or ``ALL_PROXY`` when one is set. Over https a proxy only tunnels the encrypted
connection, but a plain-http request reaches it whole, ``Authorization`` header
included. The only plain-http addresses SYNTHIA talks to are on this machine
(the local model's server, a provider on ``localhost``, the daemon), so those go
direct, never through a proxy; everything else keeps the environment's proxy.
"""

from __future__ import annotations

from typing import Final

import httpx

LOOPBACK_HOSTS: Final = ("localhost", "127.0.0.1", "::1")
LOOPBACK_ORIGINS: Final = tuple(
    f"http://[{host}]" if ":" in host else f"http://{host}" for host in LOOPBACK_HOSTS
)
# httpx's own default, so a caller that names no timeout gets what httpx gives.
DEFAULT_TIMEOUT: Final = httpx.Timeout(5.0)


def http_client(
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout: httpx.Timeout = DEFAULT_TIMEOUT,
) -> httpx.AsyncClient:
    """Return a client whose requests to this machine never go through a proxy.

    ``transport`` replaces the network for every request, a test's included.
    """
    direct: dict[str, httpx.AsyncBaseTransport | None] = dict.fromkeys(LOOPBACK_ORIGINS)
    return httpx.AsyncClient(transport=transport, timeout=timeout, mounts=direct)
