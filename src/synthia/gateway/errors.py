"""Errors the gateway raises on purpose.

Each carries ``retryable``: whether the same request, sent again unchanged, may
succeed. The retry layer decides from that flag alone, so the knowledge of which
failures are transient lives here and nowhere else.
"""

from typing import ClassVar

from synthia.kernel.errors import SynthiaError


class GatewayError(SynthiaError):
    """Base class for every model gateway error."""

    retryable: ClassVar[bool] = False


class IncompleteResponseError(GatewayError):
    """A stream ended before the completion was whole.

    Either no finish reason arrived, or a tool call never received its id or
    name. Treating such a stream as a finished answer would act on a truncated
    one.
    """


class MalformedStreamError(GatewayError):
    """The provider sent something that is not a valid completion chunk."""


class BadRequestError(GatewayError):
    """The provider rejected the request itself (HTTP 400, 404, 413, 422)."""


class AuthError(GatewayError):
    """The API key is missing, wrong or not allowed this model (HTTP 401, 403)."""


class PaymentRequiredError(GatewayError):
    """The account has no credit for this request (HTTP 402)."""


class RateLimitedError(GatewayError):
    """The provider asked us to slow down (HTTP 429).

    ``retry_after_s`` is how long it asked us to wait, when it said.
    """

    retryable: ClassVar[bool] = True

    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class ProviderError(GatewayError):
    """The provider failed on its side (HTTP 5xx or 408), or failed mid-stream."""

    retryable: ClassVar[bool] = True


class ConnectionFailedError(GatewayError):
    """The request never got a response: no connection, a timeout, a reset."""

    retryable: ClassVar[bool] = True
