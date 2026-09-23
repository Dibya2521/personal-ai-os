"""Errors the gateway raises on purpose."""

from synthia.kernel.errors import SynthiaError


class GatewayError(SynthiaError):
    """Base class for every model gateway error."""


class IncompleteResponseError(GatewayError):
    """A stream ended before the completion was whole.

    Either no finish reason arrived, or a tool call never received its id or
    name. Treating such a stream as a finished answer would act on a truncated
    one.
    """
