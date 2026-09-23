"""The error hierarchy.

Every error SYNTHIA raises on purpose derives from :class:`SynthiaError`, so a
caller can separate a known failure, whose message is safe to show a user, from a
defect, which is logged with its traceback and shown only as a generic message.
"""


class SynthiaError(Exception):
    """Base class for every error SYNTHIA raises on purpose."""


class ConfigError(SynthiaError):
    """The configuration could not be loaded or is invalid."""
