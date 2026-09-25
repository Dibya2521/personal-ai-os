"""How long a test waits for something that is expected to happen."""

# A net for a hang, not a measure of speed: a passing test never waits this long,
# and a fully loaded machine can take many seconds just to start a Python process.
HANG_TIMEOUT_S = 120.0
