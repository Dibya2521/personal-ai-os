"""How long a test waits for something that is expected to happen."""

# A net for a hang, not a measure of speed: a passing test never waits this long.
# A daemon start is 6 to 9 s on a quiet machine, but with an antivirus scanning every
# file read it took 55 s alone and over 120 s beside four test workers.
HANG_TIMEOUT_S = 300.0
