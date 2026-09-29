import os

import pytest
from hypothesis import settings

from synthia.kernel.config import ENV_PREFIX

# Hypothesis fails an example that takes over 200 ms by the wall clock, which a
# loaded machine exceeds on code that is not slow. Hangs are caught by
# tests.timing.HANG_TIMEOUT_S, and complexity by timing 4x the input against 1x.
settings.register_profile("synthia", deadline=None)
settings.load_profile("synthia")


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide the developer's own SYNTHIA_ variables, so tests see only what they set."""
    for name in [n for n in os.environ if n.upper().startswith(ENV_PREFIX)]:
        monkeypatch.delenv(name)
