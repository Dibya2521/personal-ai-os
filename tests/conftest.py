import os

import pytest

from synthia.kernel.config import ENV_PREFIX


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide the developer's own SYNTHIA_ variables, so tests see only what they set."""
    for name in [n for n in os.environ if n.upper().startswith(ENV_PREFIX)]:
        monkeypatch.delenv(name)
