"""Run llama.cpp's server for the local model.

The server listens on loopback only, on a port chosen per launch, and requires
a key made fresh for each launch, so no other program on the machine, and no
web page making requests to localhost, can use it.
"""

from __future__ import annotations

import secrets
import socket
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from pydantic import SecretStr

from synthia.kernel.errors import SynthiaError

if TYPE_CHECKING:
    from pathlib import Path

    from synthia.models.catalogue import Backend, Model

SERVER_NAMES: Final = frozenset({"llama-server", "llama-server.exe"})
LOOPBACK: Final = "127.0.0.1"
API_KEY_VARIABLE: Final = "LLAMA_API_KEY"
KEY_BYTES: Final = 32


class ServerError(SynthiaError):
    """The local model server cannot be found, started or reached."""


def find_server(runtime: Path) -> Path:
    """Return the ``llama-server`` executable inside an installed runtime.

    The layout inside a release archive is not part of its contract, so the
    directory is searched; the shallowest match wins.

    Raises:
        ServerError: If there is none.
    """
    found = [p for p in runtime.rglob("llama-server*") if p.name in SERVER_NAMES]
    if not found:
        message = f"no llama-server in {runtime}; reinstall it"
        raise ServerError(message)
    return min(found, key=lambda p: (len(p.parts), str(p)))


def free_port() -> int:
    """Return a loopback port that was free a moment ago."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((LOOPBACK, 0))
        return int(probe.getsockname()[1])


def new_key() -> SecretStr:
    """Return a key for one launch."""
    return SecretStr(secrets.token_urlsafe(KEY_BYTES))


@dataclass(frozen=True, slots=True)
class Launch:
    """Everything needed to start the server once."""

    backend: Backend
    binary: Path
    weights: Path
    projector: Path | None
    port: int
    context: int
    key: SecretStr

    @classmethod
    def of(  # noqa: PLR0913
        cls,
        backend: Backend,
        runtime: Path,
        model: Model,
        model_dir: Path,
        *,
        port: int,
        context: int,
        key: SecretStr,
    ) -> Launch:
        """Return the launch of ``model``, installed in ``model_dir``, on ``runtime``.

        Raises:
            ServerError: If the runtime has no server executable.
        """
        projector = model_dir / model.projector.name if model.projector else None
        return cls(
            backend,
            find_server(runtime),
            model_dir / model.weights.name,
            projector,
            port,
            context,
            key,
        )

    @property
    def base_url(self) -> str:
        """Return the OpenAI-compatible API root of the running server."""
        return f"http://{LOOPBACK}:{self.port}/v1"

    @property
    def health_url(self) -> str:
        """Return the address that answers 200 once the model is loaded."""
        return f"http://{LOOPBACK}:{self.port}/health"

    def command(self) -> list[str]:
        """Return the command line; the key is passed in the environment instead."""
        args = [
            str(self.binary),
            "--model",
            str(self.weights),
            "--host",
            LOOPBACK,
            "--port",
            str(self.port),
            "--ctx-size",
            str(self.context),
            "--no-webui",
            "--offline",
        ]
        if self.projector is not None:
            args += ["--mmproj", str(self.projector)]
        return args

    def environment(self, base: dict[str, str]) -> dict[str, str]:
        """Return ``base`` with the key added, for the server's process."""
        return {**base, API_KEY_VARIABLE: self.key.get_secret_value()}
