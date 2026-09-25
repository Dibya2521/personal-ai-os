"""The local model, as a chat model the router can send requests to."""

from __future__ import annotations

from contextlib import aclosing
from typing import TYPE_CHECKING

from synthia.gateway.errors import ConnectionFailedError
from synthia.gateway.openai_compat import Dialect, Endpoint, OpenAICompatibleModel
from synthia.gateway.types import ModelInfo

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    import httpx

    from synthia.gateway.types import ChatChunk, ChatRequest
    from synthia.models.catalogue import Model
    from synthia.models.server import LlamaServer


class LocalModel:
    """A :class:`~synthia.gateway.protocol.ChatModel` served by a :class:`LlamaServer`.

    Each request goes to the launch serving at that moment, with that launch's
    key, so a restart onto a new port and key is followed without notice.
    """

    def __init__(
        self,
        server: LlamaServer,
        client: httpx.AsyncClient,
        model: Model,
        context: int,
    ) -> None:
        self._server = server
        self._client = client
        self._info = ModelInfo(model.id, context, vision=model.vision, tools=True)

    @property
    def info(self) -> ModelInfo:
        """Return what the local model can do."""
        return self._info

    def ready(self) -> bool:
        """Return whether the server is serving, for the router's choice."""
        return self._server.running is not None

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        """Yield the local model's answer.

        Raises:
            ConnectionFailedError: If the server is not serving.
            GatewayError: What the server's answer raised.
        """
        launch = self._server.running
        if launch is None:
            message = "the local model is not running"
            raise ConnectionFailedError(message)
        endpoint = Endpoint(
            launch.base_url, self._info.id, launch.key, dialect=Dialect.LLAMA_CPP
        )
        adapter = OpenAICompatibleModel(
            client=self._client, endpoint=endpoint, info=self._info
        )
        async with aclosing(adapter.stream(request)) as chunks:
            async for chunk in chunks:
                yield chunk
