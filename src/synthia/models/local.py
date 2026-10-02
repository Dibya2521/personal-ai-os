"""The local model, as a chat model the router can send requests to."""

from __future__ import annotations

import asyncio
import logging
import math
from contextlib import aclosing
from typing import TYPE_CHECKING, Final, cast

import httpx

from synthia.gateway.errors import ConnectionFailedError
from synthia.gateway.openai_compat import Dialect, Endpoint, OpenAICompatibleModel
from synthia.gateway.reasoning import allowance_s
from synthia.gateway.types import ModelInfo

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from synthia.gateway.types import ChatChunk, ChatRequest
    from synthia.models.catalogue import Model
    from synthia.models.server import Launch, LlamaServer

logger = logging.getLogger(__name__)

CONTROL_SUFFIX: Final = "/control"
END_REASONING: Final = "reasoning_end"
MAX_DETAIL: Final = 200
READY_POLL_S: Final = 0.25

type Sleep = Callable[[float], Awaitable[None]]


async def end_thinking(
    client: httpx.AsyncClient, endpoint: Endpoint, completion_id: str
) -> bool:
    """Tell llama-server to end ``completion_id``'s thinking now; the answer follows.

    Returns whether the server did. A failure is logged, never raised: the
    answer still comes, only when the model stops thinking by itself. The
    server answers an unknown id with 200 and ``"success": false``, so the
    body is read, not only the status.
    """
    headers: dict[str, str] = {}
    if endpoint.api_key is not None:
        headers["Authorization"] = f"Bearer {endpoint.api_key.get_secret_value()}"
    body = {"id": completion_id, "action": END_REASONING}
    try:
        response = await client.post(
            endpoint.completions_url + CONTROL_SUFFIX, json=body, headers=headers
        )
    except httpx.TransportError as error:
        logger.warning("could not end the local thinking: %s", type(error).__name__)
        return False
    try:
        answer: object = response.json()
    except ValueError:
        answer = None
    if (
        response.is_success
        and isinstance(answer, dict)
        and cast("dict[str, object]", answer).get("success") is True
    ):
        return True
    detail = response.text[:MAX_DETAIL]
    logger.warning(
        "the local server did not end its thinking: HTTP %s %s",
        response.status_code,
        detail,
    )
    return False


class LocalModel:
    """A :class:`~synthia.gateway.protocol.ChatModel` served by a :class:`LlamaServer`.

    Each request goes to the launch serving at that moment, with that launch's
    key, so a restart onto a new port and key is followed without notice.

    Thinking is held to the request level's time allowance: once the model has
    thought that long without starting its answer, the server is told to end
    the thinking, and the answer follows. A request that arrives while the
    server is still starting waits for it. ``sleep`` waits out both.
    """

    def __init__(
        self,
        server: LlamaServer,
        client: httpx.AsyncClient,
        model: Model,
        context: int,
        *,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._server = server
        self._client = client
        self._info = ModelInfo(model.id, context, vision=model.vision, tools=True)
        self._sleep = sleep

    @property
    def info(self) -> ModelInfo:
        """Return what the local model can do."""
        return self._info

    def ready(self) -> bool:
        """Return whether the server is serving, for the router's choice."""
        return self._server.running is not None

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatChunk]:
        """Yield the local model's answer, once its server is serving.

        Raises:
            ConnectionFailedError: If the server is not serving within its
                start timeout.
            GatewayError: What the server's answer raised.
        """
        launch = await self._serving()
        endpoint = Endpoint(
            launch.base_url, self._info.id, launch.key, dialect=Dialect.LLAMA_CPP
        )
        adapter = OpenAICompatibleModel(
            client=self._client, endpoint=endpoint, info=self._info
        )
        allowance = allowance_s(request.reasoning)
        timer: asyncio.Task[None] | None = None
        answering = False
        try:
            async with aclosing(adapter.stream(request)) as chunks:
                async for chunk in chunks:
                    if chunk.text or chunk.tool_calls or chunk.finish_reason:
                        answering = True
                        if timer is not None:
                            timer.cancel()
                    elif (
                        chunk.reasoning
                        and chunk.id
                        and allowance is not None
                        and timer is None
                        and not answering
                    ):
                        timer = asyncio.create_task(
                            self._end_after(allowance, endpoint, chunk.id)
                        )
                    yield chunk
        finally:
            if timer is not None:
                timer.cancel()
                await asyncio.wait([timer])

    async def _serving(self) -> Launch:
        # Polled, not awaited as an event: the server runs on another thread's
        # loop, and its asyncio.Event cannot be awaited from this one.
        timeout = self._server.start_timeout_s
        for _ in range(math.ceil(timeout / READY_POLL_S)):
            launch = self._server.running
            if launch is not None:
                return launch
            await self._sleep(READY_POLL_S)
        launch = self._server.running
        if launch is None:
            message = f"the local model did not start within {timeout:g} s"
            raise ConnectionFailedError(message)
        return launch

    async def _end_after(
        self, seconds: float, endpoint: Endpoint, completion_id: str
    ) -> None:
        await self._sleep(seconds)
        if await end_thinking(self._client, endpoint, completion_id):
            logger.info("ended the local thinking after %g s", seconds)
