from collections.abc import AsyncIterator, Iterable

import pytest
from hypothesis import given
from hypothesis import strategies as st

from synthia.gateway.errors import IncompleteResponseError
from synthia.gateway.protocol import collect
from synthia.gateway.types import (
    ChatChunk,
    FinishReason,
    ToolCall,
    ToolCallDelta,
    Usage,
)


async def stream_of(chunks: Iterable[ChatChunk]) -> AsyncIterator[ChatChunk]:
    for chunk in chunks:
        yield chunk


async def test_text_deltas_join_in_order_with_the_last_usage() -> None:
    response = await collect(
        stream_of(
            [
                ChatChunk(text="Hel", model="qwen/actual"),
                ChatChunk(text="lo", model="ignored-later"),
                ChatChunk(finish_reason=FinishReason.STOP, usage=Usage(5, 2)),
            ]
        )
    )

    assert response.text == "Hello"
    assert response.model == "qwen/actual"
    assert response.finish_reason is FinishReason.STOP
    assert response.usage == Usage(5, 2)
    assert response.tool_calls == ()


async def test_tool_call_fragments_join_by_index_whatever_the_interleaving() -> None:
    response = await collect(
        stream_of(
            [
                ChatChunk(tool_calls=(ToolCallDelta(1, "b", "search", '{"q":'),)),
                ChatChunk(tool_calls=(ToolCallDelta(0, "a", "clock", "{"),)),
                ChatChunk(
                    tool_calls=(
                        ToolCallDelta(1, arguments='"x"}'),
                        ToolCallDelta(0, arguments="}"),
                    )
                ),
                ChatChunk(finish_reason=FinishReason.TOOL_CALLS),
            ]
        )
    )

    assert response.tool_calls == (
        ToolCall("a", "clock", "{}"),
        ToolCall("b", "search", '{"q":"x"}'),
    )


@given(st.lists(st.text(max_size=8), max_size=30))
async def test_any_split_of_the_text_collects_to_the_same_answer(
    pieces: list[str],
) -> None:
    chunks = [ChatChunk(text=p) for p in pieces]
    chunks.append(ChatChunk(finish_reason=FinishReason.STOP))

    assert (await collect(stream_of(chunks))).text == "".join(pieces)


async def test_a_stream_without_a_finish_reason_is_incomplete() -> None:
    with pytest.raises(IncompleteResponseError, match="finish reason"):
        await collect(stream_of([ChatChunk(text="the answer is")]))


async def test_a_tool_call_that_never_got_its_name_is_incomplete() -> None:
    chunks = [
        ChatChunk(tool_calls=(ToolCallDelta(0, "a", None, "{}"),)),
        ChatChunk(finish_reason=FinishReason.TOOL_CALLS),
    ]

    with pytest.raises(IncompleteResponseError, match="tool call 0"):
        await collect(stream_of(chunks))
