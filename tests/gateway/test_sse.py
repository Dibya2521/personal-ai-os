import time
import tracemalloc
from collections.abc import AsyncIterator

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from synthia.gateway.errors import MalformedStreamError
from synthia.gateway.sse import ServerSentEvent, SSEParser, aiter_events

CAP = 100


@pytest.mark.parametrize(
    "stream",
    [
        pytest.param([b"data: " + b"x" * CAP], id="one line never ended"),
        pytest.param([b"data: x\n"] * CAP, id="many data lines, no blank line"),
        pytest.param([b": keep-alive\n"] * CAP, id="comments, no blank line"),
        pytest.param([b"data: " + b"x" * 60, b"x" * 60], id="a line split in two"),
    ],
)
def test_an_event_past_the_cap_is_refused(stream: list[bytes]) -> None:
    parser = SSEParser(max_event_chars=CAP)

    with pytest.raises(MalformedStreamError, match=f"passed {CAP} characters"):
        feed_all(parser, stream)


def test_the_next_events_unfinished_line_does_not_count_against_this_one() -> None:
    parser = SSEParser(max_event_chars=CAP)

    events = parser.feed(b"data: " + b"x" * 90 + b"\n\ndata: " + b"y" * 20)

    assert [len(e.data) for e in events] == [90]


def test_a_line_in_tiny_chunks_costs_a_few_bytes_per_character() -> None:
    chunks = 200_000
    parser = SSEParser(max_event_chars=4 * chunks)
    parser.feed(b"data: ")
    tracemalloc.start()
    try:
        for _ in range(chunks):
            parser.feed(b"ab")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 4 * 2 * chunks


def feed_all(parser: SSEParser, stream: list[bytes]) -> None:
    for chunk in stream:
        parser.feed(chunk)


def test_the_cap_is_per_event_and_an_event_at_it_passes() -> None:
    parser = SSEParser(max_event_chars=CAP)
    event = b"data: " + b"x" * (CAP - 6) + b"\n\n"

    events = [e for _ in range(5) for e in parser.feed(event)]

    assert [len(e.data) for e in events] == [CAP - 6] * 5


OPENROUTER_LIKE = (
    b": OPENROUTER PROCESSING\n\n"
    b'data: {"choices":[{"delta":{"content":"Na"}}]}\n\n'
    b'data: {"choices":[{"delta":{"content":"mast\xc3\xa9 \xf0\x9f\x99\x8f"}}]}\n\n'
    b"data: [DONE]\n\n"
)


def parse_all(stream: bytes) -> list[ServerSentEvent]:
    parser = SSEParser()
    events = parser.feed(stream)
    parser.close()
    return events


def parse_split(stream: bytes, cuts: list[int]) -> list[ServerSentEvent]:
    parser = SSEParser()
    events: list[ServerSentEvent] = []
    start = 0
    for cut in sorted({c % (len(stream) + 1) for c in cuts}):
        events += parser.feed(stream[start:cut])
        start = cut
    events += parser.feed(stream[start:])
    parser.close()
    return events


def test_a_chat_stream_parses_with_its_keep_alive_ignored() -> None:
    events = parse_all(OPENROUTER_LIKE)

    assert [e.data for e in events] == [
        '{"choices":[{"delta":{"content":"Na"}}]}',
        '{"choices":[{"delta":{"content":"mastÃ© 🙏"}}]}'.replace("Ã©", "é"),
        "[DONE]",
    ]
    assert all(e.event == "message" for e in events)


@pytest.mark.parametrize("ending", [b"\n", b"\r\n", b"\r"])
def test_every_line_ending_is_accepted(ending: bytes) -> None:
    stream = ending.join([b"event: delta", b"data: one", b"data: two", b"", b""])

    assert parse_all(stream) == [ServerSentEvent(data="one\ntwo", event="delta")]


def test_fields_follow_the_specification() -> None:
    stream = (
        b"\xef\xbb\xbfid: 7\n"  # a byte order mark at the start is dropped
        b"retry: 3000\n"
        b"retry: soon\n"  # not digits: ignored
        b"data\n"  # a field with no colon has an empty value
        b"data:  two spaces\n"  # only one leading space is removed
        b"unknown: ignored\n"
        b"\n"
        b"id: bad\x00id\n"  # an id containing NULL is ignored
        b"data: next\n\n"
    )

    first, second = parse_all(stream)

    assert first == ServerSentEvent(data="\n two spaces", id="7", retry=3000)
    assert second.id == "7"


def test_a_blank_line_without_data_dispatches_nothing_and_resets_the_type() -> None:
    assert parse_all(b"event: ping\n\ndata: x\n\n") == [ServerSentEvent(data="x")]


def test_an_event_cut_off_at_the_end_is_not_dispatched() -> None:
    assert parse_all(b"data: complete\n\ndata: cut off") == [
        ServerSentEvent(data="complete")
    ]
    assert parse_all(b"data: no blank line\n") == []


def test_a_crlf_split_between_chunks_is_one_line_ending() -> None:
    parser = SSEParser()

    assert parser.feed(b"data: a\r") == []
    assert parser.feed(b"\n\r") == [ServerSentEvent(data="a")]
    assert parser.feed(b"\n") == []


def test_a_cr_dispatches_without_waiting_for_the_next_chunk() -> None:
    parser = SSEParser()

    assert parser.feed(b"data: now\r\r") == [ServerSentEvent(data="now")]


def test_a_character_split_across_chunks_survives() -> None:
    parser = SSEParser()
    emoji = "🙏".encode()

    events = parser.feed(b"data: " + emoji[:2]) + parser.feed(emoji[2:] + b"\n\n")

    assert events == [ServerSentEvent(data="🙏")]


def test_invalid_utf8_is_replaced_rather_than_fatal() -> None:
    assert parse_all(b"data: \xff\xfe\n\n") == [ServerSentEvent(data="��")]


@settings(max_examples=300)
@given(st.lists(st.integers(min_value=0, max_value=400), max_size=12))
def test_any_way_of_cutting_the_stream_yields_the_same_events(cuts: list[int]) -> None:
    stream = OPENROUTER_LIKE + b"event: x\r\ndata: y\r\rdata: z\r\n\r\n"

    assert parse_split(stream, cuts) == parse_all(stream)


@given(st.binary(max_size=200), st.lists(st.integers(0, 200), max_size=6))
def test_arbitrary_bytes_never_crash_and_splitting_never_changes_the_result(
    stream: bytes, cuts: list[int]
) -> None:
    assert parse_split(stream, cuts) == parse_all(stream)


def seconds_to_parse_one_long_line(size: int, times: int) -> float:
    payload = b"data: " + b"x" * size + b"\n\n"
    started = time.perf_counter()
    for _ in range(times):
        parser = SSEParser()
        for i in range(0, len(payload), 4096):
            parser.feed(payload[i : i + 4096])
    return time.perf_counter() - started


def test_a_long_line_in_many_chunks_costs_linear_time() -> None:
    # Both sides parse the same 8 MB, so the ratio is the cost per byte: about
    # 1 when linear, about 16 when quadratic (a line 16 times longer). Each
    # side runs tens of milliseconds and the runs alternate, so load on a
    # busy machine slows both alike.
    short: list[float] = []
    long: list[float] = []
    for _ in range(5):
        short.append(seconds_to_parse_one_long_line(125_000, times=64))
        long.append(seconds_to_parse_one_long_line(2_000_000, times=4))

    assert min(long) / min(short) < 4


async def test_aiter_events_yields_as_chunks_arrive() -> None:
    async def chunks() -> AsyncIterator[bytes]:
        yield b"data: a\n\nda"
        yield b"ta: b\n\n"

    assert [e.data async for e in aiter_events(chunks())] == ["a", "b"]
