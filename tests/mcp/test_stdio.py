import asyncio
import json

import pytest

from synthia.kernel.jsonrpc import ConnectionClosedError, Peer
from synthia.mcp.stdio import MAX_LINE_BYTES, Lines, line_sender


def framed() -> tuple[Peer, Lines, list[bytes]]:
    written: list[bytes] = []
    peer = Peer(line_sender(written.append))
    return peer, Lines(peer), written


async def test_each_message_goes_out_as_one_line() -> None:
    peer, _, written = framed()
    peer.notify("notifications/initialized")

    assert written == [b'{"jsonrpc": "2.0", "method": "notifications/initialized"}\n']


async def test_lines_split_across_reads_or_sharing_one_are_each_a_message() -> None:
    peer, lines, _ = framed()
    first = asyncio.create_task(peer.request("a", {}))
    second = asyncio.create_task(peer.request("b", {}))
    await asyncio.sleep(0)

    lines.feed(
        b'{"jsonrpc": "2.0", "id": 2, "result": "B"}\n\n{"jsonrpc": "2.0", "id": 1, "re'
    )
    lines.feed(b'sult": "A"}\r\n')

    assert (await first, await second) == ("A", "B")


async def test_a_line_past_the_limit_closes_the_peer() -> None:
    peer, lines, _ = framed()
    task = asyncio.create_task(peer.request("a", {}))
    await asyncio.sleep(0)
    lines.feed(b"x" * (MAX_LINE_BYTES + 1))

    with pytest.raises(ConnectionClosedError, match="longer than"):
        await task


async def test_a_line_that_is_not_json_is_skipped() -> None:
    peer, lines, _ = framed()
    task = asyncio.create_task(peer.request("a", {}))
    await asyncio.sleep(0)
    lines.feed(b"not json\n")
    lines.feed(json.dumps({"jsonrpc": "2.0", "id": 1, "result": "ok"}).encode() + b"\n")

    assert await task == "ok"
