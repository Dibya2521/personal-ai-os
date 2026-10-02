import asyncio
import json

import pytest

from synthia.mcp.jsonrpc import (
    MAX_LINE_BYTES,
    METHOD_NOT_FOUND,
    Connection,
    ConnectionClosedError,
    RpcError,
)


class Peer:
    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []
        self.connection = Connection(self.receive)

    def receive(self, data: bytes) -> None:
        assert data.endswith(b"\n")
        assert data.count(b"\n") == 1
        self.sent.append(json.loads(data))

    def answer(self, message: dict[str, object]) -> None:
        self.connection.feed(json.dumps({"jsonrpc": "2.0", **message}).encode() + b"\n")


async def asked(peer: Peer, method: str) -> asyncio.Task[object]:
    task = asyncio.create_task(peer.connection.request(method, {"x": 1}))
    await asyncio.sleep(0)
    return task


async def test_a_request_goes_out_as_one_line_and_its_answer_comes_back() -> None:
    peer = Peer()
    task = await asked(peer, "tools/list")
    peer.answer({"id": 1, "result": {"tools": []}})

    assert await task == {"tools": []}
    assert peer.sent == [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {"x": 1}}
    ]


async def test_answers_split_across_chunks_or_sharing_one_are_matched_by_id() -> None:
    peer = Peer()
    first, second = await asked(peer, "a"), await asked(peer, "b")
    wire = (
        b'{"jsonrpc": "2.0", "id": 2, "result": "B"}\n{"jsonrpc": "2.0", "id": 1, "re'
    )
    peer.connection.feed(wire)
    peer.connection.feed(b'sult": "A"}\r\n')

    assert (await first, await second) == ("A", "B")


async def test_an_error_answer_is_raised_with_its_code() -> None:
    peer = Peer()
    task = await asked(peer, "tools/call")
    peer.answer({"id": 1, "error": {"code": -32602, "message": "unknown tool"}})

    with pytest.raises(RpcError, match="unknown tool") as caught:
        await task
    assert caught.value.code == -32602


async def test_lines_that_are_not_answers_are_skipped() -> None:
    peer = Peer()
    task = await asked(peer, "a")
    peer.connection.feed(b"not json\n[1, 2]\n\n" + b'{"jsonrpc": "2.0", "id": 99}\n')
    peer.answer({"id": "1", "result": "a string id is not ours"})
    peer.answer({"id": 1, "result": "ok"})

    assert await task == "ok"


async def test_the_peers_ping_is_answered_and_other_requests_refused() -> None:
    peer = Peer()
    peer.answer({"id": "p", "method": "ping"})
    peer.answer({"id": 7, "method": "roots/list"})
    peer.answer({"method": "notifications/progress"})

    assert peer.sent == [
        {"jsonrpc": "2.0", "id": "p", "result": {}},
        {
            "jsonrpc": "2.0",
            "id": 7,
            "error": {"code": METHOD_NOT_FOUND, "message": "method not supported"},
        },
    ]


async def test_a_cancelled_request_tells_the_peer() -> None:
    peer = Peer()
    task = await asked(peer, "tools/call")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert peer.sent[-1] == {
        "jsonrpc": "2.0",
        "method": "notifications/cancelled",
        "params": {"requestId": 1, "reason": "cancelled"},
    }


async def test_closing_fails_waiting_and_later_requests_and_stops_sending() -> None:
    peer = Peer()
    task = await asked(peer, "a")
    peer.connection.close("the server exited with code 3")

    with pytest.raises(ConnectionClosedError, match="exited with code 3"):
        await task
    with pytest.raises(ConnectionClosedError, match="exited with code 3"):
        await peer.connection.request("b", {})
    peer.connection.notify("notifications/initialized")
    assert len(peer.sent) == 1


async def test_an_answer_that_arrived_before_closing_is_kept() -> None:
    peer = Peer()
    task = await asked(peer, "a")
    peer.answer({"id": 1, "result": "in time"})
    peer.connection.close("closed right after")

    assert await task == "in time"


async def test_a_line_past_the_limit_closes_the_connection() -> None:
    peer = Peer()
    task = await asked(peer, "a")
    peer.connection.feed(b"x" * (MAX_LINE_BYTES + 1))

    with pytest.raises(ConnectionClosedError, match="longer than"):
        await task


async def test_text_goes_out_as_ascii_so_nothing_can_break_the_line() -> None:
    peer = Peer()
    peer.connection.notify("say", {"text": "line\nbreak \N{SNOWMAN} \ud83d"})

    assert peer.sent == [
        {
            "jsonrpc": "2.0",
            "method": "say",
            "params": {"text": "line\nbreak \N{SNOWMAN} \ud83d"},
        }
    ]
