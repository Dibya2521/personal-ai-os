import asyncio
import json
import logging

import pytest

from synthia.kernel.jsonrpc import (
    INTERNAL_ERROR,
    METHOD_NOT_FOUND,
    ConnectionClosedError,
    Method,
    Notice,
    Params,
    Peer,
    RpcError,
)


class Wire:
    """One peer, with what it sends kept as parsed messages."""

    def __init__(
        self,
        methods: dict[str, Method] | None = None,
        notices: dict[str, Notice] | None = None,
    ) -> None:
        self.sent: list[dict[str, object]] = []
        self.raw: list[bytes] = []
        self.peer = Peer(self.keep, methods=methods, notices=notices)

    def keep(self, message: bytes) -> None:
        self.raw.append(message)
        self.sent.append(json.loads(message))

    def answer(self, message: dict[str, object]) -> None:
        self.peer.receive(json.dumps({"jsonrpc": "2.0", **message}))


async def asked(wire: Wire, method: str) -> asyncio.Task[object]:
    task = asyncio.create_task(wire.peer.request(method, {"x": 1}))
    await asyncio.sleep(0)
    return task


async def settled() -> None:
    for _ in range(3):
        await asyncio.sleep(0)


async def test_a_request_goes_out_as_one_message_and_its_answer_comes_back() -> None:
    wire = Wire()
    task = await asked(wire, "tools/list")
    wire.answer({"id": 1, "result": {"tools": []}})

    assert await task == {"tools": []}
    assert wire.sent == [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {"x": 1}}
    ]


async def test_answers_in_any_order_are_matched_by_id() -> None:
    wire = Wire()
    first, second = await asked(wire, "a"), await asked(wire, "b")
    wire.peer.receive(b'{"jsonrpc": "2.0", "id": 2, "result": "B"}')
    wire.answer({"id": 1, "result": "A"})

    assert (await first, await second) == ("A", "B")


@pytest.mark.parametrize(
    ("error", "code", "message"),
    [
        ({"code": -32602, "message": "unknown tool"}, -32602, "unknown tool"),
        ({"code": "bad"}, 0, "no message"),
    ],
    ids=["code-and-message", "neither-usable"],
)
async def test_an_error_answer_is_raised_with_its_code(
    error: dict[str, object], code: int, message: str
) -> None:
    wire = Wire()
    task = await asked(wire, "tools/call")
    wire.answer({"id": 1, "error": error})

    with pytest.raises(RpcError, match=f"^{message}$") as caught:
        await task
    assert caught.value.code == code


async def test_messages_that_are_not_answers_to_us_are_skipped() -> None:
    wire = Wire()
    task = await asked(wire, "a")
    for junk in (b"not json", b"[1, 2]", b'{"jsonrpc": "2.0", "id": 99}'):
        wire.peer.receive(junk)
    wire.answer({"id": "1", "result": "a string id is not ours"})
    wire.answer({"id": 1, "result": "ok"})
    wire.answer({"id": 1, "result": "a second answer"})

    assert await task == "ok"


async def test_ping_is_answered_and_unknown_or_unnamed_methods_refused() -> None:
    wire = Wire()
    wire.answer({"id": "p", "method": "ping"})
    wire.answer({"id": 7, "method": "roots/list"})
    wire.answer({"id": 8, "method": 42})
    wire.answer({"id": [1], "method": "ping"})

    refused = {"code": METHOD_NOT_FOUND, "message": "method not supported"}
    assert wire.sent == [
        {"jsonrpc": "2.0", "id": "p", "result": {}},
        {"jsonrpc": "2.0", "id": 7, "error": refused},
        {"jsonrpc": "2.0", "id": 8, "error": refused},
    ]


async def test_a_served_method_gets_its_params_and_its_result_goes_back() -> None:
    async def add(params: Params) -> object:
        return cast_int(params["a"]) + cast_int(params["b"])

    async def nothing(params: Params) -> object:
        return params

    wire = Wire(methods={"add": add, "echo": nothing})
    wire.answer({"id": 1, "method": "add", "params": {"a": 2, "b": 3}})
    wire.answer({"id": "two", "method": "echo", "params": [1, 2]})
    await settled()

    assert wire.sent == [
        {"jsonrpc": "2.0", "id": 1, "result": 5},
        {"jsonrpc": "2.0", "id": "two", "result": {}},
    ]


def cast_int(value: object) -> int:
    assert isinstance(value, int)
    return value


async def test_a_method_that_refuses_or_breaks_answers_with_an_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def refuse(_: Params) -> object:
        raise RpcError(-32602, "no such session")

    async def broken(_: Params) -> object:
        message = "a bug"
        raise LookupError(message)

    wire = Wire(methods={"refuse": refuse, "broken": broken})
    with caplog.at_level(logging.ERROR):
        wire.answer({"id": 1, "method": "refuse"})
        wire.answer({"id": 2, "method": "broken"})
        await settled()

    assert wire.sent == [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32602, "message": "no such session"},
        },
        {
            "jsonrpc": "2.0",
            "id": 2,
            "error": {"code": INTERNAL_ERROR, "message": "internal error"},
        },
    ]
    assert "method failed while serving request 2" in caplog.text
    assert "a bug" not in json.dumps(wire.sent)


async def test_a_request_the_other_side_cancels_stops_and_gets_no_answer() -> None:
    started, stopped = asyncio.Event(), asyncio.Event()

    async def slow(_: Params) -> object:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
        return None  # pragma: no cover

    wire = Wire(methods={"slow": slow})
    wire.answer({"id": 4, "method": "slow"})
    await asyncio.wait_for(started.wait(), 1)
    for request_id in (99, [4]):
        wire.answer(
            {"method": "notifications/cancelled", "params": {"requestId": request_id}}
        )
    await settled()
    assert not stopped.is_set()

    wire.answer({"method": "notifications/cancelled", "params": {"requestId": 4}})
    await asyncio.wait_for(stopped.wait(), 1)
    await settled()

    assert wire.sent == []


async def test_notifications_reach_their_handler_and_others_are_ignored() -> None:
    heard: list[Params] = []
    wire = Wire(notices={"progress": heard.append})
    wire.answer({"method": "progress", "params": {"done": 1}})
    wire.answer({"method": "progress"})
    wire.answer({"method": "unknown"})
    wire.answer({"method": 3})

    assert heard == [{"done": 1}, {}]
    assert wire.sent == []


async def test_a_cancelled_request_tells_the_other_side() -> None:
    wire = Wire()
    task = await asked(wire, "tools/call")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert wire.sent[-1] == {
        "jsonrpc": "2.0",
        "method": "notifications/cancelled",
        "params": {"requestId": 1, "reason": "cancelled"},
    }


async def test_closing_fails_requests_stops_methods_and_stops_sending() -> None:
    stopped = asyncio.Event()

    async def slow(_: Params) -> object:
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
        return None  # pragma: no cover

    wire = Wire(methods={"slow": slow})
    task = await asked(wire, "a")
    wire.answer({"id": 5, "method": "slow"})
    await settled()
    wire.peer.close("the server exited with code 3")
    wire.peer.close("a second reason is ignored")

    with pytest.raises(ConnectionClosedError, match="exited with code 3"):
        await task
    with pytest.raises(ConnectionClosedError, match="exited with code 3"):
        await wire.peer.request("b", {})
    await asyncio.wait_for(stopped.wait(), 1)
    wire.peer.notify("notifications/initialized")
    assert len(wire.sent) == 1


async def test_an_answer_that_arrived_before_closing_is_kept() -> None:
    wire = Wire()
    task = await asked(wire, "a")
    wire.answer({"id": 1, "result": "in time"})
    wire.peer.close("closed right after")

    assert await task == "in time"


async def test_text_goes_out_as_ascii_so_no_newline_or_half_emoji_is_raw() -> None:
    wire = Wire()
    wire.peer.notify("say", {"text": "line\nbreak \N{SNOWMAN} \ud83d"})

    assert wire.raw[0].isascii()
    assert b"\n" not in wire.raw[0]
    assert wire.sent == [
        {
            "jsonrpc": "2.0",
            "method": "say",
            "params": {"text": "line\nbreak \N{SNOWMAN} \ud83d"},
        }
    ]


async def test_two_peers_can_each_ask_the_other() -> None:
    async def approve(params: Params) -> object:
        return params["tool"] == "calculate"

    async def turn(params: Params) -> object:
        allowed = await server.request("approve", {"tool": params["tool"]})
        return f"{params['tool']}: {'ran' if allowed else 'refused'}"

    def to_client(message: bytes) -> None:
        client.receive(message)

    def to_server(message: bytes) -> None:
        server.receive(message)

    server = Peer(to_client, methods={"turn": turn})
    client = Peer(to_server, methods={"approve": approve})

    answers = [
        await asyncio.wait_for(client.request("turn", {"tool": tool}), 1)
        for tool in ("calculate", "run_python")
    ]

    assert answers == ["calculate: ran", "run_python: refused"]
