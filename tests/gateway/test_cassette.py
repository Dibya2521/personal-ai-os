import gzip
import json
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from synthia.gateway.cassette import CassetteError, CassetteTransport, Mode
from synthia.gateway.openai_compat import Endpoint, OpenAICompatibleModel
from synthia.gateway.protocol import collect
from synthia.gateway.types import ChatRequest, Message, ModelInfo

KEY = "sk-or-v1-cassette-test-key-000"  # pragma: allowlist secret
URL = "https://example.test/api/v1/chat/completions"
STREAM = (
    b'data: {"model":"vendor/actual","choices":[{"delta":{"content":"Hi"}}]}\n\n'
    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
    b"data: [DONE]\n\n"
)


def live_server(body: bytes = STREAM, **headers: str) -> httpx.MockTransport:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers=headers)

    return httpx.MockTransport(handler)


async def record(path: Path, inner: httpx.AsyncBaseTransport, *bodies: object) -> None:
    transport = CassetteTransport.recording(path, inner, secrets=[KEY])
    async with httpx.AsyncClient(transport=transport) as client:
        for body in bodies:
            await client.post(
                URL, json=body, headers={"Authorization": f"Bearer {KEY}"}
            )


def stored(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


async def test_a_recording_replays_with_no_network(tmp_path: Path) -> None:
    path = tmp_path / "chat.json"
    await record(path, live_server(), {"model": "m", "messages": []})

    replay = CassetteTransport.replaying(path)
    async with httpx.AsyncClient(transport=replay) as client:
        response = await client.post(URL, json={"messages": [], "model": "m"})

    assert response.status_code == 200
    assert response.content == STREAM
    assert replay.unplayed == 0


async def test_the_adapter_streams_the_same_answer_from_a_cassette(
    tmp_path: Path,
) -> None:
    path = tmp_path / "adapter.json"
    info = ModelInfo("m", 1000, vision=False, tools=False)
    endpoint = Endpoint("https://example.test/api/v1", "m", SecretStr(KEY))
    request = ChatRequest((Message.user("hello"),))

    recorder = CassetteTransport.recording(path, live_server(), secrets=[KEY])
    async with httpx.AsyncClient(transport=recorder) as client:
        live = await collect(
            OpenAICompatibleModel(client=client, endpoint=endpoint, info=info).stream(
                request
            )
        )
    async with httpx.AsyncClient(transport=CassetteTransport.replaying(path)) as client:
        replayed = await collect(
            OpenAICompatibleModel(client=client, endpoint=endpoint, info=info).stream(
                request
            )
        )

    assert replayed == live
    assert replayed.text == "Hi"


async def test_identical_requests_replay_in_the_order_they_were_recorded(
    tmp_path: Path,
) -> None:
    answers = iter([b"first", b"second"])

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=next(answers))

    path = tmp_path / "twice.json"
    await record(path, httpx.MockTransport(handler), {"q": 1}, {"q": 1})

    async with httpx.AsyncClient(transport=CassetteTransport.replaying(path)) as client:
        first = await client.post(URL, json={"q": 1})
        second = await client.post(URL, json={"q": 1})
        with pytest.raises(CassetteError, match=r"no recording in twice\.json"):
            await client.post(URL, json={"q": 1})

    assert (first.content, second.content) == (b"first", b"second")


async def test_an_unrecorded_request_fails_loudly_and_names_what_is_missing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "one.json"
    await record(path, live_server(), {"q": "recorded"})

    async with httpx.AsyncClient(transport=CassetteTransport.replaying(path)) as client:
        with pytest.raises(CassetteError) as caught:
            await client.post(URL, json={"q": "never recorded"})

    assert f"POST {URL}" in str(caught.value)
    assert "never recorded" not in str(caught.value)


async def test_the_key_and_request_headers_are_never_written(tmp_path: Path) -> None:
    echoing = live_server(
        b'{"error":{"message":"bad key ' + KEY.encode() + b'"}}',
        **{
            "content-type": "application/json",
            "set-cookie": "session=abc",
            "cf-ray": "8f00",
            "x-ratelimit-remaining": "49",
        },
    )
    path = tmp_path / "secret.json"
    await record(
        path, echoing, {"model": "m", "messages": [{"content": f"my key is {KEY}"}]}
    )

    text = path.read_text(encoding="utf-8")
    (interaction,) = stored(path)["interactions"]  # type: ignore[misc]

    assert KEY not in text
    assert "authorization" not in text.lower()
    assert interaction["response"]["headers"] == {  # type: ignore[index]
        "content-type": "application/json",
        "x-ratelimit-remaining": "49",
    }


async def test_a_compressed_response_is_stored_and_replayed_plain(
    tmp_path: Path,
) -> None:
    body = b'{"ok": true}'
    path = tmp_path / "gzip.json"
    await record(
        path, live_server(gzip.compress(body), **{"content-encoding": "gzip"}), {}
    )

    async with httpx.AsyncClient(transport=CassetteTransport.replaying(path)) as client:
        response = await client.post(URL, json={})

    assert response.content == body
    assert "content-encoding" not in response.headers


async def test_a_binary_body_round_trips(tmp_path: Path) -> None:
    body = bytes(range(256))
    path = tmp_path / "binary.json"
    await record(path, live_server(body), {})

    async with httpx.AsyncClient(transport=CassetteTransport.replaying(path)) as client:
        assert (await client.post(URL, json={})).content == body


async def test_the_file_is_stable_and_readable(tmp_path: Path) -> None:
    path = tmp_path / "stable.json"
    await record(
        path, live_server(), {"model": "qwen", "messages": [{"content": "hello"}]}
    )
    first = path.read_bytes()
    await record(
        path, live_server(), {"messages": [{"content": "hello"}], "model": "qwen"}
    )

    assert path.read_bytes() == first
    assert b"\r\n" not in first
    (interaction,) = stored(path)["interactions"]  # type: ignore[misc]
    assert interaction["request"]["summary"] == "qwen: hello"  # type: ignore[index]


@pytest.mark.parametrize(
    ("content", "problem"),
    [
        (None, "cannot read"),
        ("{not json", "cannot read"),
        ('{"version": 99}', "version"),
    ],
)
def test_a_missing_corrupt_or_foreign_cassette_is_rejected(
    tmp_path: Path, content: str | None, problem: str
) -> None:
    path = tmp_path / "bad.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")

    with pytest.raises(CassetteError, match=problem):
        CassetteTransport.replaying(path)


async def test_a_non_json_request_body_still_matches(tmp_path: Path) -> None:
    path = tmp_path / "form.json"
    transport = CassetteTransport.recording(path, live_server(b"ok"))
    async with httpx.AsyncClient(transport=transport) as client:
        await client.post(URL, content=b"raw \xff bytes")

    replay = CassetteTransport.replaying(path)
    async with httpx.AsyncClient(transport=replay) as client:
        assert (await client.post(URL, content=b"raw \xff bytes")).content == b"ok"
    (interaction,) = stored(path)["interactions"]  # type: ignore[misc]
    assert interaction["request"]["summary"] == "11 bytes"  # type: ignore[index]


def test_modes_are_named() -> None:
    assert Mode("record") is Mode.RECORD
