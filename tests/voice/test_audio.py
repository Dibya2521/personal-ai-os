import asyncio
import sys
import threading
import time
from collections import deque
from contextlib import aclosing
from dataclasses import dataclass, field
from importlib.abc import MetaPathFinder
from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

from synthia.voice import audio
from synthia.voice.audio import (
    FRAME_SAMPLES,
    AudioUnavailableError,
    Callback,
    Devices,
    Frame,
    Ring,
    devices,
    microphone,
)


def block(value: float) -> NDArray[np.float32]:
    return np.full(FRAME_SAMPLES, value, dtype=np.float32)


@given(capacity=st.integers(1, 4), ops=st.lists(st.booleans(), max_size=60))
def test_ring_behaves_like_a_bounded_queue_that_drops_the_newcomer(
    capacity: int, ops: list[bool]
) -> None:
    ring, model, dropped, kept = Ring(capacity), deque[int](), 0, list[Frame]()
    for n, put in enumerate(ops):
        if put:
            accepted = ring.put(block(n), n)
            assert accepted == (len(model) < capacity)
            if accepted:
                model.append(n)
            else:
                dropped += 1
        else:
            frame = ring.take()
            expected = model.popleft() if model else None
            assert (frame and frame.captured_ns) == expected
            if frame is not None:
                kept.append(frame)

    assert ring.overruns == dropped
    assert all(np.all(frame.samples == frame.captured_ns) for frame in kept)


def test_ring_hands_frames_across_threads_in_order() -> None:
    total, ring, taken = 5_000, Ring(8), list[int]()
    deadline = time.monotonic() + 60

    def produce() -> None:
        for n in range(total):
            ring.put(block(n), n)

    producer = threading.Thread(target=produce)
    producer.start()
    while len(taken) + ring.overruns < total:
        assert time.monotonic() < deadline, f"{len(taken)=} {ring.overruns=}"
        frame = ring.take()
        if frame is None:
            # Spinning on an empty ring holds the GIL and starves the producer.
            time.sleep(0)
            continue
        assert frame.samples[0] == frame.captured_ns
        taken.append(frame.captured_ns)
    producer.join()

    assert taken == sorted(set(taken))
    assert len(taken) + ring.overruns == total


def test_a_ring_holds_at_least_one_frame() -> None:
    with pytest.raises(ValueError, match=r"capacity=0 must be at least 1"):
        Ring(0)


@dataclass
class Timing:
    inputBufferAdcTime: float  # noqa: N815 - PortAudio's field name
    currentTime: float  # noqa: N815 - PortAudio's field name


@dataclass
class Flags:
    input_overflow: bool = False


@dataclass
class FakeStream:
    latency: float = 0.03
    events: list[str] = field(default_factory=list[str])
    callback: Callback | None = None

    def open(self, callback: Callback) -> "FakeStream":
        self.callback = callback
        return self

    def start(self) -> None:
        self.events.append("start")

    def stop(self) -> None:
        self.events.append("stop")

    def close(self) -> None:
        self.events.append("close")

    def deliver(self, value: float, timing: Timing, flags: Flags) -> None:
        assert self.callback is not None
        self.callback(block(value)[:, np.newaxis], FRAME_SAMPLES, timing, flags)  # pyright: ignore[reportArgumentType]


async def test_the_microphone_yields_each_block_with_its_capture_time() -> None:
    stream = FakeStream()
    async with (
        microphone(open_stream=stream.open) as mic,
        aclosing(mic.frames()) as frames,
    ):
        before = time.perf_counter_ns()
        thread = threading.Thread(
            target=lambda: [
                stream.deliver(1.0, Timing(10.0, 10.02), Flags()),
                stream.deliver(2.0, Timing(0.0, 10.04), Flags(input_overflow=True)),
            ]
        )
        thread.start()
        async with asyncio.timeout(2):
            first, second = await anext(frames), await anext(frames)
        after = time.perf_counter_ns()
        thread.join()

        assert first.samples.shape == (FRAME_SAMPLES,)
        assert (first.samples[0], second.samples[0]) == (1.0, 2.0)
        assert before - 20_000_000 <= first.captured_ns <= after - 20_000_000
        assert before <= second.captured_ns <= after
        assert mic.device_overflows == 1
        assert stream.events == ["start"]

    assert stream.events == ["start", "stop", "close"]


async def test_a_waiting_loop_wakes_for_the_next_block() -> None:
    stream = FakeStream()
    async with (
        microphone(open_stream=stream.open) as mic,
        aclosing(mic.frames()) as frames,
    ):
        waiting = asyncio.ensure_future(anext(frames))
        await asyncio.sleep(0)
        assert not waiting.done()

        await asyncio.to_thread(stream.deliver, 5.0, Timing(0.0, 0.0), Flags())
        async with asyncio.timeout(2):
            assert (await waiting).samples[0] == 5.0


async def test_frames_the_loop_cannot_keep_up_with_are_counted() -> None:
    stream = FakeStream()
    async with microphone(open_stream=stream.open, ring_frames=1) as mic:
        for value in (1.0, 2.0, 3.0):
            stream.deliver(value, Timing(0.0, 0.0), Flags())

        assert mic.overruns == 2
        async with aclosing(mic.frames()) as frames:
            assert (await anext(frames)).samples[0] == 1.0


def device(name: str, host: int, inputs: int, outputs: int) -> dict[str, object]:
    return {
        "name": name,
        "hostapi": host,
        "max_input_channels": inputs,
        "max_output_channels": outputs,
    }


def fake_sounddevice(default: tuple[int, int]) -> SimpleNamespace:
    found = [
        device("Mic (MME)", 0, 2, 0),
        device("Speakers (MME)", 0, 0, 2),
        device("Mic (WASAPI)", 1, 2, 0),
        device("Speakers (WASAPI)", 1, 0, 2),
        device("Headset (WASAPI)", 1, 1, 2),
    ]
    names = ("MME", "Windows WASAPI")
    return SimpleNamespace(
        query_devices=lambda: found,
        default=SimpleNamespace(device=default),
        query_hostapis=lambda index: {"name": names[index]},  # pyright: ignore[reportUnknownLambdaType]
    )


@pytest.mark.parametrize(
    ("default", "expected"),
    [
        (
            (2, 3),
            Devices(
                host_api="Windows WASAPI",
                inputs=("Mic (WASAPI)", "Headset (WASAPI)"),
                outputs=("Speakers (WASAPI)", "Headset (WASAPI)"),
                default_input="Mic (WASAPI)",
                default_output="Speakers (WASAPI)",
            ),
        ),
        (
            (-1, 1),
            Devices(
                host_api="MME",
                inputs=("Mic (MME)",),
                outputs=("Speakers (MME)",),
                default_input=None,
                default_output="Speakers (MME)",
            ),
        ),
        ((-1, -1), Devices("MME", ("Mic (MME)",), ("Speakers (MME)",), None, None)),
    ],
    ids=["default host api only", "no microphone", "no devices"],
)
def test_devices_lists_the_default_host_api_once(
    monkeypatch: pytest.MonkeyPatch, default: tuple[int, int], expected: Devices
) -> None:
    monkeypatch.setattr(audio, "_sounddevice", lambda: fake_sounddevice(default))

    assert devices() == expected


class NoPortAudio(MetaPathFinder):
    def find_spec(self, fullname: str, *_: object) -> None:
        if fullname == "sounddevice":
            message = "PortAudio library not found"
            raise OSError(message)


def test_a_missing_portaudio_is_a_known_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
    monkeypatch.setattr(sys, "meta_path", [NoPortAudio(), *sys.meta_path])

    with pytest.raises(AudioUnavailableError, match=r"PortAudio library not found"):
        devices()
