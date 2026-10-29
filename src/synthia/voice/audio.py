"""Sound from the microphone, in fixed frames, to the event loop.

PortAudio calls back on its own thread with each block of samples. That thread
must never wait, so it copies the block into a ring allocated up front and wakes
the loop; the loop takes frames out of the ring at its own pace.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

import numpy as np

from synthia.kernel.errors import SynthiaError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Callable

    from numpy.typing import NDArray
    from sounddevice import CallbackFlags, CallbackTime

SAMPLE_RATE: Final = 16_000
FRAME_SAMPLES: Final = 512
"""32 ms at 16 kHz: the frame Silero VAD reads."""
RING_FRAMES: Final = 64
"""About 2 s of sound may wait for the loop before frames are dropped."""
NS_PER_S: Final = 1_000_000_000


class AudioUnavailableError(SynthiaError):
    """PortAudio, the library under every sound device, cannot be loaded."""


@dataclass(frozen=True, slots=True)
class Frame:
    """32 ms of mono sound at 16 kHz, and when its first sample was captured.

    ``captured_ns`` is on the :func:`time.perf_counter_ns` clock: latency hops
    are tens of milliseconds, and ``monotonic`` ticks every 15.6 ms on Windows
    before Python 3.13.
    """

    samples: NDArray[np.float32]
    captured_ns: int


class Ring:
    """Frames from one producer thread to one consumer, with no lock.

    Each counter has one writer: the producer moves ``_written`` only after the
    slot is filled, and the consumer moves ``_read`` only after the slot is
    copied out, so neither side sees a half-written frame. A full ring drops
    the incoming frame and counts it in ``overruns``: dropping the oldest
    instead would make the producer move the consumer's counter.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            message = f"{capacity=} must be at least 1"
            raise ValueError(message)
        self._samples = np.zeros((capacity, FRAME_SAMPLES), dtype=np.float32)
        self._captured = np.zeros(capacity, dtype=np.int64)
        self._capacity = capacity
        self._written = 0
        self._read = 0
        self.overruns = 0

    def put(self, samples: NDArray[np.float32], captured_ns: int) -> bool:
        """Copy one frame in, from the producer only; False when the ring was full."""
        if self._written - self._read == self._capacity:
            self.overruns += 1
            return False
        slot = self._written % self._capacity
        self._samples[slot] = samples
        self._captured[slot] = captured_ns
        self._written += 1
        return True

    def take(self) -> Frame | None:
        """Copy the oldest frame out, from the consumer only; None when empty."""
        if self._read == self._written:
            return None
        slot = self._read % self._capacity
        frame = Frame(self._samples[slot].copy(), int(self._captured[slot]))
        self._read += 1
        return frame


class AudioTransport(Protocol):
    """Where SYNTHIA's sound comes from: this laptop's devices or a test's script."""

    def frames(self) -> AsyncIterator[Frame]:
        """Yield frames in capture order, for as long as the transport is open."""
        ...


type Callback = Callable[[NDArray[np.float32], int, CallbackTime, CallbackFlags], None]


class InputStream(Protocol):
    """The part of a PortAudio input stream the microphone drives."""

    @property
    def latency(self) -> float:
        """Seconds between sound reaching the device and the callback."""
        ...

    def start(self) -> None:
        """Start calling back."""
        ...

    def stop(self) -> None:
        """Stop calling back, after the blocks already captured."""
        ...

    def close(self) -> None:
        """Release the device."""
        ...


type OpenStream = Callable[[Callback], InputStream]


class Microphone:
    """The frames of an open input stream, read by the event loop."""

    def __init__(self, loop: asyncio.AbstractEventLoop, ring: Ring) -> None:
        self._loop = loop
        self._ring = ring
        self._ready = asyncio.Event()
        self.device_overflows = 0
        """Blocks PortAudio itself lost before calling back."""

    @property
    def overruns(self) -> int:
        """Frames dropped because the loop fell more than the ring behind."""
        return self._ring.overruns

    def on_audio(
        self,
        samples: NDArray[np.float32],
        _count: int,
        timing: CallbackTime,
        status: CallbackFlags,
    ) -> None:
        """Take one block on PortAudio's thread: copy it, then wake the loop."""
        if status.input_overflow:
            self.device_overflows += 1
        self._ring.put(samples[:, 0], time.perf_counter_ns() - _age_ns(timing))
        self._loop.call_soon_threadsafe(self._ready.set)

    async def frames(self) -> AsyncGenerator[Frame]:
        """Yield frames in capture order until the stream closes."""
        while True:
            frame = self._ring.take()
            if frame is not None:
                yield frame
                continue
            # A put after the take above schedules ``set`` behind this clear, so
            # no wake-up is lost.
            self._ready.clear()
            await self._ready.wait()


def _age_ns(timing: CallbackTime) -> int:
    """How long ago the block's first sample reached the device."""
    if timing.inputBufferAdcTime <= 0:
        return 0
    return max(0, round((timing.currentTime - timing.inputBufferAdcTime) * NS_PER_S))


def open_input(callback: Callback) -> InputStream:
    """Open the default microphone: mono, 16 kHz, one frame per callback.

    Raises:
        AudioUnavailableError: If PortAudio cannot be loaded.
    """
    sd = _sounddevice()
    return sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SAMPLES,
        channels=1,
        dtype="float32",
        latency="low",
        callback=callback,
    )


@asynccontextmanager
async def microphone(
    *, open_stream: OpenStream = open_input, ring_frames: int = RING_FRAMES
) -> AsyncGenerator[Microphone]:
    """Listen while the block runs; the device is released when it ends."""
    mic = Microphone(asyncio.get_running_loop(), Ring(ring_frames))
    stream = open_stream(mic.on_audio)
    stream.start()
    try:
        yield mic
    finally:
        stream.stop()
        stream.close()


@dataclass(frozen=True, slots=True)
class Devices:
    """The sound devices of the host API the default microphone belongs to."""

    host_api: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    default_input: str | None
    default_output: str | None


def devices() -> Devices:
    """List the devices PortAudio would choose from.

    Windows lists each device once per host API (MME, DirectSound, WASAPI), so
    only the default input's host API is listed.

    Raises:
        AudioUnavailableError: If PortAudio cannot be loaded.
    """
    sd = _sounddevice()
    found = sd.query_devices()
    default_in, default_out = sd.default.device
    anchor = default_in if default_in >= 0 else default_out
    host = found[anchor]["hostapi"] if anchor >= 0 else 0
    listed = [d for d in found if d["hostapi"] == host]
    return Devices(
        host_api=sd.query_hostapis(host)["name"],
        inputs=tuple(d["name"] for d in listed if d["max_input_channels"] > 0),
        outputs=tuple(d["name"] for d in listed if d["max_output_channels"] > 0),
        default_input=found[default_in]["name"] if default_in >= 0 else None,
        default_output=found[default_out]["name"] if default_out >= 0 else None,
    )


def _sounddevice():  # noqa: ANN202 - the module, typed by its stub
    # Importing sounddevice loads PortAudio, which raises OSError when the
    # shared library is missing (a Linux machine without libportaudio2).
    try:
        import sounddevice  # noqa: PLC0415
    except OSError as error:
        message = f"PortAudio could not be loaded: {error}"
        raise AudioUnavailableError(message) from error
    return sounddevice
