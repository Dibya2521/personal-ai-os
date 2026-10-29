# The part of sounddevice SYNTHIA uses: the package ships without type hints.

from collections.abc import Callable
from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

class PortAudioError(Exception): ...

class DeviceInfo(TypedDict):
    name: str
    index: int
    hostapi: int
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float

class HostApiInfo(TypedDict):
    name: str
    devices: list[int]

class CallbackFlags:
    @property
    def input_overflow(self) -> bool: ...

class CallbackTime:
    @property
    def inputBufferAdcTime(self) -> float: ...  # noqa: N802 - PortAudio's field name
    @property
    def currentTime(self) -> float: ...  # noqa: N802 - PortAudio's field name

class _Default:
    device: tuple[int, int]

default: _Default

def query_devices() -> list[DeviceInfo]: ...
def query_hostapis(index: int) -> HostApiInfo: ...

class InputStream:
    def __init__(
        self,
        *,
        samplerate: float,
        blocksize: int,
        channels: int,
        dtype: Literal["float32"],
        latency: Literal["low", "high"],
        callback: Callable[
            [NDArray[np.float32], int, CallbackTime, CallbackFlags], None
        ],
    ) -> None: ...
    @property
    def latency(self) -> float: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...
