"""How likely each 32 ms frame is speech, from Silero VAD.

Silero VAD is a small recurrent network (2.3 MB, MIT). It reads one frame of
512 samples at 16 kHz with the last 64 samples of the frame before it as
context, and carries a state of shape (2, 1, 128) from frame to frame, so the
frames of one stream must go through it in order. This module is that loop,
written after Silero's own ONNX wrapper: the context and the state live here
and the network is a pure function of them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import onnxruntime as ort

from synthia.kernel.errors import SynthiaError
from synthia.voice.audio import FRAME_SAMPLES, SAMPLE_RATE

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from numpy.typing import NDArray

    from synthia.models.catalogue import SpeechModel

CONTEXT_SAMPLES: Final = 64
STATE_SHAPE: Final = (2, 1, 128)

type Network = Callable[
    [NDArray[np.float32], NDArray[np.float32]], tuple[float, NDArray[np.float32]]
]
"""The window (1, 576) and the state in; the speech probability and the next
state out."""


class VadError(SynthiaError):
    """A voice activity model that cannot be loaded."""


class Vad:
    """One stream's speech probabilities, frame by frame."""

    def __init__(self, network: Network) -> None:
        self._network = network
        self._context = np.zeros(CONTEXT_SAMPLES, np.float32)
        self._state = np.zeros(STATE_SHAPE, np.float32)

    @classmethod
    def load(cls, directory: Path, spec: SpeechModel) -> Vad:
        """Load ``spec``'s installed network from ``directory``.

        It runs on one thread, as Silero's wrapper runs it: a frame costs too
        little for more threads to pay for their start.

        Raises:
            VadError: If the file is missing, unreadable or not an ONNX network.
        """
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        try:
            session = ort.InferenceSession(
                str(directory / spec.model.name),
                options,
                providers=["CPUExecutionProvider"],
            )
        # onnxruntime raises 15 classes with no base but Exception
        # (InvalidProtobuf for a damaged file, NoSuchFile, Fail...).
        except Exception as error:
            message = f"cannot load the voice activity model {spec.id}: {error}"
            raise VadError(message) from error
        rate = np.array(SAMPLE_RATE, np.int64)

        def network(
            window: NDArray[np.float32], state: NDArray[np.float32]
        ) -> tuple[float, NDArray[np.float32]]:
            out, after = session.run(
                None, {"input": window, "state": state, "sr": rate}
            )
            return float(out[0, 0]), np.asarray(after, np.float32)

        return cls(network)

    def probability(self, samples: NDArray[np.float32]) -> float:
        """Return how likely the next frame of the stream is speech, from 0 to 1.

        Raises:
            ValueError: If ``samples`` is not one frame.
        """
        if samples.shape != (FRAME_SAMPLES,):
            message = f"{samples.shape=} is not one frame of {FRAME_SAMPLES}"
            raise ValueError(message)
        window = np.concatenate([self._context, samples])[np.newaxis, :]
        speech, self._state = self._network(window, self._state)
        self._context = samples[-CONTEXT_SAMPLES:].copy()
        return speech

    def reset(self) -> None:
        """Forget the stream so far, before reading a new one."""
        self._context = np.zeros(CONTEXT_SAMPLES, np.float32)
        self._state = np.zeros(STATE_SHAPE, np.float32)
