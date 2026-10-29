"""Where an utterance starts and ends, from the speech probability of each frame.

Speech starts once enough frames in a row are likely speech, and ends once
enough frames in a row are likely not. Two thresholds, one to enter speech and
a lower one to leave it, keep a probability hovering near one line from
starting and ending speech on every frame. The frames just before the start
are kept, so the first syllable, quieter than the voice that follows, is not
cut off.

The defaults are LiveKit Agents' for the same model (Silero VAD, a voice
assistant listening for turns): enter at 0.5 and leave at 0.15 below it, 50 ms
of speech to start, 550 ms of quiet to end, 500 ms kept before the start.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Final

import numpy as np

from synthia.voice.audio import FRAME_SAMPLES, SAMPLE_RATE

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from synthia.voice.audio import Frame

FRAME_MS: Final = FRAME_SAMPLES * 1000 // SAMPLE_RATE
SPEECH_ON: Final = 0.5
SPEECH_OFF: Final = 0.35
MIN_SPEECH_MS: Final = 50
HANGOVER_MS: Final = 550
PREROLL_MS: Final = 500
TAIL_FRAMES: Final = 1
"""Quiet frames kept after the last speech frame, as Silero pads its own
segments by about 30 ms."""


@dataclass(frozen=True, slots=True)
class SpeechStarted:
    """The person began to speak; ``started_ns`` is the first speech frame's capture."""

    started_ns: int


@dataclass(frozen=True, slots=True)
class SpeechEnded:
    """The person stopped speaking.

    ``ended_ns`` is the capture time of the first quiet frame, when the person
    stopped, not when the quiet had lasted long enough to be sure. ``samples``
    holds the utterance, from the kept frames before the start to one frame
    past the last speech.
    """

    ended_ns: int
    samples: NDArray[np.float32]


type Boundary = SpeechStarted | SpeechEnded


class _State(Enum):
    QUIET = auto()
    SPEAKING = auto()


def _frames(ms: int) -> int:
    return max(1, -(-ms // FRAME_MS))


class Endpointer:
    """Turns frames and their speech probabilities into utterance boundaries.

    Raises:
        ValueError: If ``off`` is above ``on`` or the pre-roll is shorter than
            the speech needed to start, which would drop the first speech frames.
    """

    def __init__(
        self,
        *,
        on: float = SPEECH_ON,
        off: float = SPEECH_OFF,
        min_speech_ms: int = MIN_SPEECH_MS,
        hangover_ms: int = HANGOVER_MS,
        preroll_ms: int = PREROLL_MS,
    ) -> None:
        if off > on:
            message = f"{off=} must not be above {on=}"
            raise ValueError(message)
        self._on = on
        self._off = off
        self._start_after = _frames(min_speech_ms)
        self._end_after = _frames(hangover_ms)
        if _frames(preroll_ms) < self._start_after:
            message = f"{preroll_ms=} must cover {min_speech_ms=}"
            raise ValueError(message)
        self._before: deque[Frame] = deque(maxlen=_frames(preroll_ms))
        self._utterance: list[Frame] = []
        self._state = _State.QUIET
        self._run = 0

    def push(self, frame: Frame, probability: float) -> Boundary | None:
        """Take the next frame and its speech probability; return what it completes."""
        if self._state is _State.QUIET:
            return self._while_quiet(frame, probability)
        return self._while_speaking(frame, probability)

    def _while_quiet(self, frame: Frame, probability: float) -> Boundary | None:
        self._before.append(frame)
        self._run = self._run + 1 if probability >= self._on else 0
        if self._run < self._start_after:
            return None
        self._utterance = list(self._before)
        self._before.clear()
        self._state, self._run = _State.SPEAKING, 0
        return SpeechStarted(self._utterance[-self._start_after].captured_ns)

    def _while_speaking(self, frame: Frame, probability: float) -> Boundary | None:
        self._utterance.append(frame)
        self._run = self._run + 1 if probability < self._off else 0
        if self._run < self._end_after:
            return None
        quiet = self._run
        kept = self._utterance[: len(self._utterance) - quiet + TAIL_FRAMES]
        ended = SpeechEnded(
            self._utterance[-quiet].captured_ns,
            np.concatenate([f.samples for f in kept]),
        )
        # The quiet that ended this utterance is the lead-in of the next one,
        # should the person start again at once.
        self._before.extend(self._utterance[-quiet:])
        self._utterance = []
        self._state, self._run = _State.QUIET, 0
        return ended
