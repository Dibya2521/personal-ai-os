from collections.abc import Sequence

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from synthia.voice.audio import FRAME_SAMPLES, Frame
from synthia.voice.endpoint import (
    Boundary,
    Endpointer,
    SpeechEnded,
    SpeechStarted,
)

FRAME_NS = 32_000_000
QUIET, SPEECH, BETWEEN = 0.1, 0.9, 0.4
PREROLL_FRAMES, END_FRAMES = 16, 18


def frame(n: int) -> Frame:
    return Frame(np.full(FRAME_SAMPLES, n, dtype=np.float32), n * FRAME_NS)


def run(probabilities: Sequence[float]) -> list[tuple[int, Boundary]]:
    endpointer = Endpointer()
    found = [(n, endpointer.push(frame(n), p)) for n, p in enumerate(probabilities)]
    return [(n, b) for n, b in found if b is not None]


def first_frames(ended: SpeechEnded) -> list[int]:
    return [int(v) for v in ended.samples[::FRAME_SAMPLES]]


def test_one_loud_frame_is_not_speech() -> None:
    assert run([QUIET, SPEECH, QUIET, SPEECH, QUIET]) == []


def test_speech_starts_at_its_first_frame_once_two_in_a_row_are_speech() -> None:
    assert run([QUIET] * 5 + [SPEECH, SPEECH]) == [(6, SpeechStarted(5 * FRAME_NS))]


def test_speech_ends_after_550_ms_of_quiet_dated_by_the_first_quiet_frame() -> None:
    probabilities = [SPEECH] * 4 + [QUIET] * END_FRAMES
    found = run(probabilities)

    assert [n for n, _ in found] == [1, 3 + END_FRAMES]
    ended = found[1][1]
    assert isinstance(ended, SpeechEnded)
    assert ended.ended_ns == 4 * FRAME_NS
    assert run(probabilities[:-1])[1:] == []


def test_a_dip_above_the_lower_threshold_is_still_speech() -> None:
    almost = [SPEECH] * 2 + [QUIET] * (END_FRAMES - 1)
    found = run([*almost, BETWEEN, *([QUIET] * (END_FRAMES - 1))])

    assert [type(b) for _, b in found] == [SpeechStarted]


def test_the_utterance_keeps_500_ms_before_and_32_ms_after_the_speech() -> None:
    probabilities = [QUIET] * 30 + [SPEECH] * 5 + [QUIET] * END_FRAMES
    ended = run(probabilities)[1][1]

    assert isinstance(ended, SpeechEnded)
    first_kept = 30 + 2 - PREROLL_FRAMES
    assert first_frames(ended) == list(range(first_kept, 36))
    assert ended.samples.shape == ((36 - first_kept) * FRAME_SAMPLES,)


def test_the_quiet_after_one_utterance_leads_into_the_next() -> None:
    probabilities = (
        [SPEECH] * 3 + [QUIET] * END_FRAMES + [SPEECH] * 2 + [QUIET] * END_FRAMES
    )
    second = run(probabilities)[3][1]

    assert isinstance(second, SpeechEnded)
    assert first_frames(second)[0] == 3 + END_FRAMES + 2 - PREROLL_FRAMES


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"on": 0.3, "off": 0.4}, r"off=0.4 must not be above on=0.3"),
        ({"min_speech_ms": 200, "preroll_ms": 100}, r"preroll_ms=100 must cover"),
    ],
)
def test_settings_that_would_lose_speech_are_refused(
    settings: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Endpointer(**settings)  # pyright: ignore[reportArgumentType]


@given(st.lists(st.sampled_from([QUIET, BETWEEN, SPEECH]), max_size=300))
def test_boundaries_alternate_and_every_utterance_is_whole_frames(
    probabilities: list[float],
) -> None:
    found = [b for _, b in run(probabilities)]

    assert all(isinstance(b, SpeechStarted) for b in found[::2])
    for start, end in zip(found[::2], found[1::2], strict=False):
        assert isinstance(start, SpeechStarted)
        assert isinstance(end, SpeechEnded)
        assert start.started_ns < end.ended_ns
        assert end.samples.size > 0
        assert end.samples.size % FRAME_SAMPLES == 0
