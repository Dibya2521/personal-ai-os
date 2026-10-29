from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from synthia.kernel.config import default_home
from synthia.models.catalogue import SILERO_VAD
from synthia.models.install import BYTES_PER_GB, Installer
from synthia.voice.audio import FRAME_SAMPLES
from synthia.voice.vad import CONTEXT_SAMPLES, STATE_SHAPE, Vad, VadError


@dataclass
class RecordingNetwork:
    windows: list[NDArray[np.float32]] = field(
        default_factory=list[NDArray[np.float32]]
    )
    states: list[NDArray[np.float32]] = field(default_factory=list[NDArray[np.float32]])

    def __call__(
        self, window: NDArray[np.float32], state: NDArray[np.float32]
    ) -> tuple[float, NDArray[np.float32]]:
        self.windows.append(window.copy())
        self.states.append(state.copy())
        return float(window[0, -1]), state + 1


def ramp(start: float) -> NDArray[np.float32]:
    return np.arange(start, start + FRAME_SAMPLES, dtype=np.float32)


def test_each_frame_is_read_with_the_last_64_samples_before_it() -> None:
    network = RecordingNetwork()
    vad = Vad(network)
    first, second = ramp(0), ramp(1000)

    assert vad.probability(first) == FRAME_SAMPLES - 1
    first[:] = -1
    vad.probability(second)

    assert network.windows[0].shape == (1, CONTEXT_SAMPLES + FRAME_SAMPLES)
    assert np.all(network.windows[0][0, :CONTEXT_SAMPLES] == 0)
    expected = np.concatenate([ramp(0)[-CONTEXT_SAMPLES:], second])
    np.testing.assert_array_equal(network.windows[1][0], expected)


def test_the_state_carries_from_frame_to_frame_until_reset() -> None:
    network = RecordingNetwork()
    vad = Vad(network)
    for _ in range(3):
        vad.probability(ramp(0))
    vad.reset()
    vad.probability(ramp(5))

    assert [float(s.max()) for s in network.states] == [0.0, 1.0, 2.0, 0.0]
    assert all(s.shape == STATE_SHAPE for s in network.states)
    assert np.all(network.windows[3][0, :CONTEXT_SAMPLES] == 0)


def test_only_one_frame_at_a_time_is_read() -> None:
    with pytest.raises(ValueError, match=r"is not one frame of 512"):
        Vad(RecordingNetwork()).probability(np.zeros(480, np.float32))


def test_a_missing_model_cannot_load(tmp_path: Path) -> None:
    with pytest.raises(
        VadError, match=r"cannot load the voice activity model silero-vad"
    ):
        Vad.load(tmp_path, SILERO_VAD)


def installed_vad() -> Vad:
    installer = Installer(default_home(), BYTES_PER_GB)
    if not installer.installed(SILERO_VAD):
        pytest.skip("silero-vad is not installed (synthia models install silero-vad)")
    return Vad.load(installer.path_of(SILERO_VAD), SILERO_VAD)


def test_the_installed_model_hears_no_speech_in_silence_or_a_steady_tone() -> None:
    vad = installed_vad()
    tone = 0.3 * np.sin(2 * np.pi * 440 * np.arange(FRAME_SAMPLES) / 16_000)

    for samples in (np.zeros(FRAME_SAMPLES), tone):
        vad.reset()
        heard = [vad.probability(samples.astype(np.float32)) for _ in range(30)]
        assert max(heard) < 0.5, heard


def test_the_installed_model_gives_the_same_answers_after_a_reset() -> None:
    vad = installed_vad()
    rng = np.random.default_rng(7)
    frames = [rng.normal(0, 0.1, FRAME_SAMPLES).astype(np.float32) for _ in range(20)]

    first = [vad.probability(f) for f in frames]
    vad.reset()
    second = [vad.probability(f) for f in frames]

    assert first == second
    assert all(0.0 <= p <= 1.0 for p in first)
