"""Record the person saying each voice command once, for the speech-to-text benchmark.

    uv run python benchmarks/voice_record.py
    uv run python benchmarks/voice_record.py --only 7

Each line of ``benchmarks/data/voice_commands.txt`` is shown in turn; the
person reads it aloud and SYNTHIA's own microphone, voice activity model and
endpointer cut the utterance, so recording the set also exercises them. Each
utterance is saved as 16-bit mono WAV at 16 kHz in the SYNTHIA home, never in
the repository: a voice is personal data. A line with no speech within 15 s is
skipped and can be recorded again with ``--only``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import wave
from contextlib import aclosing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

from synthia.kernel.config import load_settings
from synthia.models.catalogue import SILERO_VAD
from synthia.models.install import BYTES_PER_GB, Installer
from synthia.voice.audio import SAMPLE_RATE, microphone
from synthia.voice.endpoint import Endpointer, SpeechEnded, SpeechStarted
from synthia.voice.vad import Vad

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from synthia.voice.audio import Microphone

COMMANDS: Final = Path(__file__).parent / "data" / "voice_commands.txt"
WAIT_S: Final = 15.0
INT16_MAX: Final = 32_767


def clips_dir(home: Path) -> Path:
    """Return where the recorded commands are kept in the SYNTHIA home ``home``."""
    return home / "bench" / "voice_eval"


def clip_path(home: Path, number: int) -> Path:
    """Return where command ``number`` (from 1) is kept."""
    return clips_dir(home) / f"{number:02d}.wav"


def commands() -> list[str]:
    """Return the reference texts, one per line."""
    return COMMANDS.read_text(encoding="utf-8").splitlines()


def save(path: Path, samples: NDArray[np.float32]) -> None:
    """Write ``samples`` as 16-bit mono WAV at 16 kHz."""
    pcm = np.clip(samples, -1.0, 1.0) * INT16_MAX
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm.astype("<i2").tobytes())


@dataclass(frozen=True, slots=True)
class Heard:
    """What one wait for an utterance found."""

    samples: NDArray[np.float32] | None
    started: bool
    peak: float
    """The highest speech probability of any frame."""


async def utterance(mic: Microphone, vad: Vad) -> Heard:
    """Wait for the next utterance; its samples are None when none ended in time."""
    vad.reset()
    endpointer = Endpointer()
    started, peak = False, 0.0
    try:
        async with asyncio.timeout(WAIT_S), aclosing(mic.frames()) as frames:
            async for frame in frames:
                probability = vad.probability(frame.samples)
                peak = max(peak, probability)
                boundary = endpointer.push(frame, probability)
                started = started or isinstance(boundary, SpeechStarted)
                if isinstance(boundary, SpeechEnded):
                    return Heard(boundary.samples, started=True, peak=peak)
    except TimeoutError:
        pass
    return Heard(None, started=started, peak=peak)


async def record(home: Path, numbers: list[int]) -> int:
    """Record each command in ``numbers``; return how many were skipped."""
    lines = commands()
    installer = Installer(home, BYTES_PER_GB)
    vad = Vad.load(installer.path_of(SILERO_VAD), SILERO_VAD)
    skipped = 0
    async with microphone() as mic:
        for number in numbers:
            print(
                f"\n[{number:02d}/{len(lines)}] say:  {lines[number - 1]}", flush=True
            )
            heard = await utterance(mic, vad)
            if heard.samples is None:
                why = "started but never ended" if heard.started else "never started"
                print(f"   skipped: speech {why} (peak {heard.peak:.2f})", flush=True)
                skipped += 1
                continue
            save(clip_path(home, number), heard.samples)
            seconds = heard.samples.size / SAMPLE_RATE
            print(f"   saved {seconds:.1f} s (peak {heard.peak:.2f})", flush=True)
    return skipped


def main() -> int:
    """Record every command, or only the one named."""
    parser = argparse.ArgumentParser(description="Record each voice command once.")
    parser.add_argument("--only", type=int, help="record only this command number")
    parser.add_argument("--now", action="store_true", help="start without waiting")
    args = parser.parse_args()
    total = len(commands())
    numbers = [args.only] if args.only else list(range(1, total + 1))
    home = load_settings().home
    print(f"Recording into {clips_dir(home)}. Read each line aloud, then pause.")
    if not args.now:
        input("Press Enter when ready. ")
    skipped = asyncio.run(record(home, numbers))
    print(f"\ndone: {len(numbers) - skipped} saved, {skipped} skipped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
