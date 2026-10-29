"""Measure speech-to-text candidates on the person's own recorded commands.

    uv run python benchmarks/voice_record.py      # once, to record the clips
    uv run python benchmarks/stt_candidates.py

Each candidate transcribes every clip recorded by ``voice_record.py`` through
its own reference implementation: Moonshine through moonshine-voice (its
native runtime on the ``.ort`` files), Whisper base.en int8 through onnx-asr.
These are the implementations SYNTHIA's own decoder must agree with, so every
transcript is written to ``stt_transcripts.jsonl`` beside the clips.

Word error rate compares words after lower-casing, dropping punctuation and
writing numbers as words ("4:30" is "four thirty"), since the references spell
numbers out and the models may not. Seconds are wall time per clip on the
machine as it is, so the CPU load at the start is printed with the table.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import time
import wave
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import psutil
from voice_record import INT16_MAX, clip_path, clips_dir, commands

from synthia.kernel.config import load_settings
from synthia.voice.audio import SAMPLE_RATE

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from numpy.typing import NDArray

MOONSHINE_CDN: Final = "download.moonshine.ai/model"
TRANSCRIPTS: Final = "stt_transcripts.jsonl"
CPU_SAMPLE_S: Final = 1.0

type Transcribe = Callable[[NDArray[np.float32]], str]

_ONES: Final = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS: Final = [
    "_",
    "_",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
]
_HUNDRED: Final = 100
_TWENTY: Final = 20
_TEN: Final = 10
_TOKEN: Final = re.compile(r"[a-z']+|\d+")


def number_words(n: int) -> list[str]:
    """Spell ``n`` (0 to 999) in words."""
    if n < _TWENTY:
        return [_ONES[n]]
    if n < _HUNDRED:
        tens, ones = divmod(n, _TEN)
        return [_TENS[tens]] + ([_ONES[ones]] if ones else [])
    hundreds, rest = divmod(n, _HUNDRED)
    return [_ONES[hundreds], "hundred"] + (number_words(rest) if rest else [])


def words(text: str) -> list[str]:
    """Lower-case words of ``text``, numbers spelled out, punctuation dropped."""
    out: list[str] = []
    for token in _TOKEN.findall(text.lower()):
        if token.isdigit() and int(token) < _HUNDRED * _TEN:
            out.extend(number_words(int(token)))
        elif token.strip("'"):
            out.append(token.strip("'"))
    return out


def edits(reference: list[str], hypothesis: list[str]) -> int:
    """Word-level Levenshtein distance."""
    row = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        previous, row[0] = row[0], i
        for j, hyp in enumerate(hypothesis, 1):
            previous, row[j] = (
                row[j],
                min(row[j] + 1, row[j - 1] + 1, previous + (ref != hyp)),
            )
    return row[-1]


def read_clip(path: Path) -> NDArray[np.float32]:
    """Return a 16-bit mono WAV as floats in [-1, 1]."""
    with wave.open(str(path), "rb") as clip:
        pcm = np.frombuffer(clip.readframes(clip.getnframes()), dtype="<i2")
    return (pcm / INT16_MAX).astype(np.float32)


def moonshine(models: Path, folder: str, arch_name: str) -> Transcribe:
    """Return moonshine-voice's transcriber for one downloaded model."""
    from moonshine_voice import ModelArch, Transcriber  # noqa: PLC0415

    transcriber = Transcriber(
        models / "moonshine" / MOONSHINE_CDN / folder, ModelArch[arch_name]
    )

    def transcribe(samples: NDArray[np.float32]) -> str:
        found = transcriber.transcribe_without_streaming(samples.tolist(), SAMPLE_RATE)
        return " ".join(line.text for line in found.lines)

    return transcribe


def whisper(models: Path) -> Transcribe:
    """Return onnx-asr's Whisper base.en int8."""
    import onnx_asr  # noqa: PLC0415

    model = onnx_asr.load_model(
        "whisper", models / "whisper-base.en", quantization="quantized"
    )

    def transcribe(samples: NDArray[np.float32]) -> str:
        return str(model.recognize(samples, sample_rate=SAMPLE_RATE))

    return transcribe


CANDIDATES: Final[dict[str, Callable[[Path], Transcribe]]] = {
    "moonshine-tiny-en": lambda m: moonshine(m, "tiny-en/quantized/tiny-en", "TINY"),
    "moonshine-tiny-streaming-en": lambda m: moonshine(
        m, "tiny-streaming-en/quantized_26_08_21", "TINY_STREAMING"
    ),
    "moonshine-small-streaming-en": lambda m: moonshine(
        m, "small-streaming-en/quantized_26_08_21", "SMALL_STREAMING"
    ),
    "whisper-base.en-int8": whisper,
}


@dataclass(frozen=True, slots=True)
class Row:
    """One candidate's result over every clip."""

    name: str
    wer: float
    errors: int
    reference_words: int
    median_s: float
    p95_s: float
    real_time_factor: float


def measure(
    name: str, models: Path, clips: list[tuple[int, str, NDArray[np.float32]]]
) -> tuple[Row, list[dict[str, object]]]:
    """Transcribe every clip with ``name``; return its row and its transcripts."""
    transcribe = CANDIDATES[name](models)
    transcribe(clips[0][2])
    seconds: list[float] = []
    errors = total = 0
    kept: list[dict[str, object]] = []
    for number, reference, samples in clips:
        began = time.perf_counter()
        text = transcribe(samples)
        seconds.append(time.perf_counter() - began)
        errors += edits(words(reference), words(text))
        total += len(words(reference))
        kept.append(
            {"model": name, "clip": number, "reference": reference, "text": text}
        )
    audio_s = sum(s.size for _, _, s in clips) / SAMPLE_RATE
    row = Row(
        name,
        errors / total,
        errors,
        total,
        statistics.median(seconds),
        statistics.quantiles(seconds, n=20)[18],
        sum(seconds) / audio_s,
    )
    return row, kept


def main() -> int:
    """Print one row per candidate and write every transcript."""
    home = load_settings().home
    lines = commands()
    clips = [
        (n, lines[n - 1], read_clip(clip_path(home, n)))
        for n in range(1, len(lines) + 1)
        if clip_path(home, n).is_file()
    ]
    if not clips:
        where = clips_dir(home)
        print(f"no clips in {where}: run benchmarks/voice_record.py", file=sys.stderr)
        return 1
    load = psutil.cpu_percent(interval=CPU_SAMPLE_S)
    print(f"{len(clips)} clips, CPU {load:.0f} % busy at the start")
    print(
        f"{'model':30} {'WER':>6} {'errors':>9} {'median s':>9} {'p95 s':>7} {'RTF':>6}"
    )
    transcripts: list[dict[str, object]] = []
    for name in CANDIDATES:
        row, kept = measure(name, home / "bench" / "stt", clips)
        transcripts.extend(kept)
        print(
            f"{row.name:30} {row.wer:6.3f} {row.errors:>4}/{row.reference_words:<4} "
            f"{row.median_s:9.3f} {row.p95_s:7.3f} {row.real_time_factor:6.3f}",
            flush=True,
        )
    written = clips_dir(home) / TRANSCRIPTS
    with written.open("w", encoding="utf-8", newline="\n") as out:
        for item in transcripts:
            out.write(json.dumps(item, allow_nan=False) + "\n")
    print(f"transcripts: {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
