# The part of moonshine-voice the speech-to-text benchmark uses: it has no type hints.

from collections.abc import Sequence
from enum import IntEnum
from pathlib import Path

class ModelArch(IntEnum):
    TINY = 0
    BASE = 1
    TINY_STREAMING = 2
    BASE_STREAMING = 3
    SMALL_STREAMING = 4
    MEDIUM_STREAMING = 5

class TranscriptLine:
    text: str

class Transcript:
    lines: list[TranscriptLine]

class Transcriber:
    def __init__(self, model_path: str | Path, model_arch: ModelArch = ...) -> None: ...
    def transcribe_without_streaming(
        self, audio_data: Sequence[float], sample_rate: int = 16000, flags: int = 0
    ) -> Transcript: ...
    def close(self) -> None: ...
