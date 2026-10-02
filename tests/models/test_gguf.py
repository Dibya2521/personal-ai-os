import struct
from pathlib import Path

import pytest

from synthia.models.gguf import MAX_TEXT_BYTES, context_length

STRING = 8
ARRAY = 9
UINT32 = 4
UINT64 = 10
FLOAT32 = 6


def text(value: str) -> bytes:
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def pair(key: str, kind: int, value: bytes) -> bytes:
    return text(key) + struct.pack("<I", kind) + value


def gguf(*pairs: bytes, version: int = 3) -> bytes:
    return b"GGUF" + struct.pack("<IQQ", version, 0, len(pairs)) + b"".join(pairs)


ARCH = pair("general.architecture", STRING, text("qwen35"))
# What a real header holds before the context: a list of strings and a float.
TOKENS = pair(
    "tokenizer.ggml.tokens",
    ARRAY,
    struct.pack("<IQ", STRING, 3) + text("a") + text("bb") + text("<|im_end|>"),
)
SCALES = pair("tokenizer.ggml.scores", ARRAY, struct.pack("<IQ3f", FLOAT32, 3, 0, 1, 2))
CONTEXT = pair("qwen35.context_length", UINT32, struct.pack("<I", 262_144))


def written(tmp_path: Path, data: bytes) -> Path:
    path = tmp_path / "model.gguf"
    path.write_bytes(data)
    return path


def test_the_trained_context_is_read_past_the_values_before_it(
    tmp_path: Path,
) -> None:
    path = written(tmp_path, gguf(ARCH, TOKENS, SCALES, CONTEXT) + b"\0" * 64)

    assert context_length(path) == 262_144


def test_a_64_bit_context_and_a_version_2_file_are_read(tmp_path: Path) -> None:
    wide = pair("qwen35.context_length", UINT64, struct.pack("<Q", 40_960))

    assert context_length(written(tmp_path, gguf(ARCH, wide, version=2))) == 40_960


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"PK\3\4" + b"\0" * 32, id="not a GGUF file"),
        pytest.param(gguf(ARCH, CONTEXT, version=1), id="version 1"),
        pytest.param(gguf(ARCH, TOKENS), id="no context key"),
        pytest.param(gguf(CONTEXT, ARCH), id="context before the architecture"),
        pytest.param(
            gguf(ARCH, pair("qwen35.context_length", FLOAT32, struct.pack("<f", 1))),
            id="context that is not an integer",
        ),
        pytest.param(
            gguf(ARCH, pair("qwen35.context_length", UINT32, struct.pack("<I", 0))),
            id="zero context",
        ),
    ],
)
def test_a_file_that_does_not_say_gives_none(tmp_path: Path, data: bytes) -> None:
    assert context_length(written(tmp_path, data)) is None


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(gguf(ARCH, CONTEXT)[:-2], id="cut inside the context value"),
        pytest.param(gguf(ARCH, TOKENS)[:-5], id="cut inside a string"),
        pytest.param(
            gguf(ARCH, pair("x", STRING, struct.pack("<Q", 1 << 62))),
            id="a string longer than the file",
        ),
        pytest.param(
            gguf(ARCH, pair("x", ARRAY, struct.pack("<IQ", UINT64, 1 << 60))),
            id="an array longer than the file",
        ),
        pytest.param(
            gguf(struct.pack("<Q", MAX_TEXT_BYTES + 1) + b"k"),
            id="a key longer than any real one",
        ),
        pytest.param(gguf(pair("x", 99, b"")), id="an unknown value type"),
        pytest.param(
            gguf(struct.pack("<Q", 2) + b"\xff\xfe" + struct.pack("<I", UINT32)),
            id="a key that is not UTF-8",
        ),
    ],
)
def test_a_damaged_header_gives_none_and_a_warning(
    tmp_path: Path, data: bytes, caplog: pytest.LogCaptureFixture
) -> None:
    assert context_length(written(tmp_path, data)) is None
    assert "could not read the GGUF header of model.gguf" in caplog.text


def test_a_missing_file_gives_none(tmp_path: Path) -> None:
    assert context_length(tmp_path / "absent.gguf") is None
