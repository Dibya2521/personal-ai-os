import asyncio
import io
import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NoReturn

import pytest
from hypothesis import given
from hypothesis import strategies as st

from synthia.kernel.config import LogFormat, LogLevel, Settings
from synthia.kernel.logs import (
    HANDLER_NAME,
    REDACTED,
    ConsoleFormatter,
    JsonFormatter,
    Redactor,
    configure_logging,
    correlation,
    current_correlation_id,
    secret_values,
)

PLANTED = "sk-or-v1-planted-0123456789abcdef"  # pragma: allowlist secret
TEST_LOGGER = "synthia.test"


def fail(error: type[Exception], message: str) -> NoReturn:
    raise error(message)


def make_settings(fmt: LogFormat, key: str | None = PLANTED) -> Settings:
    return Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        home=Path(),
        log_format=fmt,
        log_level=LogLevel.DEBUG,
        openrouter_api_key=key,  # pyright: ignore[reportArgumentType]
    )


@pytest.fixture
def captured() -> Iterator[tuple[io.StringIO, list[logging.Handler]]]:
    root = logging.getLogger()
    before, level = list(root.handlers), root.level
    stream = io.StringIO()
    yield stream, before
    root.handlers[:] = before
    root.setLevel(level)


def json_lines(stream: io.StringIO) -> list[dict[str, Any]]:
    """Return this test's records only; asyncio logs its own at DEBUG."""
    lines = [json.loads(line) for line in stream.getvalue().splitlines()]
    return [line for line in lines if line["logger"] == TEST_LOGGER]


def test_json_record_carries_the_standard_fields(
    captured: tuple[io.StringIO, list[logging.Handler]],
) -> None:
    stream, _ = captured
    configure_logging(make_settings(LogFormat.JSON), stream)

    logging.getLogger("synthia.test").info("hello %s", "world", extra={"turn": 3})

    (line,) = json_lines(stream)
    assert line["level"] == "INFO"
    assert line["logger"] == "synthia.test"
    assert line["msg"] == "hello world"
    assert line["turn"] == 3
    assert line["correlation_id"] is None
    assert line["ts"].endswith("+00:00")


def test_correlation_id_is_scoped_to_the_block_and_its_tasks(
    captured: tuple[io.StringIO, list[logging.Handler]],
) -> None:
    stream, _ = captured
    configure_logging(make_settings(LogFormat.JSON), stream)
    log = logging.getLogger("synthia.test")

    async def child() -> None:
        log.info("inside a task")

    async def turn() -> None:
        with correlation("turn-42"):
            await asyncio.create_task(child())
        log.info("after")

    asyncio.run(turn())

    first, second = json_lines(stream)
    assert first["correlation_id"] == "turn-42"
    assert second["correlation_id"] is None
    assert current_correlation_id() is None


@pytest.mark.parametrize("fmt", list(LogFormat))
def test_a_secret_is_redacted_wherever_it_enters_the_record(
    captured: tuple[io.StringIO, list[logging.Handler]], fmt: LogFormat
) -> None:
    stream, _ = captured
    configure_logging(make_settings(fmt), stream)
    log = logging.getLogger("synthia.test")

    log.warning("key in message " + PLANTED)  # noqa: G003 - the template itself is under test
    log.warning("key in args %s", PLANTED)
    log.warning("key in extra", extra={"header": f"Bearer {PLANTED}"})
    try:
        fail(ValueError, PLANTED)
    except ValueError:
        log.exception("key in traceback")

    output = stream.getvalue()
    assert PLANTED not in output
    assert output.count(REDACTED) >= 4


def test_console_format_is_one_readable_line(
    captured: tuple[io.StringIO, list[logging.Handler]],
) -> None:
    stream, _ = captured
    configure_logging(make_settings(LogFormat.CONSOLE), stream)

    with correlation("c-1"):
        logging.getLogger("synthia.test").error("failed", extra={"code": 7})

    line = stream.getvalue().rstrip("\n")
    assert "\n" not in line
    assert "ERROR   synthia.test: failed [c-1] code=7" in line


def test_configuring_twice_leaves_one_handler(
    captured: tuple[io.StringIO, list[logging.Handler]],
) -> None:
    stream, before = captured
    configure_logging(make_settings(LogFormat.JSON), stream)
    configure_logging(make_settings(LogFormat.CONSOLE), stream)

    ours = [h for h in logging.getLogger().handlers if h.get_name() == HANDLER_NAME]
    assert len(ours) == 1
    assert len(logging.getLogger().handlers) == len(before) + 1


def test_level_filters_records(
    captured: tuple[io.StringIO, list[logging.Handler]],
) -> None:
    stream, _ = captured
    settings = make_settings(LogFormat.JSON).model_copy(
        update={"log_level": LogLevel.WARNING}
    )
    configure_logging(settings, stream)

    logging.getLogger("synthia.test").info("dropped")
    logging.getLogger("synthia.test").warning("kept")

    assert [line["msg"] for line in json_lines(stream)] == ["kept"]


def test_unserialisable_extra_is_rendered_as_text() -> None:
    record = logging.LogRecord("x", logging.INFO, "", 0, "m", None, None)
    record.path = Path("a") / "b"

    assert json.loads(JsonFormatter(Redactor()).format(record))["path"] == str(
        Path("a") / "b"
    )


def test_secret_values_lists_only_secrets_that_are_set() -> None:
    assert secret_values(make_settings(LogFormat.JSON)) == [PLANTED]
    assert secret_values(make_settings(LogFormat.JSON, key=None)) == []


def test_a_secret_containing_another_is_masked_whole() -> None:
    redactor = Redactor(["abc", "xxabcxx"])

    assert redactor.redact("see xxabcxx and abc") == f"see {REDACTED} and {REDACTED}"


def test_values_inside_the_mask_are_ignored() -> None:
    assert Redactor(["", "*", "***"]).redact("text *") == "text *"


@pytest.mark.parametrize(
    ("secret", "text"), [("x*", "xx*"), ("*x", "*xx"), ("a*b", "aa*bb*b")]
)
def test_the_mask_cannot_rebuild_a_secret(secret: str, text: str) -> None:
    assert secret not in Redactor([secret]).redact(text)


@given(
    st.text(alphabet=st.sampled_from("ab*"), min_size=1),
    st.text(alphabet=st.sampled_from("ab*")),
    st.text(alphabet=st.sampled_from("ab*")),
)
def test_no_secret_survives_even_beside_mask_characters(
    secret: str, before: str, after: str
) -> None:
    redacted = Redactor([secret]).redact(before + secret + after)

    assert secret not in redacted or secret in REDACTED


@given(st.text(min_size=1), st.text(), st.text())
def test_no_registered_secret_survives_redaction(
    secret: str, before: str, after: str
) -> None:
    redacted = Redactor([secret]).redact(before + secret + after)

    assert secret not in redacted or secret in REDACTED


def test_console_formatter_appends_the_traceback() -> None:
    try:
        fail(RuntimeError, "boom")
    except RuntimeError:
        record = logging.LogRecord(
            "x", logging.ERROR, "", 0, "m", None, exc_info=sys.exc_info()
        )

    text = ConsoleFormatter(Redactor()).format(record)

    assert text.splitlines()[0].endswith("x: m")
    assert "RuntimeError: boom" in text
