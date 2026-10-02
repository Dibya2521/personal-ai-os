from datetime import UTC, datetime, timedelta, timezone

import pytest

from synthia.agent.tools import ToolError
from synthia.tools.basic import (
    MAX_EXPRESSION_CHARS,
    calculate,
    calculator_tool,
    clock_tool,
)

IST = timezone(timedelta(hours=5, minutes=30))


async def test_the_clock_tells_date_weekday_time_and_offset() -> None:
    tool = clock_tool(lambda: datetime(2026, 10, 2, 19, 5, 46, tzinfo=IST))

    assert await tool.run("{}") == "Friday 2026-10-02 19:05:46 UTC+0530"


async def test_the_clock_reads_the_real_time_by_default() -> None:
    before = datetime.now(UTC).replace(microsecond=0)

    told = await clock_tool().run("")

    moment = datetime.strptime(told.split(" ", 1)[1], "%Y-%m-%d %H:%M:%S UTC%z")
    assert before - timedelta(seconds=1) <= moment <= datetime.now(UTC)


@pytest.mark.parametrize(
    ("expression", "result"),
    [
        ("1234 * 5678", "7006652"),
        ("2 * (3 + 4)", "14"),
        ("7 / 2", "3.5"),
        ("7 // 2", "3"),
        ("-7 % 3", "2"),
        ("2 ** 10", "1024"),
        ("2 ** -2", "0.25"),
        ("-(-5)", "5"),
        ("+3", "3"),
        ("1.5e3 + 1", "1501.0"),
        ("  12 * 12  ", "144"),
        ("2 ** 4095", str(2**4095)),
        ("1 ** (10 ** 100)", "1"),
    ],
)
def test_arithmetic_is_exact(expression: str, result: str) -> None:
    assert calculate(expression) == result


@pytest.mark.parametrize(
    ("expression", "error"),
    [
        ("1 / 0", "division by zero"),
        ("5 % 0", "division by zero"),
        ("9 ** 9 ** 9", "the result is too large"),
        ("2 ** 4096", "the result is too large"),
        ("2 ** (10 ** 100)", "the result is too large"),
        ("10.0 ** 400", "the result is too large"),
        ("1e308 * 10", "the result is too large"),
        ("(10 ** 1000) * (10 ** 1000)", "the result is too large"),
        ("(-8) ** 0.5", "the result is not a real number"),
        ("2 +", "not an arithmetic expression: '2 +'"),
        ("", "not an arithmetic expression: ''"),
        (
            "__import__('os').system('echo hi')",
            (
                "only numbers and + - * / // % ** are allowed, "
                "not \"__import__('os').system('echo hi')\""
            ),
        ),
        ("x * 2", "only numbers and + - * / // % ** are allowed, not 'x'"),
        ("True + 1", "only numbers and + - * / // % ** are allowed, not 'True'"),
        ("'a' * 3", "only numbers and + - * / // % ** are allowed, not \"'a'\""),
        ("2 << 3", "only numbers and + - * / // % ** are allowed, not '2 << 3'"),
        ("1j * 1j", "only numbers and + - * / // % ** are allowed, not '1j'"),
        ("[1, 2]", "only numbers and + - * / // % ** are allowed, not '[1, 2]'"),
        ("not 1", "only numbers and + - * / // % ** are allowed, not 'not 1'"),
    ],
)
def test_anything_but_bounded_arithmetic_is_refused(
    expression: str, error: str
) -> None:
    with pytest.raises(ToolError) as raised:
        calculate(expression)

    assert str(raised.value) == error


def test_a_long_expression_is_refused_before_parsing() -> None:
    with pytest.raises(ToolError, match="longer than 200 characters"):
        calculate("1+" * (MAX_EXPRESSION_CHARS // 2) + "1")


async def test_the_calculator_tool_takes_an_expression() -> None:
    tool = calculator_tool()

    assert tool.spec.name == "calculate"
    assert await tool.run('{"expression": "6 * 7"}') == "42"
