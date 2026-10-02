"""The clock and the calculator: answers a model must never guess."""

from __future__ import annotations

import ast
import math
from datetime import datetime
from typing import TYPE_CHECKING, Final, cast

from synthia.agent.tools import Effect, FunctionTool, Reach, ToolError

if TYPE_CHECKING:
    from collections.abc import Callable

MAX_EXPRESSION_CHARS: Final = 200
# Results past 4,096 bits (about 1,233 digits) are refused before they are
# computed, so "9**9**9" cannot take the machine's memory.
MAX_RESULT_BITS: Final = 4096

# ``**`` can give a complex number ((-8) ** 0.5), so results are checked, not assumed.
_BINARY: Final[dict[type[ast.operator], Callable[[float, float], object]]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}
_UNARY: Final[dict[type[ast.unaryop], Callable[[float], float]]] = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


def clock_tool(
    now: Callable[[], datetime] = lambda: datetime.now().astimezone(),
) -> FunctionTool:
    """Return the tool that tells the current local date and time."""

    def current_time() -> str:
        """Return the current local date, time, weekday and UTC offset."""
        moment = now()
        return f"{moment:%A %Y-%m-%d %H:%M:%S} UTC{moment:%z}"

    return FunctionTool.of(current_time, reach=Reach.LOCAL, effect=Effect.READ)


def calculate(expression: str) -> str:
    """Evaluate arithmetic exactly: numbers, + - * / // % ** and parentheses.

    Raises:
        ToolError: If the expression is not plain arithmetic, too long, divides
            by zero, or its result would be too large.
    """
    if len(expression) > MAX_EXPRESSION_CHARS:
        message = f"the expression is longer than {MAX_EXPRESSION_CHARS} characters"
        raise ToolError(message)
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        message = f"not an arithmetic expression: {expression!r}"
        raise ToolError(message) from None
    try:
        return str(_value(tree.body))
    except ZeroDivisionError:
        message = "division by zero"
        raise ToolError(message) from None
    except OverflowError:
        message = "the result is too large"
        raise ToolError(message) from None


def calculator_tool() -> FunctionTool:
    """Return the calculator as a tool."""
    return FunctionTool.of(calculate, reach=Reach.LOCAL, effect=Effect.READ)


def _value(node: ast.expr) -> float:
    if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
        return cast("float", node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_value(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _value(node.left), _value(node.right)
        if isinstance(node.op, ast.Pow):
            _check_power(left, right)
        return _real(_BINARY[type(node.op)](left, right))
    allowed = "only numbers and + - * / // % ** are allowed"
    message = f"{allowed}, not {ast.unparse(node)!r}"
    raise ToolError(message)


def _real(result: object) -> float:
    if isinstance(result, int):
        if result.bit_length() > MAX_RESULT_BITS:
            raise OverflowError
        return result
    if isinstance(result, float):
        if not math.isfinite(result):
            raise OverflowError
        return result
    message = "the result is not a real number"
    raise ToolError(message)


def _check_power(base: float, exponent: float) -> None:
    """Refuse a power whose result is surely too large, before computing it."""
    if (
        isinstance(base, int)
        and isinstance(exponent, int)
        and exponent > 0
        and max(abs(base).bit_length() - 1, 0) * exponent > MAX_RESULT_BITS
    ):
        raise OverflowError
