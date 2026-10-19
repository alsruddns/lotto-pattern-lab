"""Korean, lossless display helpers for research and prediction results."""

import ast
import math
from numbers import Real

_BINARY_LABELS = {
    "add": "+",
    "sub": "−",
    "mul": "×",
    "safe_div": "÷",
    "safe_mod": "%",
    "min2": "최솟값",
    "max2": "최댓값",
    "bit_xor": "XOR",
    "bit_and": "AND",
    "bit_or": "OR",
    "shift_left": "왼쪽 시프트",
    "shift_right": "오른쪽 시프트",
    "rotate_left": "왼쪽 회전",
    "rotate_right": "오른쪽 회전",
    "concat_digits": "자릿수 연결",
}
_UNARY_LABELS = {
    "abs": "절댓값",
    "square": "제곱",
    "cube": "세제곱",
    "digit_sum": "자릿수 합",
    "reverse_digits": "자릿수 역순",
    "digital_root": "디지털 루트",
    "is_prime": "소수 판정",
    "factor_count": "약수 개수",
    "fibonacci_index": "피보나치 인덱스",
}


def _missing_label(value: object) -> str | None:
    if value is None:
        return "미수집"
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"none", "null", "미수집", "n/a"}:
            return "미수집"
        if lowered in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity"}:
            return "계산 불가"
    if isinstance(value, Real) and not math.isfinite(float(value)):
        return "계산 불가"
    return None


def format_decimal(value: object, places: int = 4, suffix: str = "") -> str:
    """Format a score without changing its stored precision or conflating zero and missing."""
    missing = _missing_label(value)
    if missing:
        return missing
    if not isinstance(value, Real) or isinstance(value, bool):
        return str(value)
    return f"{float(value):,.{places}f}{suffix}"


def format_probability(value: object, places: int = 4) -> str:
    """Format a probability or p value as a decimal rather than a percentage."""
    return format_decimal(value, places)


def format_percent(value: object, places: int = 2) -> str:
    """Format a ratio in [0, 1] as a percentage."""
    missing = _missing_label(value)
    if missing:
        return missing
    if not isinstance(value, Real) or isinstance(value, bool):
        return str(value)
    return f"{float(value) * 100:,.{places}f}%"


def format_count(value: object) -> str:
    """Format counts with grouping while preserving explicit missing values."""
    missing = _missing_label(value)
    if missing:
        return missing
    if not isinstance(value, Real) or isinstance(value, bool):
        return str(value)
    number = float(value)
    if not number.is_integer():
        return f"{number:,.4f}".rstrip("0").rstrip(".")
    return f"{int(number):,}"


def format_duration(seconds: object) -> str:
    """Format elapsed seconds as Korean hours, minutes, and seconds."""
    missing = _missing_label(seconds)
    if missing:
        return missing
    if not isinstance(seconds, Real) or isinstance(seconds, bool) or float(seconds) < 0:
        return "계산 불가"
    total = int(round(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    return f"{hours}시간 {minutes}분 {remaining_seconds}초"


def pretty_formula(formula: str) -> str:
    """Render saved GP calls as readable notation; never execute or rewrite the source."""
    try:
        expression = ast.parse(formula, mode="eval").body
    except (SyntaxError, ValueError):
        return formula

    def render(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            return ast.unparse(node)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return ("+" if isinstance(node.op, ast.UAdd) else "−") + render(node.operand)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name = node.func.id
            arguments = [render(argument) for argument in node.args]
            if name in _BINARY_LABELS and len(arguments) == 2:
                symbol = _BINARY_LABELS[name]
                if name in {"min2", "max2"}:
                    return f"{symbol}({arguments[0]}, {arguments[1]})"
                return f"({arguments[0]} {symbol} {arguments[1]})"
            if name == "abs" and len(arguments) == 1:
                return f"|{arguments[0]}|"
            if name == "square" and len(arguments) == 1:
                return f"({arguments[0]})²"
            if name == "cube" and len(arguments) == 1:
                return f"({arguments[0]})³"
            label = _UNARY_LABELS.get(name, name)
            return f"{label}({', '.join(arguments)})"
        return ast.unparse(node)

    return render(expression)
