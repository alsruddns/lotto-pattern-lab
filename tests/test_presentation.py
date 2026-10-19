import math

import pytest

from lotto_pattern_lab.presentation import (
    format_count,
    format_decimal,
    format_duration,
    format_percent,
    format_probability,
    pretty_formula,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.118318318, "0.1183점"),
        (0, "0.0000"),
        (None, "미수집"),
        (math.nan, "계산 불가"),
        ("미수집", "미수집"),
    ],
)
def test_format_decimal_distinguishes_zero_and_missing(value, expected):
    suffix = "점" if value == 0.118318318 else ""
    assert format_decimal(value, suffix=suffix) == expected


def test_formatters_use_consistent_korean_display():
    assert format_percent(0.6979) == "69.79%"
    assert format_probability(0.166666666) == "0.1667"
    assert format_count(242880) == "242,880"
    assert format_count(0) == "0"
    assert format_duration(4423) == "1시간 13분 43초"
    assert format_duration(None) == "미수집"


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("mul(x60, x21)", "(x60 × x21)"),
        ("add(x0, square(x2))", "(x0 + (x2)²)"),
        ("safe_div(sub(x1, 2), abs(x3))", "((x1 − 2) ÷ |x3|)"),
        ("unknown(x0)", "unknown(x0)"),
        ("mul(x0,", "mul(x0,"),
    ],
)
def test_pretty_formula_is_read_only_and_safe(formula, expected):
    assert pretty_formula(formula) == expected
