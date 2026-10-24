import math

import pytest

from lotto_pattern_lab.presentation import (
    format_count,
    format_decimal,
    format_duration,
    format_percent,
    format_probability,
    format_research_round,
    pretty_formula,
)


def test_queue_research_round_display_is_arrow_compatible_and_keeps_ids() -> None:
    import pandas as pd
    import pyarrow as pa

    records = [
        {"research_id": "round-two", "research_round": 2},
        {"research_id": "unassigned", "research_round": None},
        {"research_id": "round-ten", "research_round": 10},
    ]
    ordered = sorted(
        records,
        key=lambda item: (
            item["research_round"] is None,
            item["research_round"] if item["research_round"] is not None else 0,
        ),
    )
    display_rows = [
        {
            "연구 ID": item["research_id"],
            "연구 회차": format_research_round(item["research_round"]),
        }
        for item in ordered
    ]

    table = pa.Table.from_pandas(pd.DataFrame(display_rows), preserve_index=False)

    assert pa.types.is_string(table.schema.field("연구 회차").type) or pa.types.is_large_string(
        table.schema.field("연구 회차").type
    )
    assert table.column("연구 회차").to_pylist() == ["2", "10", "미배정"]
    assert table.column("연구 ID").to_pylist() == [
        "round-two",
        "round-ten",
        "unassigned",
    ]


@pytest.mark.parametrize(
    ("value", "missing_label", "expected"),
    [
        (3, "미수집", "3"),
        ("미수집", "미수집", "미수집"),
        (None, "미배정", "미배정"),
        (None, "미수집", "미수집"),
        ("7", "미수집", "7"),
        ("", "미수집", "미수집"),
        (float("nan"), "미수집", "미수집"),
    ],
)
def test_research_round_display_is_arrow_safe(value, missing_label, expected):
    import pandas as pd
    import pyarrow as pa

    table = pa.Table.from_pandas(
        pd.DataFrame(
            [{"연구 회차": format_research_round(value, missing_label=missing_label)}]
        ),
        preserve_index=False,
    )

    arrow_type = table.schema.field("연구 회차").type
    assert pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type)
    assert table.column("연구 회차").to_pylist() == [expected]


def test_research_round_display_keeps_numeric_order_and_handles_empty_data():
    import pandas as pd
    import pyarrow as pa

    records = [{"research_round": 10}, {"research_round": 2}, {"research_round": None}]
    records.sort(
        key=lambda row: (
            row["research_round"] is None,
            row["research_round"] if row["research_round"] is not None else 0,
        )
    )
    frame = pd.DataFrame(
        [
            {"연구 회차": format_research_round(row["research_round"])}
            for row in records
        ]
    )
    table = pa.Table.from_pandas(frame, preserve_index=False)
    empty_table = pa.Table.from_pandas(
        pd.DataFrame(columns=["연구 회차"]), preserve_index=False
    )

    assert table.column("연구 회차").to_pylist() == ["2", "10", "미배정"]
    assert empty_table.num_rows == 0


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
