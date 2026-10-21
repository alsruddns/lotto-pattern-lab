from __future__ import annotations

from pathlib import Path

import pyarrow as pa
from streamlit.testing.v1 import AppTest

from lotto_pattern_lab.continuous_research_view import _format_comparison_rows
from lotto_pattern_lab.research_queue import ResearchQueueStore


def test_queue_edit_comparison_values_are_arrow_safe_strings() -> None:
    rows = _format_comparison_rows(
        [
            {
                "설정": "이전 값",
                "기존": 12,
                "변경": "열두 개",
            },
            {
                "설정": "선택 값",
                "기존": None,
                "변경": True,
            },
            {
                "설정": "복합 값",
                "기존": [1, "둘"],
                "변경": {"workers": 4, "enabled": False},
            },
        ]
    )

    table = pa.Table.from_pylist(rows)

    assert table.schema.field("기존").type == pa.string()
    assert table.schema.field("변경").type == pa.string()
    assert rows[0]["기존"] == "12"
    assert rows[1]["기존"] == "해당 없음"
    assert rows[1]["변경"] == "예"
    assert rows[2]["기존"] == '[1, "둘"]'
    assert '"workers": 4' in rows[2]["변경"]


def test_reservation_saved_toast_renders_without_streamlit_error(tmp_path: Path) -> None:
    script = tmp_path / "reservation_feedback_app.py"
    script.write_text(
        "from lotto_pattern_lab.continuous_research_view import "
        "_show_reservation_saved_feedback\n"
        "_show_reservation_saved_feedback()\n",
        encoding="utf-8",
    )

    app = AppTest.from_file(script).run(timeout=30)

    assert not app.exception, [exception.message for exception in app.exception]


def test_consecutive_reservations_keep_independent_duration_snapshots(tmp_path: Path) -> None:
    store = ResearchQueueStore(tmp_path / "queue.sqlite")
    queue_id = store.create_queue()
    common_config = {
        "engine": ["intermittent_gp", "pysr_symbolic_regression"],
        "feature_profile": "bonus_lag_v1",
        "research_mode": "cumulative",
        "population_size": 48,
        "pysr": {"niterations": 30},
    }

    first_id = store.add_item(
        queue_id,
        engine="comparison",
        feature_profile="bonus_lag_v1",
        duration_seconds=300,
        config=common_config,
        request_key="reservation-one",
    )
    second_id = store.add_item(
        queue_id,
        engine="comparison",
        feature_profile="bonus_lag_v1",
        duration_seconds=7200,
        config=common_config,
        request_key="reservation-two",
    )

    items = store.list_items(queue_id)

    assert first_id != second_id
    assert [item["duration_seconds"] for item in items] == [300, 7200]
    assert items[0]["config"] == items[1]["config"] == common_config
