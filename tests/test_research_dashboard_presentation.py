from lotto_pattern_lab.research_dashboard_presentation import (
    display_engine_names,
    format_round_timestamp,
    round_grid_column_order,
)


def test_round_timestamp_converts_utc_to_seoul_and_handles_legacy_naive() -> None:
    assert format_round_timestamp("2026-10-21T13:54:27.455066+00:00") == "2026-10-21 22:54:27"
    assert format_round_timestamp("2026-10-21T13:54:27.455066") == "2026-10-21 22:54:27"
    assert format_round_timestamp(None) == "미수집"


def test_engine_labels_are_display_only_and_compact() -> None:
    assert display_engine_names(["intermittent_gp"]) == "GP"
    assert display_engine_names(["pysr_symbolic_regression"]) == "PySR"
    assert display_engine_names(["pysr_symbolic_regression", "intermittent_gp"]) == "GP + PySR"
    assert display_engine_names([]) == "미수집"


def test_round_grid_places_key_metrics_first_and_keeps_other_columns() -> None:
    columns = ["연구 ID", "2등 통계 상태", "완료 실험", "연구 회차", "실행 일시", "기타"]
    ordered = round_grid_column_order(columns)
    assert ordered[:5] == ["연구 회차", "실행 일시", "완료 실험", "연구 ID", "2등 통계 상태"]
    assert set(ordered) == set(columns)
