"""Descriptive rolling-window summaries for historical formula replay rows."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

VALID_STATUSES = {"OK", "INACTIVE"}


def analyze_temporal_performance(
    rows: Iterable[dict[str, Any]], *, window_size: int = 50, step: int | None = None
) -> dict[str, Any]:
    """Measure rolling outcomes without promoting retrospective change points to proof."""
    if window_size < 2:
        raise ValueError("구간 크기는 2회 이상이어야 합니다.")
    step_size = step if step is not None else max(1, window_size // 2)
    if step_size < 1:
        raise ValueError("구간 이동 간격은 1 이상이어야 합니다.")
    ordered = sorted(
        (
            row
            for row in rows
            if row.get("validation_status") in VALID_STATUSES
            and row.get("execution_success", True)
            and row.get("round_number") is not None
        ),
        key=lambda row: int(row["round_number"]),
    )
    summaries = []
    start_index = 0
    while start_index + window_size <= len(ordered):
        window = ordered[start_index : start_index + window_size]
        active = [
            row
            for row in window
            if row.get("activation_condition_met") is True and row.get("match_count") is not None
        ]
        hits = [int(row["match_count"]) for row in active]
        targets = [int(row["match_count"] or 0) for row in window]
        target_count = len(window)
        active_count = len(active)
        summaries.append(
            {
                "start_round": int(window[0]["round_number"]),
                "end_round": int(window[-1]["round_number"]),
                "target_round_count": target_count,
                "active_prediction_count": active_count,
                "gate_activation_rate": active_count / target_count,
                "average_hits_per_target": sum(targets) / target_count,
                "average_hits_when_active": sum(hits) / active_count if active_count else None,
                "at_least_3": sum(value >= 3 for value in hits),
                "at_least_4": sum(value >= 4 for value in hits),
                "at_least_5": sum(value >= 5 for value in hits),
                "exactly_6": sum(value == 6 for value in hits),
                "at_least_3_rate_when_active": (
                    sum(value >= 3 for value in hits) / active_count if active_count else None
                ),
            }
        )
        start_index += step_size
    change_points = []
    for previous, current in zip(summaries, summaries[1:], strict=False):
        change_points.append(
            {
                "after_round": previous["end_round"],
                "before_round": current["start_round"],
                "average_hits_per_target_delta": (
                    current["average_hits_per_target"] - previous["average_hits_per_target"]
                ),
                "active_rate_delta": (
                    current["gate_activation_rate"] - previous["gate_activation_rate"]
                ),
                "descriptive_candidate": abs(
                    current["average_hits_per_target"] - previous["average_hits_per_target"]
                )
                >= 0.5,
            }
        )
    return {
        "window_size": window_size,
        "step": step_size,
        "evaluable_round_count": len(ordered),
        "window_count": len(summaries),
        "windows": summaries,
        "descriptive_change_points": change_points,
        "interpretation": (
            "과거 구간을 선택해 계산한 기술 통계입니다. "
            "변화 후보는 사후 선택 편향을 포함할 수 있으며 "
            "추첨 원인이나 미래 성능의 증거가 아닙니다."
        ),
    }
