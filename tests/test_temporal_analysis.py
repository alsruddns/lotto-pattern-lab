from __future__ import annotations

import pytest

from lotto_pattern_lab.temporal_analysis import analyze_temporal_performance


def test_temporal_windows_track_active_hits_and_inactive_rounds_separately() -> None:
    rows = [
        {
            "round_number": round_number,
            "execution_success": True,
            "validation_status": "OK" if active else "INACTIVE",
            "activation_condition_met": active,
            "match_count": hits if active else None,
        }
        for round_number, active, hits in (
            (10, True, 1),
            (11, False, None),
            (12, True, 1),
            (13, True, 5),
            (14, False, None),
            (15, True, 3),
        )
    ]
    rows.append(
        {
            "round_number": 16,
            "execution_success": False,
            "validation_status": "INPUT_UNAVAILABLE",
            "activation_condition_met": None,
            "match_count": None,
        }
    )

    report = analyze_temporal_performance(rows, window_size=3, step=2)

    assert report["evaluable_round_count"] == 6
    assert report["window_count"] == 2
    assert report["windows"][0]["gate_activation_rate"] == pytest.approx(2 / 3)
    assert report["windows"][0]["average_hits_when_active"] == 1
    assert report["windows"][0]["average_hits_per_target"] == pytest.approx(2 / 3)
    assert report["windows"][1]["at_least_3"] == 1
    assert report["descriptive_change_points"][0]["descriptive_candidate"] is True


def test_temporal_analysis_returns_empty_for_short_ranges_and_rejects_invalid_windows() -> None:
    report = analyze_temporal_performance(
        [{"round_number": 1, "validation_status": "OK", "match_count": 0}],
        window_size=10,
    )

    assert report["windows"] == []
    assert report["descriptive_change_points"] == []
    with pytest.raises(ValueError, match="2회 이상"):
        analyze_temporal_performance([], window_size=1)
