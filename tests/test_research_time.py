from __future__ import annotations

from datetime import UTC, datetime, timedelta

from lotto_pattern_lab.research_time import (
    checkpoint_status_label,
    format_hhmmss,
    research_time_snapshot,
)


def test_hhmmss_formats_minutes_hours_and_more_than_one_day() -> None:
    assert format_hhmmss(166) == "00:02:46"
    assert format_hhmmss(3600) == "01:00:00"
    assert format_hhmmss(25 * 3600 + 7) == "25:00:07"
    assert format_hhmmss(72 * 3600 + 4 * 60 + 9) == "72:04:09"


def test_remaining_time_is_clamped_at_zero() -> None:
    elapsed, remaining = research_time_snapshot(
        {"status": "completed", "total_elapsed_seconds": 80, "target_duration_seconds": 60}
    )

    assert elapsed == 80
    assert remaining == 0


def test_running_time_advances_from_session_start_and_preserves_baseline() -> None:
    now = datetime(2026, 10, 20, 12, 0, 10, tzinfo=UTC)
    state = {
        "status": "running",
        "total_elapsed_seconds": 100,
        "session_elapsed_base_seconds": 100,
        "session_started_at": (now - timedelta(seconds=10)).isoformat(),
        "target_duration_seconds": 3600,
    }

    elapsed, remaining = research_time_snapshot(state, now=now)

    assert elapsed == 110
    assert remaining == 3490


def test_inactive_worker_time_freezes_at_last_saved_checkpoint() -> None:
    now = datetime(2026, 10, 20, 12, 0, 10, tzinfo=UTC)
    state = {
        "status": "running",
        "total_elapsed_seconds": 100,
        "session_elapsed_base_seconds": 100,
        "session_started_at": (now - timedelta(seconds=10)).isoformat(),
        "target_duration_seconds": 3600,
    }

    elapsed, remaining = research_time_snapshot(state, now=now, worker_active=False)
    assert elapsed == 100
    assert remaining == 3500


def test_completed_time_stays_fixed_and_uses_finish_time_when_elapsed_is_missing() -> None:
    now = datetime(2026, 10, 20, 12, 0, 10, tzinfo=UTC)
    state = {
        "status": "completed",
        "total_elapsed_seconds": 90061,
        "started_at": (now - timedelta(days=2)).isoformat(),
        "finished_at": (now - timedelta(seconds=5)).isoformat(),
        "target_duration_seconds": 100,
    }

    elapsed, remaining = research_time_snapshot(state, now=now)
    assert elapsed == 90061
    assert remaining == 0

    del state["total_elapsed_seconds"]
    elapsed, remaining = research_time_snapshot(state, now=now)
    assert elapsed == 172795
    assert remaining == 0


def test_unlimited_research_has_no_remaining_time() -> None:
    elapsed, remaining = research_time_snapshot(
        {"status": "running", "total_elapsed_seconds": 12, "target_duration_seconds": 0}
    )

    assert elapsed == 12
    assert remaining is None


def test_paused_research_does_not_advance_between_refreshes() -> None:
    now = datetime(2026, 10, 20, 12, 0, 10, tzinfo=UTC)
    state = {
        "status": "paused",
        "total_elapsed_seconds": 42,
        "session_started_at": (now - timedelta(minutes=10)).isoformat(),
        "session_finished_at": (now - timedelta(seconds=5)).isoformat(),
        "target_duration_seconds": 3600,
    }

    elapsed, remaining = research_time_snapshot(state, now=now)
    assert elapsed == 42
    assert remaining == 3558


def test_checkpoint_status_distinguishes_waiting_from_saved() -> None:
    assert checkpoint_status_label(False) == "첫 체크포인트 저장 대기 중"
    assert checkpoint_status_label(True, "2026-10-20T12:00:00+00:00") == (
        "저장 완료 · 2026-10-20T12:00:00+00:00"
    )
    assert checkpoint_status_label(True) == "저장 완료 · 시각 기록 없음"
