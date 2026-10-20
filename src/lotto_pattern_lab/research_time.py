"""Elapsed and remaining time calculations for continuous research runs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def format_hhmmss(seconds: object) -> str:
    """Format a duration without wrapping after 24 hours."""
    try:
        total = max(0, int(float(seconds)))
    except (TypeError, ValueError, OverflowError):
        total = 0
    hours, remainder = divmod(total, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"


def checkpoint_status_label(saved: bool, last_updated_at: object = None) -> str:
    if not saved:
        return "첫 체크포인트 저장 대기 중"
    timestamp = last_updated_at if isinstance(last_updated_at, str) else "시각 기록 없음"
    return f"저장 완료 · {timestamp}"


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def research_time_snapshot(
    state: dict[str, Any], *, now: datetime | None = None, worker_active: bool | None = None
) -> tuple[float, float | None]:
    """Return elapsed and remaining seconds using checkpointed session timing."""
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    now = now.astimezone(UTC)

    elapsed_recorded = state.get("total_elapsed_seconds")
    if elapsed_recorded is None:
        elapsed_recorded = state.get("elapsed_seconds")
    try:
        elapsed = max(0.0, float(elapsed_recorded or 0))
    except (TypeError, ValueError, OverflowError):
        elapsed = 0.0

    status = state.get("status")
    if elapsed_recorded is None and status not in {"running", "pause_requested"}:
        started = _timestamp(state.get("started_at"))
        finished = _timestamp(state.get("finished_at"))
        if started is not None and finished is not None:
            elapsed = max(0.0, (finished - started).total_seconds())
    if status in {"running", "pause_requested"} and worker_active is not False:
        session_started = _timestamp(state.get("session_started_at"))
        if session_started is not None:
            try:
                baseline = max(0.0, float(state.get("session_elapsed_base_seconds", elapsed)))
            except (TypeError, ValueError, OverflowError):
                baseline = elapsed
            elapsed = baseline + max(0.0, (now - session_started).total_seconds())
        elif state.get("session_started_at") is None:
            # Older checkpoints lack session timing metadata; advance from their last write.
            last_updated = _timestamp(state.get("last_updated_at"))
            if last_updated is not None:
                elapsed += max(0.0, (now - last_updated).total_seconds())

    try:
        target = float(state.get("target_duration_seconds", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        target = 0.0
    remaining = max(0.0, target - elapsed) if target > 0 else None
    return elapsed, remaining
