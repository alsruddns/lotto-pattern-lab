"""Formatting helpers for the research-round overview grid."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

SEOUL = ZoneInfo("Asia/Seoul")


def format_round_timestamp(value: object) -> str:
    """Format stored UTC timestamps for the Korean dashboard without altering storage.

    Round timestamps are written with ``datetime.now(UTC).isoformat()``. Legacy
    naive values therefore follow the same UTC convention.
    """
    if not isinstance(value, str) or not value.strip():
        return "미수집"
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return "미수집"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(SEOUL).strftime("%Y-%m-%d %H:%M:%S")


def display_engine_names(engines: object) -> str:
    """Map persisted engine identifiers to compact labels for display only."""
    if not isinstance(engines, (list, tuple, set)):
        return "미수집"
    values = {str(item).strip() for item in engines if str(item).strip()}
    labels = []
    if "intermittent_gp" in values:
        labels.append("GP")
    if "pysr_symbolic_regression" in values:
        labels.append("PySR")
    labels.extend(sorted(values - {"intermittent_gp", "pysr_symbolic_regression"}))
    return " + ".join(labels) if labels else "미수집"


def round_grid_column_order(columns: list[str]) -> list[str]:
    """Keep key round metrics first and preserve every remaining existing column."""
    priority = [
        "연구 회차",
        "실행 일시",
        "연구 시간",
        "상태",
        "집계 상태",
        "완료 실험",
        "실패 실험",
        "신규 패턴",
        "3개 이상 적중 (반복 평가)",
        "4개 이상 적중 (반복 평가)",
        "5개 이상 적중 (반복 평가)",
        "2등 일치",
        "6개 완전 일치 (반복 평가)",
    ]
    ordered = [column for column in priority if column in columns]
    ordered.extend(column for column in columns if column not in ordered)
    return ordered
