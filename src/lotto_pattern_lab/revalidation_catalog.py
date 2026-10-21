"""Lightweight read-only catalog for selecting a saved experiment."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

UNKNOWN = "\uc815\ubcf4 \uc5c6\uc74c"
NOT_COLLECTED = "\ubbf8\uc218\uc9d1"


def _engine_label(value: Any) -> str:
    normalized = str(value or "").lower()
    if normalized in {"intermittent_gp", "gp"}:
        return "GP"
    if normalized in {"pysr_symbolic_regression", "pysr"}:
        return "PySR"
    if "intermittent_gp" in normalized and "pysr_symbolic_regression" in normalized:
        return "GP + PySR"
    return str(value) if value else UNKNOWN


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _rate(count: Any, denominator: Any) -> float | None:
    numerator = _number(count)
    total = _number(denominator)
    if numerator is None or total is None or total <= 0:
        return None
    return numerator / total


def build_experiment_catalog(
    database_path: str | Path, experiment_directory: str | Path
) -> list[dict[str, Any]]:
    """Join existing experiment filenames to compact summaries using read-only SQLite."""
    root = Path(experiment_directory)
    available_ids = {path.stem for path in root.glob("*.json")} if root.is_dir() else set()
    database = Path(database_path).resolve()
    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=0.25)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        query = (
            "SELECT r.round_number,r.research_id, "
            "COALESCE(json_extract(r.metadata_json,'$.started_at'),r.created_at) AS research_date, "
            "COALESCE(json_extract(r.metadata_json,'$.config.feature_profile'),?) AS feature_profile, "  # noqa: E501
            "json_extract(e.value,'$.experiment_id') AS experiment_id, "
            "json_extract(e.value,'$.engine') AS engine, "
            "json_extract(e.value,'$.status') AS status, "
            "json_extract(e.value,'$.models.six_number_set.independent_test_activation_count') AS evaluation_count, "  # noqa: E501
            "json_extract(e.value,'$.models.six_number_set.independent_test_target_round_count') AS target_round_count, "  # noqa: E501
            "json_extract(e.value,'$.models.six_number_set.independent_test_score') AS independent_test_score, "  # noqa: E501
            'json_extract(e.value,\'$.hits."six_number_set:independent_test"."3_plus"\') AS hits_3_plus, '  # noqa: E501
            'json_extract(e.value,\'$.hits."six_number_set:independent_test"."4_plus"\') AS hits_4_plus, '  # noqa: E501
            'json_extract(e.value,\'$.hits."six_number_set:independent_test"."5_plus"\') AS hits_5_plus, '  # noqa: E501
            'json_extract(e.value,\'$.hits."six_number_set:independent_test"."6_exact"\') AS hits_6_exact, '  # noqa: E501
            'json_extract(e.value,\'$.hits."six_number_set:independent_test"."2_exact"\') AS second_prize, '  # noqa: E501
            'json_extract(e.value,\'$.hits."six_number_set:independent_test"."2_known"\') AS second_prize_known, '  # noqa: E501
            "json_extract(e.value,'$.p_values.six_number_set') AS random_comparison_p "
            "FROM research_rounds AS r, json_each(r.metadata_json,'$.dashboard_summary.experiment_summaries') AS e "  # noqa: E501
            "WHERE json_extract(e.value,'$.experiment_id') IS NOT NULL "
            "ORDER BY r.round_number DESC"
        )
        rows = connection.execute(query, (UNKNOWN,)).fetchall()
    finally:
        connection.close()

    catalog: dict[str, dict[str, Any]] = {}
    for row in rows:
        experiment_id = str(row["experiment_id"] or "")
        if not experiment_id or experiment_id not in available_ids:
            continue
        denominator = row["evaluation_count"]
        if denominator is None:
            denominator = row["target_round_count"]
        second_denominator = row["second_prize_known"]
        second_count = row["second_prize"]
        if _number(second_denominator) is None or _number(second_denominator) <= 0:
            second_count = None
        catalog[experiment_id] = {
            "experiment_id": experiment_id,
            "research_id": str(row["research_id"]),
            "research_round": int(row["round_number"]),
            "research_date": str(row["research_date"] or "")[:10] or None,
            "engine": _engine_label(row["engine"]),
            "feature_profile": row["feature_profile"] or UNKNOWN,
            "status": row["status"] or UNKNOWN,
            "validation_status": (
                "\ub3c5\ub9bd \ud14c\uc2a4\ud2b8 \uacb0\uacfc \uc218\uc9d1"
                if _number(denominator) is not None
                else "\uac80\uc99d \uc815\ubcf4 \ubbf8\uc218\uc9d1"
            ),
            "hits_3_plus": _number(row["hits_3_plus"]),
            "hits_4_plus": _number(row["hits_4_plus"]),
            "hits_5_plus": _number(row["hits_5_plus"]),
            "second_prize": _number(second_count),
            "second_prize_known": _number(second_denominator),
            "hits_6_exact": _number(row["hits_6_exact"]),
            "evaluation_count": _number(denominator),
            "rate_4_plus": _rate(row["hits_4_plus"], denominator),
            "rate_5_plus": _rate(row["hits_5_plus"], denominator),
            "second_prize_rate": _rate(second_count, second_denominator),
            "rate_6_exact": _rate(row["hits_6_exact"], denominator),
            "independent_test_score": _number(row["independent_test_score"]),
            "random_comparison_p": _number(row["random_comparison_p"]),
            "independent_test_available": _number(denominator) is not None,
            "file_available": True,
        }

    for experiment_id in available_ids - catalog.keys():
        catalog[experiment_id] = {
            "experiment_id": experiment_id,
            "research_id": None,
            "research_round": None,
            "research_date": None,
            "engine": UNKNOWN,
            "feature_profile": UNKNOWN,
            "status": UNKNOWN,
            "validation_status": "\uc694\uc57d \uc815\ubcf4 \ubbf8\uc218\uc9d1",
            "hits_3_plus": None,
            "hits_4_plus": None,
            "hits_5_plus": None,
            "second_prize": None,
            "second_prize_known": None,
            "hits_6_exact": None,
            "evaluation_count": None,
            "rate_4_plus": None,
            "rate_5_plus": None,
            "second_prize_rate": None,
            "rate_6_exact": None,
            "independent_test_score": None,
            "random_comparison_p": None,
            "independent_test_available": False,
            "file_available": True,
        }
    return list(catalog.values())


def experiment_choice_label(item: dict[str, Any]) -> str:
    """Display evidence while preserving experiment ID as the selectbox value."""

    def count(key: str) -> str:
        value = item.get(key)
        return NOT_COLLECTED if value is None else f"{int(value):,}\uac74"

    round_label = item.get("research_round")
    date_label = item.get("research_date") or "\ub0a0\uc9dc \uc815\ubcf4 \uc5c6\uc74c"
    round_text = (
        str(round_label) if round_label is not None else "\ud68c\ucc28 \uc815\ubcf4 \uc5c6\uc74c"
    )
    return (
        f"{round_text}\ud68c\ucc28 \u00b7 "
        f"4\uac1c+ {count('hits_4_plus')} \u00b7 5\uac1c+ {count('hits_5_plus')} \u00b7 "
        f"2\ub4f1 {count('second_prize')} \u00b7 {item.get('engine', UNKNOWN)} \u00b7 "
        f"{item.get('feature_profile', UNKNOWN)} \u00b7 "
        f"{item.get('validation_status', UNKNOWN)} \u00b7 {date_label}"
        f" \u00b7 {item.get('experiment_id', '')[:8]}"
    )


def filter_experiment_catalog(
    catalog: list[dict[str, Any]],
    *,
    engine: str | None = None,
    feature_profile: str | None = None,
    validation_status: str | None = None,
    independent_test: bool | None = None,
    hit_key: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[dict[str, Any]]:
    """Filter choice rows without changing the saved experiment records."""
    filtered = []
    for item in catalog:
        if engine and item.get("engine") != engine:
            continue
        if feature_profile and item.get("feature_profile") != feature_profile:
            continue
        if validation_status and item.get("validation_status") != validation_status:
            continue
        if (
            independent_test is not None
            and item.get("independent_test_available") != independent_test
        ):
            continue
        if hit_key and (item.get(hit_key) is None or item[hit_key] <= 0):
            continue
        if item.get("research_date"):
            item_date = date.fromisoformat(item["research_date"])
            if start_date and item_date < start_date:
                continue
            if end_date and item_date > end_date:
                continue
        elif start_date or end_date:
            continue
        filtered.append(item)
    return filtered


def sort_experiment_catalog(
    catalog: list[dict[str, Any]], *, key: str, descending: bool = True
) -> list[dict[str, Any]]:
    """Sort known values first; leave records with unknown denominators at the end."""
    known = [item for item in catalog if item.get(key) is not None]
    unknown = [item for item in catalog if item.get(key) is None]
    known.sort(key=lambda item: item[key], reverse=descending)
    return known + unknown
