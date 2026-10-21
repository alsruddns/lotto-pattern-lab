"""Defensive, non-persistent normalization for research dashboard records."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Any

_UNKNOWN_SUMMARY = {
    "completed_experiment_count": None,
    "failed_experiment_count": None,
    "candidate_count": None,
    "new_pattern_count": None,
    "duplicate_pattern_count": None,
    "inherited_pattern_count": None,
    "child_pattern_count": None,
    "exact_match_count": None,
    "model_metrics": {},
    "paired_metrics": {},
    "hit_totals": {},
    "unique_hit_round_sets": {},
    "unique_hit_totals": {},
    "p_value_distributions": {},
    "experiment_summaries": [],
}

_PATTERN_COUNT_KEYS = {
    "new_pattern_count",
    "duplicate_pattern_count",
    "registration_attempt_count",
    "child_pattern_count",
    "new_child_pattern_count",
    "modified_child_pattern_count",
    "new_modified_child_pattern_count",
    "parent_pattern_count",
}


def normalize_dashboard_summary(value: Any, *, missing: bool = False) -> tuple[dict, str | None]:
    """Return a safe copy and a diagnostic; missing aggregate values stay unknown."""
    diagnostic = None
    parsed = value
    if missing:
        parsed = None
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            parsed = None
            diagnostic = f"summary JSON 파싱 실패: {error.msg} (위치 {error.pos})"

    if parsed is None:
        summary = copy.deepcopy(_UNKNOWN_SUMMARY)
        if diagnostic is None:
            diagnostic = "summary가 아직 저장되지 않아 집계 대기 상태입니다."
    elif not isinstance(parsed, Mapping):
        summary = copy.deepcopy(_UNKNOWN_SUMMARY)
        diagnostic = f"summary 자료형이 dict가 아닙니다: {type(parsed).__name__}"
    else:
        summary = copy.deepcopy(dict(parsed))
        for key, default in _UNKNOWN_SUMMARY.items():
            summary.setdefault(key, copy.deepcopy(default))
        if (
            not summary
            or "completed_experiment_count" not in summary
            or summary.get("completed_experiment_count") is None
        ):
            diagnostic = "summary 집계 항목이 없거나 불완전하여 집계 대기 상태입니다."

    for key, expected_type in (
        ("model_metrics", dict),
        ("paired_metrics", dict),
        ("hit_totals", dict),
        ("unique_hit_round_sets", dict),
        ("unique_hit_totals", dict),
        ("p_value_distributions", dict),
        ("experiment_summaries", list),
    ):
        if key not in summary:
            summary[key] = expected_type()
        elif not isinstance(summary[key], expected_type):
            if diagnostic is None:
                diagnostic = (
                    f"summary.{key} 자료형이 {expected_type.__name__}가 아닙니다: "
                    f"{type(summary[key]).__name__}"
                )
            summary[key] = expected_type()

    for key, value in list(summary["model_metrics"].items()):
        if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
            if diagnostic is None:
                diagnostic = f"summary.model_metrics[{key!r}]의 형식이 올바르지 않습니다."
            summary["model_metrics"][key] = []
        else:
            for item in value:
                for metric, metric_value in list(item.items()):
                    if (
                        metric.endswith("_count")
                        and metric_value is not None
                        and (
                            isinstance(metric_value, bool)
                            or not isinstance(metric_value, (int, float))
                            or metric_value < 0
                        )
                    ):
                        if diagnostic is None:
                            diagnostic = (
                                f"summary.model_metrics[{key!r}].{metric} 값이 올바르지 않습니다."
                            )
                        item[metric] = None
                    elif (
                        metric.endswith(("_score", "_rate", "_p_value"))
                        and metric_value is not None
                        and not isinstance(metric_value, (int, float))
                    ):
                        if diagnostic is None:
                            diagnostic = (
                                f"summary.model_metrics[{key!r}].{metric} 값이 숫자가 아닙니다."
                            )
                        item[metric] = None
    if any(not isinstance(item, dict) for item in summary["experiment_summaries"]):
        if diagnostic is None:
            diagnostic = "summary.experiment_summaries에 dict가 아닌 항목이 있습니다."
        summary["experiment_summaries"] = [
            item for item in summary["experiment_summaries"] if isinstance(item, dict)
        ]
    if any(not isinstance(item, dict) for item in summary["hit_totals"].values()):
        if diagnostic is None:
            diagnostic = "summary.hit_totals에 dict가 아닌 항목이 있습니다."
        summary["hit_totals"] = {}
    elif any(
        isinstance(count, bool) or not isinstance(count, (int, float)) or count < 0
        for group in summary["hit_totals"].values()
        for count in group.values()
    ):
        if diagnostic is None:
            diagnostic = "summary.hit_totals에 올바르지 않은 집계 값이 있습니다."
        summary["hit_totals"] = {}
    for key in ("paired_metrics", "p_value_distributions"):
        for model, value in list(summary[key].items()):
            if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
                if diagnostic is None:
                    diagnostic = f"summary.{key}[{model!r}] 형식이 올바르지 않습니다."
                summary[key][model] = []
    for model, phases in list(summary["unique_hit_totals"].items()):
        if not isinstance(phases, dict):
            if diagnostic is None:
                diagnostic = "summary.unique_hit_totals 형식이 올바르지 않습니다."
            summary["unique_hit_totals"][model] = {}
            continue
        for phase, counts in list(phases.items()):
            if not isinstance(counts, dict):
                if diagnostic is None:
                    diagnostic = "summary.unique_hit_totals의 구간 집계 형식이 올바르지 않습니다."
                phases[phase] = {}
    for item in summary["experiment_summaries"]:
        for key in ("models", "p_values"):
            if key in item and not isinstance(item[key], dict):
                if diagnostic is None:
                    diagnostic = f"summary.experiment_summaries.{key} 형식이 올바르지 않습니다."
                item[key] = {}
        for model, metrics in list(item.get("models", {}).items()):
            if not isinstance(metrics, dict):
                if diagnostic is None:
                    diagnostic = "실험별 모델 집계 형식이 올바르지 않습니다."
                item["models"][model] = {}
            elif not isinstance(metrics.get("match_distribution", {}), dict):
                if diagnostic is None:
                    diagnostic = "실험별 적중 분포 집계 형식이 올바르지 않습니다."
                metrics["match_distribution"] = {}

    summary["summary_state"] = "available" if diagnostic is None else "pending_or_invalid"
    if diagnostic:
        summary["summary_diagnostic"] = diagnostic
    return summary, diagnostic


def validated_pattern_counts(value: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Validate library aggregation shape and values before merging it into a summary."""
    if not isinstance(value, dict):
        return None, f"패턴 집계 결과가 dict가 아닙니다: {type(value).__name__}"
    if not _PATTERN_COUNT_KEYS <= value.keys() or "lineage_event_counts" not in value:
        return None, "패턴 집계 결과에 필수 항목이 없습니다."
    for key in _PATTERN_COUNT_KEYS:
        count = value[key]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return None, f"패턴 집계 {key} 값이 올바른 0 이상의 정수가 아닙니다."
    lineage = value["lineage_event_counts"]
    if not isinstance(lineage, dict) or any(
        not isinstance(key, str)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for key, count in lineage.items()
    ):
        return None, "패턴 계보 집계의 형식 또는 값이 올바르지 않습니다."
    return copy.deepcopy(value), None


def dashboard_failure_records(record: Mapping[str, Any]) -> list[dict[str, Any]] | int | None:
    """Prefer experiment failures, then expose worker-start failures without counting trials."""
    failures = record.get("failed_runs")
    if isinstance(failures, list) and failures:
        return copy.deepcopy(failures)
    if isinstance(failures, int) and not isinstance(failures, bool) and failures > 0:
        return failures
    startup_failures = record.get("failed_run_summary")
    if isinstance(startup_failures, list):
        return copy.deepcopy(startup_failures)
    return failures if isinstance(failures, int) else None
