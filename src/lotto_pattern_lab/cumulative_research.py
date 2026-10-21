"""Evidence-bounded helpers for cumulative research and mode comparisons."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Iterable
from typing import Any

from .research_optimization_apply import validate_optimization_compatibility

INDEPENDENT_MODE = "independent"
CUMULATIVE_MODE = "cumulative"
LEGACY_MODE = "legacy"
RESEARCH_MODES = {INDEPENDENT_MODE, CUMULATIVE_MODE, LEGACY_MODE}


def recommend_optuna_settings(
    studies: Iterable[dict[str, Any]],
    *,
    feature_profile: str,
    dataset_range: list[int] | None,
    dataset_sha256: str | None,
    dataset_integrity: dict[str, Any] | None,
) -> dict[str, Any]:
    """Recommend a saved GP setting only for an exact dataset and validation policy match."""
    compatible: list[dict[str, Any]] = []
    rejected: list[str] = []
    for study in studies:
        if study.get("status") != "completed" or study.get("best_trial_number") is None:
            continue
        best_value = study.get("best_value")
        if not isinstance(best_value, (int, float)) or not math.isfinite(float(best_value)):
            continue
        if study.get("engine") != "intermittent_gp":
            continue
        if study.get("objective_policy_version") != "validation_only_v1":
            continue
        if study.get("final_test_used_for_selection") is not False:
            continue
        if not dataset_sha256 or study.get("dataset_snapshot_sha256") != dataset_sha256:
            rejected.append(str(study.get("study_name", "기록 없음")))
            continue
        parameters, reason = validate_optimization_compatibility(
            study,
            current_engines=("intermittent_gp",),
            current_dataset_range=dataset_range,
            dataset_integrity=dataset_integrity,
        )
        if reason or study.get("feature_profile") != feature_profile:
            continue
        compatible.append({**study, "normalized_parameters": parameters})
    if not compatible:
        return {
            "available": False,
            "reason": "추천할 데이터가 충분하지 않습니다.",
            "dataset_hash_missing_or_different": rejected,
        }
    best = max(
        compatible,
        key=lambda item: (
            float(item.get("best_value", float("-inf"))),
            str(item.get("study_name", "")),
        ),
    )
    return {
        "available": True,
        "reason": "동일 데이터 스냅샷과 특징 프로파일의 validation 전용 Optuna 기록입니다.",
        "study_name": best["study_name"],
        "trial_number": int(best["best_trial_number"]),
        "best_value": best.get("best_value"),
        "parameters": dict(best["normalized_parameters"]),
        "dataset_sha256": dataset_sha256,
        "feature_profile": feature_profile,
    }


def _known_sum(records: list[dict[str, Any]], key: str) -> int | None:
    values = [record.get("dashboard_summary", {}).get(key) for record in records]
    if not values or any(value is None for value in values):
        return None
    return sum(int(value) for value in values)


def cumulative_growth_summary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize recorded search volume and evidence without treating volume as accuracy."""
    rounds = list(records)
    elapsed = sum(float(record.get("total_elapsed_seconds", 0) or 0) for record in rounds)
    patterns = _known_sum(rounds, "new_pattern_count")
    inherited = _known_sum(rounds, "inherited_pattern_count")
    duplicates = _known_sum(rounds, "duplicate_pattern_count")
    modified = _known_sum(rounds, "new_modified_child_pattern_count")
    model_values: dict[str, dict[str, list[float]]] = {
        model: {metric: [] for metric in ("validation_score", "independent_test_score")}
        for model in ("one_number", "six_number_set")
    }
    random_p_values: dict[str, list[float]] = {
        "one_number": [],
        "six_number_set": [],
    }
    for record in rounds:
        for model, experiments in record.get("dashboard_summary", {}).get(
            "model_metrics", {}
        ).items():
            if model not in model_values:
                continue
            for experiment in experiments:
                for metric in model_values[model]:
                    value = experiment.get(metric)
                    if isinstance(value, (int, float)):
                        model_values[model][metric].append(float(value))
        for experiment in record.get("dashboard_summary", {}).get(
            "experiment_summaries", []
        ):
            for model, value in (experiment.get("p_values") or {}).items():
                if (
                    model in random_p_values
                    and isinstance(value, (int, float))
                    and math.isfinite(float(value))
                ):
                    random_p_values[model].append(float(value))
    means = {
        model: {
            metric: statistics.fmean(values) if values else None
            for metric, values in metrics.items()
        }
        for model, metrics in model_values.items()
    }
    efficiency_change = None
    ordered = sorted(rounds, key=lambda item: int(item.get("research_round", 0)))
    completed = [item for item in ordered if item.get("status") == "completed"]
    if completed:
        latest = completed[-1]
        latest_summary = latest.get("dashboard_summary", {})
        latest_runtime = float(latest.get("total_elapsed_seconds", 0) or 0)
        latest_new = latest_summary.get("new_pattern_count")
        if latest_runtime > 0 and latest_new is not None:
            from .research_rounds import compare_research_rounds

            for previous in reversed(completed[:-1]):
                if previous.get("research_mode", "legacy") != latest.get(
                    "research_mode", "legacy"
                ):
                    continue
                if not compare_research_rounds(previous, latest)["directly_comparable"]:
                    continue
                previous_runtime = float(previous.get("total_elapsed_seconds", 0) or 0)
                previous_new = previous.get("dashboard_summary", {}).get("new_pattern_count")
                if previous_runtime > 0 and previous_new is not None:
                    previous_rate = float(previous_new) / (previous_runtime / 3600)
                    latest_rate = float(latest_new) / (latest_runtime / 3600)
                    if previous_rate > 0:
                        efficiency_change = (latest_rate - previous_rate) / previous_rate
                    break
    return {
        "research_round_count": len(rounds),
        "candidate_count": _known_sum(rounds, "candidate_count"),
        "new_valid_formula_count": patterns,
        "reused_elite_formula_count": inherited,
        "modified_formula_count": modified,
        "duplicate_formula_count": duplicates,
        "runtime_seconds": elapsed,
        "new_valid_formulas_per_hour": (
            patterns / (elapsed / 3600)
            if patterns is not None and elapsed > 0
            else None
        ),
        "efficiency_change_vs_comparable_previous": efficiency_change,
        "mean_validation_scores": {
            model: metrics["validation_score"] for model, metrics in means.items()
        },
        "mean_independent_test_scores": {
            model: metrics["independent_test_score"] for model, metrics in means.items()
        },
        "random_control_median_p_values": {
            model: statistics.median(values) if values else None
            for model, values in random_p_values.items()
        },
        "growth_claim": "연구 횟수와 후보 수는 데이터 축적량이며 성능 향상을 뜻하지 않습니다.",
    }


def _mode_group_key(record: dict[str, Any]) -> tuple[Any, ...] | None:
    config = record.get("config", {})
    dataset_hash = record.get("dataset_sha256")
    if not dataset_hash:
        return None
    return (
        dataset_hash,
        tuple(record.get("engines", ())),
        config.get("feature_profile"),
        config.get("train_ratio"),
        config.get("validation_ratio"),
        config.get("max_lag"),
        config.get("population_size"),
        config.get("generations"),
        config.get("crossover_probability"),
        config.get("mutation_probability"),
        config.get("max_depth"),
        config.get("workers"),
        config.get("control_runs"),
        config.get("min_activation_rate"),
        config.get("hidden_states"),
        json.dumps(record.get("pysr_config") or {}, sort_keys=True),
        record.get("target_duration_seconds"),
    )


def compare_research_modes(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compare independent and cumulative runs only on shared data, settings, and seeds."""
    groups: dict[tuple[Any, ...], dict[str, dict[int, dict[str, Any]]]] = {}
    for record in records:
        mode = record.get("research_mode")
        key = _mode_group_key(record)
        if mode not in {INDEPENDENT_MODE, CUMULATIVE_MODE} or key is None:
            continue
        by_mode = groups.setdefault(key, {INDEPENDENT_MODE: {}, CUMULATIVE_MODE: {}})
        summary = record.get("dashboard_summary", {})
        for experiment in summary.get("experiment_summaries", []):
            seed = experiment.get("seed")
            if seed is None or experiment.get("status") not in {None, "completed"}:
                continue
            by_mode[mode][int(seed)] = experiment
    comparisons = []
    for key, by_mode in groups.items():
        common = sorted(set(by_mode[INDEPENDENT_MODE]) & set(by_mode[CUMULATIVE_MODE]))
        if len(common) < 2:
            continue
        differences = {metric: [] for metric in ("validation_score", "independent_test_score")}
        runtime_differences: list[float] = []
        candidate_differences: list[float] = []
        formula_differences: list[float] = []
        complexity_differences: list[float] = []
        for seed in common:
            independent_experiment = by_mode[INDEPENDENT_MODE][seed]
            cumulative_experiment = by_mode[CUMULATIVE_MODE][seed]
            for collection, key_name in (
                (runtime_differences, "runtime_seconds"),
                (candidate_differences, "candidate_count"),
                (formula_differences, "new_pattern_count"),
            ):
                new_value = cumulative_experiment.get(key_name)
                old_value = independent_experiment.get(key_name)
                if isinstance(new_value, (int, float)) and isinstance(old_value, (int, float)):
                    collection.append(float(new_value) - float(old_value))
            for metric in differences:
                independent_scores = [
                    by_mode[INDEPENDENT_MODE][seed].get("models", {}).get(model, {}).get(metric)
                    for model in ("one_number", "six_number_set")
                ]
                cumulative_scores = [
                    by_mode[CUMULATIVE_MODE][seed].get("models", {}).get(model, {}).get(metric)
                    for model in ("one_number", "six_number_set")
                ]
                pairs = [
                    (float(new), float(old))
                    for new, old in zip(cumulative_scores, independent_scores, strict=True)
                    if isinstance(new, (int, float)) and isinstance(old, (int, float))
                ]
                if pairs:
                    differences[metric].append(
                        statistics.fmean(new - old for new, old in pairs)
                    )
            new_complexities = [
                cumulative_experiment.get("models", {}).get(model, {}).get("complexity")
                for model in ("one_number", "six_number_set")
            ]
            old_complexities = [
                independent_experiment.get("models", {}).get(model, {}).get("complexity")
                for model in ("one_number", "six_number_set")
            ]
            complexity_pairs = [
                (float(new), float(old))
                for new, old in zip(new_complexities, old_complexities, strict=True)
                if isinstance(new, (int, float)) and isinstance(old, (int, float))
            ]
            if complexity_pairs:
                complexity_differences.append(
                    statistics.fmean(new - old for new, old in complexity_pairs)
                )
        random_p_medians: dict[str, dict[str, float | None]] = {}
        for mode in (INDEPENDENT_MODE, CUMULATIVE_MODE):
            random_p_medians[mode] = {}
            for model in ("one_number", "six_number_set"):
                values = [
                    by_mode[mode][seed].get("p_values", {}).get(model)
                    for seed in common
                ]
                numeric_values = [
                    float(value)
                    for value in values
                    if isinstance(value, (int, float)) and math.isfinite(float(value))
                ]
                random_p_medians[mode][model] = (
                    statistics.median(numeric_values) if numeric_values else None
                )
        comparisons.append(
            {
                "paired_seed_count": len(common),
                "validation_median_delta": (
                    statistics.median(differences["validation_score"])
                    if differences["validation_score"]
                    else None
                ),
                "independent_test_median_delta": (
                    statistics.median(differences["independent_test_score"])
                    if differences["independent_test_score"]
                    else None
                ),
                "runtime_median_delta_seconds": (
                    statistics.median(runtime_differences) if runtime_differences else None
                ),
                "candidate_median_delta": (
                    statistics.median(candidate_differences) if candidate_differences else None
                ),
                "new_formula_median_delta": (
                    statistics.median(formula_differences) if formula_differences else None
                ),
                "complexity_median_delta": (
                    statistics.median(complexity_differences)
                    if complexity_differences
                    else None
                ),
                "dataset_sha256": key[0],
                "engine_ids": list(key[1]),
                "feature_profile": key[2],
                "target_duration_seconds": key[-1],
                "random_control_median_p_values": random_p_medians,
            }
        )
    if not comparisons:
        return {
            "available": False,
            "reason": (
                "동일 데이터·설정·실행 예산에서 두 방식의 공통 시드를 "
                "2개 이상 확보해야 합니다."
            ),
            "comparisons": [],
        }
    return {
        "available": True,
        "reason": "공통 시드별 차이의 중앙값입니다. 독립 테스트는 연구 선택에 사용하지 않습니다.",
        "comparisons": comparisons,
    }
