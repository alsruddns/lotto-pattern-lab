"""Compatibility and provenance helpers for applying Optuna GP settings."""

from __future__ import annotations

import math
from typing import Any

OPTUNA_TO_RESEARCH_KEYS = {
    "population_size": "new_population_size",
    "generations": "new_generations",
    "crossover_probability": "new_crossover_probability",
    "mutation_probability": "new_mutation_probability",
    "max_depth": "new_max_depth",
    "max_lag": "new_max_lag",
    "min_activation_rate": "new_min_activation_rate",
}
INTEGER_PARAMETERS = {"population_size", "generations", "max_depth", "max_lag"}


def validate_optimization_compatibility(
    study: dict[str, Any],
    *,
    current_engines: tuple[str, ...],
    current_dataset_range: list[int] | None,
    dataset_integrity: dict[str, Any] | None,
    current_dataset_sha256: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return normalized GP parameters only when saved study metadata is compatible."""
    if study.get("engine") != "intermittent_gp":
        return None, "저장된 Optuna 연구 엔진이 GP로 확인되지 않습니다."
    if "intermittent_gp" not in current_engines:
        return None, "이 설정은 GP 전용입니다. 현재 PySR 단독 실행에는 적용할 수 없습니다."
    if current_dataset_range is None or study.get("dataset_range") != current_dataset_range:
        return None, "현재 데이터 회차 범위가 최적화 연구의 저장 범위와 다릅니다."
    if current_dataset_sha256 is not None and study.get(
        "dataset_snapshot_sha256"
    ) != current_dataset_sha256:
        return None, (
            "현재 데이터 스냅샷 해시가 최적화 연구의 저장값과 다르거나 기록되지 않았습니다."
        )
    profile = study.get("feature_profile")
    if profile not in {"legacy", "bonus_lag_v1", "calendar_round"}:
        return None, "최적화 연구의 특징 프로파일을 확인할 수 없습니다."
    if dataset_integrity is None:
        return None, "현재 데이터 무결성 정보를 확인할 수 없습니다."
    if profile == "bonus_lag_v1" and not dataset_integrity.get("all_draws_have_bonus"):
        return None, "보너스 번호 특징에 필요한 데이터가 현재 데이터에 없습니다."
    if profile == "calendar_round" and not dataset_integrity.get("all_draws_dated"):
        return None, "회차·날짜 특징에 필요한 추첨일 데이터가 현재 데이터에 없습니다."

    params = study.get("best_parameters")
    if not isinstance(params, dict):
        return None, "최고 Trial의 파라미터 기록을 읽을 수 없습니다."
    missing = sorted(set(OPTUNA_TO_RESEARCH_KEYS) - set(params))
    extra = sorted(set(params) - set(OPTUNA_TO_RESEARCH_KEYS))
    if missing:
        return None, "최고 Trial에 필요한 파라미터가 없습니다: " + ", ".join(missing)
    if extra:
        return None, "현재 연구 설정과 호환되지 않는 파라미터가 있습니다: " + ", ".join(extra)

    normalized: dict[str, Any] = {}
    try:
        for name, value in params.items():
            numeric = float(value)
            if not math.isfinite(numeric):
                return None, f"{name} 값이 유한한 숫자가 아닙니다."
            if name in INTEGER_PARAMETERS:
                if not numeric.is_integer():
                    return None, f"{name} 값이 정수가 아닙니다."
                normalized[name] = int(numeric)
            else:
                normalized[name] = numeric
    except (TypeError, ValueError, OverflowError):
        return None, "최고 Trial의 파라미터 형식이 올바르지 않습니다."
    return normalized, None


def changed_optimization_settings(
    applied: dict[str, Any], current: dict[str, Any]
) -> bool:
    """Compare the persisted applied snapshot with the current research controls."""
    return any(applied.get(key) != current.get(key) for key in applied)


def optimization_origin_record(
    *,
    study: dict[str, Any],
    parameters: dict[str, Any],
    research_settings: dict[str, Any],
    applied_at: str,
    modified: bool,
) -> dict[str, Any]:
    """Create immutable-at-start provenance for one newly launched research run."""
    return {
        "optuna_study_id": study.get("optuna_study_id"),
        "study_name": str(study["study_name"]),
        "trial_number": int(study["best_trial_number"]),
        "parameters": dict(parameters),
        "research_settings": dict(research_settings),
        "feature_profile": str(study["feature_profile"]),
        "dataset_range": list(study["dataset_range"]),
        "applied_at": applied_at,
        "modified_by_user": bool(modified),
    }
