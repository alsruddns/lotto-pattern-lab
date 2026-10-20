"""Statistical summaries for matched random-data GP search controls."""

from __future__ import annotations

import hashlib
import random
from typing import Any

MODEL_KEYS = ("one_number", "six_number_set")


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _holm_adjust(p_values: dict[str, float | None]) -> dict[str, float | None]:
    valid = sorted(
        ((key, value) for key, value in p_values.items() if value is not None),
        key=lambda item: item[1],
    )
    adjusted: dict[str, float | None] = dict.fromkeys(p_values)
    running_max = 0.0
    count = len(valid)
    for index, (key, value) in enumerate(valid):
        running_max = max(running_max, min(1.0, value * (count - index)))
        adjusted[key] = running_max
    return adjusted


def summarize_matched_random_controls(
    experiment: dict[str, Any], *, bootstrap_samples: int = 2000
) -> dict[str, Any]:
    """Compare validation-selected GP candidates against matched random-data GP searches.

    The random replicates replay the same GP search budget. Their best candidate is chosen
    using validation scores, then only its independent test score enters this summary.
    """
    if bootstrap_samples < 100:
        raise ValueError("신뢰구간 부트스트랩은 100회 이상이어야 합니다.")
    controls = experiment.get("random_controls", [])
    seed_text = f"{experiment.get('experiment_id', '')}:{experiment.get('dataset_sha256', '')}"
    random_seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "big")
    rng = random.Random(random_seed)
    raw: dict[str, dict[str, Any]] = {}
    raw_p: dict[str, float | None] = {}
    for model_key in MODEL_KEYS:
        actual_model = experiment.get(f"{model_key}_model") or {}
        selected = actual_model.get("selected") or {}
        observed = selected.get("test_score")
        scores = []
        for control in controls:
            model = control.get(model_key) or {}
            control_selected = model.get("selected") or {}
            score = control_selected.get("test_score")
            if score is not None:
                scores.append(float(score))
        if observed is None or not scores:
            raw[model_key] = {
                "observed_test_score": observed,
                "control_replicates": len(scores),
                "control_mean": None,
                "control_95_interval": None,
                "control_mean_95_ci": None,
                "score_difference": None,
                "empirical_p_value": None,
                "p_value_resolution": None,
                "interpretation_status": "insufficient_controls",
            }
            raw_p[model_key] = None
            continue
        observed_value = float(observed)
        control_mean = sum(scores) / len(scores)
        p_value = (1 + sum(score >= observed_value for score in scores)) / (len(scores) + 1)
        bootstrap_means = [
            sum(rng.choice(scores) for _ in scores) / len(scores)
            for _ in range(bootstrap_samples)
        ]
        raw[model_key] = {
            "observed_test_score": observed_value,
            "control_replicates": len(scores),
            "control_mean": control_mean,
            "control_95_interval": [_percentile(scores, 0.025), _percentile(scores, 0.975)],
            "control_mean_95_ci": [
                _percentile(bootstrap_means, 0.025),
                _percentile(bootstrap_means, 0.975),
            ],
            "score_difference": observed_value - control_mean,
            "empirical_p_value": p_value,
            "p_value_resolution": 1 / (len(scores) + 1),
            "interpretation_status": "tentative" if len(scores) < 99 else "replicated_controls",
        }
        raw_p[model_key] = p_value
    adjusted = _holm_adjust(raw_p)
    for model_key in MODEL_KEYS:
        raw[model_key]["holm_adjusted_p_value"] = adjusted[model_key]
        raw[model_key]["hypothesis_family_size"] = sum(
            value is not None for value in raw_p.values()
        )
    return {
        "method": "matched random 6/45 datasets; same GP search and validation selection",
        "final_test_used_for_selection": False,
        "bootstrap_samples": bootstrap_samples,
        "random_seed": random_seed,
        "models": raw,
        "limitations": [
            "대조 반복 수가 적으면 꼬리 확률과 구간 추정이 거칩니다.",
            "구간은 대조군 평균의 부트스트랩 구간이며 미래 당첨확률의 신뢰구간이 아닙니다.",
            "다중 검정 보정은 이 표의 한 번호·6개 집합 두 비교에만 적용됩니다.",
        ],
    }


def control_score_rows(experiment: dict[str, Any], model_key: str) -> list[dict[str, Any]]:
    """Return the selected independent-test score for each matched random replicate."""
    if model_key not in MODEL_KEYS:
        raise ValueError(f"Unsupported model key: {model_key}")
    rows = []
    for control in experiment.get("random_controls", []):
        model = control.get(model_key) or {}
        selected = model.get("selected") or {}
        score = selected.get("test_score")
        if score is not None:
            rows.append(
                {
                    "반복": int(control.get("replicate", len(rows) + 1)),
                    "데이터 시드": control.get("data_seed"),
                    "독립 테스트 점수": float(score),
                }
            )
    return rows
