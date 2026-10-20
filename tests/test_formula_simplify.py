from __future__ import annotations

from lotto_pattern_lab.dataset import make_random_dataset
from lotto_pattern_lab.formula_simplify import (
    formula_metrics,
    simplify_candidate,
    simplify_formula,
    verify_candidate_simplification,
)


def test_safe_local_simplification_and_duplicate_fingerprint() -> None:
    result = simplify_formula("add(mul(x0, 1), 0)")
    assert result["simplified_formula"] == "x0"
    assert formula_metrics("add(x0, x1)")["normalized_sha256"] == formula_metrics(
        "add(x1, x0)"
    )["normalized_sha256"]
    assert simplify_formula("safe_div(x0, 0.0)")["simplified_formula"] == "safe_div(x0, 0.0)"


def test_candidate_simplification_requires_full_evaluation_equivalence() -> None:
    candidate = {
        "activation_condition": "add(x0, 1.0)",
        "formulas": ["add(x0, 0.0)"] * 12,
        "rule_count": 2,
        "outputs_per_event": 6,
    }
    simplified, metrics = simplify_candidate(candidate)
    dataset = make_random_dataset(24, seed=3, first_round=100)
    experiment = {
        "config": {"feature_profile": "legacy", "max_lag": 10},
        "feature_schema": {},
    }

    report = verify_candidate_simplification(candidate, simplified, experiment, dataset)

    assert simplified["formulas"] == ["x0"] * 12
    assert metrics["simplified_operator_count"] < metrics["original_operator_count"]
    assert report["equivalent"]
    assert report["evaluation_rounds"] > 0


def test_gate_change_is_detected_across_full_dataset() -> None:
    candidate = {
        "activation_condition": "add(x0, 1.0)",
        "formulas": ["add(x0, 0.0)"] * 12,
        "rule_count": 2,
        "outputs_per_event": 6,
    }
    altered = {**candidate, "activation_condition": "add(x0, 2.0)"}
    report = verify_candidate_simplification(
        candidate,
        altered,
        {"config": {"max_lag": 10}, "feature_schema": {}},
        make_random_dataset(24, seed=3, first_round=100),
    )
    assert not report["equivalent"]
    assert any(item["field"] == "activation_rule" for item in report["differences"])
