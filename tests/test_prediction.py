import json
import math
from pathlib import Path

import pytest
from deap import gp
from streamlit.testing.v1 import AppTest

from lotto_pattern_lab.dataset import DrawDataset, DrawRecord, make_random_dataset
from lotto_pattern_lab.draw import Draw
from lotto_pattern_lab.features import build_features_for_target, feature_names
from lotto_pattern_lab.gp_engine import _make_primitive_set, _predict
from lotto_pattern_lab.prediction import (
    FormulaError,
    audit_candidate_history,
    compare_recommendation,
    create_pysr_recommendation,
    create_recommendation,
    evaluate_formula,
    evaluate_gate,
    evaluate_pysr_formula,
    load_recommendation,
    save_comparison,
    save_recommendation,
    summarize_prediction_history,
)


def _experiment(one_gate: str = "x0", six_gate: str = "add(x0, 1)") -> dict:
    one_selected = {
        "activation_condition": one_gate,
        "formulas": ["add(x0, 0)", "sub(x0, 0)"],
        "rule_count": 2,
        "outputs_per_event": 1,
        "training_hit_rate": 0.15,
        "validation_hit_rate": 0.2,
        "test_hit_rate": 0.1,
        "training_score": 0.01,
        "validation_score": 0.02,
        "test_score": 0.0,
    }
    six_selected = {
        "activation_condition": six_gate,
        "formulas": ["x0"] * 12,
        "rule_count": 2,
        "outputs_per_event": 6,
        "training_hit_rate": 0.14,
        "validation_hit_rate": 0.18,
        "test_hit_rate": 0.13,
        "training_score": 0.01,
        "validation_score": 0.02,
        "test_score": 0.0,
    }
    return {
        "experiment_id": "test-experiment",
        "experiment_type": "intermittent",
        "status": "completed",
        "config": {"max_lag": 10, "population_size": 8, "seed": 7},
        "splits": {"train_rounds": [1247, 1247]},
        "one_number_model": {"selected": one_selected, "train_rounds": list(range(753, 1053))},
        "six_number_set_model": {
            "selected": six_selected,
            "train_rounds": list(range(753, 1053)),
        },
    }


def _dataset(first_round: int = 1237, length: int = 10) -> DrawDataset:
    return make_random_dataset(length, seed=911, first_round=first_round)


def test_saved_formula_parser_matches_existing_gp_operations() -> None:
    pset = _make_primitive_set(70)
    values = tuple(index / 3 + 0.25 for index in range(70))
    formulas = (
        "add(sub(mul(x0, x1), safe_div(x2, x3)), safe_mod(abs(x4), max2(1, min2(x5, x6))))",
        "square(cube(-2))",
        "min2(max2(x8, x9), abs(sub(x10, x11)))",
    )

    for formula in formulas:
        compiled = gp.compile(formula, pset)
        expected_value = float(compiled(*values))
        actual = evaluate_formula(formula, values, feature_names())

        assert math.isclose(float(actual["value"]), expected_value)
        assert actual["number"] == _predict(compiled, values)
        assert actual["steps"]


def test_formula_parser_rejects_python_execution_and_maps_feature_names() -> None:
    values = tuple(float(index) for index in range(70))
    names = feature_names()

    result = evaluate_formula("add(x0, x57)", values, names)

    assert result["variable_values"]["x0"]["feature"] == "round_scaled"
    assert result["variable_values"]["x57"]["feature"] == names[57]
    with pytest.raises(FormulaError):
        evaluate_formula("__import__('os').system('whoami')", values, names)
    with pytest.raises(FormulaError):
        evaluate_formula("x999", values, names)


def test_pysr_formula_executor_uses_safe_arithmetic_and_unavailable_candidates_fail() -> None:
    assert evaluate_pysr_formula("square(x0) + abs(x1) / 2", (3.0, -4.0), ("a", "b")) == 11
    with pytest.raises(FormulaError):
        evaluate_pysr_formula("__import__('os')", (1.0,), ("a",))


def test_pysr_recommendation_reexecutes_saved_target_formulas() -> None:
    from lotto_pattern_lab.feature_schema import feature_mapping

    dataset = _dataset(first_round=1237, length=12)
    feature_names_by_id = feature_mapping("legacy", 10)
    candidates = [
        {
            "target": f"main_position_{position}",
            "formula": f"{position + 10}",
            "fingerprint": f"fingerprint-{position}",
            "classification": "RETAINED",
            "evaluation": {"validation": {"mae": 0.5}},
        }
        for position in range(1, 7)
    ]
    experiment = {
        "experiment_id": "pysr-test",
        "status": "completed",
        "engine": "pysr_symbolic_regression",
        "engine_version": "pysr-engine-v1",
        "config": {"target": "main_positions", "feature_profile": "legacy", "max_lag": 10},
        "feature_mapping": feature_names_by_id,
        "dataset_sha256": "saved-dataset-hash",
        "candidates": candidates,
    }

    result = create_pysr_recommendation(
        experiment,
        dataset,
        dataset.records[-1].round_number + 1,
        official_latest_round=dataset.records[-1].round_number,
    )

    assert result["recommended_numbers"] == [11, 12, 13, 14, 15, 16]
    assert result["replay_status"] == "verified_saved_formula_reexecution"
    assert result["feature_schema"]["mapping"] == feature_names_by_id


def test_gate_modulo_three_branches_match_existing_engine_policy() -> None:
    assert evaluate_gate(1.99)["branch"] == "rule_1"
    assert evaluate_gate(2.99)["branch"] == "rule_2"
    assert evaluate_gate(3.0)["branch"] == "inactive"
    assert evaluate_gate(float("inf"))["branch"] == "inactive"


def test_target_features_use_only_exact_prior_rounds() -> None:
    original = _dataset(first_round=1237, length=12)
    features = build_features_for_target(original, 1247)
    changed = list(original.records)
    changed[10] = DrawRecord(1247, Draw.from_numbers((1, 2, 3, 4, 5, 6)))
    changed[11] = DrawRecord(1248, Draw.from_numbers((7, 8, 9, 10, 11, 12)))
    future_mutation = DrawDataset(tuple(changed), "synthetic", "future rows changed")

    assert features == build_features_for_target(future_mutation, 1247)
    with pytest.raises(ValueError, match="contiguous history"):
        build_features_for_target(_dataset(first_round=1238, length=9), 1247)


def test_recommendation_runs_selected_rules_and_reuses_duplicate_policy() -> None:
    recommendation = create_recommendation(
        _experiment(),
        _dataset(),
        1247,
        "six_number_set",
        random_seed=44,
        official_latest_round=1246,
    )

    assert recommendation["recommendation_status"] == "pre_draw_prediction"
    assert recommendation["input_through_round"] == 1246
    assert recommendation["selected_rule"] == 2
    assert recommendation["recommended_numbers"] == [2, 3, 4, 5, 6, 7]
    assert recommendation["feature_mapping"]["x0"] == "round_scaled"
    assert len(recommendation["formula_fingerprint"]) == 64
    assert recommendation["feature_schema"]["profile"] == "legacy"
    assert len(set(recommendation["recommended_numbers"])) == 6
    assert all(
        item["duplicate_adjustments"] == list(range(2, 2 + index))
        for index, item in enumerate(recommendation["formula_calculations"])
    )
    assert (
        recommendation["random_baseline"]
        == create_recommendation(
            _experiment(),
            _dataset(),
            1247,
            "six_number_set",
            random_seed=44,
            official_latest_round=1246,
        )["random_baseline"]
    )


def test_inactive_gate_does_not_force_recommendation_numbers() -> None:
    recommendation = create_recommendation(
        _experiment(one_gate="mul(x0, 3)"),
        _dataset(),
        1247,
        "one_number",
        random_seed=5,
        official_latest_round=1246,
    )

    assert recommendation["gate_result"]["branch"] == "inactive"
    assert recommendation["recommended_numbers"] == []
    assert recommendation["formula_calculations"] == []


def test_prediction_excludes_target_and_later_draws_and_marks_reproduction() -> None:
    data = _dataset(first_round=1237, length=12)
    experiment = _experiment()
    first = create_recommendation(
        experiment, data, 1247, "one_number", random_seed=5, official_latest_round=1246
    )
    with_future_result = create_recommendation(
        experiment, data, 1247, "one_number", random_seed=5, official_latest_round=1247
    )

    assert first["feature_values"] == with_future_result["feature_values"]
    assert first["gate_result"] == with_future_result["gate_result"]
    assert first["recommended_numbers"] == with_future_result["recommended_numbers"]
    assert (
        first["data_provenance"]["used_draws_sha256"]
        == with_future_result["data_provenance"]["used_draws_sha256"]
    )
    assert first["recommendation_status"] == "historical_reproduction"


def test_recommendation_is_immutable_and_actual_comparison_is_separate(tmp_path) -> None:
    recommendation = create_recommendation(
        _experiment(),
        _dataset(),
        1247,
        "six_number_set",
        random_seed=1247,
        official_latest_round=1246,
    )
    saved_path = save_recommendation(recommendation, tmp_path / "predictions")
    before = saved_path.read_bytes()
    loaded = json.loads(saved_path.read_text(encoding="utf-8"))
    actual = Draw.from_numbers((2, 7, 13, 22, 35, 44))
    comparison = compare_recommendation(loaded, actual.numbers, actual_bonus=11)
    comparison_path = save_comparison(comparison, tmp_path / "comparisons")

    assert load_recommendation(saved_path) == loaded
    assert comparison["ai_hits"] == [2, 7]
    assert comparison["ai_hit_count"] == 2
    assert comparison["prize_tier"] is None
    assert comparison_path.is_file()
    with pytest.raises(FileExistsError):
        save_recommendation(loaded, tmp_path / "predictions")
    assert saved_path.read_bytes() == before


def test_prediction_history_summarizes_only_saved_pre_draw_records() -> None:
    data = make_random_dataset(10, 22, first_round=1237)
    target_record = DrawRecord(1247, Draw.from_numbers((1, 2, 3, 4, 5, 6)))
    with_result = DrawDataset((*data.records, target_record), "synthetic", "completed target")
    recommendation = create_recommendation(
        _experiment(), data, 1247, "one_number", random_seed=9, official_latest_round=1246
    )
    historical = {**recommendation, "recommendation_status": "historical_reproduction"}

    summary = summarize_prediction_history([recommendation, historical], with_result)

    assert summary["one_number"]["recommendations"] == 1
    assert summary["one_number"]["ai_numbers"] == 1
    assert summary["one_number"]["by_gate_branch"]["rule_1"]["actual_draws"] == 1
    assert summary["one_number"]["hit_distribution"]["1"] == 1


def test_retrospective_candidate_audit_counts_unique_rounds_once() -> None:
    dataset = _dataset(first_round=1237, length=12)

    result = audit_candidate_history(_experiment(), dataset, "six_number_set", 1)

    assert result["evaluation_rounds"] == 2
    assert sum(result["hit_distribution"].values()) == 2
    assert all(
        event["prediction_type"] == "retrospective_unique_round" for event in result["events"]
    )


def test_recommendation_streamlit_page_executes_without_script_errors() -> None:
    page = next(
        (Path(__file__).resolve().parents[1] / "src/lotto_pattern_lab/pages").glob("2_*.py")
    )
    app = AppTest.from_file(page).run(timeout=30)

    assert not app.exception, [exception.message for exception in app.exception]
    assert app.title[0].value.startswith("AI")
