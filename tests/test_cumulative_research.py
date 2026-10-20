from __future__ import annotations

import pytest

from lotto_pattern_lab.cumulative_research import (
    CUMULATIVE_MODE,
    INDEPENDENT_MODE,
    compare_research_modes,
    cumulative_growth_summary,
    recommend_optuna_settings,
)
from lotto_pattern_lab.pattern_library import PatternLibrary


def _study(**overrides):
    return {
        "status": "completed",
        "best_trial_number": 3,
        "best_value": 0.5,
        "engine": "intermittent_gp",
        "feature_profile": "legacy",
        "dataset_range": [1, 20],
        "dataset_snapshot_sha256": "snapshot-a",
        "final_test_used_for_selection": False,
        "objective_policy_version": "validation_only_v1",
        "study_name": "matching-study",
        "best_parameters": {
            "population_size": 32,
            "generations": 8,
            "crossover_probability": 0.65,
            "mutation_probability": 0.25,
            "max_depth": 5,
            "max_lag": 10,
            "min_activation_rate": 0.1,
        },
        **overrides,
    }


def test_optuna_recommendation_requires_exact_snapshot_and_validation_policy():
    valid = recommend_optuna_settings(
        [_study()],
        feature_profile="legacy",
        dataset_range=[1, 20],
        dataset_sha256="snapshot-a",
        dataset_integrity={},
    )
    assert valid["available"]
    assert valid["parameters"]["population_size"] == 32

    legacy = recommend_optuna_settings(
        [_study(dataset_snapshot_sha256=None)],
        feature_profile="legacy",
        dataset_range=[1, 20],
        dataset_sha256="snapshot-a",
        dataset_integrity={},
    )
    assert not legacy["available"]
    assert legacy["reason"] == "추천할 데이터가 충분하지 않습니다."

    test_tuned = recommend_optuna_settings(
        [_study(final_test_used_for_selection=True)],
        feature_profile="legacy",
        dataset_range=[1, 20],
        dataset_sha256="snapshot-a",
        dataset_integrity={},
    )
    assert not test_tuned["available"]


def test_growth_summary_separates_search_volume_and_test_evidence():
    summary = cumulative_growth_summary(
        [
            {
                "research_round": 1,
                "total_elapsed_seconds": 3600,
                "dashboard_summary": {
                    "candidate_count": 100,
                    "new_pattern_count": 10,
                    "inherited_pattern_count": 4,
                    "new_modified_child_pattern_count": 3,
                    "duplicate_pattern_count": 8,
                    "model_metrics": {
                        "six_number_set": [
                            {"validation_score": 0.2, "independent_test_score": 0.1}
                        ]
                    },
                },
            }
        ]
    )
    assert summary["research_round_count"] == 1
    assert summary["new_valid_formula_count"] == 10
    assert summary["reused_elite_formula_count"] == 4
    assert summary["modified_formula_count"] == 3
    assert summary["new_valid_formulas_per_hour"] == 10
    assert summary["mean_validation_scores"]["six_number_set"] == 0.2
    assert summary["mean_independent_test_scores"]["six_number_set"] == 0.1
    assert summary["random_control_median_p_values"]["six_number_set"] is None


def _mode_record(mode: str, seed: int, validation: float, test: float):
    return {
        "research_mode": mode,
        "dataset_sha256": "same-snapshot",
        "engines": ["intermittent_gp"],
        "target_duration_seconds": 3600,
        "config": {
            "feature_profile": "legacy",
            "train_ratio": 0.6,
            "validation_ratio": 0.2,
            "max_lag": 10,
            "population_size": 8,
            "generations": 2,
            "crossover_probability": 0.65,
            "mutation_probability": 0.25,
            "max_depth": 4,
            "workers": 1,
            "control_runs": 1,
            "min_activation_rate": 0.1,
            "hidden_states": 3,
        },
        "dashboard_summary": {
            "experiment_summaries": [
                {
                    "seed": seed,
                    "status": "completed",
                    "models": {
                        model: {
                            "validation_score": validation,
                            "independent_test_score": test,
                        }
                        for model in ("one_number", "six_number_set")
                    },
                }
            ]
        },
    }


def test_mode_comparison_requires_two_matched_seeds_and_reports_test_separately():
    records = [
        _mode_record(INDEPENDENT_MODE, 1, 0.2, 0.1),
        _mode_record(INDEPENDENT_MODE, 2, 0.3, 0.2),
        _mode_record(CUMULATIVE_MODE, 1, 0.25, 0.05),
        _mode_record(CUMULATIVE_MODE, 2, 0.35, 0.25),
    ]
    comparison = compare_research_modes(records)
    assert comparison["available"]
    result = comparison["comparisons"][0]
    assert result["paired_seed_count"] == 2
    assert result["validation_median_delta"] == pytest.approx(0.05)
    assert result["independent_test_median_delta"] == pytest.approx(0.0)

    insufficient = compare_research_modes(records[:3])
    assert not insufficient["available"]


def test_elite_candidate_query_is_bounded_and_does_not_read_full_library(tmp_path, monkeypatch):
    library = PatternLibrary(tmp_path / "patterns.sqlite")

    def fail_full_scan(*args, **kwargs):
        raise AssertionError("Elite selection must use the bounded query")

    monkeypatch.setattr(library, "list_patterns", fail_full_scan)
    assert library.list_elite_patterns(limit=1) == []
    selected, _ = library.select_elite_candidates(
        feature_profile="legacy", max_lag=10, max_candidates=1
    )
    assert selected == []


def test_cumulative_elite_selection_rejects_legacy_patterns_without_source_evidence(
    tmp_path,
):
    library = PatternLibrary(tmp_path / "patterns.sqlite")
    library.register_pattern(
        {
            "activation_condition": "x0",
            "formulas": ["x1"],
            "outputs_per_event": 1,
            "feature_profile": "legacy",
            "performance": {
                "training_score": 0.5,
                "validation_score": 0.5,
                "validation_activation_rate": 0.5,
            },
        }
    )
    selected, skipped = library.select_elite_candidates(
        feature_profile="legacy",
        max_lag=10,
        max_candidates=2,
        max_data_round=20,
        require_temporal_provenance=True,
        experiment_directory=tmp_path / "experiments",
    )
    assert selected == []
    assert skipped[0]["reason"] == "source experiment is unavailable"
