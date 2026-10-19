from __future__ import annotations

from datetime import date, timedelta

from lotto_pattern_lab.dataset import DrawDataset, DrawRecord
from lotto_pattern_lab.draw import Draw
from lotto_pattern_lab.hit_audit import audit_research_round, build_hit_audit_markdown


class _RoundStore:
    def get(self, research_id: str) -> dict[str, object]:
        return {
            "research_id": research_id,
            "research_round": 5,
            "completed_experiment_ids": ["exp-1", "exp-2"],
        }


class _ExperimentStore:
    def load(self, experiment_id: str) -> dict[str, object]:
        events = [
            {
                "round_number": target_round,
                "active": True,
                "rule_index": 1,
                "predicted_numbers": [1, 2, 3, 4, 5, 6],
                "winning_numbers": [1, 2, 3, 4, 5, 6],
                "hit_count": 6,
                "full_set_match": True,
            }
            for target_round in (11, 12, 13)
        ]
        return {
            "experiment_id": experiment_id,
            "config": {"feature_profile": "legacy", "max_lag": 10},
            "feature_variables": {"x0": "round_scaled"},
            "splits": {
                "train_rounds": [11, 11],
                "validation_rounds": [12, 12],
                "test_rounds": [13, 13],
                "test_set_used_for_selection": False,
            },
            "one_number_model": {
                "selected": {
                    "activation_condition": "1",
                    "formulas": ["0"],
                    "outputs_per_event": 1,
                    "activation_events": events,
                }
            },
            "six_number_set_model": {
                "selected": {
                    "activation_condition": "1",
                    "formulas": ["x0", "1", "2", "3", "4", "5"],
                    "outputs_per_event": 6,
                    "activation_events": events,
                }
            },
        }


def _dataset() -> DrawDataset:
    start = date(2020, 1, 1)
    rows = tuple(
        DrawRecord(
            round_number,
            Draw.from_numbers((1, 2, 3, 4, 5, 6)),
            start + timedelta(days=7 * round_number),
        )
        for round_number in range(1, 17)
    )
    return DrawDataset(rows, "synthetic", "synthetic audit fixture")


def test_read_only_round_audit_distinguishes_repeated_hits_and_unique_draws() -> None:
    result = audit_research_round(
        "round-five",
        _dataset(),
        experiment_store=_ExperimentStore(),  # type: ignore[arg-type]
        rounds_store=_RoundStore(),  # type: ignore[arg-type]
    )

    six_model = result["models"]["six_number_set"]
    assert result["linked_experiment_count"] == result["loaded_experiment_count"] == 2
    assert six_model["repeated_hit_counts"]["independent_test"]["6_exact"] == 2
    assert six_model["unique_round_counts"]["independent_test"]["6_exact"] == 1
    assert six_model["unique_hit_rounds"]["independent_test"]["6_exact"] == [13]
    assert six_model["formula_prediction_reproduction"] == {
        "reproduced": 6,
        "compared": 6,
    }
    assert six_model["winning_numbers_match_official"] == {"verified": 6, "compared": 6}
    assert six_model["hit_formula_rankings"][0]["independent_test_4_plus"] == 2
    assert six_model["hit_formula_rankings"][0]["independent_test_unique_rounds"] == [13]
    assert six_model["high_hits"][0]["used_feature_variables"] == {"x0": "round_scaled"}
    assert result["leakage_review"]["test_set_used_for_selection_values"] == [False]


def test_hit_audit_report_labels_validation_and_test_selection_bias() -> None:
    result = audit_research_round(
        "round-five",
        _dataset(),
        experiment_store=_ExperimentStore(),  # type: ignore[arg-type]
        rounds_store=_RoundStore(),  # type: ignore[arg-type]
    )

    report = build_hit_audit_markdown(result, detailed=True)
    assert "test_set_used_for_selection" in report
    assert "4+" in report
    assert "x0" in report
