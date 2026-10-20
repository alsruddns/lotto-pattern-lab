from __future__ import annotations

import json

import pytest

from lotto_pattern_lab.dataset import DrawDataset, DrawRecord
from lotto_pattern_lab.draw import Draw
from lotto_pattern_lab.feature_schema import PRIMITIVE_SET_VERSION, feature_mapping
from lotto_pattern_lab.formula_replay import replay_formula


def _dataset(start: int = 753, end: int = 782) -> DrawDataset:
    return DrawDataset(
        tuple(
            DrawRecord(
                round_number,
                Draw.from_numbers(
                    tuple(sorted(((round_number + offset - 1) % 45) + 1 for offset in range(6)))
                ),
            )
            for round_number in range(start, end + 1)
        ),
        "official_verified",
        "fixture",
    )


def _experiment() -> dict:
    return {
        "experiment_id": "source-experiment",
        "research_id": "source-research",
        "config": {"max_lag": 10, "feature_profile": "legacy"},
        "feature_schema": {
            "feature_profile": "legacy",
            "primitive_set_version": PRIMITIVE_SET_VERSION,
            "feature_mapping": feature_mapping("legacy", 10),
        },
        "six_number_set_model": {
            "train_rounds": list(range(763, 775)),
            "validation_rounds": list(range(775, 779)),
            "test_rounds": list(range(779, 783)),
        },
    }


def _candidate() -> dict:
    return {
        "activation_condition": "1.0",
        "formulas": ["1.0"] * 12,
        "outputs_per_event": 6,
        "rule_count": 2,
    }


def test_full_replay_marks_missing_lags_and_preserves_source(tmp_path) -> None:
    experiment = _experiment()
    original = json.dumps(experiment, sort_keys=True)
    result = replay_formula(
        dataset=_dataset(),
        experiment=experiment,
        model_key="six_number_set_model",
        candidate=_candidate(),
        formula_id="formula-a",
        start_round=753,
        end_round=782,
        output_directory=tmp_path,
    )

    assert result["target_round_count"] == 30
    assert result["input_unavailable_count"] == 10
    assert result["evaluable_round_count"] == 20
    assert result["execution_error_count"] == 0
    assert set(result["match_distribution_exact"]) == {str(index) for index in range(7)}
    assert sum(result["match_distribution_exact"].values()) == 20
    assert all(len(row["predicted_numbers"]) == 6 for row in result["rows"][10:])
    assert all(len(set(row["predicted_numbers"])) == 6 for row in result["rows"][10:])
    assert result["rows"][0]["validation_status"] == "INPUT_UNAVAILABLE"
    assert result["rows"][10]["evaluation_segment"] == "학습"
    assert result["rows"][26]["evaluation_segment"] == "원본 실험 테스트"
    assert json.dumps(experiment, sort_keys=True) == original
    assert (tmp_path / f"{result['run_id']}.json").exists()


def test_replay_rejects_one_number_model(tmp_path) -> None:
    with pytest.raises(ValueError, match="6개 집합"):
        replay_formula(
            dataset=_dataset(),
            experiment=_experiment(),
            model_key="one_number_model",
            candidate=_candidate(),
            formula_id="formula-a",
            start_round=753,
            end_round=782,
            output_directory=tmp_path,
        )


def test_replay_rejects_a_different_feature_mapping(tmp_path) -> None:
    experiment = _experiment()
    experiment["feature_schema"]["feature_mapping"]["x0"]["feature_id"] = "changed.feature"
    with pytest.raises(ValueError, match="특징 매핑"):
        replay_formula(
            dataset=_dataset(),
            experiment=experiment,
            model_key="six_number_set_model",
            candidate=_candidate(),
            formula_id="formula-a",
            start_round=753,
            end_round=782,
            output_directory=tmp_path,
        )
