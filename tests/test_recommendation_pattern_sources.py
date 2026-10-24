import json
import sqlite3
from datetime import date

import pytest

from lotto_pattern_lab.dataset import make_random_dataset
from lotto_pattern_lab.feature_schema import feature_schema_version
from lotto_pattern_lab.pattern_library import PatternLibrary
from lotto_pattern_lab.prediction import create_pattern_recommendation
from lotto_pattern_lab.research_rounds import list_saved_research_predictions


def _pattern_payload(pattern_id: str = "", *, rule_index: int = 1) -> dict:
    return {
        "pattern_id": pattern_id,
        "activation_condition": "1",
        "formulas": ["x0"] * 6,
        "outputs_per_event": 6,
        "model_type": "six_number_set",
        "feature_profile": "legacy",
        "feature_schema": {"feature_profile": "legacy"},
        "origin_research_id": "source-research",
        "research_id": "source-research",
        "experiment_id": "source-experiment",
        "experiment_config": {"feature_profile": "legacy", "max_lag": 10},
        "candidate_rank": 2,
        "rule_index": rule_index,
        "engine": "deap_intermittent_gp",
    }


def test_hit_pattern_queries_deduplicate_rounds_and_require_main_set_exact_match(tmp_path):
    library = PatternLibrary(tmp_path / "patterns.sqlite")
    exact_id, _ = library.register_pattern(_pattern_payload())
    mismatched_id, _ = library.register_pattern(
        {
            **_pattern_payload(),
            "experiment_id": "mismatch-experiment",
            "formulas": ["add(x0,1)"] * 6,
        }
    )
    library.record_observation(
        exact_id,
        915,
        "retrospective",
        {
            "full_set_match": True,
            "hit_count": 6,
            "prediction_created_before_draw": False,
            "predicted_numbers": [2, 6, 11, 13, 22, 37],
            "winning_numbers": [2, 6, 11, 13, 22, 37],
            "classification": "retrospective_search_fit",
        },
    )
    library.record_observation(
        exact_id,
        915,
        "validation",
        {
            "full_set_match": False,
            "hit_count": 5,
            "predicted_numbers": [2, 6, 11, 13, 22, 38],
            "winning_numbers": [2, 6, 11, 13, 22, 37],
        },
    )
    library.record_observation(
        mismatched_id,
        916,
        "retrospective",
        {
            "full_set_match": True,
            "hit_count": 6,
            "bonus_match": True,
            "prediction_created_before_draw": False,
            "predicted_numbers": [1, 2, 3, 4, 5, 6],
            "winning_numbers": [2, 6, 11, 13, 22, 37],
        },
    )

    five_plus = library.list_hit_pattern_candidates(minimum_hits=5)
    exact_six = library.list_hit_pattern_candidates(minimum_hits=6)

    assert {item["pattern_id"] for item in five_plus} == {exact_id, mismatched_id}
    five_plus_exact = next(item for item in five_plus if item["pattern_id"] == exact_id)
    assert five_plus_exact["hit_count"] == 2
    assert five_plus_exact["unique_hit_rounds"] == 1
    exact = next(item for item in exact_six if item["pattern_id"] == exact_id)
    assert exact["hit_count"] == 1
    assert exact["unique_hit_rounds"] == 1
    assert exact["exact_six_count"] == 1
    assert five_plus_exact["phases"] == ["retrospective", "validation"]
    assert exact["hit_rounds"][0]["prediction_created_before_draw"] is False
    assert mismatched_id not in {item["pattern_id"] for item in exact_six}


def test_read_only_library_access_does_not_create_or_migrate_database(tmp_path):
    path = tmp_path / "missing.sqlite"

    with pytest.raises(sqlite3.OperationalError):
        PatternLibrary.read_only(path).list_hit_pattern_candidates(minimum_hits=6)

    assert not path.exists()


def test_recommendation_pattern_table_formats_missing_research_round_for_arrow(
    tmp_path, monkeypatch
):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    from lotto_pattern_lab import paths

    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    library = PatternLibrary(tmp_path / "data/database/patterns.sqlite")
    for rule_index, research_id, research_round, formula in [
        (1, "research-with-round", 8, "x0"),
        (2, "research-without-round", None, "add(x0,1)"),
    ]:
        payload = _pattern_payload(rule_index=rule_index)
        payload.update(
            origin_research_id=research_id,
            research_id=research_id,
            formulas=[formula] * 6,
        )
        if research_round is not None:
            payload["research_round"] = research_round
        pattern_id, _ = library.register_pattern(payload)
        library.record_observation(
            pattern_id,
            1400,
            "validation",
            {
                "hit_count": 5,
                "full_set_match": False,
                "predicted_numbers": [1, 2, 3, 4, 5, 6],
                "winning_numbers": [1, 2, 3, 4, 5, 7],
                "prediction_created_before_draw": False,
            },
        )

    page = next(
        (Path(__file__).resolve().parents[1] / "src/lotto_pattern_lab/pages").glob(
            "2_*.py"
        )
    )
    app = AppTest.from_file(page).run(timeout=60)

    assert not app.exception, [error.message for error in app.exception]
    assert len(app.dataframe) == 1
    assert set(app.dataframe[0].value["연구 회차"]) == {"8", "미수집"}


def test_pattern_replay_uses_saved_six_formula_and_rejects_other_gate_branch():
    dataset = make_random_dataset(10, seed=7, first_round=1237)
    pattern = _pattern_payload(pattern_id="pattern-1")

    recommendation = create_pattern_recommendation(
        pattern,
        dataset,
        1247,
        official_latest_round=1246,
        random_seed=3,
    )

    assert recommendation["recommendation_status"] == "pre_draw_prediction"
    assert recommendation["source_pattern_id"] == "pattern-1"
    assert recommendation["research_id"] == "source-research"
    assert len(recommendation["recommended_numbers"]) == 6
    assert len(set(recommendation["recommended_numbers"])) == 6
    assert all(1 <= number <= 45 for number in recommendation["recommended_numbers"])
    assert recommendation["feature_schema"]["profile"] == "legacy"
    with pytest.raises(ValueError, match="규칙 2 전용"):
        create_pattern_recommendation(
            {**pattern, "rule_index": 2},
            dataset,
            1247,
            official_latest_round=1246,
            random_seed=3,
        )


def test_pattern_replay_supports_integrated_features_without_target_draw_inputs():
    dataset = make_random_dataset(12, seed=17, first_round=1235, include_bonus=True)
    pattern = {
        **_pattern_payload(pattern_id="integrated-pattern"),
        "feature_profile": "integrated_v1",
        "feature_schema_version": feature_schema_version("integrated_v1"),
        "experiment_config": {
            "feature_profile": "integrated_v1",
            "max_lag": 10,
        },
    }

    recommendation = create_pattern_recommendation(
        pattern,
        dataset,
        1247,
        official_latest_round=1246,
        target_date=date(2026, 10, 31),
        random_seed=9,
    )

    assert recommendation["feature_schema"]["profile"] == "integrated_v1"
    assert recommendation["data_provenance"]["data_used_through_round"] == 1246
    assert recommendation["recommendation_status"] == "pre_draw_prediction"
    assert len(recommendation["recommended_numbers"]) == 6
    with pytest.raises(ValueError, match="저장 특징 스키마"):
        create_pattern_recommendation(
            {**pattern, "feature_schema_version": 1},
            dataset,
            1247,
            official_latest_round=1246,
            target_date=date(2026, 10, 31),
            random_seed=9,
        )
    with pytest.raises(ValueError, match="6개 집합"):
        create_pattern_recommendation(
            {**pattern, "model_type": "one_number", "outputs_per_event": 1},
            dataset,
            1247,
            official_latest_round=1246,
        )


def test_saved_research_predictions_match_research_id_or_legacy_experiment_link(tmp_path):
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    records = {
        "direct": {"recommendation_id": "direct", "research_id": "research-a"},
        "legacy": {"recommendation_id": "legacy", "experiment_id": "exp-a"},
        "other": {
            "recommendation_id": "other",
            "research_id": "research-b",
            "experiment_id": "exp-a",
        },
    }
    for name, payload in records.items():
        (predictions / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")

    found, errors = list_saved_research_predictions(
        "research-a", ["exp-a"], predictions
    )

    assert {item["recommendation_id"] for item in found} == {"direct", "legacy"}
    assert all(item["research_id"] == "research-a" for item in found)
    assert errors == []


def test_saved_research_prediction_storage_errors_are_distinct_from_empty(tmp_path):
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    (predictions / "bad.json").write_text("{broken", encoding="utf-8")

    found, errors = list_saved_research_predictions("research-a", [], predictions)

    assert found == []
    assert errors
