from __future__ import annotations

import csv
import math
from pathlib import Path

from lotto_pattern_lab.dataset import make_random_dataset
from lotto_pattern_lab.feature_schema import feature_mapping
from lotto_pattern_lab.research_recommendations import (
    generate_research_recommendations,
    list_research_recommendation_candidates,
)
from lotto_pattern_lab.storage import ExperimentStore


def _gp_experiment(experiment_id: str = "gp-candidate") -> dict:
    candidate = {
        "activation_condition": "add(x0, 1)",
        "formulas": ["x0"] * 12,
        "rule_count": 2,
        "outputs_per_event": 6,
        "validation_score": 0.2,
        "validation_hit_rate": 0.3,
        "test_score": 0.1,
        "test_hit_rate": 0.25,
        "test_full_set_matches": 0,
    }
    return {
        "experiment_id": experiment_id,
        "experiment_type": "intermittent",
        "engine": "intermittent_gp",
        "status": "completed",
        "created_at": "2026-10-20T12:00:00+00:00",
        "config": {"feature_profile": "legacy", "max_lag": 10},
        "dataset_sha256": "source-dataset-hash",
        "splits": {
            "train_rounds": [1237, 1238],
            "validation_rounds": [1239, 1240],
            "test_rounds": [1241, 1242],
        },
        "independent_test_used_for_selection": False,
        "six_number_set_model": {
            "selected": candidate,
            "candidates": [candidate],
            "train_rounds": [1237, 1242],
        },
    }


def test_only_completed_compatible_candidates_with_time_ordered_holdout_are_listed(
    tmp_path: Path,
) -> None:
    store = ExperimentStore(tmp_path)
    store.save(_gp_experiment())
    bad_order = _gp_experiment("bad-order")
    bad_order["splits"]["test_rounds"] = [1238, 1240]
    store.save(bad_order)
    unfinished = _gp_experiment("unfinished")
    unfinished["status"] = "running"
    store.save(unfinished)

    candidates = list_research_recommendation_candidates(tmp_path, feature_profile="legacy")

    assert [(item["experiment_id"], item["engine"]) for item in candidates] == [
        ("gp-candidate", "intermittent_gp")
    ]
    assert candidates[0]["test_hit_rate"] == 0.25
    assert list_research_recommendation_candidates(tmp_path, feature_profile="bonus_lag_v1") == []


def test_generation_replays_saved_formula_and_returns_unsaved_six_number_draft(
    tmp_path: Path,
) -> None:
    store = ExperimentStore(tmp_path)
    store.save(_gp_experiment())
    candidates = list_research_recommendation_candidates(tmp_path, feature_profile="legacy")
    dataset = make_random_dataset(12, seed=22, first_round=1235)

    drafts, failures = generate_research_recommendations(
        candidates,
        dataset,
        1247,
        3,
        feature_profile="legacy",
        random_seed=7,
        official_latest_round=1246,
    )

    assert failures == []
    assert len(drafts) == 1  # only one distinct eligible source was available
    draft = drafts[0]
    assert draft["recommendation_status"] == "pre_draw_prediction"
    assert len(draft["recommended_numbers"]) == 6
    assert len(set(draft["recommended_numbers"])) == 6
    assert draft["input_through_round"] == 1246
    assert draft["research_selection"]["actual_win_probability_claimed"] is False
    assert (
        draft["research_selection"]["supporting_candidates"][0]["experiment_id"]
        == "gp-candidate"
    )
    assert not list(tmp_path.glob("predictions/*.json"))


def test_candidate_metrics_require_validation_and_holdout_evidence(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path)
    experiment = _gp_experiment()
    experiment["six_number_set_model"]["selected"].pop("test_hit_rate")
    experiment["six_number_set_model"]["selected"].pop("test_full_set_matches")
    experiment["six_number_set_model"]["selected"].pop("test_score")
    store.save(experiment)

    assert list_research_recommendation_candidates(tmp_path) == []


def test_pysr_candidate_is_listed_and_replayed_only_with_six_verified_targets(
    tmp_path: Path,
) -> None:
    mapping = feature_mapping("legacy", 10)
    experiment = {
        "experiment_id": "pysr-candidate",
        "research_id": "pysr-research",
        "status": "completed",
        "engine": "pysr_symbolic_regression",
        "engine_version": "pysr-engine-v1",
        "created_at": "2026-10-20T13:00:00+00:00",
        "config": {
            "target": "main_positions",
            "feature_profile": "legacy",
            "max_lag": 10,
        },
        "feature_mapping": mapping,
        "dataset_sha256": "pysr-dataset-hash",
        "splits": {
            "train_rounds": [1237, 1238],
            "validation_rounds": [1239, 1240],
            "test_rounds": [1241, 1242],
        },
        "candidates": [
            {
                "target": f"main_position_{position}",
                "formula": str(10 + position),
                "fingerprint": f"fingerprint-{position}",
                "classification": "RETAINED",
                "evaluation": {
                    "validation": {"mae": 0.5, "sample_count": 2},
                    "independent_test": {"mae": 0.7, "sample_count": 2},
                },
            }
            for position in range(1, 7)
        ],
    }
    ExperimentStore(tmp_path).save(experiment)
    candidates = list_research_recommendation_candidates(tmp_path)
    dataset = make_random_dataset(12, seed=22, first_round=1235)

    drafts, failures = generate_research_recommendations(
        candidates,
        dataset,
        1247,
        1,
        feature_profile="legacy",
        official_latest_round=1246,
    )

    assert failures == []
    assert len(candidates) == 1
    assert candidates[0]["engine"] == "pysr_symbolic_regression"
    assert candidates[0]["position_validation_mae"] == 0.5
    assert math.isclose(candidates[0]["position_holdout_mae"], 0.7)
    assert len(drafts) == 1
    assert drafts[0]["engine"] == "pysr_symbolic_regression"
    assert len(set(drafts[0]["recommended_numbers"])) == 6


def test_recommendation_page_renders_automatic_research_mode_without_user_data(
    monkeypatch, tmp_path: Path
) -> None:
    from streamlit.testing.v1 import AppTest

    import lotto_pattern_lab.paths as paths

    root = tmp_path / "project"
    for directory in ("experiments", "predictions", "data/raw", "data/official"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    ExperimentStore(root / "experiments").save(_gp_experiment())
    dataset = make_random_dataset(12, seed=31, first_round=1235)
    with (root / "data/raw/lotto_draws_753_1246_with_bonus.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.writer(output)
        writer.writerow(("round", "n1", "n2", "n3", "n4", "n5", "n6", "bonus"))
        for record in dataset.records:
            writer.writerow((record.round_number, *record.draw.numbers, record.draw.bonus or ""))
    monkeypatch.setattr(
        paths,
        "project_path",
        lambda value: Path(value) if Path(value).is_absolute() else root / Path(value),
    )
    page = next(
        (Path(__file__).resolve().parents[1] / "src/lotto_pattern_lab/pages").glob("2_AI_*.py")
    )

    app = AppTest.from_file(page).run(timeout=30)

    assert not app.exception, [exception.message for exception in app.exception]
    assert len(app.subheader) > 0
    assert any(item.key == "generate_research_based_recommendations" for item in app.button)
