import json
from datetime import date, timedelta
from pathlib import Path

import pytest
from deap import gp
from streamlit.testing.v1 import AppTest

from lotto_pattern_lab.atomic_io import interprocess_lock
from lotto_pattern_lab.backtest import chronological_split
from lotto_pattern_lab.dataset import (
    DrawDataset,
    DrawRecord,
    dataset_integrity_report,
    make_random_dataset,
)
from lotto_pattern_lab.draw import Draw
from lotto_pattern_lab.features import build_examples, feature_names
from lotto_pattern_lab.gp_engine import _make_primitive_set
from lotto_pattern_lab.intermittent import IntermittentConfig, run_intermittent_search
from lotto_pattern_lab.intermittent_experiment import run_intermittent_experiment
from lotto_pattern_lab.pattern_library import (
    PatternLibrary,
    canonical_formula,
    import_historical_experiment,
    register_experiment_candidates,
    register_experiment_exact_matches,
    register_prospective_comparison,
)
from lotto_pattern_lab.periodicity import periodicity_report
from lotto_pattern_lab.prediction import create_recommendation, evaluate_formula
from lotto_pattern_lab.pysr_backend import optional_backend_status
from lotto_pattern_lab.research import research_worker_active, run_research_schedule
from lotto_pattern_lab.research_features import (
    ResearchFeatureConfig,
    build_research_examples,
    build_research_features_for_target,
    research_feature_names,
)
from lotto_pattern_lab.research_rounds import ResearchRoundStore
from lotto_pattern_lab.storage import ExperimentStore
from lotto_pattern_lab.tabular_models import (
    evaluate_weighted_ensemble,
    run_tabular_model,
    validation_ensemble_weights,
)


def _dated_dataset(length: int = 35) -> DrawDataset:
    source = make_random_dataset(length, seed=51, first_round=753)
    first_date = date(2017, 5, 6)
    records = tuple(
        DrawRecord(record.round_number, record.draw, first_date + timedelta(days=7 * index))
        for index, record in enumerate(source.records)
    )
    return DrawDataset(records, "synthetic", "Synthetic numbers with explicit date fixtures.")


def test_shared_research_slot_reports_other_research_or_optimizer(tmp_path, monkeypatch) -> None:
    import lotto_pattern_lab.research as research

    monkeypatch.setattr(research, "GLOBAL_RESEARCH_LOCK", tmp_path / "shared-slot")
    assert research_worker_active() is False
    with interprocess_lock(
        research.GLOBAL_RESEARCH_LOCK, timeout=0, scope="research-global"
    ):
        assert research_worker_active() is True


def test_random_forest_baseline_uses_chronological_splits_and_saves_model(tmp_path) -> None:
    from lotto_pattern_lab.tabular_models import model_availability

    if not model_availability()["random_forest"]["available"]:
        pytest.skip("scikit-learn is an optional dependency")
    result = run_tabular_model(
        _dated_dataset(48),
        "random_forest",
        config=IntermittentConfig(
            workers=1,
            seed=27,
            max_lag=4,
            train_ratio=0.6,
            validation_ratio=0.2,
        ),
        directory=tmp_path / "model-experiments",
        model_directory=tmp_path / "models",
    )

    assert result["target_type"] == "six_number_set"
    assert result["split_rounds"]["test_used_for_selection"] is False
    assert Path(result["model_path"]).is_file()
    assert result["independent_test"]["evaluation_round_count"] > 0
    assert all(
        len(row["predicted_numbers"]) == 6
        for row in result["independent_test"]["predictions"]
    )
    assert all(len(row["number_scores"]) == 45 for row in result["validation"]["predictions"])


def test_ensemble_weights_come_from_validation_and_test_rounds_are_not_tuned() -> None:
    def predictions(round_number: int, scores: list[float]) -> dict:
        numbers = sorted((np.argsort(scores, kind="stable")[-6:] + 1).tolist())
        return {
            "round_number": round_number,
            "predicted_numbers": numbers,
            "winning_numbers": [1, 2, 3, 4, 5, 6],
            "number_scores": scores,
        }

    import numpy as np

    scores_a = [float(45 - value) for value in range(45)]
    scores_b = [float(value) for value in range(45)]
    model_results = {
        "random_forest": {
            "validation": {
                "average_hits": 2.0,
                "predictions": [predictions(10, scores_a)],
            },
            "independent_test": {"predictions": [predictions(20, scores_a)]},
        },
        "xgboost": {
            "validation": {
                "average_hits": 1.0,
                "predictions": [predictions(10, scores_b)],
            },
            "independent_test": {"predictions": [predictions(20, scores_b)]},
        },
    }

    weights = validation_ensemble_weights(model_results)
    test_result = evaluate_weighted_ensemble(model_results, weights)

    assert weights == {"random_forest": pytest.approx(2 / 3), "xgboost": pytest.approx(1 / 3)}
    assert test_result["weight_selection_phase"] == "validation"
    assert test_result["independent_test_used_for_weight_selection"] is False
    assert test_result["evaluation_round_count"] == 1


def test_calendar_round_features_require_dates_and_use_known_target_date() -> None:
    undated = make_random_dataset(25, 11, first_round=753)
    config = ResearchFeatureConfig()
    with pytest.raises(ValueError, match="date for every record"):
        build_research_examples(undated, config=config)

    dataset = _dated_dataset(25)
    examples = build_research_examples(dataset, config=config)
    names = research_feature_names(config)
    assert len(examples[0].features) == len(feature_names()) + len(names)
    assert examples[0].features[-len(names)] == float(2017)
    assert examples[0].features[-len(names) + 1] == 7.0
    assert names[0] == "draw_year"
    assert names[6] == "draw_yyyymmdd"
    unverified_dates = DrawDataset(dataset.records, "csv_unverified", "unverified date fixture")
    with pytest.raises(ValueError, match="official verified date source"):
        build_research_examples(unverified_dates, config=config)

    features = build_research_features_for_target(
        dataset, dataset.records[-1].round_number + 1, date(2017, 12, 23)
    )
    assert features[-len(names)] == 2017.0
    with pytest.raises(ValueError, match="explicitly supplied target date"):
        build_research_features_for_target(dataset, 778, None)


def test_calendar_prediction_requires_explicit_date_and_restores_feature_mapping() -> None:
    dataset = _dated_dataset(10)
    experiment = {
        "experiment_id": "calendar-test",
        "experiment_type": "intermittent",
        "status": "completed",
        "config": {"max_lag": 10, "feature_profile": "calendar_round"},
        "splits": {"train_rounds": [753, 760]},
        "one_number_model": {
            "train_rounds": [753, 760],
            "selected": {
                "activation_condition": "x70",
                "formulas": ["x70", "x0"],
                "rule_count": 2,
                "outputs_per_event": 1,
            },
        },
    }
    with pytest.raises(ValueError, match="explicitly supplied target date"):
        create_recommendation(experiment, dataset, 763, "one_number")
    result = create_recommendation(
        experiment,
        dataset,
        763,
        "one_number",
        target_date=date(2017, 7, 15),
        official_latest_round=762,
        random_seed=763,
    )
    assert result["feature_mapping"]["x70"] == "draw_year"
    assert result["feature_values"]["x70"] == 2017.0
    assert result["input_through_round"] == 762


def test_bounded_bit_and_number_operators_match_deap_evaluator() -> None:
    pset = _make_primitive_set(70)
    values = tuple(float(index + 1) for index in range(70))
    names = feature_names()
    formulas = (
        "bit_xor(15, 7)",
        "bit_and(15, 7)",
        "bit_or(8, 3)",
        "shift_left(3, 2)",
        "shift_left(1, 31)",
        "shift_right(28, 2)",
        "rotate_left(1, 4)",
        "rotate_right(16, 4)",
        "concat_digits(123, 45)",
        "digit_sum(12345)",
        "reverse_digits(1203)",
        "digital_root(9999)",
        "is_prime(97)",
        "factor_count(28)",
        "fibonacci_index(55)",
    )
    for formula in formulas:
        expected = gp.compile(formula, pset)(*values)
        actual = evaluate_formula(formula, values, names)
        assert float(actual["value"]) == float(expected)
    assert evaluate_formula("shift_left(1,31)", values, feature_names())["value"] == 2147483648.0
    assert evaluate_formula("factor_count(1000000)", values, feature_names())["value"] == 49.0


def test_formula_canonicalization_collapses_commutative_duplicates() -> None:
    assert canonical_formula(" add ( x0 , x1 ) ") == canonical_formula("add(x1,x0)")
    assert canonical_formula("sub(x0,x1)") != canonical_formula("sub(x1,x0)")


def test_dataset_integrity_reports_missing_rounds_and_date_coverage() -> None:
    source = make_random_dataset(3, 19, first_round=753)
    dataset = DrawDataset(
        (
            source.records[0],
            DrawRecord(755, source.records[2].draw, date(2017, 5, 20)),
        ),
        "synthetic",
        "gapped fixture",
    )
    report = dataset_integrity_report(dataset)
    assert report["rounds_contiguous"] is False
    assert report["missing_rounds"] == [754]
    assert report["all_draws_dated"] is False


def test_pattern_library_preserves_retrospective_exact_matches_and_history(tmp_path) -> None:
    library = PatternLibrary(tmp_path / "patterns.sqlite")
    event = {
        "round_number": 800,
        "active": True,
        "rule_index": 1,
        "predicted_numbers": [1, 2, 3, 4, 5, 6],
        "winning_numbers": [1, 2, 3, 4, 5, 6],
        "full_set_match": True,
        "hit_count": 6,
    }
    selected = {
        "activation_condition": "add(x0,1)",
        "formulas": ["x0"] * 12,
        "outputs_per_event": 6,
        "complexity": 12,
        "activation_events": [event],
    }
    result = {
        "experiment_id": "historical-test",
        "dataset": {"round_range": [753, 810]},
        "config": {"seed": 4},
        "feature_variables": {"x0": "round_scaled"},
        "six_number_set_model": {"selected": selected, "candidates": [selected]},
    }

    discoveries = register_experiment_exact_matches(result, library)
    assert len(discoveries) == 1
    assert discoveries[0]["created"] is True
    exact = library.list_exact_matches()
    assert len(exact) == 1
    assert exact[0]["match"]["classification"] == "retrospective_search_fit"
    assert exact[0]["match"]["prediction_created_before_draw"] is False
    patterns = register_experiment_candidates(result, library)
    assert len(patterns) == 2

    pattern_id = discoveries[0]["pattern_id"]
    duplicate_id, created = library.register_pattern(
        {
            "activation_condition": "add(1,x0)",
            "formulas": ["x0"] * 6,
            "engine": "test",
            "status": "new_discovery",
        }
    )
    assert duplicate_id == pattern_id
    assert created is False
    library.set_status(pattern_id, "refuted", "No prospective replication.")
    stored = library.get_pattern(pattern_id)
    assert stored["status"] == "pending_validation"
    assert stored["current_status"] == "refuted"
    assert stored["events"][-1]["payload"]["reason"] == "No prospective replication."


def test_legacy_experiment_import_recovers_missing_winning_numbers_without_editing(
    tmp_path,
) -> None:
    source = make_random_dataset(12, 63, first_round=800)
    winning = Draw.from_numbers((1, 2, 3, 4, 5, 6))
    dataset = DrawDataset(
        (DrawRecord(800, winning), *source.records[1:]),
        "synthetic",
        "legacy experiment import fixture",
    )
    candidate = {
        "activation_condition": "add(x0,1)",
        "formulas": ["x0"] * 12,
        "outputs_per_event": 6,
        "complexity": 12,
        "activation_events": [
            {
                "round_number": 800,
                "active": True,
                "rule_index": 1,
                "predicted_numbers": [1, 2, 3, 4, 5, 6],
                "hit_count": 6,
            }
        ],
    }
    experiment = {
        "experiment_id": "old-format",
        "experiment_type": "intermittent",
        "status": "completed",
        "dataset": {"round_range": [790, 812]},
        "config": {},
        "six_number_set_model": {"selected": candidate, "candidates": [candidate]},
    }
    original_event = dict(candidate["activation_events"][0])
    library = PatternLibrary(tmp_path / "patterns.sqlite")

    summary = import_historical_experiment(experiment, dataset, library)

    assert summary["exact_matches"] == 1
    assert experiment["six_number_set_model"]["selected"]["activation_events"][0] == original_event
    assert library.list_exact_matches()[0]["match"]["winning_numbers"] == [1, 2, 3, 4, 5, 6]


def test_prospective_pattern_evidence_requires_pre_draw_prediction(tmp_path) -> None:
    library = PatternLibrary(tmp_path / "patterns.sqlite")
    pattern_id, _ = library.register_pattern(
        {
            "activation_condition": "x0",
            "formulas": ["x0"],
            "status": "pending_validation",
        }
    )
    with pytest.raises(ValueError, match="recorded before the draw"):
        library.record_observation(pattern_id, 1000, "prospective", {"full_set_match": True})
    assert library.record_observation(
        pattern_id,
        1000,
        "prospective",
        {"prediction_created_before_draw": True, "full_set_match": False},
    )


def test_frozen_pre_draw_recommendation_result_is_appended_to_pattern_history(tmp_path) -> None:
    library = PatternLibrary(tmp_path / "patterns.sqlite")
    formulas = ["x0"] * 6
    pattern_id, _ = library.register_pattern(
        {
            "activation_condition": "add(x0,1)",
            "formulas": formulas,
            "engine": "deap_intermittent_gp",
            "status": "pending_validation",
        }
    )
    recommendation = {
        "recommendation_id": "saved-before-draw",
        "target_round": 1247,
        "experiment_id": "fixed-experiment",
        "recommendation_status": "pre_draw_prediction",
        "official_latest_round_at_creation": 1246,
        "data_provenance": {"data_used_through_round": 1246},
        "model_type": "six_number_set",
        "activation_condition": "add(x0,1)",
        "used_formulas": formulas,
    }
    comparison = {"ai_hit_count": 6, "ai_hits": [1, 2, 3, 4, 5, 6]}
    assert register_prospective_comparison(recommendation, comparison, library) == pattern_id
    assert library.get_pattern(pattern_id)["observations"][0]["phase"] == "prospective"

    historical = {**recommendation, "recommendation_status": "historical_reproduction"}
    assert register_prospective_comparison(historical, comparison, library) is None


def test_research_scheduler_saves_checkpoint_and_resumes_at_next_trial(tmp_path) -> None:
    checkpoint = tmp_path / "checks" / "research.json"
    dataset = _dated_dataset(40)
    config = IntermittentConfig(
        population_size=8,
        generations=1,
        crossover_probability=0.7,
        mutation_probability=0.15,
        max_depth=2,
        seed=31,
        workers=1,
        train_ratio=0.55,
        validation_ratio=0.25,
        max_lag=7,
        control_runs=0,
        min_activation_rate=0.15,
        hidden_states=4,
    )
    first = run_research_schedule(
        dataset,
        config,
        duration_seconds=120,
        checkpoint_path=checkpoint,
        experiment_directory=tmp_path / "experiments",
        pattern_database=tmp_path / "patterns.sqlite",
        max_runs=1,
        research_id="79ecbfa6-5505-4ed2-8ba9-39fe9cd7ffb1",
    )
    assert first["research_id"] == "79ecbfa6-5505-4ed2-8ba9-39fe9cd7ffb1"
    assert first["status"] == "completed"
    assert first["research_round"] == 1
    assert first["completed_runs"] == 1
    assert first["attempts"] == 1
    assert first["dataset_sha256"]
    assert first["checkpoint_path"] == str(checkpoint.resolve())
    assert first["rng_resume"] == {"mode": "per_attempt_seed", "base_seed": 31, "next_seed": 32}
    assert first["completed_experiments"][0]["seed"] == 31
    assert first["completed_experiments"][0]["experiment_id"] == first["last_experiment_id"]
    saved_config = first["config"]
    assert saved_config["crossover_probability"] == 0.7
    assert saved_config["mutation_probability"] == 0.15
    assert saved_config["max_depth"] == 2
    assert saved_config["train_ratio"] == 0.55
    assert saved_config["validation_ratio"] == 0.25
    assert saved_config["max_lag"] == 7
    assert saved_config["min_activation_rate"] == 0.15
    assert saved_config["hidden_states"] == 4
    saved_experiment = json.loads(
        (tmp_path / "experiments" / f"{first['last_experiment_id']}.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved_experiment["config"] == saved_config
    assert saved_experiment["pattern_library_candidates"]
    assert "full_set_match_discoveries" in saved_experiment

    # The first run hit max_runs; model an explicit pause before testing resume.
    paused_state = json.loads(checkpoint.read_text(encoding="utf-8"))
    paused_state["status"] = "paused"
    checkpoint.write_text(json.dumps(paused_state), encoding="utf-8")

    second = run_research_schedule(
        dataset,
        config,
        duration_seconds=120,
        checkpoint_path=checkpoint,
        experiment_directory=tmp_path / "experiments",
        pattern_database=tmp_path / "patterns.sqlite",
        resume=True,
        max_runs=2,
    )
    assert second["completed_runs"] == 2
    assert second["attempts"] == 2
    assert second["resume_count"] == 1
    assert second["research_round"] == first["research_round"]
    assert second["last_experiment_id"] != first["last_experiment_id"]
    assert [item["seed"] for item in second["completed_experiments"]] == [31, 32]
    registry = ResearchRoundStore(tmp_path / "patterns.sqlite")
    assert registry.get(first["research_id"])["research_round"] == 1
    assert (
        registry.get(first["research_id"])["dashboard_summary"]["completed_experiment_count"] == 2
    )
    assert second["rng_resume"]["next_seed"] == 33
    assert len(PatternLibrary(tmp_path / "patterns.sqlite").list_patterns()) >= 4

    paused_state = json.loads(checkpoint.read_text(encoding="utf-8"))
    paused_state["status"] = "paused"
    checkpoint.write_text(json.dumps(paused_state), encoding="utf-8")

    changed = _dated_dataset(40)
    changed = DrawDataset(
        (
            *changed.records[:-1],
            DrawRecord(
                792,
                Draw.from_numbers((1, 3, 5, 7, 9, 11)),
                changed.records[-1].draw_date,
            ),
        ),
        "synthetic",
        "changed round",
    )
    with pytest.raises(ValueError, match="dataset hash differs"):
        run_research_schedule(
            changed,
            config,
            duration_seconds=1,
            checkpoint_path=checkpoint,
            experiment_directory=tmp_path / "experiments",
            pattern_database=tmp_path / "patterns.sqlite",
            resume=True,
        )


def test_integrated_profile_executes_and_resumes_without_changing_schema(tmp_path) -> None:
    source = make_random_dataset(40, seed=902, first_round=753, include_bonus=True)
    first_date = date(2017, 5, 6)
    dataset = DrawDataset(
        tuple(
            DrawRecord(record.round_number, record.draw, first_date + timedelta(days=7 * index))
            for index, record in enumerate(source.records)
        ),
        "synthetic",
        "Integrated profile checkpoint fixture.",
    )
    checkpoint = tmp_path / "integrated.json"
    config = IntermittentConfig(
        population_size=8,
        generations=1,
        max_depth=2,
        seed=41,
        workers=1,
        train_ratio=0.55,
        validation_ratio=0.25,
        max_lag=7,
        control_runs=0,
        hidden_states=3,
        feature_profile="integrated_v1",
    )
    first = run_research_schedule(
        dataset,
        config,
        duration_seconds=120,
        checkpoint_path=checkpoint,
        experiment_directory=tmp_path / "experiments",
        pattern_database=tmp_path / "patterns.sqlite",
        max_runs=1,
        research_id="00000000-0000-4000-8000-000000000104",
    )
    experiment = json.loads(
        (tmp_path / "experiments" / f"{first['last_experiment_id']}.json").read_text(
            encoding="utf-8"
        )
    )
    assert first["config"]["feature_profile"] == "integrated_v1"
    assert experiment["feature_schema"]["feature_schema_version"] == 3
    assert len(experiment["feature_schema"]["feature_mapping"]) == 76
    assert experiment["feature_variables"]["x75"] == "lag7_bonus"
    assert experiment["six_number_set_model"]["test_rounds"]
    target_date = dataset.records[-1].draw_date + timedelta(days=7)
    recommendation = create_recommendation(
        experiment,
        dataset,
        dataset.records[-1].round_number + 1,
        "six_number_set",
        target_date=target_date,
    )
    assert recommendation["feature_schema"]["profile"] == "integrated_v1"
    assert len(recommendation["feature_values"]) == 76

    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    saved["status"] = "paused"
    checkpoint.write_text(json.dumps(saved), encoding="utf-8")
    resumed = run_research_schedule(
        dataset,
        config,
        duration_seconds=120,
        checkpoint_path=checkpoint,
        experiment_directory=tmp_path / "experiments",
        pattern_database=tmp_path / "patterns.sqlite",
        resume=True,
        max_runs=2,
    )
    assert resumed["config"]["feature_profile"] == "integrated_v1"
    assert resumed["resume_count"] == 1
    assert resumed["completed_runs"] == 2


def test_calendar_profile_runs_same_search_on_real_and_random_dates(tmp_path) -> None:
    config = IntermittentConfig(
        population_size=8,
        generations=1,
        max_depth=2,
        seed=77,
        workers=1,
        control_runs=1,
        feature_profile="calendar_round",
    )
    result = run_intermittent_experiment(
        _dated_dataset(45), config, ExperimentStore(tmp_path / "experiments")
    )
    assert result["feature_variables"]["x70"] == "draw_year"
    assert len(result["random_controls"]) == 1
    assert result["random_controls"][0]["data_seed"] == config.seed + 100_003
    selected = result["six_number_set_model"]["selected"]
    assert set(selected["match_distribution"]) == {str(number) for number in range(7)}
    assert isinstance(selected["test_full_set_matches"], int)


def test_windows_spawn_worker_search_completes_with_multiple_workers() -> None:
    examples = build_examples(make_random_dataset(36, seed=119))
    splits = chronological_split(examples, 0.6, 0.2)
    result = run_intermittent_search(
        splits,
        IntermittentConfig(
            population_size=8,
            generations=1,
            max_depth=2,
            workers=2,
            seed=120,
            control_runs=0,
        ),
        outputs_per_event=6,
    )
    assert result.selected.outputs_per_event == 6
    assert result.selected.test_full_set_matches >= 0


def test_research_dashboard_streamlit_page_executes() -> None:
    page = next(
        (Path(__file__).resolve().parents[1] / "src/lotto_pattern_lab/pages").glob("3_*.py")
    )
    app = AppTest.from_file(page).run(timeout=30)
    assert not app.exception, [exception.message for exception in app.exception]
    assert app.title[0].value.startswith("지속적")


def test_research_round_dashboard_streamlit_page_executes() -> None:
    page = next(
        (Path(__file__).resolve().parents[1] / "src/lotto_pattern_lab/pages").glob("5_*.py")
    )
    app = AppTest.from_file(page).run(timeout=30)

    assert not app.exception, [exception.message for exception in app.exception]
    assert app.title[0].value == "연구 회차 대시보드"


def test_periodicity_report_finds_a_repeat_and_labels_diagnostic_only() -> None:
    values = [float(index % 7) for index in range(70)]
    result = periodicity_report(values, max_lag=20)
    assert result["observation_count"] == 70
    assert result["autocorrelation_by_lag"][7] > 0.89
    assert result["interpretation"].startswith("Exploratory")


def test_pysr_backend_is_optional_and_does_not_initialize_julia() -> None:
    status = optional_backend_status()
    assert (
        status["python_version_requirement"] == ">=3.9 (PySR 2.7.x)"
    )
    assert status["runtime_verified"] is False
    if not status["installed"]:
        assert status["install_command"]
