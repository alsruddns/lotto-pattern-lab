import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lotto_pattern_lab.pattern_library import PatternLibrary
from lotto_pattern_lab.research_rounds import (
    ResearchRoundStore,
    build_chatgpt_report,
    compare_research_rounds,
    legacy_research_records,
    model_phase_comparison_stats,
    paired_comparison_summary,
    rebuild_dashboard_summary,
    rebuild_dashboard_summary_from_store,
    summarize_experiment,
)
from lotto_pattern_lab.storage import ExperimentStore


def test_research_round_migration_orders_history_and_resume_keeps_number(tmp_path) -> None:
    checkpoints = tmp_path / "checkpoints"
    experiments = tmp_path / "experiments"
    checkpoints.mkdir()
    experiments.mkdir()
    for research_id, created_at in (
        ("study-b", "2025-02-01T00:00:00+00:00"),
        ("study-a", "2025-01-01T00:00:00+00:00"),
    ):
        (checkpoints / f"research_{research_id}.json").write_text(
            json.dumps(
                {
                    "research_id": research_id,
                    "created_at": created_at,
                    "status": "paused",
                    "completed_experiments": [],
                    "config": {"feature_profile": "legacy"},
                }
            ),
            encoding="utf-8",
        )
    (experiments / "orphan.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp-orphan",
                "experiment_type": "intermittent",
                "status": "completed",
                "started_at": "2025-03-01T00:00:00+00:00",
                "research_id": "study-orphan",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    rounds = ResearchRoundStore(tmp_path / "patterns.sqlite")

    rounds.migrate_legacy(checkpoints, experiments)
    records = {record["research_id"]: record for record in rounds.list()}
    assert records["study-a"]["research_round"] == 1
    assert records["study-b"]["research_round"] == 2
    assert records["study-orphan"]["research_round"] == 3
    assert rounds.experiment_ids("study-orphan") == ["exp-orphan"]
    assert rounds.ensure_round("study-a", {}, checkpoints, experiments) == 1

    next_round = rounds.ensure_round("study-new", {"status": "queued"}, checkpoints, experiments)
    assert next_round == 4


def test_concurrent_research_round_registration_has_unique_numbers(tmp_path) -> None:
    database = tmp_path / "patterns.sqlite"
    checkpoints = tmp_path / "checkpoints"
    experiments = tmp_path / "experiments"
    checkpoints.mkdir()
    experiments.mkdir()
    registry = ResearchRoundStore(database)
    ids = [f"research-{index}" for index in range(12)]

    def register(research_id: str) -> int:
        return ResearchRoundStore(database).ensure_round(
            research_id, {"status": "running"}, checkpoints, experiments
        )

    with ThreadPoolExecutor(max_workers=6) as workers:
        numbers = list(workers.map(register, ids))
    assert len(set(numbers)) == len(ids)
    assert {item["research_id"] for item in registry.list()} == set(ids)


def test_round_overview_omits_experiment_details_without_writing(tmp_path) -> None:
    database = tmp_path / "patterns.sqlite"
    registry = ResearchRoundStore(database)
    checkpoints = tmp_path / "checkpoints"
    experiments = tmp_path / "experiments"
    checkpoints.mkdir()
    experiments.mkdir()
    registry.ensure_round(
        "overview-round",
        {
            "status": "completed",
            "completed_experiment_ids": ["experiment-1"],
            "dashboard_summary": {
                "summary_schema_version": 3,
                "completed_experiment_count": 1,
                "failed_experiment_count": 0,
                "hit_totals": {"six_number_set:independent_test": {"4_plus": 2}},
                "experiment_summaries": [
                    {"experiment_id": "experiment-1", "payload": "x" * 10000}
                ],
                "model_metrics": {
                    "six_number_set": [{"independent_test_score": 0.1}]
                },
            },
        },
        checkpoints,
        experiments,
    )
    before = registry.get("overview-round")

    overview = registry.list_overview()

    assert len(overview) == 1
    summary = overview[0]["dashboard_summary"]
    assert summary["completed_experiment_count"] == 1
    assert summary["hit_totals"]["six_number_set:independent_test"]["4_plus"] == 2
    assert "experiment_summaries" not in summary
    assert summary["model_metrics"]["six_number_set"][0]["independent_test_score"] == 0.1
    assert registry.get("overview-round") == before


def test_legacy_experiment_metadata_scan_does_not_load_detail_payload(tmp_path) -> None:
    experiment_dir = tmp_path / "experiments"
    experiment_dir.mkdir()
    (experiment_dir / "record.json").write_text(
        '{\n  "experiment_id": "exp-1",\n  "started_at": "2024-01-01",\n'
        '  "status": "completed",\n  "research_id": "legacy-1",\n'
        '  "large_payload": {"huge": [1, 2, 3]}\n}\n',
        encoding="utf-8",
    )
    records = legacy_research_records(tmp_path / "missing", experiment_dir)
    assert records[0]["research_id"] == "legacy-1"
    assert records[0]["experiment_ids"] == ["exp-1"]


def _experiment(experiment_id: str, score: float, *, elite: bool) -> dict:
    one_selected = {
        "training_score": score + 0.1,
        "validation_score": score,
        "test_score": score - 0.1,
        "training_activation_rate": 0.5,
        "validation_activation_rate": 0.25,
        "test_activation_rate": 0.5,
        "training_hit_count": 4,
        "validation_hit_count": 2,
        "test_hit_count": 3,
        "training_full_set_matches": 0,
        "validation_full_set_matches": 0,
        "test_full_set_matches": 0,
        "match_distribution_by_split": {
            "training": {"0": 3, "1": 2, "3": 1},
            "validation": {"0": 2, "4": 1},
            "independent_test": {"0": 2, "5": 1},
        },
        "activation_events": [
            {"round_number": 1, "active": True},
            {"round_number": 2, "active": True},
            {"round_number": 3, "active": True},
        ],
        "activation_condition": "add(x0,1)",
        "formulas": ["x1"],
        "complexity": 2,
    }
    one = {"selected": one_selected, "candidates": [one_selected]}
    six_selected = {
        **one_selected,
        "formulas": ["x1"] * 6,
        "outputs_per_event": 6,
    }
    six = {"selected": six_selected, "candidates": [six_selected]}
    paired = (
        {
            "same_dataset_and_splits": True,
            "one_number_model": {
                "selected": {"validation_score": score - 0.2, "test_score": score - 0.3}
            },
            "six_number_set_model": {
                "selected": {"validation_score": score - 0.4, "test_score": score - 0.5}
            },
        }
        if elite
        else None
    )
    return {
        "experiment_id": experiment_id,
        "status": "completed",
        "seed": 11,
        "runtime_seconds": 5.0,
        "config": {
            "population_size": 8,
            "feature_profile": "legacy",
            "train_ratio": 0.6,
            "validation_ratio": 0.2,
            "max_lag": 10,
            "control_runs": 3,
        },
        "progress": [{}, {}],
        "splits": {"train_rounds": [1, 1], "validation_rounds": [2, 2], "test_rounds": [3, 3]},
        "one_number_model": one,
        "six_number_set_model": six,
        "elite_seeding": {
            "one_number_search": {"seeded_count": 1 if elite else 0},
            "six_number_set_search": {"seeded_count": 2 if elite else 0},
        },
        "paired_unseeded_comparison": paired,
        "random_control_empirical_p_value": {"one_number": 0.5, "six_number_set": 0.75},
        "random_control_replicates": 3,
        "pattern_library_candidates": [
            {"created": True, "parent_pattern_ids": ["parent"]},
            {"created": False, "parent_pattern_ids": []},
        ],
        "full_set_match_discoveries": [
            {
                "pattern_id": "exact-1",
                "round_number": 1,
                "winning_numbers": [1, 2, 3, 4, 5, 6],
                "activation_condition": "add(x0,1)",
                "formulas": ["x1"] * 6,
            }
        ],
    }


def test_summary_separates_models_counts_hits_and_elite_seeding() -> None:
    experiment = _experiment("exp-1", 0.4, elite=True)
    summary = summarize_experiment(experiment)
    assert summary["candidate_count"] == 16
    assert summary["new_pattern_count"] == 1
    assert summary["duplicate_pattern_count"] == 1
    assert summary["child_pattern_count"] == 1
    assert summary["inherited_pattern_count"] == 3
    assert round(summary["models"]["one_number"]["independent_test_score"], 6) == 0.3
    assert round(summary["models"]["six_number_set"]["independent_test_score"], 6) == 0.3
    assert summary["hits"]["one_number:independent_test"]["5_plus"] == 1
    assert summary["paired"]["one_number"]["same_dataset_and_splits"] is True


def test_compressed_event_reference_uses_distribution_for_activation_counts() -> None:
    experiment = _experiment("exp-pointer", 0.4, elite=True)
    selected = experiment["six_number_set_model"]["selected"]
    expected = sum(
        int(value)
        for value in selected["match_distribution_by_split"]["validation"].values()
    )
    selected["activation_events"] = {"$detail_ref": "/events", "item_count": 3}
    summary = summarize_experiment(experiment)
    assert summary["models"]["six_number_set"]["validation_activation_count"] == expected


def test_hit_repetition_and_unique_round_counts_are_separate() -> None:
    first = _experiment("exp-a", 0.4, elite=False)
    second = _experiment("exp-b", 0.4, elite=False)
    for experiment in (first, second):
        selected = experiment["six_number_set_model"]["selected"]
        selected["match_distribution_by_split"]["validation"] = {"5": 1}
        selected["activation_events"] = [
            {"round_number": 2, "active": True, "hit_count": 5}
        ]
    summary = rebuild_dashboard_summary([first, second])
    repeated = summary["hit_totals"]["six_number_set:validation"]["5_plus"]
    unique = summary["unique_hit_totals"]["six_number_set"]["validation"]["5_plus"]
    assert repeated == 2
    assert unique == 1


def test_paired_comparison_summarizes_delta_wins_losses_and_time() -> None:
    experiment = _experiment("exp-paired", 0.4, elite=True)
    experiment["paired_unseeded_comparison"]["six_number_set_model"]["candidates"] = [
        {"activation_condition": "add(x0,1)", "formulas": ["x1"] * 6},
        {"activation_condition": "add(x0,1)", "formulas": ["x1"] * 6},
    ]
    compact = summarize_experiment(experiment)
    summary = rebuild_dashboard_summary([compact])
    stats = paired_comparison_summary(summary)["six_number_set"]
    assert stats["validation_mean_delta"] == 0.4
    assert stats["validation_wins"] == 1
    assert stats["test_wins"] == 1
    assert stats["actual_runtime_mean"] is None
    assert stats["unseeded_candidate_count_mean"] == 2
    assert stats["unseeded_unique_formula_count_mean"] == 1


def test_report_summarizes_p_values_instead_of_dumping_each_observation() -> None:
    experiment = _experiment("exp-p", 0.4, elite=False)
    summary = rebuild_dashboard_summary([experiment, experiment])
    record = {
        "research_round": 1,
        "research_id": "study",
        "config": {**experiment["config"], "control_runs": 3},
        "dashboard_summary": summary,
    }
    report = build_chatgpt_report(record)
    assert "0.25" in report
    assert "p값은 실험당 대조 반복이 적어 분해능이 거칩니다" in report
    assert "[0.5, 0.5" not in report


def test_pattern_registration_and_child_counts_come_from_library_events(tmp_path) -> None:
    library = PatternLibrary(tmp_path / "patterns.sqlite")
    parent_id, _ = library.register_pattern(
        {
            "activation_condition": "x0",
            "formulas": ["x1"],
            "research_id": "prior-study",
            "origin_research_id": "prior-study",
        }
    )
    child_payload = {
        "activation_condition": "x0",
        "formulas": ["add(x1,1)"],
        "research_id": "current-study",
        "origin_research_id": "current-study",
        "experiment_id": "exp-1",
    }
    child_id, created = library.register_pattern(child_payload)
    duplicate_id, duplicate_created = library.register_pattern(child_payload)
    library.record_lineage(
        parent_id,
        child_id,
        research_id="current-study",
        generation=1,
        method="mutation",
        operators=["mutation"],
        parent_formula="x1",
        child_formula="add(x1,1)",
    )
    assert created and not duplicate_created and duplicate_id == child_id
    assert library.research_pattern_counts("current-study") == {
        "new_pattern_count": 1,
        "duplicate_pattern_count": 1,
        "registration_attempt_count": 2,
        "child_pattern_count": 1,
        "new_child_pattern_count": 1,
        "modified_child_pattern_count": 1,
        "new_modified_child_pattern_count": 1,
        "parent_pattern_count": 1,
        "lineage_event_counts": {"mutation": 1},
    }


def test_store_reaggregation_expands_sidecars_and_counts_distinct_hit_rounds(tmp_path) -> None:
    experiment = _experiment("exp-detail", 0.4, elite=False)
    experiment["splits"] = {
        "train_rounds": [1, 10],
        "validation_rounds": [11, 30],
        "test_rounds": [31, 40],
    }
    event_rows = [
        {
            "round_number": round_number,
            "active": True,
            "hit_count": 5 if round_number in (11, 12) else 1,
            "full_set_match": False,
        }
        for round_number in range(1, 41)
    ]
    for model_key in ("one_number_model", "six_number_set_model"):
        selected = experiment[model_key]["selected"]
        selected["activation_events"] = event_rows
        selected["match_distribution_by_split"] = {
            "training": {"1": 10},
            "validation": {"1": 18, "5": 2},
            "independent_test": {"1": 10},
        }
    store = ExperimentStore(tmp_path / "experiments")
    store.save(experiment)
    record = {
        "research_id": "study",
        "completed_experiment_ids": ["exp-detail"],
        "failed_runs": [],
    }
    summary = rebuild_dashboard_summary_from_store(record, store.directory)
    assert summary["model_metrics"]["six_number_set"][0]["validation_activation_count"] == 20
    assert summary["hit_totals"]["six_number_set:validation"]["5_plus"] == 2
    assert summary["unique_hit_totals"]["six_number_set"]["validation"]["5_plus"] == 2


def test_missing_legacy_registration_data_is_not_reported_as_zero() -> None:
    experiment = _experiment("legacy-exp", 0.4, elite=False)
    del experiment["pattern_library_candidates"]
    summary = summarize_experiment(experiment)
    assert summary["registration_metrics_available"] is False
    assert summary["new_pattern_count"] is None
    assert summary["duplicate_pattern_count"] is None


def test_report_missing_fields_are_explicit_and_comparison_flags_differences() -> None:
    experiment = _experiment("exp-1", 0.4, elite=False)
    summary = rebuild_dashboard_summary([experiment], failed_experiment_count=1)
    round_record = {
        "research_round": 2,
        "research_id": "uuid-2",
        "status": "completed",
        "dashboard_summary": summary,
        "config": experiment["config"],
        "failed_runs": 0,
    }
    report = build_chatgpt_report(round_record)
    assert "uuid-2" in report
    assert "\ubbf8\uc218\uc9d1" in report
    assert "\uc0ac\uc804 \uc608\uce21" in report
    other = {**round_record, "config": {**experiment["config"], "population_size": 32}}
    comparison = compare_research_rounds(round_record, other)
    assert comparison["directly_comparable"] is False
    assert comparison["differences"]["population_size"] == {"left": 8, "right": 32}


def test_profile_comparison_requires_matching_dataset_range_seed_and_budget() -> None:
    base = {
        "dataset_sha256": "same-data",
        "dataset_round_range": [753, 1246],
        "target_duration_seconds": 3600,
        "status": "completed",
        "engines": ["intermittent_gp"],
        "dashboard_summary": {"completed_experiment_count": 5},
        "config": {
            "feature_profile": "bonus_lag_v1",
            "seed": 17,
            "train_ratio": 0.6,
            "validation_ratio": 0.2,
            "population_size": 64,
            "generations": 10,
            "max_lag": 10,
        },
    }
    integrated = {
        **base,
        "config": {**base["config"], "feature_profile": "integrated_v1"},
    }

    comparison = compare_research_rounds(base, integrated)

    assert comparison["directly_comparable"] is True
    assert comparison["profile_comparison"] is True
    assert comparison["left_feature_profile"] == "bonus_lag_v1"
    assert comparison["right_feature_profile"] == "integrated_v1"

    different_seed = {**integrated, "config": {**integrated["config"], "seed": 18}}
    assert compare_research_rounds(base, different_seed)["directly_comparable"] is False
    incomplete = {**integrated, "status": "running"}
    assert compare_research_rounds(base, incomplete)["directly_comparable"] is False


def test_report_ranks_one_number_and_six_number_candidates_separately() -> None:
    experiment = _experiment("exp-separated", 0.4, elite=False)
    summary = rebuild_dashboard_summary([experiment])
    report = build_chatgpt_report(
        {
            "research_round": 1,
            "research_id": "separated",
            "status": "completed",
            "config": experiment["config"],
            "dashboard_summary": summary,
        },
        top_candidates_per_model=1,
    )

    assert "### 한 번호 모델" in report
    assert "### 6개 집합 모델" in report
    assert "## 상위 후보 (검증 기반 정렬)" not in report
    assert "실행 가능 후보 여부는 판정되지 않았습니다" in report


def test_report_keeps_retrospective_recommendations_out_of_prospective_section() -> None:
    report = build_chatgpt_report(
        {
            "research_round": 1,
            "research_id": "retrospective",
            "config": {"feature_profile": "integrated_v1"},
            "prediction_candidates": [
                {
                    "recommendation_status": "historical_reproduction",
                    "target_round": 1246,
                    "recommended_numbers": [1, 2, 3, 4, 5, 6],
                }
            ],
        }
    )

    assert "저장된 사전 추천 없음" in report
    assert "회고 분석 추천 기록" in report
    assert "사전 예측 성과에 미포함" in report


def test_profile_comparison_stats_include_rounds_predictions_and_hits() -> None:
    stats = model_phase_comparison_stats(
        {
            "model_metrics": {
                "six_number_set": [
                    {
                        "independent_test_score": 0.25,
                        "independent_test_target_round_count": 12,
                        "independent_test_activation_count": 10,
                        "independent_test_hit_count": 23,
                    }
                ]
            },
            "experiment_summaries": [
                {"split_rounds": {"test_rounds": [31, 42]}}
            ],
            "hit_totals": {
                "six_number_set:independent_test": {
                    "3_plus": 4,
                    "4_plus": 1,
                    "5_plus": 0,
                    "2_exact": 1,
                    "6_exact": 0,
                }
            },
        }
    )

    assert stats["six_number_set"]["independent_test"]["evaluation_round_count"] == 12
    assert (
        stats["six_number_set"]["independent_test"]["unique_evaluation_round_count"]
        == 12
    )
    assert stats["six_number_set"]["independent_test"]["active_prediction_count"] == 10
    assert stats["six_number_set"]["independent_test"]["hit_count"] == 23
    assert stats["six_number_set"]["independent_test"]["2_exact"] == 1


def test_report_includes_worker_start_failure_code_without_trial_counts() -> None:
    report = build_chatgpt_report(
        {
            "research_round": 13,
            "research_id": "b2c54ef1-763c-4885-afe9-f673a3ea3161",
            "status": "failed",
            "config": {},
            "failed_runs": 0,
            "failed_run_summary": [
                {
                    "error_type": "WorkerStartError",
                    "worker_exit_code": 2,
                    "error": "워커가 체크포인트 생성 전에 종료했습니다.",
                }
            ],
        }
    )
    assert "failed" in report
    assert "worker_exit_code" in report
    assert "2" in report
    assert "체크포인트 생성 전에 종료" in report


def test_clipboard_button_has_browser_confirmation_and_fallback() -> None:
    component = (
        Path(__file__).resolve().parents[1] / "src/lotto_pattern_lab/clipboard_component/index.html"
    ).read_text(encoding="utf-8")
    assert "navigator.clipboard.writeText(markdown)" in component
    assert 'document.execCommand("copy")' in component
    assert 'status: "copied"' in component
    assert 'status: "error"' in component
