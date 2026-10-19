import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lotto_pattern_lab.research_rounds import (
    ResearchRoundStore,
    build_chatgpt_report,
    compare_research_rounds,
    legacy_research_records,
    rebuild_dashboard_summary,
    summarize_experiment,
)


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
    }
    report = build_chatgpt_report(round_record)
    assert "uuid-2" in report
    assert "\ubbf8\uc218\uc9d1" in report
    assert "\uc0ac\uc804 \uc608\uce21" in report
    other = {**round_record, "config": {**experiment["config"], "population_size": 32}}
    comparison = compare_research_rounds(round_record, other)
    assert comparison["directly_comparable"] is False
    assert comparison["differences"]["population_size"] == {"left": 8, "right": 32}


def test_clipboard_button_has_browser_confirmation_and_fallback() -> None:
    component = (
        Path(__file__).resolve().parents[1] / "src/lotto_pattern_lab/clipboard_component/index.html"
    ).read_text(encoding="utf-8")
    assert "navigator.clipboard.writeText(markdown)" in component
    assert 'document.execCommand("copy")' in component
    assert 'status: "copied"' in component
    assert 'status: "error"' in component
