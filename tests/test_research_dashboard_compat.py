from __future__ import annotations

import copy
from pathlib import Path

import pytest

from lotto_pattern_lab.draw import Draw
from lotto_pattern_lab.research_dashboard import (
    dashboard_failure_records,
    normalize_dashboard_summary,
    research_aggregation_status,
    stringify_dashboard_table_rows,
    validated_pattern_counts,
)
from lotto_pattern_lab.research_rounds import (
    _attach_verified_bonus_outcomes,
    rebuild_dashboard_summary,
    safely_reaggregate_second_prize,
    summarize_experiment,
)
from lotto_pattern_lab.research_startup import research_startup_message


def _counts() -> dict:
    return {
        "new_pattern_count": 0,
        "duplicate_pattern_count": 0,
        "registration_attempt_count": 0,
        "child_pattern_count": 0,
        "new_child_pattern_count": 0,
        "modified_child_pattern_count": 0,
        "new_modified_child_pattern_count": 0,
        "parent_pattern_count": 0,
        "lineage_event_counts": {},
    }


@pytest.mark.parametrize(
    ("value", "missing", "state", "has_diagnostic"),
    [
        (
            {"completed_experiment_count": 2, "model_metrics": {}},
            False,
            "available",
            False,
        ),
        (None, False, "pending_or_invalid", True),
        (None, True, "pending_or_invalid", True),
        ({}, False, "pending_or_invalid", True),
        (
            '{"completed_experiment_count": 1, "model_metrics": {}}',
            False,
            "available",
            False,
        ),
        ("{broken json", False, "pending_or_invalid", True),
        (["unexpected"], False, "pending_or_invalid", True),
        (
            {"completed_experiment_count": 1, "model_metrics": None},
            False,
            "pending_or_invalid",
            True,
        ),
    ],
)
def test_summary_legacy_and_invalid_shapes_are_normalized_without_mutation(
    value, missing, state, has_diagnostic
):
    original = copy.deepcopy(value)
    summary, diagnostic = normalize_dashboard_summary(value, missing=missing)

    assert isinstance(summary, dict)
    assert summary["summary_state"] == state
    assert bool(diagnostic) is has_diagnostic
    assert summary["completed_experiment_count"] is None or isinstance(
        summary["completed_experiment_count"], int
    )
    if not missing:
        assert value == original


def test_summary_nested_records_and_hit_totals_are_safely_diagnosed():
    summary, diagnostic = normalize_dashboard_summary(
        {
            "completed_experiment_count": 1,
            "hit_totals": {"six_number_set:validation": {"3_plus": "bad"}},
            "experiment_summaries": [{"models": None, "p_values": []}],
        }
    )

    assert diagnostic
    assert summary["hit_totals"] == {}
    assert summary["experiment_summaries"][0]["models"] == {}
    assert summary["experiment_summaries"][0]["p_values"] == {}


def test_numeric_p_value_distribution_is_a_valid_aggregate_field():
    summary, diagnostic = normalize_dashboard_summary(
        {
            "summary_schema_version": 2,
            "completed_experiment_count": 1,
            "p_value_distributions": {"one_number": [0.1, 0.25]},
        }
    )

    assert diagnostic is None
    assert summary["summary_state"] == "available"
    assert summary["p_value_distributions"]["one_number"] == [0.1, 0.25]


@pytest.mark.parametrize(
    ("record", "summary", "diagnostic", "expected"),
    [
        ({"status": "completed"}, {"completed_experiment_count": 2}, None, "집계 완료"),
        (
            {"status": "completed", "completed_experiment_ids": ["e1"]},
            {"completed_experiment_count": None},
            "summary missing",
            "집계 대기",
        ),
        ({"status": "failed"}, {"completed_experiment_count": 0}, None, "집계 대상 아님"),
        ({"status": "failed"}, {"completed_experiment_count": 2}, None, "집계 완료"),
        ({"status": "paused"}, {"completed_experiment_count": 2}, None, "부분 집계"),
        (
            {"status": "completed"},
            {"completed_experiment_count": 2},
            "optional metric invalid",
            "부분 집계",
        ),
        ({"status": "failed"}, {}, "invalid JSON", "집계 오류"),
    ],
)
def test_research_aggregate_status_is_independent_from_execution_status(
    record, summary, diagnostic, expected
):
    assert research_aggregation_status(record, summary, diagnostic) == expected


def test_valid_summary_is_copied_before_hydration_updates():
    original = {"completed_experiment_count": 1, "model_metrics": {}}
    summary, diagnostic = normalize_dashboard_summary(original)

    assert diagnostic is None
    assert summary is not original
    summary["new_pattern_count"] = 4
    assert "new_pattern_count" not in original


def test_queued_or_failed_without_experiments_is_not_an_aggregation_error():
    assert (
        research_aggregation_status(
            {"status": "queued"}, {}, "summary가 아직 저장되지 않아 집계 대기 상태입니다."
        )
        == "집계 대상 아님"
    )
    assert (
        research_aggregation_status(
            {"status": "failed"}, {}, "summary가 아직 저장되지 않아 집계 대기 상태입니다."
        )
        == "집계 대상 아님"
    )
    assert research_aggregation_status({"status": "running"}, {}, None) == "집계 진행 중"


def test_pattern_counts_accept_complete_dict_and_reject_missing_or_wrong_values():
    normalized, diagnostic = validated_pattern_counts(_counts())
    assert diagnostic is None
    assert normalized == _counts()

    for invalid in ({}, None, [], {**_counts(), "new_pattern_count": -1}):
        result, error = validated_pattern_counts(invalid)
        assert result is None
        assert error


def test_dashboard_exposes_startup_failure_without_counting_it_as_an_experiment():
    startup_failure = [{"error_type": "WorkerStartError", "worker_exit_code": 2}]
    first_result = dashboard_failure_records(
        {"failed_runs": [], "failed_run_summary": startup_failure}
    )
    assert first_result == startup_failure
    assert dashboard_failure_records({"failed_runs": 0, "failed_run_summary": startup_failure}) == (
        startup_failure
    )
    assert dashboard_failure_records({"failed_runs": 0}) == 0


def test_comparison_table_converts_missing_and_numeric_values_to_arrow_safe_display_strings():
    import pandas as pd
    import pyarrow as pa

    rows = stringify_dashboard_table_rows(
        [
            {"일반 GP 검증": 0.125, "일반 GP 독립 테스트": None, "대조 반복": 4},
            {"일반 GP 검증": None, "일반 GP 독립 테스트": 0.5, "대조 반복": None},
        ]
    )
    frame = pd.DataFrame(rows)

    assert rows[0]["일반 GP 검증"] == "0.125"
    assert rows[1]["일반 GP 검증"] == "해당 없음"
    assert rows[0]["대조 반복"] == "4"
    assert all(
        all(isinstance(value, str) for value in frame[column].tolist())
        for column in frame.columns
    )
    assert pa.Table.from_pandas(frame, preserve_index=False).num_rows == 2


def test_research_summary_counts_exact_second_prizes_without_changing_five_plus():
    experiment = {
        "status": "completed",
        "experiment_id": "test",
        "dataset": {"dataset_sha256": "snapshot"},
        "splits": {"train_rounds": [1, 1], "validation_rounds": [2, 2], "test_rounds": [3, 3]},
        "six_number_set_model": {
            "selected": {
                "match_distribution_by_split": {
                    "validation": {"5": 1},
                    "independent_test": {"0": 1},
                },
                "activation_events": [
                    {
                        "round_number": 2,
                        "active": True,
                        "predicted_numbers": [1, 5, 12, 20, 33, 17],
                        "winning_numbers": [1, 5, 12, 20, 33, 40],
                        "hit_count": 5,
                        "full_set_match": False,
                        "actual_bonus_number": 17,
                        "prize_tier": 2,
                    }
                ],
            }
        },
    }
    compact = summarize_experiment(experiment)
    summary = rebuild_dashboard_summary([compact])
    assert summary["hit_totals"]["six_number_set:validation"]["5_plus"] == 1
    assert summary["hit_totals"]["six_number_set:validation"]["2_exact"] == 1
    assert summary["unique_hit_totals"]["six_number_set"]["validation"]["2_exact"] == 1


def test_pysr_prize_metrics_stay_separate_from_gp_repeated_hit_totals():
    compact = summarize_experiment(
        {
            "status": "completed",
            "engine": "pysr_symbolic_regression",
            "candidate_count": 1,
            "six_number_set_metrics": {
                "validation": {
                    "three_plus": 4,
                    "four_plus": 1,
                    "five_plus": 0,
                    "six_exact": 0,
                    "second_prize_exact": 0,
                    "second_prize_known": 10,
                    "second_prize_unknown": 0,
                }
            },
        }
    )
    summary = rebuild_dashboard_summary([compact])
    assert summary["hit_totals"] == {}
    assert summary["experiment_summaries"][0]["six_number_set_metrics"]["validation"][
        "second_prize_known"
    ] == 10


def test_historical_bonus_reconstruction_requires_exact_snapshot_and_main_draw_match(
    tmp_path, monkeypatch
):
    source = tmp_path / "bonus.csv"
    source.touch()
    monkeypatch.setattr(
        "lotto_pattern_lab.research_rounds.project_path",
        lambda relative: source if "raw" in str(relative) else tmp_path / "missing.csv",
    )
    monkeypatch.setattr(
        "lotto_pattern_lab.research_rounds._official_bonus_dataset",
        lambda _: ("exact-snapshot", {1: Draw.from_numbers((1, 5, 12, 20, 33, 40), 17)}),
    )
    matching = {
        "dataset": {"dataset_sha256": "exact-snapshot"},
        "six_number_set_model": {
            "selected": {
                "activation_events": [
                    {
                        "round_number": 1,
                        "active": True,
                        "predicted_numbers": [1, 5, 12, 20, 33, 17],
                        "winning_numbers": [1, 5, 12, 20, 33, 40],
                        "hit_count": 5,
                    }
                ]
            }
        },
    }
    _attach_verified_bonus_outcomes(matching)
    event = matching["six_number_set_model"]["selected"]["activation_events"][0]
    assert event["actual_bonus_number"] == 17
    assert event["prize_tier"] == 2

    mismatch = {
        "dataset": {"dataset_sha256": "different-snapshot"},
        "six_number_set_model": {
            "selected": {
                "activation_events": [
                    {
                        "round_number": 1,
                        "active": True,
                        "predicted_numbers": [1, 5, 12, 20, 33, 17],
                        "winning_numbers": [1, 5, 12, 20, 33, 40],
                        "hit_count": 5,
                    }
                ]
            }
        },
    }
    _attach_verified_bonus_outcomes(mismatch)
    mismatch_event = mismatch["six_number_set_model"]["selected"]["activation_events"][0]
    assert "actual_bonus_number" not in mismatch_event


def test_safe_second_prize_reaggregation_backs_up_and_preserves_existing_totals(
    tmp_path, monkeypatch
):
    import json
    import sqlite3

    database = tmp_path / "patterns.sqlite"
    metadata = {
        "research_id": "research-1",
        "dashboard_summary": {
            "summary_schema_version": 2,
            "completed_experiment_count": 1,
            "hit_totals": {
                "six_number_set:validation": {
                    "3_plus": 3,
                    "4_plus": 2,
                    "5_plus": 1,
                    "6_exact": 0,
                }
            },
        },
    }
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE research_rounds (research_id TEXT PRIMARY KEY, metadata_json TEXT)"
    )
    connection.execute(
        "INSERT INTO research_rounds VALUES (?, ?)",
        ("research-1", json.dumps(metadata)),
    )
    connection.commit()
    connection.close()

    rebuilt = {
        "hit_totals": {
            "six_number_set:validation": {
                "3_plus": 3,
                "4_plus": 2,
                "5_plus": 1,
                "6_exact": 0,
                "2_exact": 1,
                "2_known": 10,
                "2_unknown": 0,
            }
        },
        "unique_hit_totals": {"six_number_set": {"validation": {"2_exact": 1}}},
    }
    monkeypatch.setattr(
        "lotto_pattern_lab.research_rounds.rebuild_dashboard_summary_from_store",
        lambda *_: rebuilt,
    )
    monkeypatch.setattr(
        "lotto_pattern_lab.research_rounds.project_path",
        lambda path: (
            tmp_path / "global.lock"
            if str(path) == "checkpoints/research_global"
            else tmp_path / "backups"
            if str(path) == "data/backups"
            else Path(path)
        ),
    )

    result = safely_reaggregate_second_prize(
        {"research_id": "research-1"}, tmp_path / "experiments", database
    )
    assert Path(result["backup_path"]).is_file()
    updated = result["summary"]["hit_totals"]["six_number_set:validation"]
    assert [updated[key] for key in ("3_plus", "4_plus", "5_plus", "6_exact")] == [3, 2, 1, 0]
    assert updated["2_exact"] == 1
    assert updated["2_known"] == 10

    check = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    assert check.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    check.close()


@pytest.mark.parametrize(
    ("phase", "severity", "message_part"),
    [
        ("ready", "info", "준비됨"),
        ("environment_checking", "info", "검사하고"),
        ("environment_passed", "success", "통과"),
        ("environment_failed", "error", "실패"),
        ("background_starting", "info", "요청"),
        ("process_verifying", "info", "확인"),
        ("first_checkpoint_waiting", "info", "체크포인트"),
        ("running", "success", "실행"),
        ("completed", "success", "완료"),
        ("failed", "error", "실패"),
        ("paused", "warning", "일시정지"),
        ("interrupted", "warning", "중단"),
        ("start_unconfirmed", "warning", "확인되지"),
        ("start_failed", "error", "실패"),
    ],
)
def test_research_startup_states_have_korean_phase_specific_messages(phase, severity, message_part):
    actual_severity, message = research_startup_message(phase)
    assert actual_severity == severity
    assert message_part in message
