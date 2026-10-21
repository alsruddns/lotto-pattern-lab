from __future__ import annotations

import copy

import pytest

from lotto_pattern_lab.research_dashboard import (
    dashboard_failure_records,
    normalize_dashboard_summary,
    stringify_dashboard_table_rows,
    validated_pattern_counts,
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


def test_valid_summary_is_copied_before_hydration_updates():
    original = {"completed_experiment_count": 1, "model_metrics": {}}
    summary, diagnostic = normalize_dashboard_summary(original)

    assert diagnostic is None
    assert summary is not original
    summary["new_pattern_count"] = 4
    assert "new_pattern_count" not in original


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
