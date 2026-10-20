from __future__ import annotations

from lotto_pattern_lab.randomized_controls import (
    control_score_rows,
    summarize_matched_random_controls,
)


def _search(test_score: float) -> dict:
    return {"selected": {"test_score": test_score}}


def test_matched_random_summary_reports_empirical_p_and_holm_adjustment() -> None:
    experiment = {
        "experiment_id": "experiment-test",
        "dataset_sha256": "data-hash",
        "one_number_model": {"selected": {"test_score": 0.2}},
        "six_number_set_model": {"selected": {"test_score": 0.2}},
        "random_controls": [
            {
                "replicate": 1,
                "data_seed": 20,
                "one_number": _search(0.1),
                "six_number_set": _search(0.1),
            },
            {
                "replicate": 2,
                "data_seed": 21,
                "one_number": _search(0.2),
                "six_number_set": _search(0.1),
            },
            {
                "replicate": 3,
                "data_seed": 22,
                "one_number": _search(0.0),
                "six_number_set": _search(0.1),
            },
        ],
    }

    report = summarize_matched_random_controls(experiment, bootstrap_samples=100)
    one_number = report["models"]["one_number"]
    six_numbers = report["models"]["six_number_set"]

    assert report["final_test_used_for_selection"] is False
    assert one_number["empirical_p_value"] == 0.5
    assert six_numbers["empirical_p_value"] == 0.25
    assert one_number["holm_adjusted_p_value"] == 0.5
    assert six_numbers["holm_adjusted_p_value"] == 0.5
    assert one_number["control_replicates"] == 3
    assert one_number["interpretation_status"] == "tentative"
    assert len(one_number["control_mean_95_ci"]) == 2


def test_random_control_score_rows_keep_replicate_and_seed_provenance() -> None:
    experiment = {
        "random_controls": [
            {"replicate": 7, "data_seed": 123, "six_number_set": _search(-0.2)}
        ]
    }

    assert control_score_rows(experiment, "six_number_set") == [
        {"반복": 7, "데이터 시드": 123, "독립 테스트 점수": -0.2}
    ]
