from pathlib import Path

import pytest

from lotto_pattern_lab.research_reports import build_research_report, save_completed_research_report


def test_report_includes_reproducibility_and_interpretation_fields() -> None:
    report = build_research_report(
        {
            "research_id": "study-1",
            "status": "completed",
            "dataset_sha256": "abc123",
            "dataset_round_range": [753, 1246],
            "engines": ["intermittent_gp"],
            "config": {"population_size": 32, "generations": 4, "workers": 2},
            "dashboard_summary": {"candidate_count": 7, "completed_experiment_count": 3},
        }
    )

    assert report.startswith("# ")
    assert "abc123" in report
    assert "7" in report
    assert "\uc0ac\uc804 \uc608\uce21" in report
    assert "\ufffd" not in report


def test_report_save_is_immutable_and_completed_only(tmp_path: Path) -> None:
    state = {"research_id": "study-2", "status": "completed"}
    target = save_completed_research_report(state, directory=tmp_path)
    first_content = target.read_bytes()
    target.write_bytes(b"existing immutable report")

    assert save_completed_research_report(state, directory=tmp_path) == target
    assert target.read_bytes() == b"existing immutable report"
    assert first_content.startswith(b"# ")
    with pytest.raises(ValueError):
        save_completed_research_report({**state, "status": "paused"}, directory=tmp_path)


def test_report_rejects_windows_path_separators(tmp_path: Path) -> None:
    state = {"research_id": "study" + chr(92) + "outside", "status": "completed"}
    with pytest.raises(ValueError):
        save_completed_research_report(state, directory=tmp_path)
