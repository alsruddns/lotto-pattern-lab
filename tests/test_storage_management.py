"""Storage quotas, archive round trips, and guarded checkpoint cleanup."""

import json
import os
from pathlib import Path

from lotto_pattern_lab.storage import ExperimentStore
from lotto_pattern_lab.storage_management import (
    checkpoint_cleanup_plan,
    compress_experiment,
    experiment_compression_plan,
    load_storage_settings,
    save_storage_settings,
    storage_inventory,
    storage_over_quota,
)


def _large_experiment(experiment_id: str = "sample") -> dict:
    return {
        "experiment_id": experiment_id,
        "status": "completed",
        "dataset": {"source_kind": "synthetic", "dataset_sha256": "abc123"},
        "config": {"population_size": 10},
        "one_number_model": {
            "selected": {"formulas": ["x0"], "activation_condition": "x1"},
            "candidates": [
                {
                    "formulas": ["x0"],
                    "activation_events": [
                        {"round_number": round_number, "active": round_number % 2 == 0}
                        for round_number in range(100)
                    ],
                    "active_gaps": list(range(100)),
                }
            ],
        },
        "random_controls": [],
        "progress": [{"generation": index} for index in range(100)],
    }


def test_large_experiments_save_summary_and_lossless_compressed_details(tmp_path: Path) -> None:
    original = _large_experiment()
    store = ExperimentStore(tmp_path, minimum_compress_bytes=1)
    summary_path = store.save(original)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["storage"]["format"] == "summary+gzip-details-v1"
    assert summary["storage"]["dataset_ref"] == "abc123"
    assert summary["storage"]["statistics"]["progress_event_count"] == 100
    assert summary["one_number_model"]["candidates"][0]["activation_events"]["item_count"] == 100
    assert (tmp_path / "sample.details.json.gz").is_file()
    assert store.load("sample") == original


def test_existing_large_json_compression_is_previewed_and_reversible(tmp_path: Path) -> None:
    original = _large_experiment()
    source = tmp_path / "legacy.json"
    original["experiment_id"] = "legacy"
    source.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")

    plan = experiment_compression_plan(tmp_path, threshold=1)
    assert [Path(item["path"]).name for item in plan["candidates"]] == ["legacy.json"]
    assert plan["expected_bytes_freed"] > 0
    result = compress_experiment(source)

    assert result["archived"] is True
    assert source.is_file()
    assert (tmp_path / "legacy.details.json.gz").is_file()
    assert ExperimentStore(tmp_path).load("legacy") == original


def test_running_or_unfinished_experiments_are_not_archive_candidates(tmp_path: Path) -> None:
    value = _large_experiment("running")
    value["status"] = "running"
    path = tmp_path / "running.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    plan = experiment_compression_plan(tmp_path, threshold=1)
    assert not plan["candidates"]
    assert plan["protected"][0]["path"] == str(path)


def test_checkpoint_plan_protects_recent_resumable_active_and_exact_records(
    tmp_path: Path,
) -> None:
    now = 2_000_000_000
    for index in range(1, 11):
        status = "completed"
        record = {
            "research_id": f"r-{index}",
            "status": status,
            "last_updated_at": "2026-01-01T00:00:00+00:00",
            "resumable": index in {6, 7, 8},
        }
        if index == 6:
            record["status"] = "paused"
        if index == 7:
            record["worker_pid"] = os.getpid()
        if index == 8:
            record["discovered_exact_matches"] = [{"round_number": 800}]
        path = tmp_path / f"research_{index}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        os.utime(path, (now - index, now - index))

    plan = checkpoint_cleanup_plan(tmp_path, keep=5)
    candidate_names = {Path(item["path"]).name for item in plan["candidates"]}
    protected_names = {Path(item["path"]).name for item in plan["protected"]}
    assert candidate_names == {"research_9.json", "research_10.json"}
    assert {"research_6.json", "research_7.json", "research_8.json"} <= protected_names


def test_checkpoint_cleanup_preserves_resumable_failed_and_paused_runs(tmp_path: Path) -> None:
    for index, status in enumerate(("failed", "paused", "running"), start=1):
        record = {
            "research_id": f"recoverable-{index}",
            "status": status,
            "last_updated_at": "2026-10-19T00:00:00+00:00",
            "resumable": True,
        }
        (tmp_path / f"research_{index}.json").write_text(json.dumps(record), encoding="utf-8")

    plan = checkpoint_cleanup_plan(tmp_path, keep=0)

    assert not plan["candidates"]
    assert len(plan["protected"]) == 3


def test_storage_inventory_and_quota_warning(tmp_path: Path) -> None:
    folder = tmp_path / "experiments"
    folder.mkdir()
    (folder / "result.json").write_bytes(b"x" * 17)

    report = storage_inventory(tmp_path)
    assert report["total_bytes"] == 17
    assert report["folders"][0]["file_count"] == 1
    assert storage_over_quota(report, 16)
    assert not storage_over_quota(report, 17)


def test_storage_settings_default_safe_and_round_trip(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings = load_storage_settings(settings_path)
    assert settings["automatic_checkpoint_cleanup"] is False
    assert settings["checkpoint_keep"] == 5

    save_storage_settings(
        {
            "quota_bytes": 1234,
            "automatic_checkpoint_cleanup": True,
            "checkpoint_keep": 6,
        },
        settings_path,
    )
    saved = load_storage_settings(settings_path)
    assert saved["quota_bytes"] == 1234
    assert saved["automatic_checkpoint_cleanup"] is True
    assert saved["checkpoint_keep"] == 6


def test_streamlit_storage_settings_page_executes() -> None:
    from streamlit.testing.v1 import AppTest

    from lotto_pattern_lab.paths import PROJECT_ROOT

    page = PROJECT_ROOT / "src/lotto_pattern_lab/pages/4_저장_용량_관리.py"
    app = AppTest.from_file(str(page)).run(timeout=30)

    assert not app.exception


def test_research_page_uses_project_paths_from_another_working_directory(
    tmp_path: Path, monkeypatch
) -> None:
    from streamlit.testing.v1 import AppTest

    from lotto_pattern_lab.paths import PROJECT_ROOT

    monkeypatch.chdir(tmp_path)
    page = PROJECT_ROOT / "src/lotto_pattern_lab/pages/3_지속적_패턴_연구.py"

    app = AppTest.from_file(str(page)).run(timeout=30)

    assert not app.exception
    assert "다른 작업 디렉터리" not in str(app.exception)
