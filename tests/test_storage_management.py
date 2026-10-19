"""Storage quotas, archive round trips, and guarded checkpoint cleanup."""

import json
import os
from pathlib import Path

from lotto_pattern_lab.storage import ExperimentStore
from lotto_pattern_lab.storage_management import (
    checkpoint_cleanup_plan,
    compress_experiment,
    compress_pysr_artifacts,
    create_storage_backup,
    experiment_compression_plan,
    load_storage_settings,
    pysr_artifact_compression_plan,
    save_storage_settings,
    storage_inventory,
    storage_over_quota,
    verify_storage_backup,
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


def test_storage_inventory_counts_sqlite_pysr_artifacts_and_research_usage(
    tmp_path: Path,
) -> None:
    database = tmp_path / "data" / "database" / "patterns.sqlite"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"db" * 7)
    pysr_file = tmp_path / "artifacts" / "pysr_runs" / "model.pkl"
    pysr_file.parent.mkdir(parents=True)
    pysr_file.write_bytes(b"pysr" * 5)
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "trial.json").write_text(
        json.dumps({"research_id": "research-x"}), encoding="utf-8"
    )

    report = storage_inventory(tmp_path)

    assert report["file_types"]["sqlite"]["bytes"] == 14
    assert report["file_types"]["pkl"]["bytes"] == 20
    assert report["research_usage"]["research-x"]["file_count"] == 1


def test_storage_backup_creates_and_verifies_hash_manifest(tmp_path: Path) -> None:
    source = tmp_path / "experiments" / "study.json"
    source.parent.mkdir()
    source.write_text(json.dumps({"research_id": "r1", "status": "completed"}), encoding="utf-8")

    result = create_storage_backup(tmp_path / "backups", root=tmp_path)

    assert result["file_count"] == 1
    assert verify_storage_backup(result["path"]) == {"valid": True, "verified_files": 1}
    assert source.is_file()


def test_pysr_artifact_plan_preserves_recoverable_runs_and_checkpoints(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    pysr_root = tmp_path / "artifacts" / "pysr_runs"
    completed_id = "research-completed"
    paused_id = "research-paused"
    for research_id, status in ((completed_id, "completed"), (paused_id, "paused")):
        (checkpoints / f"research_{research_id}.json").write_text(
            json.dumps({"research_id": research_id, "status": status}), encoding="utf-8"
        )
        run_dir = pysr_root / research_id / "seed-2"
        run_dir.mkdir(parents=True)
        (run_dir / "hall_of_fame.csv").write_text("equation,loss\nx0,1\n", encoding="utf-8")
        (run_dir / "checkpoint.pkl").write_bytes(b"model state")

    plan = pysr_artifact_compression_plan(pysr_root, checkpoints)
    assert len(plan["candidates"]) == 1
    assert any("recoverable" in item["reason"] for item in plan["protected"])

    result = compress_pysr_artifacts(pysr_root, checkpoints)
    completed_dir = pysr_root / completed_id / "seed-2"
    assert len(result["compressed"]) == 1
    assert not (completed_dir / "hall_of_fame.csv").exists()
    assert (completed_dir / "hall_of_fame.csv.gz").is_file()
    assert (completed_dir / "checkpoint.pkl").read_bytes() == b"model state"
    assert (pysr_root / paused_id / "seed-2" / "hall_of_fame.csv").is_file()


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
