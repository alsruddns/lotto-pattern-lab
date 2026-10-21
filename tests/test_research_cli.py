from __future__ import annotations

from lotto_pattern_lab import __main__ as app_cli
from lotto_pattern_lab.research_startup import worker_start_failure_metadata


def test_research_cli_accepts_cumulative_mode_and_optimization_source(monkeypatch, capsys):
    dataset = object()
    captured = {}
    monkeypatch.setattr(app_cli, "load_csv", lambda _path: dataset)

    def fake_run_research_schedule(actual_dataset, config, **kwargs):
        captured["dataset"] = actual_dataset
        captured["config"] = config
        captured.update(kwargs)
        return {"status": "test-only"}

    monkeypatch.setattr(app_cli, "run_research_schedule", fake_run_research_schedule)
    app_cli._run_research_cli(
        [
            "--csv",
            "unused.csv",
            "--duration-minutes",
            "480",
            "--checkpoint",
            "unused.json",
            "--research-id",
            "b2c54ef1-763c-4885-afe9-f673a3ea3161",
            "--engine",
            "intermittent_gp",
            "--feature-profile",
            "bonus_lag_v1",
            "--research-mode",
            "cumulative",
            "--optimization-origin-json",
            '{"study_name":"test-study"}',
        ]
    )

    assert captured["dataset"] is dataset
    assert captured["research_id"] == "b2c54ef1-763c-4885-afe9-f673a3ea3161"
    assert captured["research_mode"] == "cumulative"
    assert captured["optimization_origin"] == {"study_name": "test-study"}
    assert "test-only" in capsys.readouterr().out


def test_worker_exit_before_checkpoint_records_failure_without_fake_results():
    metadata = worker_start_failure_metadata(
        4242,
        2,
        "checkpoints/research_worker.log",
        failed_at="2026-10-21T00:55:28+00:00",
    )

    assert metadata["status"] == "failed"
    assert metadata["failure_reason"] == "worker_exited_before_checkpoint"
    assert metadata["worker_pid"] == 4242
    assert metadata["worker_exit_code"] == 2
    assert metadata["failed_run_summary"][0]["worker_exit_code"] == 2
    assert "exit code 2" in metadata["failed_run_summary"][0]["error"]
    assert "completed_runs" not in metadata
    assert "completed_experiment_ids" not in metadata
