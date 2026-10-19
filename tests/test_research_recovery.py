from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

from lotto_pattern_lab.dataset import DrawDataset, DrawRecord, make_random_dataset
from lotto_pattern_lab.engines import get_engine
from lotto_pattern_lab.intermittent import IntermittentConfig
from lotto_pattern_lab.research import run_research_schedule


def _dated_dataset(length: int = 35) -> DrawDataset:
    source = make_random_dataset(length, seed=51, first_round=753)
    first_date = date(2017, 5, 6)
    return DrawDataset(
        tuple(
            DrawRecord(
                item.round_number,
                item.draw,
                first_date + timedelta(days=7 * offset),
            )
            for offset, item in enumerate(source.records)
        ),
        "synthetic",
        "dated synthetic recovery test",
    )


def test_resume_retries_failed_seeds_before_next_seed_and_restores_elapsed_time(
    tmp_path: Path, monkeypatch
) -> None:
    import lotto_pattern_lab.research as research

    checkpoint = tmp_path / "checkpoints" / "recovery.json"
    dataset = _dated_dataset(40)
    config = IntermittentConfig(
        population_size=8,
        generations=1,
        max_depth=2,
        seed=90,
        workers=1,
        train_ratio=0.55,
        validation_ratio=0.25,
        max_lag=5,
        control_runs=1,
        feature_profile="calendar_round",
    )
    real_engine = get_engine("intermittent_gp")
    calls: list[int] = []

    class FailThenRun:
        def run(self, context):
            calls.append(context.config.seed)
            if len(calls) <= 3:
                time.sleep(0.01)
                raise PermissionError(5, "simulated temporary storage lock")
            return real_engine.run(context)

    monkeypatch.setattr(research, "get_engine", lambda _engine: FailThenRun())
    first = run_research_schedule(
        dataset,
        config,
        duration_seconds=60,
        checkpoint_path=checkpoint,
        experiment_directory=tmp_path / "experiments",
        pattern_database=tmp_path / "patterns.sqlite",
    )
    assert first["status"] == "failed"
    assert [item["seed"] for item in first["failed_runs"]] == [90, 91, 92]
    assert all(item["failure_category"] == "experiment" for item in first["failed_runs"])
    assert all(item["engine"] == "intermittent_gp" for item in first["failed_runs"])
    assert all(item["research_id"] == first["research_id"] for item in first["failed_runs"])
    assert all(item["feature_profile"] == "calendar_round" for item in first["failed_runs"])
    assert all(item["stage"] == "engine_start" for item in first["failed_runs"])
    assert all("PermissionError" in item["traceback"] for item in first["failed_runs"])
    assert all("research.py" in item["traceback"] for item in first["failed_runs"])
    assert first["pending_retry_seeds"] == [90, 91, 92]
    assert first["rng_resume"]["next_seed"] == 93
    prior_elapsed = first["total_elapsed_seconds"]

    resumed = run_research_schedule(
        dataset,
        config,
        duration_seconds=60,
        checkpoint_path=checkpoint,
        experiment_directory=tmp_path / "experiments",
        pattern_database=tmp_path / "patterns.sqlite",
        resume=True,
        max_runs=1,
    )

    assert calls == [90, 91, 92, 90]
    assert [item["seed"] for item in resumed["completed_experiments"]] == [90]
    assert resumed["pending_retry_seeds"] == [91, 92]
    assert resumed["rng_resume"]["next_seed"] == 93
    assert resumed["total_elapsed_seconds"] >= prior_elapsed
    assert resumed["completed_runs"] == 1
    assert len(resumed["dashboard_summary"]["p_value_distributions"]["one_number"]) == 1
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["research_id"] == first["research_id"]
    assert saved["research_round"] == first["research_round"]
