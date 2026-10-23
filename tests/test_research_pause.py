from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

from lotto_pattern_lab.dataset import DrawDataset, DrawRecord, make_random_dataset
from lotto_pattern_lab.engines import get_engine
from lotto_pattern_lab.intermittent import IntermittentConfig
from lotto_pattern_lab.research import request_pause, run_research_schedule
from lotto_pattern_lab.research_rounds import ResearchRoundStore


def _dataset(length: int = 36) -> DrawDataset:
    source = make_random_dataset(length, seed=921, first_round=753)
    first_date = date(2017, 5, 6)
    return DrawDataset(
        tuple(
            DrawRecord(item.round_number, item.draw, first_date + timedelta(days=7 * index))
            for index, item in enumerate(source.records)
        ),
        "synthetic",
        "pause unit test",
    )


def _config() -> IntermittentConfig:
    return IntermittentConfig(
        population_size=8,
        generations=1,
        max_depth=2,
        workers=1,
        seed=61,
        train_ratio=0.55,
        validation_ratio=0.25,
        max_lag=5,
        control_runs=0,
        feature_profile="legacy",
    )


def test_pause_request_waits_for_experiment_save_and_resumes_next_seed(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoints" / "pause.json"
    experiments = tmp_path / "experiments"
    database = tmp_path / "patterns.sqlite"
    pause_sent = False

    def ask_for_pause(_event: dict[str, object]) -> None:
        nonlocal pause_sent
        if not pause_sent:
            pause_sent = True
            request_pause(checkpoint, pattern_database=database)

    first = run_research_schedule(
        _dataset(),
        _config(),
        duration_seconds=60,
        checkpoint_path=checkpoint,
        experiment_directory=experiments,
        pattern_database=database,
        progress=ask_for_pause,
    )

    assert first["status"] == "paused"
    assert first["completed_runs"] == 1
    assert len(first["completed_experiments"]) == 1
    assert not Path(f"{checkpoint}.pause").exists()
    assert ResearchRoundStore(database).get(first["research_id"])["status"] == "paused"
    paused_elapsed = first["total_elapsed_seconds"]

    time.sleep(0.03)
    resumed = run_research_schedule(
        _dataset(),
        _config(),
        duration_seconds=60,
        checkpoint_path=checkpoint,
        experiment_directory=experiments,
        pattern_database=database,
        resume=True,
        max_runs=2,
    )
    assert resumed["research_id"] == first["research_id"]
    assert resumed["research_round"] == first["research_round"]
    assert resumed["status"] == "completed"
    assert resumed["completed_runs"] == 2
    assert [item["seed"] for item in resumed["completed_experiments"]] == [61, 62]
    assert resumed["total_elapsed_seconds"] >= paused_elapsed


def test_pause_message_marker_is_consumed_only_after_safe_boundary(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checks" / "pause-boundary.json"
    database = tmp_path / "patterns.sqlite"
    engine = get_engine("intermittent_gp")

    class PauseDuringTrial:
        def run(self, context):
            result = engine.run(context)
            request_pause(checkpoint, pattern_database=database)
            return result

    import lotto_pattern_lab.research as research

    original_get_engine = research.get_engine
    research.get_engine = lambda _engine: PauseDuringTrial()
    try:
        result = run_research_schedule(
            _dataset(),
            _config(),
            duration_seconds=60,
            checkpoint_path=checkpoint,
            experiment_directory=tmp_path / "experiments",
            pattern_database=database,
        )
    finally:
        research.get_engine = original_get_engine

    assert result["status"] == "paused"
    assert result["completed_runs"] == 1
    assert Path(result["completed_experiments"][0]["path"]).is_file()
    assert not Path(f"{checkpoint}.pause").exists()


def test_experiment_save_failure_does_not_become_paused(tmp_path: Path, monkeypatch) -> None:
    from lotto_pattern_lab.storage import ExperimentStore

    checkpoint = tmp_path / "checks" / "pause-save-failure.json"
    database = tmp_path / "patterns.sqlite"
    original_save = ExperimentStore.save
    failed_once = False

    def fail_result_save(self, record):
        nonlocal failed_once
        if not failed_once and record.get("experiment_type") == "intermittent":
            failed_once = True
            raise PermissionError(5, "simulated result.json.tmp sharing violation")
        return original_save(self, record)

    monkeypatch.setattr(ExperimentStore, "save", fail_result_save)
    pause_sent = False

    def ask_for_pause(_event: dict[str, object]) -> None:
        nonlocal pause_sent
        if not pause_sent:
            pause_sent = True
            request_pause(checkpoint, pattern_database=database)

    result = run_research_schedule(
        _dataset(),
        _config(),
        duration_seconds=60,
        checkpoint_path=checkpoint,
        experiment_directory=tmp_path / "experiments",
        pattern_database=database,
        progress=ask_for_pause,
    )

    assert result["status"] == "failed"
    assert result["failure_reason"] == "experiment_storage_failure"
    assert result["failed_runs"][-1]["failure_category"] == "storage"
    assert Path(f"{checkpoint}.pause").exists()
