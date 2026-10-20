from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

import pytest

from lotto_pattern_lab.dataset import DrawDataset, DrawRecord, make_random_dataset
from lotto_pattern_lab.engines import get_engine
from lotto_pattern_lab.intermittent import IntermittentConfig
from lotto_pattern_lab.pysr_backend import PySRConfig
from lotto_pattern_lab.research import _pysr_config_for_attempt, run_research_schedule


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


def test_pysr_attempt_config_uses_string_paths_and_distinct_attempt_directories(
    tmp_path: Path, monkeypatch
) -> None:
    import lotto_pattern_lab.research as research

    monkeypatch.setattr(research, "project_path", lambda path: tmp_path / path)
    base = PySRConfig(run_directory=None)
    first = _pysr_config_for_attempt(base, "research-1", 12456, 112)
    second = _pysr_config_for_attempt(base, "research-1", 12456, 113)

    assert first is not None and second is not None
    assert isinstance(first.run_directory, str)
    assert first.run_directory == str(
        tmp_path / "artifacts/pysr_runs/research-1/seed-12456-attempt-112"
    )
    assert first.run_directory != second.run_directory
    assert _pysr_config_for_attempt(None, "research-1", 1, 1) is None


def test_pysr_attempt_config_rejects_tuple_path_instead_of_stringifying_it() -> None:
    config = PySRConfig(run_directory=("some/path",))  # type: ignore[arg-type]

    try:
        _pysr_config_for_attempt(config, "research-1", 1, 1)
    except TypeError as error:
        assert "must be a string path or None" in str(error)
    else:
        raise AssertionError("tuple paths must be rejected, not converted to strings")


def test_completed_checkpoint_retries_only_failed_seeds_when_explicitly_requested(
    tmp_path: Path, monkeypatch
) -> None:
    import lotto_pattern_lab.research as research

    checkpoint = tmp_path / "checkpoints" / "completed-failures.json"
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
        control_runs=0,
        feature_profile="calendar_round",
    )
    real_engine = get_engine("intermittent_gp")
    calls: list[tuple[str, int]] = []

    class FailThenRun:
        def __init__(self, engine_id: str) -> None:
            self.engine_id = engine_id

        def run(self, context):
            calls.append((self.engine_id, context.config.seed))
            if len(calls) <= 3:
                raise TypeError("simulated path failure with full traceback")
            return real_engine.run(context)

    monkeypatch.setattr(research, "get_engine", lambda engine_id: FailThenRun(engine_id))
    engines = ("intermittent_gp", "pysr_symbolic_regression")
    pysr_config = PySRConfig(niterations=1)
    first = run_research_schedule(
        dataset,
        config,
        duration_seconds=60,
        checkpoint_path=checkpoint,
        experiment_directory=tmp_path / "experiments",
        pattern_database=tmp_path / "patterns.sqlite",
        engines=engines,
        pysr_config=pysr_config,
    )
    assert first["status"] == "failed"
    assert first["completed_runs"] == 0
    assert first["pending_retry_seeds"] == [90, 91, 92]
    assert [item["engine"] for item in first["failed_runs"]] == [
        "intermittent_gp",
        "pysr_symbolic_regression",
        "intermittent_gp",
    ]
    assert all("TypeError" in failed["traceback"] for failed in first["failed_runs"])

    completed = json.loads(checkpoint.read_text(encoding="utf-8"))
    completed["status"] = "completed"
    completed["stop_reason"] = "time_budget_reached"
    checkpoint.write_text(json.dumps(completed), encoding="utf-8")
    with pytest.raises(ValueError, match="explicitly requesting retry_failed_runs"):
        run_research_schedule(
            dataset,
            config,
            duration_seconds=60,
            checkpoint_path=checkpoint,
            experiment_directory=tmp_path / "experiments",
            pattern_database=tmp_path / "patterns.sqlite",
            engines=engines,
            pysr_config=pysr_config,
            resume=True,
            max_runs=1,
        )

    resumed = run_research_schedule(
        dataset,
        config,
        duration_seconds=60,
        checkpoint_path=checkpoint,
        experiment_directory=tmp_path / "experiments",
        pattern_database=tmp_path / "patterns.sqlite",
        engines=engines,
        pysr_config=pysr_config,
        resume=True,
        retry_failed_runs=True,
        max_runs=2,
    )

    assert calls == [
        ("intermittent_gp", 90),
        ("pysr_symbolic_regression", 91),
        ("intermittent_gp", 92),
        ("intermittent_gp", 90),
        ("pysr_symbolic_regression", 91),
    ]
    assert resumed["research_id"] == first["research_id"]
    assert resumed["completed_runs"] == 2
    assert [item["seed"] for item in resumed["completed_experiments"]] == [90, 91]
    assert len(resumed["failed_runs"]) == 3
    resolved = {
        item["seed"]: item["resolved_by_experiment_id"]
        for item in resumed["failed_runs"]
        if item.get("resolved_by_experiment_id")
    }
    assert resolved[90] == resumed["completed_experiments"][0]["experiment_id"]
    assert resolved[91] == resumed["completed_experiments"][1]["experiment_id"]
    assert 92 not in resolved
    assert resumed["pending_retry_seeds"] == [92]


def test_completed_checkpoint_can_retry_only_user_selected_seed(
    tmp_path: Path, monkeypatch
) -> None:
    import lotto_pattern_lab.research as research

    checkpoint = tmp_path / "checkpoints" / "selected-failure.json"
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
        control_runs=0,
        feature_profile="calendar_round",
    )
    real_engine = get_engine("intermittent_gp")
    calls: list[tuple[str, int]] = []

    class FailThenRun:
        def __init__(self, engine_id: str) -> None:
            self.engine_id = engine_id

        def run(self, context):
            calls.append((self.engine_id, context.config.seed))
            if len(calls) <= 3:
                raise RuntimeError("test-only seed failure")
            return real_engine.run(context)

    monkeypatch.setattr(research, "get_engine", lambda engine_id: FailThenRun(engine_id))
    engines = ("intermittent_gp", "pysr_symbolic_regression")
    pysr_config = PySRConfig(niterations=1)
    first = run_research_schedule(
        dataset,
        config,
        duration_seconds=60,
        checkpoint_path=checkpoint,
        experiment_directory=tmp_path / "experiments",
        pattern_database=tmp_path / "patterns.sqlite",
        engines=engines,
        pysr_config=pysr_config,
    )
    assert first["pending_retry_seeds"] == [90, 91, 92]
    completed = json.loads(checkpoint.read_text(encoding="utf-8"))
    completed["status"] = "completed"
    checkpoint.write_text(json.dumps(completed), encoding="utf-8")

    retried = run_research_schedule(
        dataset,
        config,
        duration_seconds=60,
        checkpoint_path=checkpoint,
        experiment_directory=tmp_path / "experiments",
        pattern_database=tmp_path / "patterns.sqlite",
        engines=engines,
        pysr_config=pysr_config,
        resume=True,
        retry_seeds=(91,),
    )

    assert calls == [
        ("intermittent_gp", 90),
        ("pysr_symbolic_regression", 91),
        ("intermittent_gp", 92),
        ("pysr_symbolic_regression", 91),
    ]
    assert retried["stop_reason"] == "selected_failed_seed_retries_completed"
    assert [item["seed"] for item in retried["completed_experiments"]] == [91]
    assert retried["pending_retry_seeds"] == [90, 92]


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
