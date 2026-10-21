from __future__ import annotations

from pathlib import Path

import pytest

import lotto_pattern_lab.research_optimization as optimization
from lotto_pattern_lab.intermittent import IntermittentConfig


class _Trial:
    number = 4

    def suggest_int(self, name, low, high, **kwargs):
        return high if name == "population_size" else low

    def suggest_float(self, name, low, high):
        return (low + high) / 2


def test_optimization_duration_and_eta_require_real_samples() -> None:
    assert optimization.format_optimization_duration(5 * 3600 + 7) == "05:00:07"
    assert optimization.format_optimization_duration(26 * 3600 + 3) == "26:00:03"
    assert optimization.estimate_optimization_remaining_seconds(4, 30.0, 1) is None
    assert optimization.estimate_optimization_remaining_seconds(4, 30.0, 2, 15) == 105
    assert optimization.estimate_optimization_remaining_seconds(0, None, 0) == 0


def test_quick_optimizer_trials_use_bounded_configs_and_disable_nested_controls() -> None:
    base = IntermittentConfig(population_size=16, generations=2, max_lag=10, control_runs=5)

    config = optimization._trial_config(_Trial(), base, "quick")

    assert config.population_size == 32
    assert config.generations == 1
    assert 1 <= config.max_lag <= 10
    assert config.control_runs == 0
    assert config.seed == base.seed + 4


def test_optimizer_objective_uses_validation_training_and_complexity_not_test() -> None:
    selected = {
        "training_score": 0.2,
        "validation_score": 0.1,
        "independent_test_score": 1000.0,
        "complexity": 10,
    }
    experiment = {
        "one_number_model": {"selected": selected},
        "six_number_set_model": {"selected": selected},
    }

    objective, components = optimization.validation_objective(experiment)

    assert objective == pytest.approx(0.1 - 0.01 - 0.005)
    assert components == {
        "validation_mean": 0.1,
        "training_validation_gap": pytest.approx(0.1),
        "mean_complexity": 10.0,
    }


def test_optimizer_fails_clearly_when_optional_optuna_is_missing(tmp_path, monkeypatch) -> None:
    from lotto_pattern_lab.dataset import make_random_dataset

    monkeypatch.setattr(
        optimization,
        "optuna_status",
        lambda: {
            "available": False,
            "install_extra": "install command",
        },
    )

    with pytest.raises(RuntimeError, match="install command"):
        optimization.optimize_gp(
            make_random_dataset(20, seed=31),
            IntermittentConfig(population_size=8, generations=1),
            study_name="missing-optuna",
            n_trials=1,
            storage_directory=tmp_path / "storage",
        )


def test_optimization_history_reads_sqlite_and_json_without_mutating_files(tmp_path) -> None:
    import hashlib
    import json
    import sqlite3

    root = tmp_path / "data" / "optimizations"
    root.mkdir(parents=True)
    database = root / "study.sqlite"
    state = root / "study.state.json"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE studies (study_id INTEGER PRIMARY KEY, study_name TEXT);
            CREATE TABLE study_user_attributes (study_id INTEGER, key TEXT, value_json TEXT);
            CREATE TABLE study_directions (study_id INTEGER, direction TEXT, objective INTEGER);
            CREATE TABLE trials (
                trial_id INTEGER PRIMARY KEY, study_id INTEGER, number INTEGER,
                state TEXT, datetime_start TEXT, datetime_complete TEXT
            );
            CREATE TABLE trial_values (trial_id INTEGER, objective INTEGER, value REAL);
            CREATE TABLE trial_params (trial_id INTEGER, param_name TEXT, param_value REAL);
            INSERT INTO studies VALUES (1, 'saved-study');
            INSERT INTO study_user_attributes VALUES (1, 'feature_profile', '"bonus_lag_v1"');
            INSERT INTO study_user_attributes VALUES (1, 'dataset_range', '[753, 1246]');
            INSERT INTO study_directions VALUES (1, 'MAXIMIZE', 0);
            INSERT INTO trials VALUES (1, 1, 0, 'COMPLETE', '2026-01-01', '2026-01-02');
            INSERT INTO trial_values VALUES (1, 0, 0.25);
            INSERT INTO trial_params VALUES (1, 'population_size', 56);
            """
        )
    state.write_text(
        json.dumps({"study_name": "saved-study", "status": "completed"}), encoding="utf-8"
    )
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (database, state)
    }

    records = optimization.list_optimization_studies(root)

    assert len(records) == 1
    assert records[0]["completed_trials"] == 1
    assert records[0]["best_trial_number"] == 0
    assert records[0]["best_value"] == 0.25
    assert records[0]["best_parameters"] == {"population_size": 56}
    assert records[0]["feature_profile"] == "bonus_lag_v1"
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (database, state)
    } == before


def test_new_optimization_refuses_to_reuse_existing_name_without_touching_state(
    tmp_path, monkeypatch
) -> None:
    import hashlib

    from lotto_pattern_lab.dataset import make_random_dataset

    root = tmp_path / "optimizations"
    root.mkdir()
    state_path = root / "duplicate.state.json"
    state_path.write_text('{"status":"completed"}', encoding="utf-8")
    before = hashlib.sha256(state_path.read_bytes()).hexdigest()
    monkeypatch.setattr(optimization, "optuna_status", lambda: {"available": True})

    with pytest.raises(FileExistsError, match="이미 있습니다"):
        optimization.optimize_gp(
            make_random_dataset(20, seed=35),
            IntermittentConfig(population_size=8, generations=1),
            study_name="duplicate",
            action="new",
            storage_directory=root,
        )

    assert hashlib.sha256(state_path.read_bytes()).hexdigest() == before


def test_resume_requires_existing_optuna_database(tmp_path, monkeypatch) -> None:
    from lotto_pattern_lab.dataset import make_random_dataset

    monkeypatch.setattr(optimization, "optuna_status", lambda: {"available": True})
    with pytest.raises(FileNotFoundError, match="SQLite"):
        optimization.optimize_gp(
            make_random_dataset(20, seed=36),
            IntermittentConfig(population_size=8, generations=1),
            study_name="missing-study",
            action="resume",
            storage_directory=tmp_path / "missing",
        )


def test_new_optuna_study_records_exact_dataset_snapshot_and_resume_is_immutable(
    tmp_path, monkeypatch
) -> None:
    import hashlib

    from lotto_pattern_lab.data_audit import dataset_hashes
    from lotto_pattern_lab.dataset import make_random_dataset

    dataset = make_random_dataset(30, seed=37)
    config = IntermittentConfig(
        population_size=8, generations=1, max_depth=2, workers=1, control_runs=0
    )
    monkeypatch.setattr(optimization, "optuna_status", lambda: {"available": True})
    optimization.optimize_gp(
        dataset,
        config,
        study_name="snapshot-study",
        mode="quick",
        action="new",
        n_trials=1,
        storage_directory=tmp_path,
        experiment_directory=tmp_path / "experiments",
    )
    record = optimization.list_optimization_studies(tmp_path)[0]
    assert record["dataset_snapshot_sha256"] == dataset_hashes(dataset)["bonus_inclusive_sha256"]
    assert record["dataset_hash_algorithm"] == dataset_hashes(dataset)["algorithm_version"]

    database = tmp_path / "snapshot-study.sqlite"
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    changed = make_random_dataset(30, seed=38)
    with pytest.raises(ValueError, match="데이터 스냅샷"):
        optimization.optimize_gp(
            changed,
            config,
            study_name="snapshot-study",
            mode="quick",
            action="resume",
            n_trials=1,
            storage_directory=tmp_path,
            experiment_directory=tmp_path / "experiments",
        )
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


def test_optimization_stop_is_safe_and_persisted_study_can_resume(tmp_path, monkeypatch) -> None:
    import json

    from lotto_pattern_lab.dataset import make_random_dataset

    monkeypatch.setattr(optimization, "optuna_status", lambda: {"available": True})
    selected = {
        "training_score": 0.2,
        "validation_score": 0.1,
        "independent_test_score": None,
        "complexity": 3,
    }
    monkeypatch.setattr(
        optimization,
        "run_intermittent_experiment",
        lambda *_args, **_kwargs: {
            "experiment_id": "mock-experiment",
            "runtime_seconds": 0.01,
            "one_number_model": {"selected": selected},
            "six_number_set_model": {"selected": selected},
        },
    )
    state_path = tmp_path / "safe-stop.state.json"
    run_id = "test-run-1"

    def request_after_first_trial(_event):
        if not state_path.exists():
            return
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "running":
            optimization.request_optimization_stop(state_path, run_id=run_id)

    result = optimization.optimize_gp(
        make_random_dataset(30, seed=41),
        IntermittentConfig(population_size=8, generations=1, max_depth=2, workers=1),
        study_name="safe-stop",
        mode="quick",
        action="new",
        n_trials=4,
        storage_directory=tmp_path,
        experiment_directory=tmp_path / "experiments",
        progress=request_after_first_trial,
        run_id=run_id,
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert result["status"] == "stopped"
    assert state["status"] == "stopped"
    assert state["run_id"] == run_id
    assert state["completed_trials"] == 1
    assert not Path(f"{state_path}.stop.json").exists()
    snapshot = optimization.optimization_monitor_snapshot(
        {"study_name": "safe-stop", "database_path": str(tmp_path / "safe-stop.sqlite")}
    )
    assert snapshot["available"] is True
    assert snapshot["counts"]["complete"] == 1
    assert snapshot["trials"][0]["number"] == 0
