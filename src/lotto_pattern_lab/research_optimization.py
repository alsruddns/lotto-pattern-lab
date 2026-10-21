"""Optional Optuna-backed GP configuration search using validation-only objectives."""

from __future__ import annotations

import importlib.util
import json
import re
import sqlite3
import statistics
import uuid
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_bytes
from .data_audit import dataset_hashes
from .dataset import DrawDataset
from .intermittent import IntermittentConfig
from .intermittent_experiment import run_intermittent_experiment
from .paths import project_path
from .storage import ExperimentStore

OPTIMIZATION_SCHEMA_VERSION = 1


def list_optimization_studies(
    storage_directory: str | Path = "data/optimizations",
) -> list[dict[str, Any]]:
    """Read Optuna studies and sidecar state without opening databases for writing."""
    root = project_path(storage_directory)
    if not root.is_dir():
        return []
    records: dict[str, dict[str, Any]] = {}
    for state_path in root.glob("*.state.json"):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            name = str(state.get("study_name") or state_path.name.removesuffix(".state.json"))
            records[name] = {
                **state,
                "study_name": name,
                "state_path": str(state_path),
                "database_path": str(root / f"{_study_slug(name)}.sqlite"),
                "database_available": False,
                "best_parameters": {},
            }
        except (OSError, ValueError, TypeError):
            records[state_path.name.removesuffix(".state.json")] = {
                "study_name": state_path.name.removesuffix(".state.json"),
                "status": "state_unreadable",
                "state_path": str(state_path),
                "database_path": str(
                    root / f"{state_path.name.removesuffix('.state.json')}.sqlite"
                ),
                "database_available": False,
                "best_parameters": {},
            }

    for database_path in root.glob("*.sqlite"):
        try:
            uri = database_path.resolve().as_uri() + "?mode=ro"
            with sqlite3.connect(uri, uri=True, timeout=1) as connection:
                rows = connection.execute(
                    "SELECT study_id, study_name FROM studies ORDER BY study_id"
                ).fetchall()
                for study_id, name_value in rows:
                    name = str(name_value)
                    record = records.setdefault(
                        name,
                        {
                            "study_name": name,
                            "status": "기록 확인 필요",
                            "completed_trials": 0,
                            "best_parameters": {},
                        },
                    )
                    record.update(database_path=str(database_path), database_available=True)
                    record["optuna_study_id"] = int(study_id)
                    for key, value in connection.execute(
                        "SELECT key, value_json FROM study_user_attributes WHERE study_id=?",
                        (study_id,),
                    ).fetchall():
                        try:
                            record[str(key)] = json.loads(value)
                        except (TypeError, ValueError):
                            record[str(key)] = value
                    if (
                        "engine" not in record
                        and record.get("objective_policy_version") == "validation_only_v1"
                    ):
                        # This policy version is written only by optimize_gp in this package.
                        record["engine"] = "intermittent_gp"
                        record["engine_metadata_source"] = "optimizer implementation metadata"
                    trials = connection.execute(
                        """SELECT t.trial_id, t.number, t.state, tv.value
                           FROM trials t LEFT JOIN trial_values tv ON tv.trial_id=t.trial_id
                           WHERE t.study_id=? AND tv.objective=0""",
                        (study_id,),
                    ).fetchall()
                    timing = connection.execute(
                        "SELECT MIN(datetime_start), MAX(datetime_complete) "
                        "FROM trials WHERE study_id=?",
                        (study_id,),
                    ).fetchone()
                    if timing and timing[0]:
                        record.setdefault("started_at", str(timing[0]))
                    if timing and timing[1]:
                        record.setdefault("finished_at", str(timing[1]))
                    completed = [
                        row for row in trials if row[2] == "COMPLETE" and row[3] is not None
                    ]
                    record["completed_trials"] = len(completed)
                    directions = connection.execute(
                        "SELECT direction FROM study_directions "
                        "WHERE study_id=? ORDER BY objective",
                        (study_id,),
                    ).fetchall()
                    maximize = not directions or directions[0][0] == "MAXIMIZE"
                    if completed:
                        best = (max if maximize else min)(completed, key=lambda row: float(row[3]))
                        record.update(best_trial_number=int(best[1]), best_value=float(best[3]))
                        params = connection.execute(
                            """SELECT param_name, param_value FROM trial_params
                               WHERE trial_id=? ORDER BY param_name""",
                            (best[0],),
                        ).fetchall()
                        record["best_parameters"] = {
                            str(key): (
                                int(value)
                                if str(key)
                                in {"population_size", "generations", "max_depth", "max_lag"}
                                and float(value).is_integer()
                                else float(value)
                            )
                            for key, value in params
                        }
                    elif trials:
                        record["best_trial_number"] = None
                        record["best_value"] = None
                    record.setdefault("mode", "기록 없음")
                    record.setdefault("status", "completed" if completed else "진행 기록 없음")
                    record["last_modified"] = datetime.fromtimestamp(
                        database_path.stat().st_mtime, UTC
                    ).isoformat()
        except (OSError, sqlite3.Error):
            continue
    return sorted(records.values(), key=lambda item: str(item.get("study_name", "")).casefold())


def list_optimization_run_states(
    storage_directory: str | Path = "data/optimizations",
) -> list[dict[str, Any]]:
    """Read only the small JSON state files used to identify live optimization workers."""
    root = project_path(storage_directory)
    if not root.is_dir():
        return []
    records = []
    for state_path in root.glob("*.state.json"):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                continue
            name = str(state.get("study_name") or state_path.name.removesuffix(".state.json"))
            records.append(
                {
                    **state,
                    "study_name": name,
                    "state_path": str(state_path),
                    "database_path": str(root / f"{_study_slug(name)}.sqlite"),
                }
            )
        except (OSError, ValueError, TypeError):
            continue
    return records


def optuna_status() -> dict[str, Any]:
    available = importlib.util.find_spec("optuna") is not None
    return {
        "available": available,
        "status": "available" if available else "optional_dependency_missing",
        "install_extra": '.venv\\Scripts\\python.exe -m pip install -e ".[optimization]"',
    }


def _study_slug(study_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", study_name).strip(".-")
    if not slug:
        raise ValueError("최적화 이름에 사용할 수 있는 문자가 없습니다.")
    return slug[:80]


def _trial_config(trial: Any, base: IntermittentConfig, mode: str) -> IntermittentConfig:
    ranges = {
        "quick": ((8, 32), (1, 5)),
        "standard": ((32, 128), (2, 20)),
        "deep": ((32, 256), (5, 100)),
    }
    if mode not in ranges:
        raise ValueError("최적화 모드는 quick, standard, deep 중 하나여야 합니다.")
    width, generations = ranges[mode]
    return replace(
        base,
        population_size=trial.suggest_int("population_size", *width, step=8),
        generations=trial.suggest_int("generations", *generations),
        crossover_probability=trial.suggest_float("crossover_probability", 0.45, 0.85),
        mutation_probability=trial.suggest_float("mutation_probability", 0.05, 0.45),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        max_lag=trial.suggest_int("max_lag", 1, min(10, max(1, base.max_lag))),
        min_activation_rate=trial.suggest_float("min_activation_rate", 0.05, 0.3),
        control_runs=0,
        seed=base.seed + int(trial.number),
    )


def validation_objective(experiment: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Build the Optuna objective without reading any final test field."""
    selected_models = [
        model
        for model in (
            experiment.get("one_number_model", {}).get("selected"),
            experiment.get("six_number_set_model", {}).get("selected"),
        )
        if isinstance(model, dict)
    ]
    if len(selected_models) != 2:
        raise ValueError("최적화 실험에서 두 모델의 선택 결과를 찾지 못했습니다.")
    validation_scores = [float(model["validation_score"]) for model in selected_models]
    train_scores = [float(model["training_score"]) for model in selected_models]
    complexities = [float(model.get("complexity", 0)) for model in selected_models]
    validation_mean = sum(validation_scores) / len(validation_scores)
    generalization_gap = sum(
        max(0.0, train - validation)
        for train, validation in zip(train_scores, validation_scores, strict=True)
    ) / len(train_scores)
    complexity_mean = sum(complexities) / len(complexities)
    objective = validation_mean - 0.1 * generalization_gap - 0.0005 * complexity_mean
    return objective, {
        "validation_mean": validation_mean,
        "training_validation_gap": generalization_gap,
        "mean_complexity": complexity_mean,
    }


def optimize_gp(
    dataset: DrawDataset,
    base_config: IntermittentConfig,
    *,
    study_name: str,
    mode: str = "standard",
    action: str = "resume",
    n_trials: int = 10,
    timeout_seconds: float | None = None,
    storage_directory: str | Path = "data/optimizations",
    experiment_directory: str | Path = "experiments/optimization_trials",
    progress: Callable[[dict[str, Any]], None] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run/resume a persisted Optuna study; study best-trial selection uses validation only."""
    if mode not in {"quick", "standard", "deep"}:
        raise ValueError("최적화 모드는 quick, standard, deep 중 하나여야 합니다.")
    if action not in {"new", "resume", "add_trials"}:
        raise ValueError("최적화 실행 방식은 new, resume, add_trials 중 하나여야 합니다.")
    if n_trials < 1:
        raise ValueError("최적화 Trial 수는 1 이상이어야 합니다.")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("시간 제한은 0보다 커야 합니다.")
    if not optuna_status()["available"]:
        raise RuntimeError(
            "Optuna가 설치되지 않았습니다. 선택 기능 설치 명령: " + optuna_status()["install_extra"]
        )
    import optuna

    slug = _study_slug(study_name)
    storage_root = project_path(storage_directory)
    storage_root.mkdir(parents=True, exist_ok=True)
    study_db = storage_root / f"{slug}.sqlite"
    storage_uri = f"sqlite:///{study_db.as_posix()}"
    state_file = storage_root / f"{slug}.state.json"
    stop_file = Path(f"{state_file}.stop.json")
    active_run_id = run_id or str(uuid.uuid4())
    if action == "new" and (study_db.exists() or state_file.exists()):
        raise FileExistsError(
            "같은 이름의 최적화 기록이 이미 있습니다. 기존 연구 이어하기를 선택하세요."
        )
    if action in {"resume", "add_trials"} and not study_db.is_file():
        raise FileNotFoundError("이어서 실행할 Optuna SQLite 연구 기록을 찾지 못했습니다.")
    study = optuna.create_study(
        study_name=slug,
        storage=storage_uri,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=base_config.seed),
        load_if_exists=True,
    )
    snapshot_hashes = dataset_hashes(dataset)
    study_metadata = {
        "mode": mode,
        "engine": "intermittent_gp",
        "feature_profile": base_config.feature_profile,
        "dataset_range": [dataset.records[0].round_number, dataset.records[-1].round_number],
        "dataset_snapshot_sha256": snapshot_hashes["bonus_inclusive_sha256"],
        "dataset_hash_algorithm": snapshot_hashes["algorithm_version"],
        "final_test_used_for_selection": False,
        "objective_policy_version": "validation_only_v1",
    }
    if action == "new":
        for key, value in study_metadata.items():
            study.set_user_attr(key, value)
    else:
        stored = study.user_attrs
        immutable_keys = (
            "engine",
            "feature_profile",
            "dataset_range",
            "dataset_snapshot_sha256",
            "dataset_hash_algorithm",
        )
        mismatched = [
            key
            for key in immutable_keys
            if stored.get(key) is not None and stored[key] != study_metadata[key]
        ]
        if mismatched:
            raise ValueError(
                "기존 Optuna 기록의 엔진·프로파일·데이터 스냅샷과 다릅니다. "
                "기록을 변경하지 않고 실행을 중단했습니다: " + ", ".join(mismatched)
            )
    trial_store = ExperimentStore(Path(experiment_directory) / slug)
    started_at = datetime.now(UTC).isoformat()
    starting_trial_count = len(study.trials)
    starting_state_counts = _trial_state_counts(study.trials)

    def write_state(event: dict[str, Any]) -> None:
        try:
            previous = json.loads(state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            previous = {}
        payload = {**previous, **event}
        atomic_write_bytes(
            state_file,
            (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        )

    write_state(
        {
            "study_name": slug,
            "run_id": active_run_id,
            "mode": mode,
            "status": "running",
            "completed_trials": starting_state_counts["complete"],
            "failed_trials": starting_state_counts["failed"],
            "pruned_trials": starting_state_counts["pruned"],
            "requested_additional_trials": n_trials,
            "starting_trial_count": starting_trial_count,
            "starting_state_counts": starting_state_counts,
            "study_database": str(study_db),
            "action": action,
            "started_at": started_at,
            "updated_at": datetime.now(UTC).isoformat(),
        }
    )

    def objective(trial: Any) -> float:
        write_state(
            {
                "study_name": slug,
                "run_id": active_run_id,
                "status": "running",
                "current_trial_number": trial.number,
                "current_trial_started_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
                "starting_trial_count": starting_trial_count,
                "requested_additional_trials": n_trials,
                "study_database": str(study_db),
            }
        )
        config = _trial_config(trial, base_config, mode)
        experiment = run_intermittent_experiment(dataset, config, trial_store)
        value, components = validation_objective(experiment)
        trial.set_user_attr("experiment_id", str(experiment["experiment_id"]))
        trial.set_user_attr("objective_components", components)
        trial.set_user_attr("runtime_seconds", float(experiment.get("runtime_seconds", 0.0)))
        return value

    def report(study_state: Any, trial: Any) -> None:
        state_counts = _trial_state_counts(study_state.trials)
        event = {
            "study_name": slug,
            "run_id": active_run_id,
            "mode": mode,
            "status": "running",
            "completed_trials": state_counts["complete"],
            "failed_trials": state_counts["failed"],
            "pruned_trials": state_counts["pruned"],
            "starting_trial_count": starting_trial_count,
            "starting_state_counts": starting_state_counts,
            "requested_additional_trials": n_trials,
            "best_value": study_state.best_value if study_state.best_trial else None,
            "best_trial_number": study_state.best_trial.number if study_state.best_trial else None,
            "best_parameters": study_state.best_trial.params if study_state.best_trial else {},
            "last_trial_number": trial.number,
            "last_trial_state": str(trial.state),
            "current_trial_number": None,
            "current_trial_started_at": None,
            "study_database": str(study_db),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        if progress:
            progress(event)
        stop_requested = _optimization_stop_requested(stop_file, active_run_id)
        event["status"] = "stopping" if stop_requested else "running"
        event["stop_requested_at"] = (
            json.loads(stop_file.read_text(encoding="utf-8")).get("requested_at")
            if stop_requested
            else None
        )
        write_state(event)
        if stop_requested:
            study_state.stop()

    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout_seconds,
            callbacks=[report],
            n_jobs=1,
            gc_after_trial=True,
        )
    except Exception as error:
        failed_state_counts = _trial_state_counts(study.trials)
        write_state(
            {
                "study_name": slug,
                "run_id": active_run_id,
                "mode": mode,
                "status": "failed",
                "completed_trials": failed_state_counts["complete"],
                "failed_trials": failed_state_counts["failed"],
                "pruned_trials": failed_state_counts["pruned"],
                "error_type": type(error).__name__,
                "error": str(error),
                "study_database": str(study_db),
                "finished_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )
        if _optimization_stop_requested(stop_file, active_run_id):
            stop_file.unlink(missing_ok=True)
        raise
    from optuna.trial import TrialState

    best_trial = next(
        (trial for trial in reversed(study.trials) if trial.state == TrialState.COMPLETE), None
    )
    if best_trial is not None:
        best_trial = study.best_trial
    was_stopped = _optimization_stop_requested(stop_file, active_run_id)
    final_status = (
        "stopped"
        if was_stopped
        else "completed"
        if best_trial is not None
        else "failed_no_completed_trials"
    )
    final_state_counts = _trial_state_counts(study.trials)
    write_state(
        {
            "study_name": slug,
            "run_id": active_run_id,
            "mode": mode,
            "status": final_status,
            "completed_trials": final_state_counts["complete"],
            "failed_trials": final_state_counts["failed"],
            "pruned_trials": final_state_counts["pruned"],
            "starting_trial_count": starting_trial_count,
            "starting_state_counts": starting_state_counts,
            "requested_additional_trials": n_trials,
            "best_trial_number": best_trial.number if best_trial else None,
            "best_value": best_trial.value if best_trial else None,
            "best_parameters": best_trial.params if best_trial else {},
            "started_at": started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "study_database": str(study_db),
        }
    )
    stop_file.unlink(missing_ok=True)
    return {
        "study_name": slug,
        "status": final_status,
        "mode": mode,
        "database_path": str(study_db),
        "completed_trial_count": final_state_counts["complete"],
        "best_trial_number": best_trial.number if best_trial else None,
        "best_value": best_trial.value if best_trial else None,
        "best_parameters": best_trial.params if best_trial else {},
        "best_experiment_id": best_trial.user_attrs.get("experiment_id") if best_trial else None,
        "final_test_used_for_selection": False,
        "resume_supported": True,
    }


def _trial_state_counts(trials: list[Any]) -> dict[str, int]:
    counts = {"complete": 0, "failed": 0, "pruned": 0, "running": 0, "waiting": 0}
    for trial in trials:
        state = str(trial.state).split(".")[-1].casefold()
        if state in counts:
            counts[state] += 1
    return counts


def _optimization_stop_requested(path: Path, run_id: str) -> bool:
    try:
        request = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False
    except (OSError, ValueError, TypeError):
        return False
    return request.get("run_id") == run_id


def request_optimization_stop(state_path: str | Path, *, run_id: str) -> Path:
    """Request a safe stop after the current trial reaches its boundary."""
    state_file = Path(state_path)
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError("최적화 상태를 읽을 수 없어 중지를 요청하지 못했습니다.") from error
    if state.get("run_id") != run_id or state.get("status") not in {"running", "stopping"}:
        raise RuntimeError("해당 최적화 실행이 더 이상 진행 중이지 않습니다.")
    stop_file = Path(f"{state_file}.stop.json")
    request = {"run_id": run_id, "requested_at": datetime.now(UTC).isoformat()}
    atomic_write_bytes(stop_file, (json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
    state.update(status="stopping", stop_requested_at=request["requested_at"])
    atomic_write_bytes(
        state_file,
        (json.dumps(state, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    return stop_file


def format_optimization_duration(seconds: int | None) -> str:
    if seconds is None:
        return "산출 불가"
    days, remainder = divmod(max(0, seconds), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{days * 24 + hours:02d}:{minutes:02d}:{secs:02d}"


def estimate_optimization_remaining_seconds(
    remaining_trials: int,
    mean_trial_seconds: float | None,
    sampled_trial_durations: int,
    current_trial_elapsed_seconds: int = 0,
) -> int | None:
    if remaining_trials <= 0:
        return 0
    if mean_trial_seconds is None or sampled_trial_durations < 2:
        return None
    return max(
        0,
        int(remaining_trials * mean_trial_seconds) - max(0, current_trial_elapsed_seconds),
    )


def optimization_monitor_snapshot(
    record: dict[str, Any], *, recent_limit: int = 20
) -> dict[str, Any]:
    """Read compact trial progress and recent results from an Optuna SQLite study."""
    database = Path(str(record.get("database_path", "")))
    if not database.is_file():
        return {"available": False, "trials": [], "counts": {}}
    try:
        uri = database.resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=1) as connection:
            connection.row_factory = sqlite3.Row
            study_row = connection.execute(
                "SELECT study_id FROM studies WHERE study_name=?", (record["study_name"],)
            ).fetchone()
            if study_row is None:
                return {"available": False, "trials": [], "counts": {}}
            study_id = int(study_row["study_id"])
            rows = connection.execute(
                "SELECT state, COUNT(*) AS count FROM trials WHERE study_id=? GROUP BY state",
                (study_id,),
            ).fetchall()
            counts = {str(row["state"]).casefold(): int(row["count"]) for row in rows}
            trials = connection.execute(
                """SELECT t.trial_id, t.number, t.state, t.datetime_start, t.datetime_complete,
                          tv.value
                   FROM trials t LEFT JOIN trial_values tv
                     ON tv.trial_id=t.trial_id AND tv.objective=0
                   WHERE t.study_id=? ORDER BY t.number DESC LIMIT ?""",
                (study_id, max(1, min(100, recent_limit))),
            ).fetchall()
            results = []
            for row in trials:
                params = connection.execute(
                    "SELECT param_name, param_value FROM trial_params "
                    "WHERE trial_id=? ORDER BY param_name",
                    (row["trial_id"],),
                ).fetchall()
                system_attrs = connection.execute(
                    "SELECT key, value_json FROM trial_system_attributes WHERE trial_id=?",
                    (row["trial_id"],),
                ).fetchall()
                failure_reason = None
                for attribute in system_attrs:
                    if str(attribute["key"]).casefold() in {"fail_reason", "failure_reason"}:
                        try:
                            failure_reason = json.loads(attribute["value_json"])
                        except (TypeError, ValueError):
                            failure_reason = attribute["value_json"]
                        break
                started, finished = row["datetime_start"], row["datetime_complete"]
                duration = None
                if started and finished:
                    try:
                        duration = max(
                            0.0,
                            (
                                datetime.fromisoformat(finished) - datetime.fromisoformat(started)
                            ).total_seconds(),
                        )
                    except ValueError:
                        pass
                results.append(
                    {
                        "number": int(row["number"]),
                        "state": str(row["state"]),
                        "started_at": started,
                        "finished_at": finished,
                        "duration_seconds": duration,
                        "value": float(row["value"]) if row["value"] is not None else None,
                        "failure_reason": failure_reason,
                        "parameters": {
                            str(parameter["param_name"]): float(parameter["param_value"])
                            for parameter in params
                        },
                    }
                )
            durations = [
                item["duration_seconds"]
                for item in results
                if item["state"] == "COMPLETE" and item["duration_seconds"] is not None
            ]
            return {
                "available": True,
                "counts": counts,
                "trials": results,
                "mean_trial_seconds": statistics.mean(durations) if len(durations) >= 2 else None,
                "sampled_trial_durations": len(durations),
            }
    except (OSError, sqlite3.Error):
        return {"available": False, "trials": [], "counts": {}}
