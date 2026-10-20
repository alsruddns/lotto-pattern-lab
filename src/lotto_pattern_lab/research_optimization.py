"""Optional Optuna-backed GP configuration search using validation-only objectives."""

from __future__ import annotations

import importlib.util
import json
import re
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_bytes
from .dataset import DrawDataset
from .intermittent import IntermittentConfig
from .intermittent_experiment import run_intermittent_experiment
from .paths import project_path
from .storage import ExperimentStore

OPTIMIZATION_SCHEMA_VERSION = 1


def optuna_status() -> dict[str, Any]:
    available = importlib.util.find_spec("optuna") is not None
    return {
        "available": available,
        "status": "available" if available else "optional_dependency_missing",
        "install_extra": ".venv\\Scripts\\python.exe -m pip install -e \".[optimization]\"",
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
    n_trials: int = 10,
    timeout_seconds: float | None = None,
    storage_directory: str | Path = "data/optimizations",
    experiment_directory: str | Path = "experiments/optimization_trials",
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run/resume a persisted Optuna study; study best-trial selection uses validation only."""
    if mode not in {"quick", "standard", "deep"}:
        raise ValueError("최적화 모드는 quick, standard, deep 중 하나여야 합니다.")
    if n_trials < 1:
        raise ValueError("최적화 Trial 수는 1 이상이어야 합니다.")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("시간 제한은 0보다 커야 합니다.")
    if not optuna_status()["available"]:
        raise RuntimeError(
            "Optuna가 설치되지 않았습니다. 선택 기능 설치 명령: "
            + optuna_status()["install_extra"]
        )
    import optuna

    slug = _study_slug(study_name)
    storage_root = project_path(storage_directory)
    storage_root.mkdir(parents=True, exist_ok=True)
    study_db = storage_root / f"{slug}.sqlite"
    storage_uri = f"sqlite:///{study_db.as_posix()}"
    state_file = storage_root / f"{slug}.state.json"
    study = optuna.create_study(
        study_name=slug,
        storage=storage_uri,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=base_config.seed),
        load_if_exists=True,
    )
    study.set_user_attr("mode", mode)
    study.set_user_attr("feature_profile", base_config.feature_profile)
    study.set_user_attr(
        "dataset_range", [dataset.records[0].round_number, dataset.records[-1].round_number]
    )
    study.set_user_attr("final_test_used_for_selection", False)
    study.set_user_attr("objective_policy_version", "validation_only_v1")
    trial_store = ExperimentStore(Path(experiment_directory) / slug)

    def write_state(event: dict[str, Any]) -> None:
        atomic_write_bytes(
            state_file,
            (json.dumps(event, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        )

    write_state(
        {
            "study_name": slug,
            "mode": mode,
            "status": "running",
            "completed_trials": len(study.trials),
            "requested_additional_trials": n_trials,
            "study_database": str(study_db),
            "updated_at": datetime.now(UTC).isoformat(),
        }
    )

    def objective(trial: Any) -> float:
        config = _trial_config(trial, base_config, mode)
        experiment = run_intermittent_experiment(dataset, config, trial_store)
        value, components = validation_objective(experiment)
        trial.set_user_attr("experiment_id", str(experiment["experiment_id"]))
        trial.set_user_attr("objective_components", components)
        trial.set_user_attr("runtime_seconds", float(experiment.get("runtime_seconds", 0.0)))
        return value

    def report(study_state: Any, trial: Any) -> None:
        event = {
            "study_name": slug,
            "mode": mode,
            "completed_trials": len(study_state.trials),
            "requested_additional_trials": n_trials,
            "best_value": study_state.best_value if study_state.best_trial else None,
            "best_trial_number": study_state.best_trial.number if study_state.best_trial else None,
            "last_trial_number": trial.number,
            "last_trial_state": str(trial.state),
            "study_database": str(study_db),
        }
        write_state({**event, "status": "running"})
        if progress:
            progress(event)

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
        write_state(
            {
                "study_name": slug,
                "mode": mode,
                "status": "failed",
                "completed_trials": len(study.trials),
                "error_type": type(error).__name__,
                "error": str(error),
                "study_database": str(study_db),
            }
        )
        raise
    from optuna.trial import TrialState

    best_trial = next(
        (trial for trial in reversed(study.trials) if trial.state == TrialState.COMPLETE), None
    )
    if best_trial is not None:
        best_trial = study.best_trial
    final_status = "completed" if best_trial is not None else "failed_no_completed_trials"
    write_state(
        {
            "study_name": slug,
            "mode": mode,
            "status": final_status,
            "completed_trials": len(study.trials),
            "best_trial_number": best_trial.number if best_trial else None,
            "best_value": best_trial.value if best_trial else None,
            "study_database": str(study_db),
        }
    )
    return {
        "study_name": slug,
        "status": final_status,
        "mode": mode,
        "database_path": str(study_db),
        "completed_trial_count": len(study.trials),
        "best_trial_number": best_trial.number if best_trial else None,
        "best_value": best_trial.value if best_trial else None,
        "best_parameters": best_trial.params if best_trial else {},
        "best_experiment_id": best_trial.user_attrs.get("experiment_id") if best_trial else None,
        "final_test_used_for_selection": False,
        "resume_supported": True,
    }
