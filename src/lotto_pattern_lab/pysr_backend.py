"""Optional PySR symbolic-regression backend with leakage-safe Lotto targets."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata, util
from pathlib import Path
from typing import Any

import numpy as np

from .backtest import chronological_split
from .dataset import DrawDataset
from .evaluation_cache import (
    EvaluationCache,
    canonical_pysr_formula,
    classify_candidate,
    evaluation_fingerprint,
)
from .feature_schema import feature_mapping, feature_schema_version
from .features import Example, build_bonus_examples, build_examples
from .paths import project_path
from .research_features import ResearchFeatureConfig, build_research_examples
from .storage import ExperimentStore

PYSR_VERSION_RANGE = ">=2.7.0,<3"
SAFE_BINARY_OPERATORS = ("+", "-", "*", "/")
SAFE_UNARY_OPERATORS = ("square", "abs")
PYSR_ENGINE_VERSION = "pysr-engine-v1"


def _prepare_julia_environment() -> None:
    """Use the PySR venv's Julia project and app depot when they are available."""
    venv_project = Path(sys.prefix) / "julia_env"
    if "JULIA_PROJECT" not in os.environ and (venv_project / "Project.toml").is_file():
        os.environ["JULIA_PROJECT"] = str(venv_project)

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data and "JULIA_DEPOT_PATH" not in os.environ:
        app_depot = Path(local_app_data) / "LottoPatternLab" / "julia-depot"
        if app_depot.is_dir():
            os.environ["JULIA_DEPOT_PATH"] = str(app_depot)


@dataclass(frozen=True, slots=True)
class PySRConfig:
    """PySR-native settings; none of the DEAP-specific probabilities are reused."""

    target: str = "main_positions"
    feature_profile: str = "legacy"
    train_ratio: float = 0.6
    validation_ratio: float = 0.2
    max_lag: int = 10
    timeout_seconds: float = 60.0
    niterations: int = 20
    population_size: int = 32
    populations: int = 4
    maxsize: int = 24
    maxdepth: int = 8
    binary_operators: tuple[str, ...] = SAFE_BINARY_OPERATORS
    unary_operators: tuple[str, ...] = SAFE_UNARY_OPERATORS
    workers: int = 1
    seed: int = 12345
    random_control_replicates: int = 3
    guesses: tuple[str, ...] = ()
    warm_start: bool = False
    run_directory: str | None = None

    def __post_init__(self) -> None:
        if self.target not in {"one_number", "main_positions", "bonus"}:
            raise ValueError("PySR target must be one_number, main_positions, or bonus.")
        if self.feature_profile not in {"legacy", "bonus_lag_v1", "calendar_round"}:
            raise ValueError("Unsupported feature profile.")
        if self.target == "bonus" and self.feature_profile == "legacy":
            raise ValueError("Bonus-target research must use bonus_lag_v1 or calendar_round.")
        if not 0 < self.train_ratio < 1 or not 0 < self.validation_ratio < 1:
            raise ValueError("Train and validation ratios must be between zero and one.")
        if self.train_ratio + self.validation_ratio >= 1:
            raise ValueError("Train plus validation ratios must be less than one.")
        if not 1 <= self.max_lag <= 10:
            raise ValueError("max_lag must be between 1 and 10.")
        if self.timeout_seconds <= 0 or self.niterations < 1:
            raise ValueError("PySR timeout and iteration count must be positive.")
        if self.population_size < 8 or self.populations < 1:
            raise ValueError("PySR population settings are too small.")
        if not 4 <= self.maxsize <= 100 or not 1 <= self.maxdepth <= self.maxsize:
            raise ValueError("Invalid PySR complexity limits.")
        if not 1 <= self.workers <= 4:
            raise ValueError("PySR workers must be between 1 and 4.")
        if not 0 <= self.random_control_replicates <= 100:
            raise ValueError("Random-control repetitions must be between 0 and 100.")
        if not self.binary_operators or any(
            operator not in SAFE_BINARY_OPERATORS for operator in self.binary_operators
        ):
            raise ValueError("Unsupported PySR binary operator.")
        if any(operator not in SAFE_UNARY_OPERATORS for operator in self.unary_operators):
            raise ValueError("Unsupported PySR unary operator.")
        for guess in self.guesses:
            canonical_pysr_formula(guess)


def optional_backend_status() -> dict[str, Any]:
    """Inspect the optional dependency without importing Julia or mutating its environment."""
    if util.find_spec("pysr") is None:
        return {
            "available": False,
            "installed": False,
            "python_version_requirement": ">=3.9 (PySR 2.7.x)",
            "detail": "Install the optional symbolic extra in an isolated environment.",
            "install_command": (
                "create a separate Python 3.13 venv, activate it, then run "
                "python -m pip install -e '.[symbolic]'"
            ),
            "runtime_verified": False,
        }
    try:
        version = metadata.version("pysr")
    except metadata.PackageNotFoundError:
        version = "installed-without-package-metadata"
    return {
        "available": True,
        "installed": True,
        "version": version,
        "python_version_requirement": ">=3.9 (PySR 2.7.x)",
        "detail": "PySR package is installed; Julia runtime is checked only when selected.",
        "runtime_verified": False,
    }


def initialize_optional_backend() -> tuple[Any | None, dict[str, Any]]:
    """Load PySR only when explicitly requested, returning backend failures as data."""
    status = optional_backend_status()
    if not status["installed"]:
        return None, status
    try:
        _prepare_julia_environment()
        from pysr import PySRRegressor
    except Exception as error:  # Julia setup can fail independently from the GP application.
        return None, {
            **status,
            "available": False,
            "runtime_verified": True,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    return PySRRegressor, {**status, "runtime_verified": True, "error": None}


def _examples(dataset: DrawDataset, config: PySRConfig) -> tuple[Example, ...]:
    if config.feature_profile == "calendar_round":
        return build_research_examples(
            dataset,
            config.max_lag,
            ResearchFeatureConfig(include_calendar=True, include_round_digits=True),
        )
    if config.target == "bonus" or config.feature_profile == "bonus_lag_v1":
        if config.target == "bonus":
            return build_bonus_examples(dataset, config.max_lag)
        bonus = build_bonus_examples(dataset, config.max_lag)
        base = build_examples(dataset, config.max_lag)
        return tuple(
            Example(left.round_number, right.features, left.winning_numbers)
            for left, right in zip(base, bonus, strict=True)
        )
    return build_examples(dataset, config.max_lag)


def _target_values(dataset: DrawDataset, examples: tuple[Example, ...], target: str) -> np.ndarray:
    if target == "bonus":
        by_round = {item.round_number: item.draw.bonus for item in dataset.records}
        values = [by_round[item.round_number] for item in examples]
        if any(value is None for value in values):
            raise ValueError("Bonus target requires a bonus value on every target draw.")
        return np.asarray(values, dtype=np.float64)
    by_round = {item.round_number: sorted(item.draw.numbers) for item in dataset.records}
    positions = [by_round[item.round_number] for item in examples]
    if target == "one_number":
        return np.asarray([row[0] for row in positions], dtype=np.float64)
    if target.startswith("main_position_"):
        position = int(target.rsplit("_", 1)[1])
        if not 1 <= position <= 6:
            raise ValueError("Main-number position must be between 1 and 6.")
        return np.asarray([row[position - 1] for row in positions], dtype=np.float64)
    raise ValueError(f"Unsupported numeric target {target!r}.")


def _legal_lotto_number(value: float, used: set[int]) -> int:
    if not math.isfinite(value):
        return 1
    center = min(45, max(1, int(math.floor(value + 0.5))))
    return min(
        (candidate for candidate in range(1, 46) if candidate not in used),
        key=lambda x: (abs(x - center), x),
    )


def _score(model: Any, x: np.ndarray, y: np.ndarray) -> dict[str, float | int | None]:
    if len(y) == 0:
        return {"sample_count": 0, "mae": None, "number_hit_rate": None}
    predicted = np.asarray(model.predict(x), dtype=float).reshape(-1)
    valid = np.isfinite(predicted)
    if not valid.any():
        return {"sample_count": len(y), "mae": None, "number_hit_rate": 0.0}
    predicted_safe = np.nan_to_num(predicted, nan=1.0, posinf=45.0, neginf=1.0)
    rounded = np.clip(np.floor(predicted_safe + 0.5), 1, 45).astype(int)
    return {
        "sample_count": int(len(y)),
        "mae": float(np.mean(np.abs(predicted[valid] - y[valid]))),
        "number_hit_rate": float(np.mean(rounded == y)),
    }


def _dataset_sha256(dataset: DrawDataset) -> str:
    has_bonus = any(item.draw.bonus is not None for item in dataset.records)
    payload = [
        {
            "round": item.round_number,
            "date": item.draw_date.isoformat() if item.draw_date else None,
            "numbers": item.draw.numbers,
            **({"bonus": item.draw.bonus} if has_bonus else {}),
        }
        for item in dataset.records
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _prediction_score(predicted: np.ndarray, y: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(predicted, dtype=float).reshape(-1)
    valid = np.isfinite(values)
    rounded = np.clip(
        np.floor(np.nan_to_num(values, nan=1.0, posinf=45.0, neginf=1.0) + 0.5), 1, 45
    ).astype(int)
    return {
        "sample_count": int(len(y)),
        "mae": float(np.mean(np.abs(values[valid] - y[valid]))) if valid.any() else None,
        "number_hit_rate": float(np.mean(rounded == y)) if len(y) else None,
    }


def _random_set_controls(
    examples: tuple[Example, ...], observed_hits: list[int], replicates: int, seed: int
) -> dict[str, Any]:
    if not replicates:
        return {
            "random_control_replicates": 0,
            "random_control_mean_hits": None,
            "empirical_p_value": None,
        }
    rng = np.random.default_rng(seed)
    replicate_means = [
        float(
            np.mean(
                [
                    len(
                        set(rng.choice(np.arange(1, 46), size=6, replace=False))
                        & set(example.winning_numbers)
                    )
                    for example in examples
                ]
            )
        )
        for _ in range(replicates)
    ]
    observed_mean = float(np.mean(observed_hits)) if observed_hits else 0.0
    return {
        "random_control_replicates": replicates,
        "random_control_mean_hits": float(np.mean(replicate_means)),
        "random_control_replicate_means": replicate_means,
        "empirical_p_value": float(
            (1 + sum(value >= observed_mean for value in replicate_means)) / (1 + replicates)
        ),
        "control_note": (
            "random baseline scores are descriptive; PySR internal candidate search is not "
            "randomized as a control"
        ),
    }


def _random_numeric_controls(
    actual: np.ndarray, predicted: np.ndarray, replicates: int, seed: int
) -> dict[str, Any]:
    if not replicates:
        return {
            "random_control_replicates": 0,
            "random_control_hit_rate": None,
            "empirical_p_value": None,
        }
    rng = np.random.default_rng(seed)
    target = np.asarray(actual, dtype=float)
    observed = np.asarray(predicted, dtype=float)
    observed_rate = float(
        np.mean(np.clip(np.floor(np.nan_to_num(observed, nan=1.0) + 0.5), 1, 45) == target)
    )
    replicate_rates = [
        float(np.mean(rng.integers(1, 46, size=len(target)) == target))
        for _ in range(replicates)
    ]
    return {
        "random_control_replicates": replicates,
        "random_control_hit_rate": float(np.mean(replicate_rates)),
        "random_control_replicate_rates": replicate_rates,
        "empirical_p_value": float(
            (1 + sum(value >= observed_rate for value in replicate_rates)) / (1 + replicates)
        ),
    }


def run_pysr_experiment(
    dataset: DrawDataset,
    config: PySRConfig,
    store: ExperimentStore,
    progress: Any = None,
    cache_path: str | None = None,
) -> dict[str, Any]:
    """Run a real PySR fit on one leakage-safe target and persist its result."""
    regressor, status = initialize_optional_backend()
    if regressor is None:
        raise RuntimeError(f"PySR runtime unavailable: {status.get('error', status.get('detail'))}")
    examples = _examples(dataset, config)
    splits = chronological_split(examples, config.train_ratio, config.validation_ratio)
    target_names = (
        tuple(f"main_position_{index}" for index in range(1, 7))
        if config.target == "main_positions"
        else (config.target,)
    )
    index = {item.round_number: position for position, item in enumerate(examples)}
    x_all = np.asarray([item.features for item in examples], dtype=np.float64)
    train_idx = np.asarray([index[item.round_number] for item in splits.train], dtype=int)
    validation_idx = np.asarray([index[item.round_number] for item in splits.validation], dtype=int)
    test_idx = np.asarray([index[item.round_number] for item in splits.test], dtype=int)
    candidates: list[dict[str, Any]] = []
    evaluation_cache = EvaluationCache(cache_path or "data/database/patterns.sqlite")
    cache_reused = 0
    cache_computed = 0
    target_metrics: dict[str, Any] = {}
    target_predictions: dict[str, dict[str, list[float]]] = {}
    started = time.monotonic()
    experiment_id = str(uuid.uuid4())
    pysr_run_directories: dict[str, str] = {}
    for target_index, target_name in enumerate(target_names):
        y_all = _target_values(dataset, examples, target_name)
        if config.run_directory:
            run_directory = project_path(config.run_directory)
            if config.target == "main_positions":
                run_directory = run_directory / target_name
            output_directory = run_directory.parent
            run_id = run_directory.name
        else:
            output_directory = store.directory / "pysr_runs"
            run_id = f"{experiment_id}_{target_name}"
            run_directory = output_directory / run_id
        pysr_run_directories[target_name] = str(run_directory.resolve())
        parallel_options = (
            {"parallelism": "serial"}
            if config.workers == 1
            else {"parallelism": "multiprocessing", "procs": config.workers}
        )
        model_options = dict(
            niterations=config.niterations,
            population_size=config.population_size,
            tournament_selection_n=min(10, config.population_size - 1),
            populations=config.populations,
            maxsize=config.maxsize,
            maxdepth=config.maxdepth,
            binary_operators=list(config.binary_operators),
            unary_operators=list(config.unary_operators),
            timeout_in_seconds=config.timeout_seconds / len(target_names),
            random_state=config.seed + target_index,
            deterministic=config.workers == 1,
            guesses=list(config.guesses) or None,
            warm_start=config.warm_start,
            output_directory=str(output_directory),
            run_id=run_id,
            verbosity=0,
            progress=False,
            delete_tempfiles=True,
            precision=64,
            **parallel_options,
        )
        if config.warm_start and run_directory.is_dir():
            resume_options = dict(model_options)
            resume_options.pop("binary_operators")
            resume_options.pop("unary_operators")
            model = regressor.from_file(
                run_directory=str(run_directory),
                **resume_options,
            )
        else:
            model = regressor(**model_options)
        model.fit(
            x_all[train_idx],
            y_all[train_idx],
            variable_names=[f"x{i}" for i in range(x_all.shape[1])],
        )
        target_metrics[target_name] = {
            "train": _score(model, x_all[train_idx], y_all[train_idx]),
            "validation": _score(model, x_all[validation_idx], y_all[validation_idx]),
            "independent_test": _score(model, x_all[test_idx], y_all[test_idx]),
        }
        target_predictions[target_name] = {
            phase: np.nan_to_num(
                np.asarray(model.predict(x_all[indices]), dtype=float),
                nan=1.0,
                posinf=45.0,
                neginf=1.0,
            ).tolist()
            for phase, indices in (
                ("training", train_idx),
                ("validation", validation_idx),
                ("independent_test", test_idx),
            )
        }
        equations = getattr(model, "equations_", None)
        if equations is not None:
            data_hash = _dataset_sha256(dataset)
            for equation_index, row in enumerate(equations.to_dict(orient="records")):
                formula = str(row.get("equation", ""))
                conditions = {
                    "engine_version": PYSR_ENGINE_VERSION,
                    "pysr_version": status.get("version"),
                    "dataset_sha256": data_hash,
                    "feature_profile": config.feature_profile,
                    "feature_schema_version": feature_schema_version(config.feature_profile),
                    "max_lag": config.max_lag,
                    "feature_names": list(
                        feature_mapping(config.feature_profile, config.max_lag).values()
                    ),
                    "primitive_set_version": "pysr_safe_float_v1",
                    "numeric_semantics": "float64 with finite numeric predictions",
                    "target": target_name,
                    "activation_condition": "always",
                    "output_transform": "round_half_up_and_clamp_1_to_45",
                    "train_rounds": [splits.train[0].round_number, splits.train[-1].round_number],
                    "validation_rounds": [
                        splits.validation[0].round_number,
                        splits.validation[-1].round_number,
                    ],
                    "test_rounds": [splits.test[0].round_number, splits.test[-1].round_number],
                    "evaluation_policy": "mae_and_rounded_number_hit_rate_v1",
                }
                cached = evaluation_cache.get(formula, conditions)
                reused_existing = cached is not None
                if cached is None:
                    cached = {
                        "train": _prediction_score(
                            model.predict(x_all[train_idx], index=equation_index), y_all[train_idx]
                        ),
                        "validation": _prediction_score(
                            model.predict(x_all[validation_idx], index=equation_index),
                            y_all[validation_idx],
                        ),
                        "independent_test": _prediction_score(
                            model.predict(x_all[test_idx], index=equation_index), y_all[test_idx]
                        ),
                    }
                    evaluation_cache.put(formula, conditions, cached)
                    cache_computed += 1
                else:
                    cache_reused += 1
                fingerprint = evaluation_fingerprint(formula, conditions)
                classification, classification_reason = classify_candidate(
                    validation_mae=cached.get("validation", {}).get("mae"),
                    validation_sample_count=int(
                        cached.get("validation", {}).get("sample_count", 0)
                    ),
                    complexity=int(row.get("complexity", 0)),
                )
                if reused_existing or fingerprint in {item["fingerprint"] for item in candidates}:
                    classification, classification_reason = (
                        "REDUNDANT",
                        "same normalized formula and evaluation conditions were evaluated before",
                    )
                evaluation_cache.classify(
                    fingerprint,
                    classification,
                    classification_reason,
                    policy_version="candidate_quality_v1",
                    dataset_sha256=data_hash,
                )
                candidates.append(
                    {
                        "target": target_name,
                        "formula": formula,
                        "normalized_formula": canonical_pysr_formula(formula),
                        "fingerprint": fingerprint,
                        "complexity": int(row.get("complexity", 0)),
                        "loss": float(row["loss"]) if row.get("loss") is not None else None,
                        "score": float(row["score"]) if row.get("score") is not None else None,
                        "evaluation": cached,
                        "classification": classification,
                        "classification_reason": classification_reason,
                        "classification_policy": "candidate_quality_v1",
                    }
                )
        if progress:
            progress({"stage": "pysr_target_complete", "target": target_name})
    duration = time.monotonic() - started
    set_metrics: dict[str, Any] = {}
    if config.target == "main_positions":
        for split_name, target_rounds in (
            ("training", splits.train),
            ("validation", splits.validation),
            ("independent_test", splits.test),
        ):
            hit_counts: list[int] = []
            for round_index, example in enumerate(target_rounds):
                used: set[int] = set()
                for position in range(1, 7):
                    number = _legal_lotto_number(
                        target_predictions[f"main_position_{position}"][split_name][round_index],
                        used,
                    )
                    used.add(number)
                hit_counts.append(len(used & set(example.winning_numbers)))
            set_metrics[split_name] = {
                "target_round_count": len(target_rounds),
                "mean_hits": float(np.mean(hit_counts)) if hit_counts else None,
                "three_plus": sum(value >= 3 for value in hit_counts),
                "four_plus": sum(value >= 4 for value in hit_counts),
                "five_plus": sum(value >= 5 for value in hit_counts),
                "six_exact": sum(value == 6 for value in hit_counts),
                "match_distribution": {
                    str(value): hit_counts.count(value) for value in range(7)
                },
            }
            set_metrics[split_name].update(
                _random_set_controls(
                    target_rounds,
                    hit_counts,
                    config.random_control_replicates,
                    config.seed + 10_000 + len(set_metrics),
                )
            )
    numeric_controls: dict[str, Any] = {}
    if config.target != "main_positions":
        y_values = _target_values(dataset, examples, target_names[0])
        for phase, indices in (
            ("training", train_idx),
            ("validation", validation_idx),
            ("independent_test", test_idx),
        ):
            numeric_controls[phase] = _random_numeric_controls(
                y_values[indices],
                np.asarray(target_predictions[target_names[0]][phase]),
                config.random_control_replicates,
                config.seed + 20_000 + len(numeric_controls),
            )
    result: dict[str, Any] = {
        "experiment_id": experiment_id,
        "status": "completed",
        "engine": "pysr_symbolic_regression",
        "engine_version": PYSR_ENGINE_VERSION,
        "pysr_version": status.get("version"),
        "pysr_run_directories": pysr_run_directories,
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": duration,
        "seed": config.seed,
        "config": asdict(config),
        "target": config.target,
        "feature_mapping": feature_mapping(config.feature_profile, config.max_lag),
        "dataset": {
            "source_kind": dataset.source_kind,
            "verification_status": dataset.verification_status,
            "dataset_sha256": _dataset_sha256(dataset),
            "round_range": [dataset.records[0].round_number, dataset.records[-1].round_number],
            "target_data_leakage_policy": (
                "features use only prior draw values; target draw excluded"
            ),
        },
        "splits": {
            "train_rounds": [item.round_number for item in splits.train],
            "validation_rounds": [item.round_number for item in splits.validation],
            "test_rounds": [item.round_number for item in splits.test],
        },
        "metrics": target_metrics,
        "numeric_random_controls": numeric_controls or None,
        "six_number_set_metrics": set_metrics or None,
        "candidate_count": len(candidates),
        "evaluation_cache": {"reused": cache_reused, "computed": cache_computed},
        "evaluation_cache_scope": (
            "reuses holdout metric calculations after PySR search; PySR internal population "
            "fitness evaluations are not short-circuited"
        ),
        "dataset_sha256": _dataset_sha256(dataset),
        "candidates": candidates,
        "prediction_policy": (
            "regress sorted main-number position then round/clamp to 1..45"
            if config.target != "bonus"
            else "regress bonus number then round/clamp to 1..45"
        ),
        "six_set_policy": (
            "six independently fitted sorted positions, rounded to 1..45, collision-repaired "
            "to distinct numbers, then scored as an unordered set"
            if config.target == "main_positions"
            else "not produced by this numeric target"
        ),
        "resume_policy": (
            "PySR from_file warm-start used only when enabled and a previous run directory exists; "
            "process-restart recovery depends on a valid PySR pickle"
        ),
    }
    if progress:
        progress({"stage": "pysr_fit_complete", "candidates": len(candidates)})
    store.save(result)
    return result
