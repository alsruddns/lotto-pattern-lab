"""Orchestrate GP search, chronological evaluation, and matched random controls."""

import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from .backtest import chronological_split
from .dataset import DrawDataset, make_random_dataset
from .features import build_examples, feature_names
from .gp_engine import SearchConfig, SearchResult, run_search
from .storage import ExperimentStore


def result_to_dict(result: SearchResult) -> dict[str, object]:
    return {
        "formula": result.formula,
        "training_accuracy": result.training_accuracy,
        "validation_accuracy": result.validation_accuracy,
        "test_accuracy": result.test_accuracy,
        "training_fitness": result.training_fitness,
        "complexity": result.complexity,
        "height": result.height,
        "train_rounds": list(result.train_rounds),
        "validation_rounds": list(result.validation_rounds),
        "test_rounds": list(result.test_rounds),
        "candidate_formulas": [
            {
                "formula": candidate.formula,
                "training_accuracy": candidate.training_accuracy,
                "validation_accuracy": candidate.validation_accuracy,
                "test_accuracy": candidate.test_accuracy,
                "training_fitness": candidate.training_fitness,
                "complexity": candidate.complexity,
                "height": candidate.height,
            }
            for candidate in result.candidates
        ],
        "random_baseline": result.random_baseline,
    }


def run_experiment(
    dataset: DrawDataset,
    config: SearchConfig,
    store: ExperimentStore,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    experiment_id = str(uuid.uuid4())
    started_at = datetime.now(UTC).isoformat()
    started_clock = time.perf_counter()
    actual_splits = chronological_split(
        build_examples(dataset, config.max_lag), config.train_ratio, config.validation_ratio
    )
    events: list[dict[str, object]] = []

    def record_progress(event: dict[str, int | float | str]) -> None:
        snapshot = {**event, "timestamp": datetime.now(UTC).isoformat()}
        events.append(snapshot)
        if progress:
            progress(snapshot)
        store.save(
            {
                "experiment_id": experiment_id,
                "status": "running",
                "started_at": started_at,
                "dataset": dataset_metadata(dataset),
                "config": config_to_plain(config),
                "progress": events,
            }
        )

    actual = run_search(actual_splits, config, record_progress, "dataset")
    control_results: list[dict[str, object]] = []
    for control_index in range(config.control_runs):
        control_seed = config.seed + 100_003 + control_index
        control = make_random_dataset(
            len(dataset.records), control_seed, dataset.records[0].round_number
        )
        control_config = SearchConfig(
            **{**config_to_plain(config), "seed": config.seed + 10_007 + control_index}
        )
        control_examples = build_examples(control, control_config.max_lag)
        control_splits = chronological_split(
            control_examples, control_config.train_ratio, control_config.validation_ratio
        )
        control_result = run_search(
            control_splits, control_config, record_progress, f"random_control_{control_index + 1}"
        )
        control_results.append(
            {
                "run": control_index + 1,
                "data_seed": control_seed,
                "search_seed": control_config.seed,
                **result_to_dict(control_result),
            }
        )

    runtime_seconds = time.perf_counter() - started_clock
    control_scores = [float(item["test_accuracy"]) for item in control_results]
    control_max = max(control_scores, default=0.0)
    empirical_p_value = (
        (1 + sum(score >= actual.test_accuracy for score in control_scores))
        / (1 + len(control_scores))
        if control_scores
        else None
    )
    result: dict[str, object] = {
        "experiment_id": experiment_id,
        "status": "completed",
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": runtime_seconds,
        "dataset": dataset_metadata(dataset),
        "config": config_to_plain(config),
        "seed": config.seed,
        "candidate": result_to_dict(actual),
        "feature_variables": {
            f"x{index}": name for index, name in enumerate(feature_names(config.max_lag))
        },
        "random_controls": control_results,
        "random_control_test_scores": control_scores,
        "random_control_max_test_accuracy": control_max,
        "random_control_empirical_p_value": empirical_p_value,
        "random_baseline_accuracy": 6 / 45,
        "progress": events,
        "test_set_used_for_selection": False,
    }
    store.save(result)
    return result


def dataset_metadata(dataset: DrawDataset) -> dict[str, object]:
    return {
        "source_kind": dataset.source_kind,
        "source_description": dataset.source_description,
        "source_url": dataset.source_url,
        "verification_status": dataset.verification_status,
        "verified_at": dataset.verified_at,
        "first_round": dataset.records[0].round_number,
        "last_round": dataset.records[-1].round_number,
        "first_date": dataset.records[0].draw_date.isoformat()
        if dataset.records[0].draw_date
        else None,
        "last_date": dataset.records[-1].draw_date.isoformat()
        if dataset.records[-1].draw_date
        else None,
        "draw_count": len(dataset.records),
    }


def config_to_plain(config: SearchConfig) -> dict[str, int | float]:
    return {
        "population_size": config.population_size,
        "generations": config.generations,
        "crossover_probability": config.crossover_probability,
        "mutation_probability": config.mutation_probability,
        "max_depth": config.max_depth,
        "seed": config.seed,
        "workers": config.workers,
        "train_ratio": config.train_ratio,
        "validation_ratio": config.validation_ratio,
        "max_lag": config.max_lag,
        "control_runs": config.control_runs,
    }


def load_result(store: ExperimentStore, experiment_id: str) -> dict[str, object]:
    return store.load(experiment_id)
