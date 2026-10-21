"""Short, isolated environment checks for long-running research engines."""

from __future__ import annotations

import json
import random
import tempfile
import time
import traceback
import uuid
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deap import gp

from .atomic_io import atomic_write_bytes
from .dataset import DrawDataset
from .engines import GP_ENGINE_ID, PYSR_ENGINE_ID
from .experiment import dataset_metadata
from .gp_engine import _make_primitive_set, validate_primitive_set
from .intermittent import (
    IntermittentConfig,
    _ensure_individual_types,
    _initial_population,
    _mate_individuals,
    _mutate_individual,
    run_intermittent_search,
)
from .intermittent_experiment import _splits_for
from .preflight_cache import load_preflight_cache, preflight_cache_key, save_preflight_cache
from .pysr_backend import PySRConfig, run_pysr_experiment
from .storage import ExperimentStore


def _tree_type_errors(individual: list) -> list[str]:
    return [
        f"tree {tree_index} node {node_index} has {node.ret.__name__}, expected float"
        for tree_index, tree in enumerate(individual)
        for node_index, node in enumerate(tree)
        if node.ret is not float
    ]


def _storage_round_trip(
    root: Path, store: ExperimentStore, engine_id: str, config: dict[str, Any], dataset: DrawDataset
) -> tuple[str, bool]:
    experiment_id = str(uuid.uuid4())
    record = {
        "experiment_id": experiment_id,
        "status": "preflight",
        "engine": engine_id,
        "config": config,
        "dataset": dataset_metadata(dataset),
    }
    store.save(record)
    if store.load(experiment_id).get("engine") != engine_id:
        raise ValueError(f"{engine_id} experiment JSON round-trip failed.")
    checkpoint = root / "checkpoints" / f"{experiment_id}.json"
    checkpoint_payload = json.dumps(record, ensure_ascii=False, allow_nan=False).encode("utf-8")
    atomic_write_bytes(checkpoint, checkpoint_payload)
    restored = json.loads(checkpoint.read_text(encoding="utf-8"))
    if restored.get("experiment_id") != experiment_id:
        raise ValueError(f"{engine_id} checkpoint round-trip failed.")
    return experiment_id, True


def _run_gp_checks(
    dataset: DrawDataset,
    config: IntermittentConfig,
    elite_seeds: tuple[dict[str, object], ...],
) -> dict[str, Any]:
    examples, splits = _splits_for(dataset, config)
    if not examples or not splits.train:
        raise ValueError("Selected GP feature profile did not produce a training split.")
    pset = _make_primitive_set(len(examples[0].features))
    validate_primitive_set(pset)
    _ensure_individual_types()
    random.seed(config.seed)

    for max_depth in range(0, config.max_depth + 1):
        for generator in (gp.genFull, gp.genGrow):
            for _ in range(10):
                tree = gp.PrimitiveTree(generator(pset=pset, min_=0, max_=max_depth))
                if tree.height > max_depth or any(node.ret is not float for node in tree):
                    raise ValueError(f"{generator.__name__} generated an invalid float tree.")

    population_size_config = replace(config, population_size=100, workers=1)
    population = [
        _initial_population(pset, output_count, population_size_config, elite_seeds)
        for output_count in (1, 6)
    ]
    generated_count = 0
    mutated_count = 0
    crossed_count = 0
    for (group, seed_report), output_count in zip(population, (1, 6), strict=True):
        if seed_report.get("seeded_count", 0) > seed_report.get("requested_elite_count", 0):
            raise ValueError("Elite restoration produced more seeds than the configured ratio.")
        for individual in group:
            errors = _tree_type_errors(individual)
            if errors:
                raise ValueError("Initial population: " + "; ".join(errors[:3]))
            generated_count += 1
            _mutate_individual(individual, pset, config.max_depth, 35, generation=1)
            errors = _tree_type_errors(individual)
            if errors:
                raise ValueError("Mutation: " + "; ".join(errors[:3]))
            mutated_count += 1
        for first, second in zip(group[::2], group[1::2], strict=False):
            _mate_individuals(first, second, config.max_depth, 35, generation=1)
            if _tree_type_errors(first) or _tree_type_errors(second):
                raise ValueError("Crossover generated a type-incompatible GP tree.")
            crossed_count += 1

    short_config = replace(config, population_size=8, generations=1, workers=1, control_runs=0)
    one_number = run_intermittent_search(
        splits, short_config, outputs_per_event=1, elite_seeds=elite_seeds
    )
    six_number_set = run_intermittent_search(
        splits, short_config, outputs_per_event=6, elite_seeds=elite_seeds
    )
    with tempfile.TemporaryDirectory(prefix="lotto-gp-preflight-") as temporary:
        root = Path(temporary)
        store = ExperimentStore(root / "experiments")
        experiment_id, checkpoint_round_trip = _storage_round_trip(
            root, store, GP_ENGINE_ID, asdict(short_config), dataset
        )
    return {
        "status": "passed",
        "engine": GP_ENGINE_ID,
        "feature_profile": config.feature_profile,
        "input_feature_count": len(examples[0].features),
        "float_terminal_count": len(pset.terminals.get(float, ())),
        "float_primitive_count": len(pset.primitives.get(float, ())),
        "generated_individuals": generated_count,
        "mutations": mutated_count,
        "crossovers": crossed_count,
        "elite_one_number": one_number.seeding,
        "elite_six_number_set": six_number_set.seeding,
        "one_number_smoke_candidates": len(one_number.candidates),
        "six_number_smoke_candidates": len(six_number_set.candidates),
        "experiment_id": experiment_id,
        "experiment_json_round_trip": True,
        "checkpoint_round_trip": checkpoint_round_trip,
        "train_round_count": len(splits.train),
        "validation_round_count": len(splits.validation),
        "test_round_count": len(splits.test),
    }


def _run_pysr_check(dataset: DrawDataset, config: PySRConfig) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="lotto-pysr-preflight-") as temporary:
        root = Path(temporary)
        store = ExperimentStore(root / "experiments")
        smoke_config = replace(
            config,
            target="one_number",
            timeout_seconds=min(config.timeout_seconds, 10.0),
            niterations=1,
            population_size=8,
            populations=1,
            maxsize=8,
            maxdepth=3,
            workers=1,
            random_control_replicates=0,
            guesses=(),
            warm_start=False,
            run_directory=str(root / "pysr_runs" / "smoke"),
        )
        result = run_pysr_experiment(
            dataset,
            smoke_config,
            store,
            cache_path=str(root / "preflight-cache.sqlite"),
        )
        persisted = store.load(str(result["experiment_id"]))
        if persisted.get("engine") != PYSR_ENGINE_ID or not persisted.get("candidates"):
            raise ValueError("PySR smoke fit was not persisted as a symbolic-regression result.")
        _storage_round_trip(root, store, PYSR_ENGINE_ID, asdict(smoke_config), dataset)
        return {
            "status": "passed",
            "engine": PYSR_ENGINE_ID,
            "pysr_version": result.get("pysr_version"),
            "julia_search": True,
            "smoke_target": smoke_config.target,
            "candidate_count": result.get("candidate_count", 0),
            "experiment_json_round_trip": True,
            "checkpoint_round_trip": True,
            "dataset": dataset_metadata(dataset),
            "runtime_seconds": result.get("runtime_seconds"),
        }


def run_research_preflight(
    dataset: DrawDataset,
    config: IntermittentConfig,
    engines: tuple[str, ...],
    pysr_config: PySRConfig | None = None,
    elite_seeds: tuple[dict[str, object], ...] = (),
) -> dict[str, Any]:
    """Run brief per-engine checks without touching research checkpoints or libraries."""
    report: dict[str, Any] = {
        "preflight_id": str(uuid.uuid4()),
        "dataset_sha256": dataset_metadata(dataset).get("dataset_sha256"),
        "engines": {},
    }
    for engine_id in engines:
        started_at = time.perf_counter()
        try:
            if engine_id == GP_ENGINE_ID:
                result = _run_gp_checks(dataset, config, elite_seeds)
            elif engine_id == PYSR_ENGINE_ID:
                if pysr_config is None:
                    raise ValueError("PySR configuration is required for the selected engine.")
                result = _run_pysr_check(dataset, pysr_config)
            else:
                raise ValueError(f"No preflight is defined for engine {engine_id!r}.")
            result["runtime_seconds"] = round(time.perf_counter() - started_at, 3)
            report["engines"][engine_id] = result
        except Exception as error:
            report["engines"][engine_id] = {
                "status": "failed",
                "engine": engine_id,
                "research_id": None,
                "seed": config.seed,
                "feature_profile": config.feature_profile,
                "stage": "preflight",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "runtime_seconds": round(time.perf_counter() - started_at, 3),
            }
    report["status"] = (
        "passed"
        if report["engines"] and all(
            result["status"] == "passed" for result in report["engines"].values()
        )
        else "failed"
    )
    return report


def research_preflight_cache_key(
    dataset: DrawDataset,
    config: IntermittentConfig,
    engines: tuple[str, ...],
    pysr_config: PySRConfig | None = None,
    elite_seeds: tuple[dict[str, object], ...] = (),
    *,
    dataset_sha256: str | None = None,
) -> str:
    """Key expensive smoke checks only by inputs that can affect their outcome."""
    metadata = dataset_metadata(dataset) if dataset_sha256 is None else {}
    settings: dict[str, Any] = {
        "dataset_sha256": dataset_sha256 or metadata.get("dataset_sha256"),
        "engines": list(engines),
    }
    if GP_ENGINE_ID in engines:
        settings["gp"] = {
            key: getattr(config, key)
            for key in (
                "crossover_probability",
                "mutation_probability",
                "max_depth",
                "train_ratio",
                "validation_ratio",
                "max_lag",
                "min_activation_rate",
                "hidden_states",
                "seed",
                "feature_profile",
                "elite_seeding_enabled",
                "elite_seed_ratio",
            )
        }
        settings["elite_seeds"] = [
            {
                key: seed.get(key)
                for key in ("pattern_id", "candidate_id", "expression", "feature_profile")
                if key in seed
            }
            for seed in elite_seeds
        ]
    if PYSR_ENGINE_ID in engines:
        if pysr_config is None:
            settings["pysr"] = None
        else:
            settings["pysr"] = {
                key: getattr(pysr_config, key)
                for key in (
                    "feature_profile",
                    "train_ratio",
                    "validation_ratio",
                    "max_lag",
                    "seed",
                    "binary_operators",
                    "unary_operators",
                )
            }
    return preflight_cache_key(settings)


def get_or_run_research_preflight(
    dataset: DrawDataset,
    config: IntermittentConfig,
    engines: tuple[str, ...],
    pysr_config: PySRConfig | None = None,
    elite_seeds: tuple[dict[str, object], ...] = (),
    *,
    dataset_sha256: str | None = None,
    force: bool = False,
    cache_path: str | Path | None = None,
) -> dict[str, Any]:
    """Reuse a fresh engine smoke report or run and persist a new one."""
    key = research_preflight_cache_key(
        dataset,
        config,
        engines,
        pysr_config,
        elite_seeds,
        dataset_sha256=dataset_sha256,
    )
    if not force:
        cached = load_preflight_cache(key, cache_path, engines=engines)
        if cached is not None:
            return {**cached, "cache_used": True, "runtime_seconds": 0.0}
    started = time.perf_counter()
    report = run_research_preflight(dataset, config, engines, pysr_config, elite_seeds)
    report["runtime_seconds"] = round(time.perf_counter() - started, 3)
    checked_at = datetime.now(UTC).isoformat()
    try:
        save_preflight_cache(key, report, cache_path, engines=engines)
        cache_saved = True
    except OSError:
        cache_saved = False
    return {
        "report": report,
        "checked_at": checked_at,
        "cache_key": key,
        "cache_used": False,
        "cache_saved": cache_saved,
        "runtime_seconds": report["runtime_seconds"],
    }
