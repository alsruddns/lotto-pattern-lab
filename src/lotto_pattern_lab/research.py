"""Time-budgeted, checkpointed orchestration over the existing intermittent GP engine."""

import hashlib
import json
import os
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .dataset import DrawDataset
from .engines import ENGINE_REGISTRY, ResearchRunContext, get_engine
from .intermittent import IntermittentConfig
from .pattern_library import (
    PatternLibrary,
    register_experiment_candidates,
    register_experiment_exact_matches,
)
from .storage import ExperimentStore

SUPPORTED_ENGINES = tuple(
    engine_id for engine_id, engine in ENGINE_REGISTRY.items() if engine.implemented
)


def dataset_digest(dataset: DrawDataset) -> str:
    payload = [
        {
            "round": record.round_number,
            "date": record.draw_date.isoformat() if record.draw_date else None,
            "numbers": record.draw.numbers,
        }
        for record in dataset.records
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f"{path.stem}_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)


def request_stop(checkpoint_path: str | Path) -> Path:
    stop_path = Path(f"{checkpoint_path}.stop")
    stop_path.parent.mkdir(parents=True, exist_ok=True)
    stop_path.write_text(datetime.now(UTC).isoformat(), encoding="utf-8")
    return stop_path


def run_research_schedule(
    dataset: DrawDataset,
    config: IntermittentConfig,
    *,
    duration_seconds: float,
    checkpoint_path: str | Path = "checkpoints/research.json",
    experiment_directory: str | Path = "experiments",
    pattern_database: str | Path = "data/database/patterns.sqlite",
    resume: bool = False,
    max_runs: int | None = None,
    engines: tuple[str, ...] = SUPPORTED_ENGINES,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run complete seeded experiments until time, count, or a stop request is reached.

    A checkpoint resumes between experiments, not inside a DEAP generation. This avoids
    serializing DEAP internals and still preserves every completed/failed trial atomically.
    """
    if duration_seconds <= 0:
        raise ValueError("Research duration must be positive.")
    if max_runs is not None and max_runs < 1:
        raise ValueError("max_runs must be positive when provided.")
    if not engines or any(engine not in SUPPORTED_ENGINES for engine in engines):
        raise ValueError(f"Supported engines are: {', '.join(SUPPORTED_ENGINES)}")
    checkpoint = Path(checkpoint_path)
    stop_path = Path(f"{checkpoint}.stop")
    data_hash = dataset_digest(dataset)
    config_payload = asdict(config)
    store = ExperimentStore(experiment_directory)
    if resume:
        state = load_checkpoint(checkpoint)
        if state.get("dataset_sha256") != data_hash:
            raise ValueError("Cannot resume: checkpoint dataset hash differs from selected data.")
        if state.get("config") != config_payload or tuple(state.get("engines", ())) != engines:
            raise ValueError("Cannot resume: checkpoint configuration or engine list differs.")
    else:
        if checkpoint.exists():
            raise FileExistsError(f"Checkpoint already exists: {checkpoint}; use resume=True.")
        state = {
            "research_id": str(uuid.uuid4()),
            "status": "queued",
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_sha256": data_hash,
            "dataset_source_kind": dataset.source_kind,
            "dataset_round_range": [
                dataset.records[0].round_number,
                dataset.records[-1].round_number,
            ],
            "config": config_payload,
            "engines": list(engines),
            "completed_runs": 0,
            "attempts": 0,
            "failed_runs": [],
            "discovered_exact_matches": [],
            "last_experiment_id": None,
            "resume_count": 0,
            "resume_semantics": (
                "resume at next seeded experiment boundary; no mid-generation restore"
            ),
        }
    previous_checkpoint_time = state.get("last_updated_at", state.get("created_at"))
    if resume and state.get("in_progress"):
        interrupted = dict(state["in_progress"])
        interrupted.update(
            {
                "failed_at": datetime.now(UTC).isoformat(),
                "error_type": "InterruptedProcess",
                "error": "Process stopped before the current GP experiment boundary.",
            }
        )
        state["failed_runs"].append(interrupted)
        state.pop("in_progress", None)
    state["status"] = "running"
    state["worker_pid"] = os.getpid()
    state["started_at"] = state.get("started_at", datetime.now(UTC).isoformat())
    state["resume_count"] += int(resume)
    state["last_updated_at"] = datetime.now(UTC).isoformat()
    _atomic_json(checkpoint, state)
    if stop_path.exists():
        try:
            stop_time = datetime.fromisoformat(stop_path.read_text(encoding="utf-8").strip())
            checkpoint_time = datetime.fromisoformat(previous_checkpoint_time)
        except (OSError, ValueError):
            stop_time = datetime.min.replace(tzinfo=UTC)
            checkpoint_time = datetime.now(UTC)
        if stop_time <= checkpoint_time and state.get("status") == "running":
            # Keep a new request made during a restart; consume only an old handled request.
            stop_path.unlink(missing_ok=True)

    started = time.monotonic()
    library = PatternLibrary(pattern_database)
    for failed in state["failed_runs"]:
        if failed.get("error_type") != "InterruptedProcess" or not failed.get("experiment_id"):
            continue
        try:
            orphan = store.load(str(failed["experiment_id"]))
        except FileNotFoundError:
            continue
        if orphan.get("status") == "running":
            orphan.update(
                {
                    "status": "failed",
                    "finished_at": datetime.now(UTC).isoformat(),
                    "failure": failed,
                }
            )
            store.save(orphan)
    consecutive_failures = 0
    while time.monotonic() - started < duration_seconds:
        if stop_path.exists():
            state["status"] = "paused"
            state["pause_reason"] = "stop_requested_at_experiment_boundary"
            break
        if max_runs is not None and state["completed_runs"] >= max_runs:
            state["status"] = "completed"
            state["stop_reason"] = "run_count_reached"
            break
        run_index = int(state["attempts"])
        run_config = replace(config, seed=config.seed + run_index)
        state["attempts"] = run_index + 1
        state["in_progress"] = {
            "seed": run_config.seed,
            "attempt": state["attempts"],
            "started_at": datetime.now(UTC).isoformat(),
        }
        _atomic_json(checkpoint, state)

        def report(event: dict[str, Any]) -> None:
            if event.get("experiment_id"):
                state["in_progress"]["experiment_id"] = event["experiment_id"]
            state["current_progress"] = {
                "engine": "intermittent_gp",
                "run_index": run_index + 1,
                "event": event,
                "updated_at": datetime.now(UTC).isoformat(),
            }
            state["last_updated_at"] = datetime.now(UTC).isoformat()
            _atomic_json(checkpoint, state)
            if progress:
                progress(state["current_progress"])

        try:
            engine = get_engine(engines[(run_index) % len(engines)])
            result = engine.run(ResearchRunContext(dataset, run_config, store, report))
            registered = register_experiment_candidates(result, library)
            exact_matches = register_experiment_exact_matches(result, library)
            state["completed_runs"] += 1
            state["last_experiment_id"] = result["experiment_id"]
            state["discovered_exact_matches"].extend(exact_matches)
            state["last_run_summary"] = {
                "experiment_id": result["experiment_id"],
                "runtime_seconds": result["runtime_seconds"],
                "engine": "intermittent_gp",
                "exact_matches": len(exact_matches),
                "patterns_registered": len(registered),
                "one_number_test_score": result["one_number_model"]["selected"]["test_score"],
                "six_number_test_score": result["six_number_set_model"]["selected"]["test_score"],
            }
            state.pop("in_progress", None)
            consecutive_failures = 0
        except Exception as error:
            consecutive_failures += 1
            state["failed_runs"].append(
                {
                    "seed": run_config.seed,
                    "failed_at": datetime.now(UTC).isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            state.pop("in_progress", None)
            if consecutive_failures >= 3:
                state["status"] = "failed"
                state["failure_reason"] = "three_consecutive_experiment_failures"
                break
        state["last_updated_at"] = datetime.now(UTC).isoformat()
        _atomic_json(checkpoint, state)
        if max_runs is not None and state["completed_runs"] >= max_runs:
            state["status"] = "completed"
            state["stop_reason"] = "run_count_reached"
            break
    else:
        state["status"] = "completed"
        state["stop_reason"] = "time_budget_reached"

    if state["status"] == "running":
        state["status"] = "paused"
        state["pause_reason"] = "time_budget_reached_before_boundary_completion"
    state["last_updated_at"] = datetime.now(UTC).isoformat()
    state["elapsed_seconds"] = time.monotonic() - started
    _atomic_json(checkpoint, state)
    if state.get("pause_reason") == "stop_requested_at_experiment_boundary":
        stop_path.unlink(missing_ok=True)
    return state
