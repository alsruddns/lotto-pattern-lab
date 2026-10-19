"""Time-budgeted, checkpointed orchestration over the existing intermittent GP engine."""

import hashlib
import json
import math
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
from .feature_schema import PRIMITIVE_SET_VERSION, feature_schema_version
from .intermittent import IntermittentConfig
from .paths import locate_checkpoint, project_path
from .pattern_library import (
    PatternLibrary,
    register_experiment_candidates,
    register_experiment_exact_matches,
)
from .research_rounds import (
    SUMMARY_SCHEMA_VERSION,
    ResearchRoundStore,
    empty_dashboard_summary,
    merge_unique_hit_rounds,
    rebuild_dashboard_summary_from_store,
    summarize_experiment,
)
from .storage import ExperimentStore
from .storage_management import run_automatic_checkpoint_cleanup

SUPPORTED_ENGINES = tuple(
    engine_id for engine_id, engine in ENGINE_REGISTRY.items() if engine.implemented
)


def dataset_digest(dataset: DrawDataset) -> str:
    payload = []
    include_bonus = any(record.draw.bonus is not None for record in dataset.records)
    for record in dataset.records:
        item = {
            "round": record.round_number,
            "date": record.draw_date.isoformat() if record.draw_date else None,
            "numbers": record.draw.numbers,
        }
        if include_bonus:
            item["bonus"] = record.draw.bonus
        payload.append(item)
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f"{path.stem}_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path, candidates = locate_checkpoint(path)
    if not checkpoint_path.is_file() and candidates:
        raise FileNotFoundError(
            f"Checkpoint path is incorrect: {checkpoint_path}. "
            f"A same-named checkpoint exists at {candidates[0]}"
        )
    try:
        source_file = checkpoint_path.open(encoding="utf-8")
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}") from error
    with source_file as source:
        return json.load(source)


def request_stop(checkpoint_path: str | Path) -> Path:
    stop_path = Path(f"{project_path(checkpoint_path)}.stop")
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
    checkpoint = project_path(checkpoint_path)
    run_automatic_checkpoint_cleanup()
    stop_path = Path(f"{checkpoint}.stop")
    data_hash = dataset_digest(dataset)
    config_payload = asdict(config)
    store = ExperimentStore(experiment_directory)
    library = PatternLibrary(pattern_database)
    round_store = ResearchRoundStore(pattern_database)
    if resume:
        state = load_checkpoint(checkpoint)
        if not state.get("research_id"):
            raise ValueError(f"Cannot resume: checkpoint has no research_id: {checkpoint}")
        recorded_path = state.get("checkpoint_path")
        if recorded_path and project_path(recorded_path) != checkpoint:
            raise ValueError(
                "Cannot resume: checkpoint research_id "
                f"{state['research_id']} points to {recorded_path}, not {checkpoint}."
            )
        progress_research_id = state.get("current_progress", {}).get("research_id")
        if progress_research_id and progress_research_id != state["research_id"]:
            raise ValueError(
                "Cannot resume: current progress belongs to research ID "
                f"{progress_research_id}, but checkpoint ID is {state['research_id']}."
            )
        in_progress_research_id = state.get("in_progress", {}).get("research_id")
        if in_progress_research_id and in_progress_research_id != state["research_id"]:
            raise ValueError(
                "Cannot resume: in-progress experiment belongs to research ID "
                f"{in_progress_research_id}, but checkpoint ID is {state['research_id']}."
            )
        mismatched_runs = [
            item.get("experiment_id")
            for item in state.get("completed_experiments", [])
            if item.get("research_id") not in (None, state["research_id"])
        ]
        if mismatched_runs:
            raise ValueError(
                "Cannot resume: completed experiment history contains records for a different "
                f"research ID: {mismatched_runs}"
            )
        if state.get("dataset_sha256") != data_hash:
            raise ValueError("Cannot resume: checkpoint dataset hash differs from selected data.")
        stored_config = {**asdict(IntermittentConfig()), **state.get("config", {})}
        if stored_config != config_payload or tuple(state.get("engines", ())) != engines:
            raise ValueError("Cannot resume: checkpoint configuration or engine list differs.")
        elite_state = state.get("elite_seeding") or {
            "enabled": False,
            "ratio": 0.2,
            "selected_patterns": [],
            "source_pattern_ids": [],
            "skipped": [],
            "policy": "legacy_unseeded",
        }
        if bool(elite_state.get("enabled")) != config.elite_seeding_enabled:
            raise ValueError("Cannot resume: Elite Seeding setting differs from the checkpoint.")
        if float(elite_state.get("ratio", 0.2)) != config.elite_seed_ratio:
            raise ValueError("Cannot resume: Elite Seeding ratio differs from the checkpoint.")
        state["elite_seeding"] = elite_state
    else:
        if checkpoint.exists():
            raise FileExistsError(f"Checkpoint already exists: {checkpoint}; use resume=True.")
        research_id = str(uuid.uuid4())
        if config.elite_seeding_enabled:
            selected_patterns, skipped_patterns = library.select_elite_candidates(
                feature_profile=config.feature_profile,
                max_lag=config.max_lag,
                max_candidates=math.ceil(config.population_size * config.elite_seed_ratio) * 2,
            )
        else:
            selected_patterns, skipped_patterns = [], []
        elite_state = {
            "enabled": config.elite_seeding_enabled,
            "ratio": config.elite_seed_ratio,
            "source_pattern_ids": [item["pattern_id"] for item in selected_patterns],
            "selected_patterns": selected_patterns,
            "skipped": skipped_patterns,
            "feature_profile": config.feature_profile,
            "feature_schema_version": feature_schema_version(config.feature_profile),
            "primitive_set_version": PRIMITIVE_SET_VERSION,
            "policy": "validation_stability_complexity_diversity_v1",
            "independent_test_used_for_selection": False,
        }
        state = {
            "research_id": research_id,
            "checkpoint_path": str(checkpoint),
            "resumable": True,
            "status": "queued",
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_sha256": data_hash,
            "dataset_source_kind": dataset.source_kind,
            "dataset_verification_status": dataset.verification_status,
            "dataset_round_range": [
                dataset.records[0].round_number,
                dataset.records[-1].round_number,
            ],
            "config": config_payload,
            "elite_seeding": elite_state,
            "engines": list(engines),
            "completed_runs": 0,
            "attempts": 0,
            "completed_experiments": [],
            "rng_resume": {
                "mode": "per_attempt_seed",
                "base_seed": config.seed,
                "next_seed": config.seed,
            },
            "failed_runs": [],
            "discovered_exact_matches": [],
            "last_experiment_id": None,
            "resume_count": 0,
            "resume_semantics": (
                "resume at next seeded experiment boundary; no mid-generation restore"
            ),
        }
    state["checkpoint_path"] = str(checkpoint)
    state["config"] = config_payload
    state["resumable"] = True
    state.setdefault("completed_experiments", [])
    round_number = round_store.ensure_round(
        str(state["research_id"]),
        {
            "created_at": state.get("created_at"),
            "checkpoint_path": str(checkpoint),
            "status": state.get("status", "queued"),
        },
        checkpoint.parent,
        store.directory,
    )
    state["research_round"] = round_number
    if state.get("dashboard_summary", {}).get("summary_schema_version") != SUMMARY_SCHEMA_VERSION:
        state["dashboard_summary"] = rebuild_dashboard_summary_from_store(
            {
                **state,
                "completed_experiment_ids": [
                    item.get("experiment_id")
                    for item in state.get("completed_experiments", [])
                    if item.get("experiment_id")
                ],
            },
            store.directory,
        )
    state["rng_resume"] = {
        "mode": "per_attempt_seed",
        "base_seed": config.seed,
        "next_seed": config.seed + int(state.get("attempts", 0)),
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
    round_store.update(str(state["research_id"]), _round_metadata(state))
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
    elite_seeds = tuple(state.get("elite_seeding", {}).get("selected_patterns", ()))
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
        state["rng_resume"]["next_seed"] = config.seed + state["attempts"]
        state["in_progress"] = {
            "seed": run_config.seed,
            "attempt": state["attempts"],
            "research_id": state["research_id"],
            "started_at": datetime.now(UTC).isoformat(),
        }
        _atomic_json(checkpoint, state)

        def report(event: dict[str, Any]) -> None:
            if event.get("experiment_id"):
                state["in_progress"]["experiment_id"] = event["experiment_id"]
            state["current_progress"] = {
                "engine": "intermittent_gp",
                "research_id": state["research_id"],
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
            result = engine.run(ResearchRunContext(dataset, run_config, store, report, elite_seeds))
            result["research_id"] = state["research_id"]
            result["research_attempt"] = state["attempts"]
            store.save(result)
            seeding = result.get("elite_seeding", {})
            used_pattern_ids = sorted(
                {
                    str(pattern_id)
                    for model_key in ("one_number_search", "six_number_set_search")
                    for pattern_id in seeding.get(model_key, {}).get("source_pattern_ids", [])
                }
            )
            if used_pattern_ids:
                library.record_elite_use(used_pattern_ids, state["research_id"], data_hash)
            registered = register_experiment_candidates(result, library)
            exact_matches = register_experiment_exact_matches(result, library)
            result["pattern_library_candidates"] = registered
            result["full_set_match_discoveries"] = exact_matches
            store.save(result)
            compact_summary = summarize_experiment(result, registered, exact_matches)
            dashboard_summary = state["dashboard_summary"]
            dashboard_summary["completed_experiment_count"] += 1
            for key in (
                "candidate_count",
                "inherited_pattern_count",
                "exact_match_count",
            ):
                dashboard_summary[key] += compact_summary[key]
            if dashboard_summary.get("registration_metrics_available", True):
                dashboard_summary["new_pattern_count"] += compact_summary["new_pattern_count"]
                dashboard_summary["duplicate_pattern_count"] += compact_summary[
                    "duplicate_pattern_count"
                ]
                dashboard_summary["child_pattern_count"] += compact_summary[
                    "child_pattern_count"
                ]
            else:
                pattern_counts = library.research_pattern_counts(str(state["research_id"]))
                dashboard_summary["new_pattern_count"] = pattern_counts["new_pattern_count"]
                dashboard_summary["child_pattern_count"] = pattern_counts["child_pattern_count"]
                dashboard_summary["duplicate_pattern_count"] = None
            dashboard_summary["model_metrics"]["one_number"].append(
                compact_summary["models"]["one_number"]
            )
            dashboard_summary["model_metrics"]["six_number_set"].append(
                compact_summary["models"]["six_number_set"]
            )
            dashboard_summary["experiment_summaries"].append(compact_summary)
            merge_unique_hit_rounds(
                dashboard_summary, compact_summary.get("unique_hit_rounds", {})
            )
            for model_key, metrics in compact_summary["paired"].items():
                dashboard_summary["paired_metrics"].setdefault(model_key, []).append(metrics)
            for key, values in compact_summary["hits"].items():
                totals = dashboard_summary["hit_totals"].setdefault(
                    key, {"3_plus": 0, "4_plus": 0, "5_plus": 0, "6_exact": 0}
                )
                for metric, value in values.items():
                    totals[metric] += int(value)
            state["completed_runs"] += 1
            state["last_experiment_id"] = result["experiment_id"]
            state["completed_experiments"].append(
                {
                    "attempt": state["attempts"],
                    "seed": run_config.seed,
                    "experiment_id": result["experiment_id"],
                    "research_id": state["research_id"],
                    "path": str(store.directory / f"{result['experiment_id']}.json"),
                    "status": "completed",
                }
            )
            state["discovered_exact_matches"].extend(exact_matches)
            state["last_run_summary"] = {
                "experiment_id": result["experiment_id"],
                "runtime_seconds": result["runtime_seconds"],
                "engine": "intermittent_gp",
                "exact_matches": len(exact_matches),
                "patterns_registered": len(registered),
                "one_number_test_score": result["one_number_model"]["selected"]["test_score"],
                "six_number_test_score": result["six_number_set_model"]["selected"]["test_score"],
                "bonus_number_test_score": (
                    result.get("bonus_number_model", {}).get("selected", {}).get("test_score")
                    if result.get("bonus_number_model")
                    else None
                ),
                "prize_tier_backtest": result.get("prize_tier_backtest"),
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
            state["dashboard_summary"]["failed_experiment_count"] = len(state["failed_runs"])
            state.pop("in_progress", None)
            if consecutive_failures >= 3:
                state["status"] = "failed"
                state["failure_reason"] = "three_consecutive_experiment_failures"
                break
        state["last_updated_at"] = datetime.now(UTC).isoformat()
        _atomic_json(checkpoint, state)
        round_store.update(str(state["research_id"]), _round_metadata(state))
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
    elapsed_this_session = time.monotonic() - started
    state["elapsed_seconds"] = elapsed_this_session
    state["total_elapsed_seconds"] = (
        float(state.get("total_elapsed_seconds", 0.0)) + elapsed_this_session
    )
    _atomic_json(checkpoint, state)
    round_store.update(str(state["research_id"]), _round_metadata(state))
    run_automatic_checkpoint_cleanup()
    if state.get("pause_reason") == "stop_requested_at_experiment_boundary":
        stop_path.unlink(missing_ok=True)
    return state


def _round_metadata(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_round": state.get("research_round"),
        "status": state.get("status"),
        "created_at": state.get("created_at"),
        "started_at": state.get("started_at"),
        "last_updated_at": state.get("last_updated_at"),
        "total_elapsed_seconds": state.get("total_elapsed_seconds", state.get("elapsed_seconds")),
        "checkpoint_path": state.get("checkpoint_path"),
        "dataset_sha256": state.get("dataset_sha256"),
        "dataset_source_kind": state.get("dataset_source_kind"),
        "dataset_verification_status": state.get("dataset_verification_status"),
        "dataset_round_range": state.get("dataset_round_range"),
        "config": state.get("config", {}),
        "engines": state.get("engines", []),
        "completed_runs": state.get("completed_runs", 0),
        "failed_runs": len(state.get("failed_runs", [])),
        "completed_experiment_ids": [
            item.get("experiment_id")
            for item in state.get("completed_experiments", [])
            if item.get("experiment_id")
        ],
        "failed_run_summary": [
            {key: item.get(key) for key in ("seed", "error_type", "error", "failed_at")}
            for item in state.get("failed_runs", [])[-20:]
        ],
        "dashboard_summary": state.get("dashboard_summary", empty_dashboard_summary()),
        "elite_seeding": {
            key: value
            for key, value in (state.get("elite_seeding") or {}).items()
            if key not in {"selected_patterns", "skipped"}
        },
        "next_seed": state.get("rng_resume", {}).get("next_seed"),
        "discovered_exact_matches": state.get("discovered_exact_matches", []),
    }
