"""Time-budgeted, checkpointed orchestration over the existing intermittent GP engine."""

import hashlib
import json
import math
import os
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any

from .atomic_io import AtomicWriteError, atomic_write_bytes, interprocess_lock
from .dataset import DrawDataset, load_csv
from .engines import ENGINE_REGISTRY, ResearchRunContext, get_engine
from .evaluation_cache import EvaluationCache
from .feature_schema import PRIMITIVE_SET_VERSION, feature_schema_version
from .formula_replay_queue import enqueue_experiment_high_hit_candidates
from .intermittent import IntermittentConfig
from .paths import locate_checkpoint, project_path
from .pattern_library import (
    PatternLibrary,
    register_experiment_candidates,
    register_experiment_exact_matches,
)
from .pysr_backend import PySRConfig
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
DEFAULT_ENGINES = ("intermittent_gp",)
GLOBAL_RESEARCH_LOCK = project_path("checkpoints/research_global")


def research_worker_active() -> bool:
    """Report whether a research or optimization process holds the shared CPU slot."""
    try:
        with interprocess_lock(GLOBAL_RESEARCH_LOCK, timeout=0, scope="research-global"):
            return False
    except TimeoutError:
        return True


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
    content = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    atomic_write_bytes(path, content)


def _single_research_worker(function: Callable[..., dict[str, Any]]):
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
        checkpoint = project_path(kwargs.get("checkpoint_path", "checkpoints/research.json"))
        with interprocess_lock(checkpoint, timeout=0, scope="research-worker"):
            Path(f"{checkpoint}.launch").unlink(missing_ok=True)
            with interprocess_lock(
                GLOBAL_RESEARCH_LOCK, timeout=0, scope="research-global"
            ):
                return function(*args, **kwargs)

    return wrapped


def _unresolved_failure_count(state: dict[str, Any]) -> int:
    completed = {
        int(item["seed"])
        for item in state.get("completed_experiments", [])
        if item.get("seed") is not None
    }
    failed = {
        int(item["seed"]) for item in state.get("failed_runs", []) if item.get("seed") is not None
    }
    return len(failed - completed)


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


def pause_request_path(checkpoint_path: str | Path) -> Path:
    return Path(f"{project_path(checkpoint_path)}.pause")


def request_pause(
    checkpoint_path: str | Path,
    *,
    pattern_database: str | Path = "data/database/patterns.sqlite",
) -> Path:
    """Persist a pause intent; the worker consumes it at an experiment boundary."""
    checkpoint = project_path(checkpoint_path)
    state = load_checkpoint(checkpoint)
    if state.get("status") != "running":
        raise ValueError(f"Cannot request pause while research is {state.get('status')}.")
    if not state.get("research_id"):
        raise ValueError("Cannot pause a checkpoint without a research ID.")
    pause_path = pause_request_path(checkpoint)
    request = {
        "research_id": state["research_id"],
        "requested_at": datetime.now(UTC).isoformat(),
    }
    with interprocess_lock(checkpoint, scope="research-control"):
        state = load_checkpoint(checkpoint)
        if state.get("status") != "running":
            raise ValueError(f"Cannot request pause while research is {state.get('status')}.")
        if state.get("research_id") != request["research_id"]:
            raise ValueError("Research ID changed while requesting a pause.")
        atomic_write_bytes(
            pause_path,
            (json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"),
        )
        requested_state = {
            **state,
            "status": "pause_requested",
            "pause_requested_at": request["requested_at"],
            "last_updated_at": request["requested_at"],
        }
        try:
            ResearchRoundStore(pattern_database).update(
                str(state["research_id"]), _round_metadata(requested_state)
            )
        except Exception:
            pause_path.unlink(missing_ok=True)
            raise
    return pause_path


def request_stop(checkpoint_path: str | Path) -> Path:
    """Backward-compatible alias for the persistent safe-pause request."""
    return request_pause(checkpoint_path)


def _pause_requested(checkpoint: Path, research_id: str) -> bool:
    pause_path = pause_request_path(checkpoint)
    if not pause_path.is_file():
        # Accept stop markers written by older app versions.
        legacy_stop = Path(f"{checkpoint}.stop")
        return legacy_stop.is_file()
    try:
        request = json.loads(pause_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True
    return request.get("research_id") in (None, research_id)


@_single_research_worker
def run_research_schedule(
    dataset: DrawDataset,
    config: IntermittentConfig,
    *,
    duration_seconds: float,
    checkpoint_path: str | Path = "checkpoints/research.json",
    dataset_path: str | Path | None = None,
    research_id: str | None = None,
    experiment_directory: str | Path = "experiments",
    pattern_database: str | Path = "data/database/patterns.sqlite",
    resume: bool = False,
    max_runs: int | None = None,
    engines: tuple[str, ...] = DEFAULT_ENGINES,
    pysr_config: PySRConfig | None = None,
    candidate_replay_threshold: int = 5,
    candidate_replay_start_round: int = 753,
    candidate_replay_end_round: int = 1246,
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
    if "pysr_symbolic_regression" in engines and pysr_config is None:
        raise ValueError("PySR engine requires an explicit PySRConfig.")
    if candidate_replay_threshold not in {3, 4, 5, 6}:
        raise ValueError("후보 등록 기준은 3, 4, 5, 6개 중 하나여야 합니다.")
    if (
        candidate_replay_start_round < 1
        or candidate_replay_end_round < candidate_replay_start_round
    ):
        raise ValueError("자동 재검증 회차 범위가 올바르지 않습니다.")
    checkpoint = project_path(checkpoint_path)
    run_automatic_checkpoint_cleanup()
    pause_path = pause_request_path(checkpoint)
    data_hash = dataset_digest(dataset)
    config_payload = asdict(config)
    store = ExperimentStore(experiment_directory)
    library = PatternLibrary(pattern_database)
    round_store = ResearchRoundStore(pattern_database)
    if resume:
        state = load_checkpoint(checkpoint)
        if state.get("status") == "completed":
            raise ValueError("Cannot resume a completed research run.")
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
        stored_pysr_config = state.get("pysr_config")
        requested_pysr_config = (
            json.loads(json.dumps(asdict(pysr_config))) if pysr_config is not None else None
        )
        if stored_pysr_config != requested_pysr_config:
            raise ValueError("Cannot resume: PySR configuration differs from the checkpoint.")
        stored_replay = state.get(
            "candidate_replay",
            {"threshold": 5, "start_round": 753, "end_round": 1246},
        )
        requested_replay = {
            "threshold": candidate_replay_threshold,
            "start_round": candidate_replay_start_round,
            "end_round": candidate_replay_end_round,
        }
        if stored_replay != requested_replay:
            raise ValueError("Cannot resume: candidate replay settings differ from the checkpoint.")
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
        research_id = str(uuid.UUID(research_id)) if research_id else str(uuid.uuid4())
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
            "dataset_path": str(project_path(dataset_path)) if dataset_path else None,
            "dataset_source_kind": dataset.source_kind,
            "dataset_verification_status": dataset.verification_status,
            "dataset_round_range": [
                dataset.records[0].round_number,
                dataset.records[-1].round_number,
            ],
            "config": config_payload,
            "target_duration_seconds": float(duration_seconds),
            "elite_seeding": elite_state,
            "engines": list(engines),
            "pysr_config": asdict(pysr_config) if pysr_config is not None else None,
            "candidate_replay": {
                "threshold": candidate_replay_threshold,
                "start_round": candidate_replay_start_round,
                "end_round": candidate_replay_end_round,
            },
            "candidate_revalidation": {"queued_jobs": 0, "job_ids": [], "errors": []},
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
    state.setdefault(
        "candidate_replay",
        {
            "threshold": candidate_replay_threshold,
            "start_round": candidate_replay_start_round,
            "end_round": candidate_replay_end_round,
        },
    )
    state.setdefault("candidate_revalidation", {"queued_jobs": 0, "job_ids": [], "errors": []})
    state["config"] = config_payload
    state.setdefault(
        "target_duration_seconds",
        float(state.get("total_elapsed_seconds", 0.0)) + float(duration_seconds),
    )
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
    else:
        summary = state["dashboard_summary"]
        summary["p_value_distributions"] = {"one_number": [], "six_number_set": []}
        for compact in summary.get("experiment_summaries", []):
            for model, value in compact.get("p_values", {}).items():
                if value is not None and model in summary["p_value_distributions"]:
                    summary["p_value_distributions"][model].append(value)
    state["rng_resume"] = {
        "mode": "per_attempt_seed",
        "base_seed": config.seed,
        "next_seed": int(
            state.get("rng_resume", {}).get(
                "next_seed", config.seed + int(state.get("attempts", 0))
            )
        ),
    }
    previous_status = state.get("status")
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
    with interprocess_lock(checkpoint, scope="research-control"):
        _atomic_json(checkpoint, state)
        round_store.update(str(state["research_id"]), _round_metadata(state))
        if resume and previous_status in {
            "paused",
            "failed",
            "interrupted",
            "pause_requested",
        }:
            pause_path.unlink(missing_ok=True)
            Path(f"{checkpoint}.stop").unlink(missing_ok=True)

    baseline_elapsed = float(state.get("total_elapsed_seconds", 0.0))
    session_budget = min(
        float(duration_seconds),
        max(0.0, float(state["target_duration_seconds"]) - baseline_elapsed),
    )
    completed_seed_set = {
        int(item["seed"])
        for item in state.get("completed_experiments", [])
        if item.get("seed") is not None
    }
    pending_seeds = []
    recovered_orphan = False
    for item in state.get("failed_runs", []):
        seed = item.get("seed")
        if seed is None or int(seed) in completed_seed_set:
            continue
        if int(seed) in pending_seeds:
            continue
        experiment_id = item.get("experiment_id")
        if experiment_id:
            try:
                existing = store.load(str(experiment_id), expand_details=False)
            except (OSError, ValueError):
                existing = {}
            if (
                existing.get("status") == "completed"
                and existing.get("research_id") == state["research_id"]
                and int(existing.get("config", {}).get("seed", -1)) == int(seed)
            ):
                state["completed_experiments"].append(
                    {
                        "attempt": existing.get("research_attempt"),
                        "seed": int(seed),
                        "experiment_id": experiment_id,
                        "research_id": state["research_id"],
                        "path": str(store.directory / f"{experiment_id}.json"),
                        "status": "completed",
                        "recovered_from_failed_save": True,
                    }
                )
                completed_seed_set.add(int(seed))
                recovered_orphan = True
                continue
        pending_seeds.append(int(seed))
    if recovered_orphan:
        state["completed_runs"] = len(state["completed_experiments"])
        for failed in state.get("failed_runs", []):
            if failed.get("seed") is not None and int(failed["seed"]) in completed_seed_set:
                failed["resolved_by_experiment_id"] = next(
                    item["experiment_id"]
                    for item in state["completed_experiments"]
                    if int(item.get("seed", -1)) == int(failed["seed"])
                )
        state["dashboard_summary"] = rebuild_dashboard_summary_from_store(
            {
                "completed_experiment_ids": [
                    item["experiment_id"] for item in state["completed_experiments"]
                ],
                "failed_runs": _unresolved_failure_count(state),
            },
            store.directory,
        )
    state["pending_retry_seeds"] = pending_seeds
    retry_queue = list(pending_seeds)
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
    while time.monotonic() - started < session_budget:
        if _pause_requested(checkpoint, str(state["research_id"])):
            state["status"] = "paused"
            state["pause_reason"] = "pause_requested_at_experiment_boundary"
            break
        if max_runs is not None and state["completed_runs"] >= max_runs:
            state["status"] = "completed"
            state["stop_reason"] = "run_count_reached"
            break
        run_index = int(state["attempts"])
        retry_seed = retry_queue.pop(0) if retry_queue else None
        run_seed = retry_seed if retry_seed is not None else int(state["rng_resume"]["next_seed"])
        run_config = replace(config, seed=run_seed)
        state["attempts"] = run_index + 1
        if retry_seed is None:
            state["rng_resume"]["next_seed"] = run_seed + 1
        state["in_progress"] = {
            "seed": run_config.seed,
            "attempt": state["attempts"],
            "research_id": state["research_id"],
            "engine": engines[run_index % len(engines)],
            "feature_profile": config.feature_profile,
            "stage": "engine_start",
            "started_at": datetime.now(UTC).isoformat(),
        }
        _atomic_json(checkpoint, state)

        def report(event: dict[str, Any]) -> None:
            if event.get("experiment_id"):
                state["in_progress"]["experiment_id"] = event["experiment_id"]
            state["in_progress"]["stage"] = str(
                event.get("stage") or event.get("run") or "experiment_running"
            )
            state["current_progress"] = {
                "engine": engines[run_index % len(engines)],
                "research_id": state["research_id"],
                "run_index": run_index + 1,
                "event": event,
                "updated_at": datetime.now(UTC).isoformat(),
            }
            state["last_updated_at"] = datetime.now(UTC).isoformat()
            elapsed_now = time.monotonic() - started
            state["elapsed_seconds"] = elapsed_now
            state["total_elapsed_seconds"] = baseline_elapsed + elapsed_now
            if _pause_requested(checkpoint, str(state["research_id"])):
                state["status"] = "pause_requested"
                state.setdefault("pause_requested_at", datetime.now(UTC).isoformat())
            _atomic_json(checkpoint, state)
            if progress:
                progress(state["current_progress"])

        try:
            engine = get_engine(engines[(run_index) % len(engines)])
            trial_pysr_config = (
                replace(
                    pysr_config,
                    seed=run_seed,
                    run_directory=(
                        pysr_config.run_directory
                        or str(
                            project_path("artifacts/pysr_runs")
                            / str(state["research_id"])
                            / (
                                f"seed-{run_seed}"
                                if pysr_config.warm_start
                                else f"seed-{run_seed}-attempt-{state['attempts']}"
                            )
                        ),
                    ),
                )
                if pysr_config is not None
                else None
            )
            result = engine.run(
                ResearchRunContext(
                    dataset,
                    run_config,
                    store,
                    report,
                    elite_seeds,
                    trial_pysr_config,
                    project_path(pattern_database),
                )
            )
            result["research_id"] = state["research_id"]
            result["research_attempt"] = state["attempts"]
            if result.get("engine") == "pysr_symbolic_regression":
                cache = EvaluationCache(pattern_database)
                schema_version = feature_schema_version(
                    str(result.get("config", {}).get("feature_profile", "legacy"))
                )
                result["symbolic_pattern_ids"] = [
                    cache.register_symbolic_candidate(
                        candidate,
                        experiment_id=str(result["experiment_id"]),
                        research_id=str(state["research_id"]),
                        dataset_sha256=str(result["dataset_sha256"]),
                        engine_version=str(result["engine_version"]),
                        feature_profile=str(
                            result.get("config", {}).get("feature_profile", "legacy")
                        ),
                        feature_schema_version=schema_version,
                    )
                    for candidate in result.get("candidates", [])
                ]
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
            candidate_revalidation = {
                "experiment_id": result["experiment_id"],
                "status": "skipped",
                "reason": None,
                "job_ids": [],
                "eligible_candidate_count": 0,
            }
            if result.get("engine") == "intermittent_gp" and dataset_path:
                try:
                    source = project_path(dataset_path)
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    if dataset_digest(load_csv(source)) != data_hash:
                        raise ValueError("자동 재검증 원본 데이터가 현재 연구 데이터와 다릅니다.")
                    queued = enqueue_experiment_high_hit_candidates(
                        result,
                        dataset_path=source,
                        candidate_replay_threshold=candidate_replay_threshold,
                        start_round=candidate_replay_start_round,
                        end_round=candidate_replay_end_round,
                    )
                    candidate_revalidation.update(
                        {
                            "status": (
                                "queued"
                                if queued["eligible_candidate_count"]
                                else "no_candidates"
                            ),
                            "eligible_candidate_count": queued["eligible_candidate_count"],
                            "registered_job_count": queued["registered_job_count"],
                            "deduplicated_job_count": queued["deduplicated_job_count"],
                            "job_ids": queued["job_ids"],
                            "threshold": candidate_replay_threshold,
                            "range": queued["range"],
                        }
                    )
                    summary = state["candidate_revalidation"]
                    summary["queued_jobs"] += queued["registered_job_count"]
                    summary["job_ids"] = sorted(set(summary["job_ids"] + queued["job_ids"]))
                except Exception as replay_error:
                    candidate_revalidation.update(
                        {"status": "registration_failed", "reason": str(replay_error)}
                    )
                    state["candidate_revalidation"].setdefault("errors", []).append(
                        {
                            "experiment_id": result["experiment_id"],
                            "error_type": type(replay_error).__name__,
                            "error": str(replay_error),
                            "recorded_at": datetime.now(UTC).isoformat(),
                        }
                    )
            elif result.get("engine") != "intermittent_gp":
                candidate_revalidation["reason"] = (
                    "현재 자동 후보 수집은 6개 집합 GP 결과에 한정됩니다."
                )
            else:
                candidate_revalidation["reason"] = "기준 데이터 파일 경로가 지정되지 않았습니다."
            result["candidate_revalidation"] = candidate_revalidation
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
                dashboard_summary["child_pattern_count"] += compact_summary["child_pattern_count"]
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
            merge_unique_hit_rounds(dashboard_summary, compact_summary.get("unique_hit_rounds", {}))
            for model, value in compact_summary.get("p_values", {}).items():
                if value is not None and model in dashboard_summary["p_value_distributions"]:
                    dashboard_summary["p_value_distributions"][model].append(value)
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
            completed_seed_set.add(run_config.seed)
            for failed in state.get("failed_runs", []):
                if failed.get("seed") is not None and int(failed["seed"]) == run_config.seed:
                    failed["resolved_by_experiment_id"] = result["experiment_id"]
            state["pending_retry_seeds"] = [
                pending for pending in state["pending_retry_seeds"] if pending != run_config.seed
            ]
            retry_queue = [pending for pending in retry_queue if pending != run_config.seed]
            state["discovered_exact_matches"].extend(exact_matches)
            dashboard_summary["failed_experiment_count"] = _unresolved_failure_count(state)
            state["last_run_summary"] = {
                "experiment_id": result["experiment_id"],
                "runtime_seconds": result["runtime_seconds"],
                "engine": result.get("engine", "intermittent_gp"),
                "exact_matches": len(exact_matches),
                "patterns_registered": len(registered),
                "one_number_test_score": result.get("one_number_model", {})
                .get("selected", {})
                .get("test_score"),
                "six_number_test_score": result.get("six_number_set_model", {})
                .get("selected", {})
                .get("test_score"),
                "engine_metrics": result.get("metrics"),
                "six_number_set_metrics": result.get("six_number_set_metrics"),
                "prize_tier_backtest": result.get("prize_tier_backtest"),
            }
            state.pop("in_progress", None)
            consecutive_failures = 0
        except Exception as error:
            consecutive_failures += 1
            in_progress = state.get("in_progress", {})
            current_progress = state.get("current_progress", {})
            progress_event = current_progress.get("event", {})
            storage_failure = isinstance(error, AtomicWriteError) or (
                isinstance(error, PermissionError) and ".tmp" in str(error)
            )
            state["failed_runs"].append(
                {
                    "seed": run_config.seed,
                    "attempt": state["attempts"],
                    "experiment_id": state.get("in_progress", {}).get("experiment_id"),
                    "research_id": state["research_id"],
                    "engine": in_progress.get(
                        "engine", engines[run_index % len(engines)]
                    ),
                    "feature_profile": in_progress.get("feature_profile", config.feature_profile),
                    "stage": in_progress.get(
                        "stage", progress_event.get("run", "engine_start")
                    ),
                    "failed_at": datetime.now(UTC).isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "failure_category": "storage" if storage_failure else "experiment",
                    "retryable": True,
                }
            )
            if run_config.seed not in state["pending_retry_seeds"]:
                state["pending_retry_seeds"].append(run_config.seed)
            state["dashboard_summary"]["failed_experiment_count"] = _unresolved_failure_count(state)
            state.pop("in_progress", None)
            if storage_failure:
                state["status"] = "failed"
                state["failure_reason"] = "experiment_storage_failure"
                break
            if consecutive_failures >= 3:
                state["status"] = "failed"
                state["failure_reason"] = "three_consecutive_experiment_failures"
                break
        state["last_updated_at"] = datetime.now(UTC).isoformat()
        _atomic_json(checkpoint, state)
        round_store.update(str(state["research_id"]), _round_metadata(state))
        if _pause_requested(checkpoint, str(state["research_id"])):
            with interprocess_lock(checkpoint, scope="research-control"):
                state["status"] = "paused"
                state["pause_reason"] = "pause_requested_at_experiment_boundary"
                state["last_updated_at"] = datetime.now(UTC).isoformat()
                state["elapsed_seconds"] = time.monotonic() - started
                state["total_elapsed_seconds"] = baseline_elapsed + state["elapsed_seconds"]
                _atomic_json(checkpoint, state)
                round_store.update(str(state["research_id"]), _round_metadata(state))
                pause_path.unlink(missing_ok=True)
                Path(f"{checkpoint}.stop").unlink(missing_ok=True)
            break
        if max_runs is not None and state["completed_runs"] >= max_runs:
            state["status"] = "completed"
            state["stop_reason"] = "run_count_reached"
            break
    else:
        state["status"] = "completed"
        state["stop_reason"] = "time_budget_reached"

    with interprocess_lock(checkpoint, scope="research-control"):
        if state["status"] != "failed" and (
            state["status"] == "pause_requested"
            or _pause_requested(checkpoint, str(state["research_id"]))
        ):
            state["status"] = "paused"
            state["pause_reason"] = "pause_requested_at_experiment_boundary"
        elif state["status"] == "running":
            state["status"] = "paused"
            state["pause_reason"] = "time_budget_reached_before_boundary_completion"
        state["last_updated_at"] = datetime.now(UTC).isoformat()
        elapsed_this_session = time.monotonic() - started
        state["elapsed_seconds"] = elapsed_this_session
        state["total_elapsed_seconds"] = baseline_elapsed + elapsed_this_session
        _atomic_json(checkpoint, state)
        round_store.update(str(state["research_id"]), _round_metadata(state))
    run_automatic_checkpoint_cleanup()
    if state.get("status") == "paused" and state.get("pause_reason") == (
        "pause_requested_at_experiment_boundary"
    ):
        pause_path.unlink(missing_ok=True)
        Path(f"{checkpoint}.stop").unlink(missing_ok=True)
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
        "pysr_config": state.get("pysr_config"),
        "completed_runs": state.get("completed_runs", 0),
        "failed_runs": _unresolved_failure_count(state),
        "target_duration_seconds": state.get("target_duration_seconds"),
        "pending_retry_seeds": state.get("pending_retry_seeds", []),
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
