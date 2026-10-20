"""Persistent, non-blocking queue for immutable full-range formula replays."""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_bytes, interprocess_lock
from .dataset import load_csv
from .formula_replay import replay_formula
from .paths import project_path
from .pattern_library import pattern_fingerprint
from .storage import ExperimentStore

MAX_REPLAY_WORKERS = 1
_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_REPLAY_WORKERS, thread_name_prefix="formula-replay")
_FUTURES: dict[str, Future] = {}
_DISPATCH_LOCK = threading.RLock()


def _jobs_directory(directory: str | Path | None = None) -> Path:
    path = project_path(directory or "data/replay_jobs")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _job_path(job_id: str, directory: str | Path | None = None) -> Path:
    return _jobs_directory(directory) / f"{job_id}.json"


def _write_job(job: dict[str, Any], directory: str | Path | None = None) -> None:
    path = _job_path(str(job["job_id"]), directory)
    payload = json.dumps(job, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    atomic_write_bytes(path, payload)


def list_replay_jobs(directory: str | Path | None = None) -> list[dict[str, Any]]:
    jobs = []
    for path in _jobs_directory(directory).glob("*.json"):
        try:
            jobs.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, json.JSONDecodeError):
            jobs.append(
                {
                    "job_id": path.stem,
                    "status": "corrupt",
                    "error": "작업 상태 JSON을 읽지 못했습니다.",
                }
            )
    return sorted(jobs, key=lambda item: str(item.get("created_at", "")), reverse=True)


def _dispatch(directory: str | Path | None = None) -> None:
    jobs_root = _jobs_directory(directory)
    with _DISPATCH_LOCK:
        for job_id, future in tuple(_FUTURES.items()):
            if future.done():
                _FUTURES.pop(job_id, None)
        capacity = MAX_REPLAY_WORKERS - sum(not future.done() for future in _FUTURES.values())
        if capacity <= 0:
            return
        running_ids = set(_FUTURES)
        queued = [
            job
            for job in list_replay_jobs(jobs_root)
            if job.get("status") == "queued" and job.get("job_id") not in running_ids
        ]
        for job in queued[:capacity]:
            job_id = str(job["job_id"])
            future = _EXECUTOR.submit(_execute_job, job_id, jobs_root)
            _FUTURES[job_id] = future
            future.add_done_callback(lambda _future, root=jobs_root: _dispatch(root))


def enqueue_formula_replay(
    *,
    dataset_path: str | Path,
    dataset_sha256: str,
    experiment: dict[str, Any],
    model_key: str,
    candidate: dict[str, Any],
    formula_id: str,
    start_round: int,
    end_round: int,
    discovery_event: dict[str, Any] | None = None,
    directory: str | Path | None = None,
    dispatch: bool = True,
) -> dict[str, Any]:
    """Persist one queue job; structurally identical jobs share one execution."""
    if model_key != "six_number_set_model":
        raise ValueError("전체 회차 재검증 대기열은 6개 집합 모델만 지원합니다.")
    if start_round < 1 or end_round < start_round:
        raise ValueError("재검증 회차 범위가 올바르지 않습니다.")
    source_path = project_path(dataset_path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != dataset_sha256:
        raise ValueError("작업 등록 전에 기준 데이터 파일이 변경되었습니다.")
    profile = (experiment.get("feature_schema") or {}).get(
            "feature_profile", experiment.get("config", {}).get("feature_profile", "legacy")
        )
    formula_key = hashlib.sha256(
        pattern_fingerprint(
            str(candidate.get("activation_condition", "")),
            tuple(map(str, candidate.get("formulas", ()))),
        ).encode("utf-8")
    ).hexdigest()
    key_payload = {
        "formula_content_sha256": formula_key,
        "profile": profile,
        "dataset_sha256": dataset_sha256,
        "start_round": start_round,
        "end_round": end_round,
    }
    job_key = hashlib.sha256(
        json.dumps(key_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()
    jobs_root = _jobs_directory(directory)
    with interprocess_lock(jobs_root / "queue", scope="formula-replay-queue"):
        for existing in list_replay_jobs(jobs_root):
            if existing.get("job_key") == job_key and existing.get("status") in {
                "queued",
                "running",
                "completed",
                "interrupted",
            }:
                existing["duplicate_request_count"] = int(
                    existing.get("duplicate_request_count", 0)
                ) + 1
                if discovery_event is not None:
                    discoveries = existing.setdefault("discovery_events", [])
                    discovery_key = str(discovery_event.get("discovery_key", ""))
                    if discovery_key and all(
                        item.get("discovery_key") != discovery_key for item in discoveries
                    ):
                        discoveries.append(discovery_event)
                        existing["discovery_count"] = len(discoveries)
                        existing["repeated_hit_events"] = sum(
                            int(item.get("hit_event_count", 0) or 0) for item in discoveries
                        )
                        existing["original_hit_rounds"] = sorted(
                            {
                                int(round_number)
                                for item in discoveries
                                for round_number in item.get("hit_rounds", [])
                            }
                        )
                        existing["origin_experiment_ids"] = sorted(
                            {
                                str(item.get("experiment_id"))
                                for item in discoveries
                                if item.get("experiment_id")
                            }
                        )
                _write_job(existing, jobs_root)
                return {**existing, "deduplicated": True}
        model = experiment.get(model_key) or {}
        experiment_snapshot = {
            key: experiment.get(key)
            for key in (
                "experiment_id",
                "research_id",
                "experiment_type",
                "status",
                "config",
                "feature_schema",
            )
        }
        experiment_snapshot[model_key] = {
            key: model.get(key, [])
            for key in ("train_rounds", "validation_rounds", "test_rounds")
        }
        now = datetime.now(UTC).isoformat()
        job = {
            "job_id": str(uuid.uuid4()),
            "job_key": job_key,
            "created_at": now,
            "updated_at": now,
            "status": "queued",
            "dataset_path": str(source_path),
            "dataset_sha256": dataset_sha256,
            "experiment": experiment_snapshot,
            "candidate": candidate,
            "model_key": model_key,
            "formula_id": formula_id,
            "start_round": start_round,
            "end_round": end_round,
            "processed_rounds": 0,
            "total_rounds": end_round - start_round + 1,
            "elapsed_seconds": 0.0,
            "estimated_remaining_seconds": None,
            "duplicate_request_count": 0,
            "discovery_events": [discovery_event] if discovery_event else [],
            "discovery_count": 1 if discovery_event else 0,
            "repeated_hit_events": (
                int(discovery_event.get("hit_event_count", 0) or 0)
                if discovery_event
                else 0
            ),
            "original_hit_rounds": (
                sorted(set(map(int, discovery_event.get("hit_rounds", []))))
                if discovery_event
                else []
            ),
            "origin_experiment_ids": (
                [str(discovery_event["experiment_id"])]
                if discovery_event and discovery_event.get("experiment_id")
                else []
            ),
            "result_path": None,
            "error": None,
        }
        _write_job(job, jobs_root)
    if dispatch:
        _dispatch(jobs_root)
    return {**job, "deduplicated": False}


def enqueue_experiment_high_hit_candidates(
    experiment: dict[str, Any],
    *,
    dataset_path: str | Path,
    candidate_replay_threshold: int = 5,
    start_round: int = 753,
    end_round: int = 1246,
    directory: str | Path | None = None,
) -> dict[str, Any]:
    """Register saved six-number candidates as background replays after each experiment."""
    if candidate_replay_threshold not in {3, 4, 5, 6}:
        raise ValueError("후보 등록 기준은 3, 4, 5, 6개 중 하나여야 합니다.")
    if start_round < 1 or end_round < start_round:
        raise ValueError("자동 재검증 회차 범위가 올바르지 않습니다.")
    model = experiment.get("six_number_set_model") or {}
    profile = str(
        (experiment.get("feature_schema") or {}).get(
            "feature_profile", experiment.get("config", {}).get("feature_profile", "legacy")
        )
    )
    experiment_snapshot = {
        key: experiment.get(key)
        for key in (
            "experiment_id",
            "research_id",
            "experiment_type",
            "status",
            "config",
            "feature_schema",
        )
    }
    experiment_snapshot["six_number_set_model"] = {
        key: model.get(key, []) for key in ("train_rounds", "validation_rounds", "test_rounds")
    }
    eligible: dict[str, tuple[dict[str, Any], list[dict[str, Any]], int]] = {}
    for rank, candidate in enumerate(model.get("candidates", []), start=1):
        hit_events = [
            event
            for event in candidate.get("activation_events", [])
            if event.get("active")
            and int(event.get("hit_count", 0) or 0) >= candidate_replay_threshold
        ]
        split_summary = candidate.get(f"at_least_{candidate_replay_threshold}_hits_by_split") or {}
        qualifies_by_summary = any(int(value or 0) > 0 for value in split_summary.values())
        if not hit_events and not qualifies_by_summary:
            continue
        normalized = pattern_fingerprint(
            str(candidate.get("activation_condition", "")),
            tuple(map(str, candidate.get("formulas", ()))),
        )
        candidate_key = hashlib.sha256(f"{profile}:{normalized}".encode()).hexdigest()
        current = eligible.get(candidate_key)
        if current is None:
            eligible[candidate_key] = (candidate, hit_events, rank)
        else:
            current[1].extend(hit_events)
            eligible[candidate_key] = (current[0], current[1], min(current[2], rank))

    data_file = project_path(dataset_path)
    data_digest = hashlib.sha256(data_file.read_bytes()).hexdigest()
    jobs = []
    for candidate_key, (candidate, hit_events, rank) in eligible.items():
        experiment_id = str(experiment.get("experiment_id", "unknown"))
        research_id = experiment.get("research_id")
        formula_id = hashlib.sha256(
            json.dumps(
                {
                    "gate": candidate.get("activation_condition"),
                    "formulas": candidate.get("formulas"),
                    "profile": profile,
                    "candidate_key": candidate_key,
                },
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:16]
        hit_rounds = sorted(
            {
                int(event["round_number"])
                for event in hit_events
                if event.get("round_number") is not None
            }
        )
        discovery_key = hashlib.sha256(
            f"{experiment_id}:{candidate_key}:{candidate_replay_threshold}".encode()
        ).hexdigest()
        discovery = {
            "discovery_key": discovery_key,
            "discovered_at": datetime.now(UTC).isoformat(),
            "experiment_id": experiment_id,
            "research_id": research_id,
            "candidate_rank": rank,
            "threshold": candidate_replay_threshold,
            "hit_event_count": len(hit_events),
            "hit_rounds": hit_rounds,
            "hit_rounds_complete": bool(hit_events),
        }
        jobs.append(
            enqueue_formula_replay(
                dataset_path=data_file,
                dataset_sha256=data_digest,
                experiment=experiment_snapshot,
                model_key="six_number_set_model",
                candidate=candidate,
                formula_id=formula_id,
                start_round=start_round,
                end_round=end_round,
                discovery_event=discovery,
                directory=directory,
            )
        )
    return {
        "threshold": candidate_replay_threshold,
        "eligible_candidate_count": len(eligible),
        "registered_job_count": sum(not item.get("deduplicated") for item in jobs),
        "deduplicated_job_count": sum(bool(item.get("deduplicated")) for item in jobs),
        "discovery_count": len(jobs),
        "job_ids": [item["job_id"] for item in jobs],
        "dataset_sha256": data_digest,
        "range": {"start": start_round, "end": end_round},
    }


def discover_high_hit_candidates(
    experiments_directory: str | Path = "experiments", *, minimum_hits: int = 5
) -> list[dict[str, Any]]:
    """Collect 6-number GP candidates and report repeat events separately from unique rounds."""
    if minimum_hits not in {3, 4, 5, 6}:
        raise ValueError("후보 기준은 3개 이상, 4개 이상, 5개 이상, 6개 일치 중 하나여야 합니다.")
    root = project_path(experiments_directory)
    store = ExperimentStore(root)
    summaries: list[tuple[Path, int]] = []
    summary_key = f"at_least_{minimum_hits}_hits_by_split"
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            candidates = (payload.get("six_number_set_model") or {}).get("candidates", [])
            if any(
                sum(int(value or 0) for value in (candidate.get(summary_key) or {}).values())
                for candidate in candidates
            ):
                summaries.append((path, len(candidates)))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    collected: dict[str, dict[str, Any]] = {}
    for path, candidate_count in summaries:
        try:
            experiment = store.load(path.stem)
        except (OSError, ValueError, KeyError):
            continue
        model = experiment.get("six_number_set_model") or {}
        candidates = model.get("candidates") or []
        for rank, candidate in enumerate(candidates[:candidate_count], start=1):
            qualifying = [
                event
                for event in candidate.get("activation_events", [])
                if int(event.get("hit_count", 0) or 0) >= minimum_hits
            ]
            if not qualifying:
                continue
            profile = str(
                (experiment.get("feature_schema") or {}).get(
                    "feature_profile",
                    experiment.get("config", {}).get("feature_profile", "legacy"),
                )
            )
            normalized = pattern_fingerprint(
                str(candidate.get("activation_condition", "")),
                tuple(map(str, candidate.get("formulas", ()))),
            )
            dedupe_key = hashlib.sha256(f"{profile}:{normalized}".encode()).hexdigest()
            matching_rounds = {
                int(event["round_number"])
                for event in qualifying
                if event.get("round_number") is not None
            }
            entry = collected.get(dedupe_key)
            if entry is None:
                experiment_snapshot = {
                    key: experiment.get(key)
                    for key in (
                        "experiment_id",
                        "research_id",
                        "experiment_type",
                        "status",
                        "config",
                        "feature_schema",
                    )
                }
                experiment_snapshot["six_number_set_model"] = {
                    key: model.get(key, [])
                    for key in ("train_rounds", "validation_rounds", "test_rounds")
                }
                entry = {
                    "candidate_key": dedupe_key,
                    "profile": profile,
                    "formula_id": hashlib.sha256(
                        json.dumps(
                            {
                                "gate": candidate.get("activation_condition"),
                                "formulas": candidate.get("formulas"),
                                "profile": profile,
                                "experiment_id": experiment["experiment_id"],
                            },
                            sort_keys=True,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()[:16],
                    "candidate": candidate,
                    "experiment": experiment_snapshot,
                    "origin_experiment_ids": [],
                    "original_hit_rounds": [],
                    "repeated_hit_events": 0,
                    "candidate_rank": rank,
                }
                collected[dedupe_key] = entry
            entry["origin_experiment_ids"].append(str(experiment.get("experiment_id")))
            entry["repeated_hit_events"] += len(qualifying)
            entry["original_hit_rounds"] = sorted(
                set(entry["original_hit_rounds"]) | matching_rounds
            )
    return sorted(
        collected.values(),
        key=lambda item: (
            -len(item["original_hit_rounds"]),
            -item["repeated_hit_events"],
            item["candidate_key"],
        ),
    )


def _execute_job(job_id: str, directory: Path) -> None:
    path = _job_path(job_id, directory)
    with interprocess_lock(path, scope="formula-replay-job"):
        job = json.loads(path.read_text(encoding="utf-8"))
        if job.get("status") != "queued":
            return
        job.update(status="running", started_at=datetime.now(UTC).isoformat(), error=None)
        _write_job(job, directory)
    try:
        source_path = project_path(job["dataset_path"])
        data_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if data_hash != job["dataset_sha256"]:
            raise ValueError("작업 대기 중 기준 데이터가 바뀌어 재검증을 중단했습니다.")
        dataset = load_csv(source_path)

        def update_progress(progress: dict[str, Any]) -> None:
            with interprocess_lock(path, scope="formula-replay-job"):
                latest = json.loads(path.read_text(encoding="utf-8"))
                latest.update(progress)
                latest["updated_at"] = datetime.now(UTC).isoformat()
                _write_job(latest, directory)

        result = replay_formula(
            dataset=dataset,
            experiment=job["experiment"],
            model_key=job["model_key"],
            candidate=job["candidate"],
            formula_id=job["formula_id"],
            start_round=int(job["start_round"]),
            end_round=int(job["end_round"]),
            progress_callback=update_progress,
        )
        with interprocess_lock(path, scope="formula-replay-job"):
            latest = json.loads(path.read_text(encoding="utf-8"))
            latest.update(
                status="completed",
                result_path=result["result_path"],
                run_id=result["run_id"],
                completed_at=datetime.now(UTC).isoformat(),
                elapsed_seconds=result["runtime_seconds"],
                result_summary={
                    key: result[key]
                    for key in (
                        "target_round_count",
                        "evaluable_round_count",
                        "active_prediction_count",
                        "input_unavailable_count",
                        "execution_error_count",
                        "at_least_3",
                        "at_least_4",
                        "at_least_5",
                        "at_least_6",
                        "mean_hits_when_active",
                    )
                },
                updated_at=datetime.now(UTC).isoformat(),
            )
            _write_job(latest, directory)
    except Exception as error:
        with interprocess_lock(path, scope="formula-replay-job"):
            latest = json.loads(path.read_text(encoding="utf-8"))
            latest.update(
                status="failed",
                error=f"{type(error).__name__}: {error}",
                failed_at=datetime.now(UTC).isoformat(),
                updated_at=datetime.now(UTC).isoformat(),
            )
            _write_job(latest, directory)


def resume_replay_job(job_id: str, directory: str | Path | None = None) -> dict[str, Any]:
    path = _job_path(job_id, directory)
    with interprocess_lock(path, scope="formula-replay-job"):
        job = json.loads(path.read_text(encoding="utf-8"))
        if job.get("status") not in {"interrupted", "failed"}:
            raise ValueError("중단되었거나 실패한 재검증 작업만 다시 시작할 수 있습니다.")
        job.update(
            status="queued",
            processed_rounds=0,
            error=None,
            result_path=None,
            updated_at=datetime.now(UTC).isoformat(),
        )
        _write_job(job, directory)
    _dispatch(directory)
    return job


def recover_replay_jobs(directory: str | Path | None = None) -> dict[str, int]:
    """Preserve stale running records as interrupted and dispatch persisted queue items."""
    interrupted = 0
    queued = 0
    jobs_root = _jobs_directory(directory)
    for job in list_replay_jobs(jobs_root):
        if job.get("status") == "running" and job.get("job_id") not in _FUTURES:
            job.update(
                status="interrupted",
                error="프로그램이 재시작되어 진행 중이던 작업을 중단 상태로 보존했습니다.",
                updated_at=datetime.now(UTC).isoformat(),
            )
            _write_job(job, jobs_root)
            interrupted += 1
        elif job.get("status") == "queued":
            queued += 1
    _dispatch(jobs_root)
    return {"interrupted": interrupted, "queued": queued}
