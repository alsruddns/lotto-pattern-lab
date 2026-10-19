"""Detached sequential queue runner; it never resumes an interrupted job implicitly."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .paths import PROJECT_ROOT, project_path
from .research import load_checkpoint, request_pause
from .research_queue import ResearchQueueStore
from .research_rounds import ResearchRoundStore
from .storage_management import load_storage_settings, storage_inventory, storage_over_quota


def start_detached_scheduler(store: ResearchQueueStore, queue_id: str) -> int:
    """Start one hidden scheduler and return only after its database lease is live."""
    if store.lease_is_live(queue_id):
        return 0
    project = PROJECT_ROOT
    log = project / "logs" / "research_queue" / "scheduler.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as output:
        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
            subprocess, "CREATE_NO_WINDOW", 0
        )
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "lotto_pattern_lab.queue_scheduler",
                "--queue-id",
                queue_id,
                "--database",
                str(store.path),
            ],
            cwd=project,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            creationflags=flags,
            close_fds=True,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    for _ in range(30):
        if store.lease_is_live(queue_id):
            return int(process.pid)
        if process.poll() is not None:
            break
        time.sleep(0.1)
    raise RuntimeError(
        "Scheduler heartbeat was not observed; see logs/research_queue/scheduler.log"
    )


def _research_command(
    item: dict[str, Any], checkpoint: Path, research_id: str, *, resume: bool = False
) -> list[str]:
    config = item["config"]
    engines = config.get("engines", [item["engine"]])
    if item["engine"] == "comparison":
        engines = ["intermittent_gp", "pysr_symbolic_regression"]
    command = [
        sys.executable,
        "-m",
        "lotto_pattern_lab",
        "research",
        "--csv",
        str(project_path(config.get("csv", "data/official/draws.csv"))),
        "--duration-minutes",
        str(max(0.01, item["duration_seconds"] / 60)),
        "--checkpoint",
        str(checkpoint),
        "--research-id",
        research_id,
        "--feature-profile",
        item["feature_profile"],
        "--seed",
        str(config.get("seed", 12345)),
        "--population",
        str(config.get("population_size", 48)),
        "--generations",
        str(config.get("generations", 15)),
        "--max-depth",
        str(config.get("max_depth", 5)),
        "--workers",
        str(config.get("workers", 2)),
        "--controls",
        str(config.get("control_runs", 3)),
        "--crossover",
        str(config.get("crossover_probability", 0.65)),
        "--mutation",
        str(config.get("mutation_probability", 0.25)),
        "--train-ratio",
        str(config.get("train_ratio", 0.6)),
        "--validation-ratio",
        str(config.get("validation_ratio", 0.2)),
        "--max-lag",
        str(config.get("max_lag", 10)),
        "--min-activation",
        str(config.get("min_activation_rate", 0.1)),
        "--hidden-states",
        str(config.get("hidden_states", 3)),
        "--elite-seed-ratio",
        str(config.get("elite_seed_ratio", 0.2)),
    ]
    for engine in engines:
        command.extend(("--engine", engine))
    if config.get("elite_seeding", True):
        command.append("--elite-seeding")
    if resume:
        command.append("--resume")
    if "pysr_symbolic_regression" in engines:
        pysr = config.get("pysr", {})
        for flag, key, default in (
            ("--pysr-target", "target", "main_positions"),
            ("--pysr-timeout", "timeout_seconds", 60),
            ("--pysr-iterations", "niterations", 20),
            ("--pysr-population", "population_size", 32),
            ("--pysr-populations", "populations", 4),
            ("--pysr-maxsize", "maxsize", 24),
            ("--pysr-maxdepth", "maxdepth", 8),
            ("--pysr-workers", "workers", 1),
            ("--pysr-controls", "random_control_replicates", 3),
        ):
            command.extend((flag, str(pysr.get(key, default))))
        command.extend(("--pysr-guesses-json", json.dumps(pysr.get("guesses", []))))
        for key, flag, defaults in (
            ("binary_operators", "--pysr-binary-operator", ["+", "-", "*", "/"]),
            ("unary_operators", "--pysr-unary-operator", ["square", "abs"]),
        ):
            for value in pysr.get(key, defaults):
                command.extend((flag, value))
        if pysr.get("warm_start"):
            command.append("--pysr-warm-start")
    return command


def scheduler_step(
    store: ResearchQueueStore,
    queue_id: str,
    *,
    launch: Callable[[dict[str, Any], Path, str], int],
    inspect: Callable[[dict[str, Any]], tuple[bool, str | None]],
    workspace_root: Path = PROJECT_ROOT,
    round_store: ResearchRoundStore | None = None,
) -> str:
    """One deterministic scheduler tick, injectable for isolated, fast tests."""
    queue = store.queue(queue_id)
    current = store.current_item(queue_id)
    if current and current["status"] == "running":
        alive, worker_status = inspect(current)
        if alive:
            return "worker_running"
        if worker_status == "completed":
            store.finish_item(current["item_id"], "completed")
        elif worker_status == "paused":
            if queue["status"] == "pause_requested":
                store.finish_item(current["item_id"], "paused")
                store.set_queue_status(queue_id, "paused")
                return "queue_paused"
            store.finish_item(current["item_id"], "completed")
        else:
            error_text = worker_status or "worker stopped unexpectedly"
            store.finish_item(current["item_id"], "failed", error=error_text)
            integrity_error = worker_status == "interrupted" or any(
                token in error_text.lower()
                for token in ("storage", "disk", "checkpoint", "corrupt", "database")
            )
            if queue["error_policy"] == "stop" or integrity_error:
                store.set_queue_status(queue_id, "failed")
                return "queue_failed"
    if queue["status"] in {"pause_requested", "paused", "failed", "interrupted", "completed"}:
        if queue["status"] == "pause_requested":
            store.set_queue_status(queue_id, "paused")
        return store.queue(queue_id)["status"]
    if store.queue(queue_id)["status"] != "running":
        return "idle"
    try:
        disk = shutil.disk_usage(workspace_root)
        if disk.free < 512 * 1024**2:
            raise OSError("디스크 여유 공간이 512 MiB보다 적습니다.")
        settings = load_storage_settings(workspace_root / ".storage_settings.json")
        inventory = storage_inventory(workspace_root)
        if storage_over_quota(inventory, settings["quota_bytes"]):
            raise OSError("연구 데이터가 설정한 저장 용량 한도를 넘었습니다.")
    except (OSError, ValueError) as error:
        store.set_queue_error(queue_id, str(error))
        store.set_queue_status(queue_id, "failed")
        return "storage_blocked"
    resumable = store.current_item(queue_id)
    if resumable and resumable["status"] in {"paused", "interrupted"}:
        item = store.claim_resume(resumable["item_id"])
        checkpoint = Path(item["checkpoint_path"])
        try:
            pid = launch(item, checkpoint, item["research_id"])
        except Exception as error:
            store.finish_item(item["item_id"], "failed", error=str(error))
            store.set_queue_status(queue_id, "failed")
            return "queue_failed"
        store.set_item_research(
            item["item_id"], item["research_id"], item["research_round"], str(checkpoint), pid
        )
        return "worker_resumed"
    item = store.claim_next(queue_id)
    if item is None:
        return store.queue(queue_id)["status"]
    research_id = str(uuid.uuid4())
    checkpoint = workspace_root / "checkpoints" / "research_queue" / f"{item['item_id']}.json"
    round_store = round_store or ResearchRoundStore(
        workspace_root / "data" / "database" / "patterns.sqlite"
    )
    round_number = round_store.ensure_round(
        research_id,
        {
            "status": "queued",
            "queue_id": queue_id,
            "queue_item_id": item["item_id"],
            "config": item["config"],
        },
        workspace_root / "checkpoints",
        workspace_root / "experiments",
    )
    try:
        pid = launch(item, checkpoint, research_id)
    except Exception as error:
        store.set_item_research(item["item_id"], research_id, round_number, str(checkpoint))
        store.finish_item(item["item_id"], "failed", error=str(error))
        store.set_queue_status(queue_id, "failed")
        return "queue_failed"
    store.set_item_research(item["item_id"], research_id, round_number, str(checkpoint), pid)
    return "worker_started"


def run_forever(
    database: Path = Path(__file__).resolve().parents[2] / "data/database/research_queue.sqlite",
    queue_id: str = "",
) -> None:
    store = ResearchQueueStore(database)
    owner = str(uuid.uuid4())
    if not store.acquire_lease(owner, queue_id):
        return

    def launch(item: dict[str, Any], checkpoint: Path, research_id: str) -> int:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        log = PROJECT_ROOT / "logs" / "research_queue" / f"{item['item_id']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as output:
            flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
                subprocess, "CREATE_NO_WINDOW", 0
            )
            process = subprocess.Popen(
                _research_command(item, checkpoint, research_id, resume=bool(item.get("resume"))),
                cwd=PROJECT_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                creationflags=flags,
                close_fds=True,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
        return int(process.pid)

    def inspect(item: dict[str, Any]) -> tuple[bool, str | None]:
        checkpoint = Path(item["checkpoint_path"])
        if checkpoint.exists():
            state = load_checkpoint(checkpoint)
            if state.get("status") in {"completed", "paused", "failed"}:
                if state["status"] == "failed":
                    last_failure = (state.get("failed_runs") or [{}])[-1]
                    return False, ":".join(
                        (
                            "failed",
                            str(last_failure.get("failure_category", "unknown")),
                            str(state.get("failure_reason", last_failure.get("error", ""))),
                        )
                    )
                return False, str(state["status"])
        # A worker lock proves liveness. Do not infer death from a PID that may have been reused.
        from .atomic_io import interprocess_lock

        try:
            with interprocess_lock(checkpoint, timeout=0, scope="research-worker"):
                pass
        except TimeoutError:
            return True, None
        return False, "interrupted"

    try:
        while True:
            if not store.heartbeat(owner):
                return
            queue = store.queue(queue_id)
            current = store.current_item(queue_id)
            if queue["status"] == "pause_requested" and current and current["status"] == "running":
                request_pause(current["checkpoint_path"])
            scheduler_step(store, queue_id, launch=launch, inspect=inspect)
            if store.queue(queue_id)["status"] in {"completed", "failed", "paused", "interrupted"}:
                return
            time.sleep(2)
    finally:
        store.release_lease(owner)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-id", required=True)
    parser.add_argument("--database", type=Path)
    args = parser.parse_args()
    run_forever(
        database=args.database
        or Path(__file__).resolve().parents[2] / "data/database/research_queue.sqlite",
        queue_id=args.queue_id,
    )
