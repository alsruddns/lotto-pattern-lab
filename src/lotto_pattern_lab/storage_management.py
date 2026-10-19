"""Storage inventory, consent-based experiment compression, and safe checkpoint retention."""

import gzip
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .paths import PROJECT_ROOT, project_path
from .storage import _atomic_bytes, _atomic_json, _serialize_result

MANAGED_DIRS = ("experiments", "checkpoints", "artifacts", "predictions", "logs")
DEFAULT_QUOTA_BYTES = 10 * 1024**3
DEFAULT_CHECKPOINT_KEEP = 5
EXPERIMENT_COMPRESSION_THRESHOLD = 1_000_000
SETTINGS_PATH = PROJECT_ROOT / ".storage_settings.json"


def load_storage_settings(path: str | Path = SETTINGS_PATH) -> dict[str, Any]:
    settings_path = project_path(path)
    if not settings_path.exists():
        return {
            "quota_bytes": DEFAULT_QUOTA_BYTES,
            "automatic_checkpoint_cleanup": False,
            "checkpoint_keep": DEFAULT_CHECKPOINT_KEEP,
        }
    with settings_path.open(encoding="utf-8") as source:
        settings = json.load(source)
    return {
        "quota_bytes": max(1, int(settings.get("quota_bytes", DEFAULT_QUOTA_BYTES))),
        "automatic_checkpoint_cleanup": bool(settings.get("automatic_checkpoint_cleanup", False)),
        "checkpoint_keep": max(5, int(settings.get("checkpoint_keep", DEFAULT_CHECKPOINT_KEEP))),
    }


def save_storage_settings(settings: dict[str, Any], path: str | Path = SETTINGS_PATH) -> None:
    normalized = {
        "quota_bytes": max(1, int(settings.get("quota_bytes", DEFAULT_QUOTA_BYTES))),
        "automatic_checkpoint_cleanup": bool(settings.get("automatic_checkpoint_cleanup", False)),
        "checkpoint_keep": max(5, int(settings.get("checkpoint_keep", DEFAULT_CHECKPOINT_KEEP))),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(project_path(path), normalized)


def storage_inventory(root: str | Path = ".") -> dict[str, Any]:
    base = project_path(root)
    folders: list[dict[str, Any]] = []
    for name in MANAGED_DIRS:
        folder = base / name
        files = [path for path in folder.rglob("*") if path.is_file()] if folder.exists() else []
        folders.append(
            {
                "folder": name,
                "exists": folder.exists(),
                "file_count": len(files),
                "bytes": sum(path.stat().st_size for path in files),
            }
        )
    return {
        "folders": folders,
        "total_bytes": sum(folder["bytes"] for folder in folders),
        "scanned_at": datetime.now(UTC).isoformat(),
    }


def storage_over_quota(inventory: dict[str, Any], quota_bytes: int) -> bool:
    return int(inventory.get("total_bytes", 0)) > quota_bytes


def _pid_is_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def _read_checkpoint(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as source:
            checkpoint = json.load(source)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(checkpoint, dict):
        return None
    return checkpoint


def checkpoint_cleanup_plan(
    directory: str | Path = "checkpoints", *, keep: int = DEFAULT_CHECKPOINT_KEEP
) -> dict[str, Any]:
    folder = project_path(directory)
    paths = sorted(folder.rglob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    protected: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        checkpoint = _read_checkpoint(path)
        reason = None
        if index < keep:
            reason = f"최근 {keep}개 보존 정책"
        elif checkpoint is None:
            reason = "읽기 또는 JSON 검증 실패"
        elif checkpoint.get("status") not in {"completed", "failed"}:
            reason = "중단 후 재개 가능한 상태이거나 완료 여부가 불명확함"
        elif checkpoint.get("in_progress"):
            reason = "진행 중 작업 참조가 남아 있음"
        elif checkpoint.get("resumable", True):
            reason = "재개 가능한 연구 체크포인트는 보존"
        elif _pid_is_alive(checkpoint.get("worker_pid")):
            reason = "체크포인트에 기록된 작업 프로세스가 실행 중"
        elif checkpoint.get("discovered_exact_matches"):
            reason = "완전 일치 발견 이력이 있어 보존"
        elif not checkpoint.get("research_id") or not checkpoint.get("last_updated_at"):
            reason = "정상 저장 메타데이터가 부족함"
        item = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "status": checkpoint.get("status") if checkpoint else "invalid",
            "reason": reason or "검증된 종료 체크포인트이며 최근 보존 개수 초과",
        }
        (protected if reason else candidates).append(item)
    return {
        "candidates": candidates,
        "protected": protected,
        "expected_bytes_freed": sum(item["bytes"] for item in candidates),
        "keep": keep,
    }


def experiment_compression_plan(
    directory: str | Path = "experiments", *, threshold: int = EXPERIMENT_COMPRESSION_THRESHOLD
) -> dict[str, Any]:
    folder = project_path(directory)
    candidates: list[dict[str, Any]] = []
    protected: list[dict[str, Any]] = []
    for path in sorted(folder.rglob("*.json")):
        if path.name.endswith(".details.json"):
            continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            protected.append({"path": str(path), "reason": "JSON을 읽을 수 없어 보존"})
            continue
        if result.get("storage", {}).get("format") == "summary+gzip-details-v1":
            continue
        if path.stat().st_size < threshold:
            continue
        if result.get("status") not in {"completed", "failed"}:
            protected.append({"path": str(path), "reason": "실행 중이거나 미완료 상태"})
            continue
        summary, compressed = _serialize_result(result, minimum_size=0)
        if compressed is None:
            continue
        detail_path = path.with_name(f"{result.get('experiment_id', path.stem)}.details.json.gz")
        compressed_size = len(compressed)
        estimated_freed = max(0, path.stat().st_size - len(summary) - compressed_size)
        candidates.append(
            {
                "path": str(path),
                "detail_path": str(detail_path),
                "original_bytes": path.stat().st_size,
                "compressed_bytes": compressed_size,
                "summary_bytes": len(summary),
                "expected_bytes_freed": estimated_freed,
                "exact_match_count": _exact_match_count(result),
            }
        )
    return {
        "candidates": candidates,
        "protected": protected,
        "expected_bytes_freed": sum(item["expected_bytes_freed"] for item in candidates),
        "threshold_bytes": threshold,
    }


def _exact_match_count(result: dict[str, Any]) -> int:
    count = 0
    for model_key in ("one_number_model", "six_number_set_model"):
        model = result.get(model_key, {})
        for candidate in [model.get("selected", {}), *model.get("candidates", [])]:
            count += int(candidate.get("training_full_set_matches", 0) or 0)
            count += int(candidate.get("validation_full_set_matches", 0) or 0)
            count += int(candidate.get("test_full_set_matches", 0) or 0)
    return count


def compress_experiment(path: str | Path) -> dict[str, Any]:
    source_path = project_path(path)
    result = json.loads(source_path.read_text(encoding="utf-8"))
    if result.get("status") not in {"completed", "failed"}:
        raise ValueError("Only completed or failed experiments can be archived.")
    if result.get("storage", {}).get("format") == "summary+gzip-details-v1":
        raise ValueError("Experiment is already stored as summary plus compressed details.")
    summary, compressed = _serialize_result(result, minimum_size=0)
    if compressed is None:
        return {"path": str(source_path), "archived": False, "bytes_freed": 0}
    experiment_id = str(result["experiment_id"])
    detail_path = source_path.with_name(f"{experiment_id}.details.json.gz")
    if detail_path.exists():
        raise FileExistsError(f"Refusing to replace existing detail archive: {detail_path}")
    _atomic_bytes(detail_path, compressed)
    with gzip.open(detail_path, "rb") as archive:
        payload = json.loads(archive.read().decode("utf-8"))
    if payload.get("experiment_id") != experiment_id:
        detail_path.unlink(missing_ok=True)
        raise ValueError("Compressed detail archive verification failed.")
    original_bytes = source_path.stat().st_size
    _atomic_bytes(source_path, summary)
    return {
        "path": str(source_path),
        "detail_path": str(detail_path),
        "archived": True,
        "bytes_freed": max(
            0, original_bytes - source_path.stat().st_size - detail_path.stat().st_size
        ),
        "detail_sha256": hashlib.sha256(compressed).hexdigest(),
    }


def execute_checkpoint_cleanup(
    directory: str | Path = "checkpoints", *, keep: int = DEFAULT_CHECKPOINT_KEEP
) -> dict[str, Any]:
    plan = checkpoint_cleanup_plan(directory, keep=keep)
    removed: list[dict[str, Any]] = []
    for item in plan["candidates"]:
        path = Path(item["path"])
        checkpoint = _read_checkpoint(path)
        if (
            checkpoint is None
            or checkpoint.get("status") not in {"completed", "failed"}
            or checkpoint.get("in_progress")
            or checkpoint.get("resumable", True)
            or _pid_is_alive(checkpoint.get("worker_pid"))
            or checkpoint.get("discovered_exact_matches")
        ):
            continue
        byte_count = path.stat().st_size
        path.unlink()
        removed.append({"path": str(path), "bytes": byte_count})
    result = {"removed": removed, "bytes_freed": sum(item["bytes"] for item in removed)}
    audit_path = PROJECT_ROOT / "logs/storage_maintenance.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as output:
        output.write(json.dumps({"at": datetime.now(UTC).isoformat(), **result}) + "\n")
        output.flush()
        os.fsync(output.fileno())
    return result


def run_automatic_checkpoint_cleanup() -> dict[str, Any] | None:
    settings = load_storage_settings()
    if not settings["automatic_checkpoint_cleanup"]:
        return None
    return execute_checkpoint_cleanup(keep=settings["checkpoint_keep"])
