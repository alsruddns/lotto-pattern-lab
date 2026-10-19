"""Storage inventory, consent-based experiment compression, and safe checkpoint retention."""

import gzip
import hashlib
import json
import os
import sqlite3
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .paths import PROJECT_ROOT, project_path
from .storage import _atomic_bytes, _atomic_json, _serialize_result

MANAGED_DIRS = (
    "experiments",
    "checkpoints",
    "artifacts",
    "predictions",
    "logs",
    "data/database",
    "pysr_runs",
)
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
    file_types: dict[str, dict[str, int]] = {}
    research_usage: dict[str, dict[str, int]] = {}
    for name in MANAGED_DIRS:
        folder = base / name
        files = [path for path in folder.rglob("*") if path.is_file()] if folder.exists() else []
        folder_bytes = 0
        for path in files:
            size = path.stat().st_size
            folder_bytes += size
            extension = ".".join(path.name.split(".")[1:]) or "[no extension]"
            bucket = file_types.setdefault(extension, {"file_count": 0, "bytes": 0})
            bucket["file_count"] += 1
            bucket["bytes"] += size
            if path.suffix.lower() == ".json" and not path.name.endswith(".details.json"):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                research_id = payload.get("research_id") or payload.get("research_round_id")
                if research_id:
                    record = research_usage.setdefault(
                        str(research_id), {"file_count": 0, "bytes": 0}
                    )
                    record["file_count"] += 1
                    record["bytes"] += size
        folders.append(
            {
                "folder": name,
                "exists": folder.exists(),
                "file_count": len(files),
                "bytes": folder_bytes,
            }
        )
    return {
        "folders": folders,
        "total_bytes": sum(folder["bytes"] for folder in folders),
        "file_types": file_types,
        "research_usage": research_usage,
        "scanned_at": datetime.now(UTC).isoformat(),
    }


def storage_over_quota(inventory: dict[str, Any], quota_bytes: int) -> bool:
    return int(inventory.get("total_bytes", 0)) > quota_bytes


def create_storage_backup(
    destination_directory: str | Path = "backups", root: str | Path = "."
) -> dict[str, Any]:
    """Create an atomic, checksum-indexed backup of managed research files."""
    base = project_path(root)
    target_directory = project_path(destination_directory)
    target_directory.mkdir(parents=True, exist_ok=True)
    files = [
        path
        for name in MANAGED_DIRS
        for folder in [base / name]
        if folder.exists()
        for path in folder.rglob("*")
        if path.is_file()
        and not path.name.endswith((".tmp", ".lock", "-wal", "-shm"))
    ]
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    destination = target_directory / f"research_backup_{timestamp}.zip"
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}_", suffix=".tmp", dir=target_directory
    )
    os.close(handle)
    temporary = Path(temporary_name)
    manifest: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for source in files:
                relative = source.relative_to(base).as_posix()
                if source.suffix.lower() in {".sqlite", ".db"}:
                    snapshot_path = temporary.with_name(f"{temporary.name}_{source.name}")
                    source_connection = sqlite3.connect(source, timeout=30)
                    snapshot_connection = sqlite3.connect(snapshot_path)
                    try:
                        source_connection.backup(snapshot_connection)
                    finally:
                        snapshot_connection.close()
                        source_connection.close()
                    try:
                        content = snapshot_path.read_bytes()
                    finally:
                        snapshot_path.unlink(missing_ok=True)
                else:
                    content = source.read_bytes()
                archive.writestr(relative, content)
                manifest.append(
                    {
                        "path": relative,
                        "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                )
            archive.writestr(
                "_backup_manifest.json",
                json.dumps(
                    {"schema_version": 1, "files": manifest},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        with zipfile.ZipFile(temporary, "r") as archive:
            _verify_backup_archive(archive)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": str(destination),
        "file_count": len(manifest),
        "source_bytes": sum(item["bytes"] for item in manifest),
        "backup_bytes": destination.stat().st_size,
    }


def _verify_backup_archive(archive: zipfile.ZipFile) -> dict[str, Any]:
    try:
        manifest = json.loads(archive.read("_backup_manifest.json").decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Backup manifest is missing or invalid.") from error
    checked = 0
    for item in manifest.get("files", []):
        name = str(item.get("path", ""))
        if not name or name.startswith("/") or ".." in Path(name).parts:
            raise ValueError("Backup contains an unsafe file path.")
        content = archive.read(name)
        if len(content) != int(item["bytes"]):
            raise ValueError(f"Backup size verification failed for {name}.")
        if hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ValueError(f"Backup checksum verification failed for {name}.")
        checked += 1
    return {"valid": checked == len(manifest.get("files", [])), "verified_files": checked}


def verify_storage_backup(path: str | Path) -> dict[str, Any]:
    backup = project_path(path)
    with zipfile.ZipFile(backup, "r") as archive:
        return _verify_backup_archive(archive)


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


def pysr_artifact_compression_plan(
    directory: str | Path = "artifacts/pysr_runs",
    checkpoint_directory: str | Path = "checkpoints",
) -> dict[str, Any]:
    """Preview compressible PySR equation exports; protect model checkpoints and active runs."""
    folder = project_path(directory)
    checkpoint_root = project_path(checkpoint_directory)
    statuses: dict[str, str] = {}
    for checkpoint_path in checkpoint_root.glob("research*.json"):
        state = _read_checkpoint(checkpoint_path)
        if state and state.get("research_id"):
            statuses[str(state["research_id"])] = str(state.get("status", "unknown"))
    candidates: list[dict[str, Any]] = []
    protected: list[dict[str, Any]] = []
    if not folder.exists():
        return {"candidates": candidates, "protected": protected, "expected_bytes_freed": 0}
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(folder)
        research_id = relative.parts[0] if len(relative.parts) > 1 else None
        if path.name not in {"hall_of_fame.csv", "hall_of_fame.csv.bak"}:
            protected.append({"path": str(path), "reason": "model state or unknown PySR artifact"})
            continue
        if research_id is None or statuses.get(research_id) != "completed":
            protected.append(
                {
                    "path": str(path),
                    "reason": "research is active, recoverable, or has no completed checkpoint",
                }
            )
            continue
        destination = Path(f"{path}.gz")
        if destination.exists():
            protected.append({"path": str(path), "reason": "compressed copy already exists"})
            continue
        content = path.read_bytes()
        compressed = gzip.compress(content, compresslevel=6, mtime=0)
        candidates.append(
            {
                "path": str(path),
                "compressed_path": str(destination),
                "original_bytes": len(content),
                "compressed_bytes": len(compressed),
                "expected_bytes_freed": max(0, len(content) - len(compressed)),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return {
        "candidates": candidates,
        "protected": protected,
        "expected_bytes_freed": sum(item["expected_bytes_freed"] for item in candidates),
    }


def compress_pysr_artifacts(
    directory: str | Path = "artifacts/pysr_runs",
    checkpoint_directory: str | Path = "checkpoints",
) -> dict[str, Any]:
    """Losslessly gzip completed-run equation CSVs after rechecking run state and checksums."""
    plan = pysr_artifact_compression_plan(directory, checkpoint_directory)
    checkpoint_root = project_path(checkpoint_directory)
    outcomes: list[dict[str, Any]] = []
    for item in plan["candidates"]:
        source = project_path(item["path"])
        compressed_path = project_path(item["compressed_path"])
        research_id = source.relative_to(project_path(directory)).parts[0]
        current_status = None
        for checkpoint_path in checkpoint_root.glob("research*.json"):
            state = _read_checkpoint(checkpoint_path)
            if state and str(state.get("research_id")) == research_id:
                current_status = state.get("status")
                break
        if current_status != "completed":
            continue
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ValueError(f"PySR artifact changed after preview: {source}")
        compressed = gzip.compress(content, compresslevel=6, mtime=0)
        _atomic_bytes(compressed_path, compressed)
        decompressed = gzip.decompress(compressed_path.read_bytes())
        if hashlib.sha256(decompressed).hexdigest() != item["sha256"]:
            compressed_path.unlink(missing_ok=True)
            raise ValueError(f"PySR artifact compression verification failed: {source}")
        current_status = None
        for checkpoint_path in checkpoint_root.glob("research*.json"):
            state = _read_checkpoint(checkpoint_path)
            if state and str(state.get("research_id")) == research_id:
                current_status = state.get("status")
                break
        if current_status != "completed":
            continue
        source.unlink()
        outcomes.append(
            {
                "path": str(source),
                "compressed_path": str(compressed_path),
                "bytes_freed": max(0, item["original_bytes"] - item["compressed_bytes"]),
            }
        )
    return {
        "compressed": outcomes,
        "bytes_freed": sum(item["bytes_freed"] for item in outcomes),
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
