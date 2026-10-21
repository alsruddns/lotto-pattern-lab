"""Durable cache for research environment checks, keyed by all stable inputs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_bytes, interprocess_lock
from .paths import project_path

PREFLIGHT_CACHE_VERSION = 2
PREFLIGHT_LOGIC_VERSION = "research-preflight-v3"
PREFLIGHT_SUCCESS_TTL = timedelta(hours=24)
PREFLIGHT_FAILURE_TTL = timedelta(minutes=2)
_BASE_PACKAGES = ("deap", "numpy", "optuna", "streamlit")
_PYSR_PACKAGES = ("pysr", "juliacall", "julia")
_PYSR_ENVIRONMENT = (
    "JULIA_BINDIR",
    "JULIA_PROJECT",
    "JULIA_DEPOT_PATH",
    "JULIA_NUM_THREADS",
    "JULIA_CPU_THREADS",
)


def _executable_signature(path_text: str | None) -> dict[str, Any] | None:
    if not path_text:
        return None
    try:
        path = Path(path_text).resolve()
        stat = path.stat()
    except (OSError, RuntimeError):
        return {"path": path_text, "exists": False}
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def environment_fingerprint(engines: list[str] | tuple[str, ...] | None = None) -> dict[str, Any]:
    selected = set(engines or ())
    packages_to_check = _BASE_PACKAGES + (
        _PYSR_PACKAGES if "pysr_symbolic_regression" in selected else ()
    )
    packages = {}
    for name in packages_to_check:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {
        "python_version": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "python_executable": _executable_signature(sys.executable),
        "packages": packages,
        "preflight_logic_version": PREFLIGHT_LOGIC_VERSION,
    }
    if "pysr_symbolic_regression" in selected:
        julia = os.environ.get("JULIA_BINDIR")
        if julia and Path(julia).is_dir():
            julia = str(Path(julia) / ("julia.exe" if os.name == "nt" else "julia"))
        julia = julia or shutil.which("julia")
        result["julia_executable"] = _executable_signature(julia)
        result["environment"] = {
            name: os.environ.get(name) for name in _PYSR_ENVIRONMENT
        }
    return result


def preflight_cache_key(settings: dict[str, Any]) -> str:
    value = {
        "settings": settings,
        "environment": environment_fingerprint(settings.get("engines")),
    }
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_path(path: str | Path | None = None) -> Path:
    return project_path(path or "data/preflight_cache.json")


def preflight_report_is_fresh(
    report: dict[str, Any], checked_at: str | None, *, now: datetime | None = None
) -> bool:
    try:
        timestamp = datetime.fromisoformat(str(checked_at))
        if timestamp.tzinfo is None:
            return False
        age = (now or datetime.now(UTC)) - timestamp.astimezone(UTC)
    except (TypeError, ValueError):
        return False
    ttl = PREFLIGHT_SUCCESS_TTL if report.get("status") == "passed" else PREFLIGHT_FAILURE_TTL
    return timedelta(0) <= age <= ttl


def load_preflight_cache(
    cache_key: str,
    path: str | Path | None = None,
    *,
    engines: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any] | None:
    cache_path = _cache_path(path)
    if not cache_path.is_file():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("cache_version") != PREFLIGHT_CACHE_VERSION:
        return None
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        return None
    entry = entries.get(cache_key)
    if not isinstance(entry, dict) or entry.get("cache_key") != cache_key:
        return None
    recorded_engines = entry.get("engines", [])
    if not isinstance(recorded_engines, list):
        return None
    if engines is not None and recorded_engines != list(engines):
        return None
    if entry.get("environment") != environment_fingerprint(recorded_engines):
        return None
    report = entry.get("report")
    if not isinstance(report, dict):
        return None
    checked_at = entry.get("checked_at")
    if not preflight_report_is_fresh(report, checked_at):
        return None
    timestamp = datetime.fromisoformat(str(checked_at)).astimezone(UTC)
    age = datetime.now(UTC) - timestamp
    return {
        "report": report,
        "checked_at": timestamp.isoformat(),
        "cache_key": cache_key,
        "age_seconds": age.total_seconds(),
        "cache_status": "fresh",
    }


def save_preflight_cache(
    cache_key: str,
    report: dict[str, Any],
    path: str | Path | None = None,
    *,
    engines: list[str] | tuple[str, ...] | None = None,
    keep_entries: int = 32,
) -> Path:
    cache_path = _cache_path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with interprocess_lock(cache_path, scope="preflight-cache"):
        try:
            existing = json.loads(cache_path.read_text(encoding="utf-8"))
            entries = (
                existing.get("entries", {})
                if isinstance(existing, dict)
                and existing.get("cache_version") == PREFLIGHT_CACHE_VERSION
                and isinstance(existing.get("entries"), dict)
                else {}
            )
        except (OSError, ValueError, json.JSONDecodeError):
            entries = {}
        entries[cache_key] = {
            "cache_key": cache_key,
            "checked_at": datetime.now(UTC).isoformat(),
            "environment": environment_fingerprint(engines),
            "engines": list(engines or ()),
            "report": report,
        }
        valid_entries = {
            key: value for key, value in entries.items() if isinstance(value, dict)
        }
        ordered = sorted(
            valid_entries.items(),
            key=lambda item: str(item[1].get("checked_at", "")),
            reverse=True,
        )[: max(1, keep_entries)]
        payload = {"cache_version": PREFLIGHT_CACHE_VERSION, "entries": dict(ordered)}
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode(
            "utf-8"
        )
        atomic_write_bytes(cache_path, encoded + b"\n")
    return cache_path
