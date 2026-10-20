"""Durable cache for research environment checks, keyed by all stable inputs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .atomic_io import atomic_write_bytes, interprocess_lock
from .paths import project_path

PREFLIGHT_CACHE_VERSION = 1
PREFLIGHT_LOGIC_VERSION = "research-preflight-v2"


def environment_fingerprint() -> dict[str, Any]:
    packages = {}
    for name in ("deap", "numpy", "streamlit", "pytest", "ruff"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python_version": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
        "preflight_logic_version": PREFLIGHT_LOGIC_VERSION,
    }


def preflight_cache_key(settings: dict[str, Any]) -> str:
    value = {"settings": settings, "environment": environment_fingerprint()}
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_path(path: str | Path | None = None) -> Path:
    return project_path(path or "data/preflight_cache.json")


def load_preflight_cache(
    cache_key: str, path: str | Path | None = None
) -> dict[str, Any] | None:
    cache_path = _cache_path(path)
    if not cache_path.is_file():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if payload.get("cache_version") != PREFLIGHT_CACHE_VERSION:
        return None
    entry = payload.get("entries", {}).get(cache_key)
    if not isinstance(entry, dict) or entry.get("cache_key") != cache_key:
        return None
    if entry.get("environment") != environment_fingerprint():
        return None
    report = entry.get("report")
    if not isinstance(report, dict):
        return None
    return {"report": report, "checked_at": entry.get("checked_at"), "cache_key": cache_key}


def save_preflight_cache(
    cache_key: str,
    report: dict[str, Any],
    path: str | Path | None = None,
    *,
    keep_entries: int = 32,
) -> Path:
    cache_path = _cache_path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with interprocess_lock(cache_path, scope="preflight-cache"):
        try:
            existing = json.loads(cache_path.read_text(encoding="utf-8"))
            entries = existing.get("entries", {}) if existing.get("cache_version") == 1 else {}
        except (OSError, ValueError, json.JSONDecodeError):
            entries = {}
        entries[cache_key] = {
            "cache_key": cache_key,
            "checked_at": datetime.now(UTC).isoformat(),
            "environment": environment_fingerprint(),
            "report": report,
        }
        ordered = sorted(
            entries.items(), key=lambda item: str(item[1].get("checked_at", "")), reverse=True
        )[: max(1, keep_entries)]
        payload = {"cache_version": PREFLIGHT_CACHE_VERSION, "entries": dict(ordered)}
        atomic_write_bytes(
            cache_path,
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
            + b"\n",
        )
    return cache_path
