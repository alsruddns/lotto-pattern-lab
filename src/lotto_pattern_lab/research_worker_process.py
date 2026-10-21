"""Shared detached worker launch helpers used by research and optimization UI flows."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from .atomic_io import LockTimeoutError, interprocess_lock
from .paths import PROJECT_ROOT


def worker_active(checkpoint_path: Path) -> bool:
    """Check whether a worker owns the checkpoint lock without terminating it."""
    try:
        with interprocess_lock(checkpoint_path, timeout=0, scope="research-worker"):
            return False
    except LockTimeoutError:
        return True


def launch_worker(
    command: list[str], checkpoint_path: Path, log_path: Path
) -> subprocess.Popen[bytes]:
    """Start a detached, logged worker while guarding duplicate launches."""
    checkpoint_path = checkpoint_path.resolve()
    launch_path = Path(f"{checkpoint_path}.launch")
    with interprocess_lock(checkpoint_path, timeout=0, scope="research-launch"):
        if worker_active(checkpoint_path):
            raise RuntimeError("이 연구의 워커가 이미 실행 중입니다.")
        if launch_path.exists():
            try:
                request = json.loads(launch_path.read_text(encoding="utf-8"))
                created = datetime.fromisoformat(request["created_at"])
                stale = (datetime.now(UTC) - created).total_seconds() > 120
            except (OSError, ValueError, KeyError):
                stale = True
            if not stale:
                raise RuntimeError("이 연구의 워커 시작 요청이 이미 처리 중입니다.")
            launch_path.unlink(missing_ok=True)
        launch_path.write_text(
            json.dumps({"created_at": datetime.now(UTC).isoformat()}), encoding="utf-8"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a", encoding="utf-8")
        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
            subprocess, "CREATE_NO_WINDOW", 0
        )
        try:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=flags,
                close_fds=True,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
        except Exception:
            launch_path.unlink(missing_ok=True)
            raise
        finally:
            log_file.close()
        return process
