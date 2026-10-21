"""JSON-safe process metadata for Optuna optimization workers."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from .atomic_io import atomic_write_bytes


@dataclass(frozen=True)
class OptimizationProcessState:
    """Serializable launch details; the live Popen handle stays in memory only."""

    study_name: str
    pid: int
    started_at: str
    command: tuple[str, ...]

    @classmethod
    def from_process(
        cls,
        process: subprocess.Popen[bytes],
        *,
        study_name: str,
        command: list[str],
        started_at: datetime | None = None,
    ) -> OptimizationProcessState:
        pid = process.pid
        if not isinstance(pid, int) or pid <= 0:
            raise ValueError("최적화 워커의 유효한 PID를 확인할 수 없습니다.")
        timestamp = started_at or datetime.now(UTC)
        if timestamp.tzinfo is None:
            raise ValueError("시작 시각에는 시간대 정보가 필요합니다.")
        return cls(
            study_name=study_name,
            pid=pid,
            started_at=timestamp.astimezone(UTC).isoformat(),
            command=tuple(str(part) for part in command),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def write_optimization_process_state(
    process: subprocess.Popen[bytes],
    path: str | Path,
    *,
    study_name: str,
    command: list[str],
    started_at: datetime | None = None,
) -> OptimizationProcessState:
    """Persist only a JSON-safe DTO while keeping ``process`` runtime-only."""
    state = OptimizationProcessState.from_process(
        process,
        study_name=study_name,
        command=command,
        started_at=started_at,
    )
    payload = (json.dumps(state.to_dict(), ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    atomic_write_bytes(Path(path), payload)
    return state


def launch_optimization_worker(
    launcher: Callable[[list[str], Path, Path], subprocess.Popen[bytes]],
    command: list[str],
    checkpoint_path: str | Path,
    log_path: str | Path,
    process_record_path: str | Path,
    *,
    study_name: str,
) -> OptimizationProcessState:
    """Start an optimization worker and persist its JSON-safe launch record."""
    process = launcher(command, Path(checkpoint_path), Path(log_path))
    return write_optimization_process_state(
        process,
        process_record_path,
        study_name=study_name,
        command=command,
    )
