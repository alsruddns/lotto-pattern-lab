"""Korean labels for the persistent research startup state machine."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_PHASE_MESSAGES: dict[str, tuple[str, str]] = {
    "ready": ("info", "준비됨 · 환경 검사를 통과한 뒤 연구를 시작할 수 있습니다."),
    "environment_checking": ("info", "연구 환경을 검사하고 있습니다."),
    "environment_passed": ("success", "연구 환경 검사를 통과했습니다."),
    "environment_failed": ("error", "연구 환경 검사에 실패했습니다."),
    "background_starting": ("info", "백그라운드 연구 실행을 요청하고 있습니다."),
    "process_verifying": ("info", "연구 프로세스가 시작됐는지 확인하고 있습니다."),
    "first_checkpoint_waiting": (
        "info",
        "연구 프로세스가 실행 중입니다. 첫 체크포인트 저장을 기다리고 있습니다.",
    ),
    "running": ("success", "연구 프로세스 실행을 확인했습니다."),
    "completed": ("success", "연구 워커가 완료 상태를 기록했습니다."),
    "failed": ("error", "연구 워커가 실패 상태를 기록했습니다."),
    "paused": ("warning", "연구 워커가 일시정지 상태를 기록했습니다."),
    "interrupted": ("warning", "연구 워커가 중단 상태를 기록했습니다."),
    "start_unconfirmed": (
        "warning",
        "실행 요청은 전달됐지만 프로세스 시작은 아직 확인되지 않았습니다.",
    ),
    "start_failed": ("error", "연구 프로세스 시작에 실패했습니다."),
}


def research_startup_message(phase: Any) -> tuple[str, str]:
    """Map persisted phase values to a severity and Korean status message."""
    key = phase if isinstance(phase, str) else "ready"
    return _PHASE_MESSAGES.get(key, _PHASE_MESSAGES["ready"])


def worker_start_failure_metadata(
    worker_pid: int, exit_code: int, log_path: str | Path, *, failed_at: str | None = None
) -> dict[str, Any]:
    """Describe a child that exited before producing its research checkpoint."""
    timestamp = failed_at or datetime.now(UTC).isoformat()
    return {
        "status": "failed",
        "worker_pid": worker_pid,
        "finished_at": timestamp,
        "failure_reason": "worker_exited_before_checkpoint",
        "failed_run_summary": [
            {
                "error_type": "WorkerStartError",
                "error": (
                    f"워커가 체크포인트를 쓰기 전에 종료했습니다 (exit code {exit_code}). "
                    f"로그: {log_path}"
                ),
                "failed_at": timestamp,
            }
        ],
    }
