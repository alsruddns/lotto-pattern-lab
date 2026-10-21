"""Persistent research controls with separate new-run and resume workflows."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from typing import Any

import streamlit as st

from lotto_pattern_lab.atomic_io import LockTimeoutError, interprocess_lock
from lotto_pattern_lab.cumulative_research import (
    CUMULATIVE_MODE,
    INDEPENDENT_MODE,
    recommend_optuna_settings,
)
from lotto_pattern_lab.data_audit import dataset_hashes
from lotto_pattern_lab.dataset import dataset_integrity_report, load_csv
from lotto_pattern_lab.feature_schema import feature_schema_version
from lotto_pattern_lab.intermittent import IntermittentConfig
from lotto_pattern_lab.paths import PROJECT_ROOT, locate_checkpoint, project_path
from lotto_pattern_lab.pattern_library import PatternLibrary
from lotto_pattern_lab.preflight import run_research_preflight
from lotto_pattern_lab.preflight_cache import (
    load_preflight_cache,
    preflight_cache_key,
    save_preflight_cache,
)
from lotto_pattern_lab.presentation import format_count, format_duration
from lotto_pattern_lab.pysr_backend import PySRConfig, optional_backend_status
from lotto_pattern_lab.queue_scheduler import start_detached_scheduler
from lotto_pattern_lab.research import (
    _round_metadata,
    dataset_digest,
    load_checkpoint,
    pause_request_path,
    request_pause,
    research_worker_active,
)
from lotto_pattern_lab.research_duration import (
    duration_seconds_from_minutes,
    recommended_preset_minutes,
    validate_duration_minutes,
)
from lotto_pattern_lab.research_optimization import list_optimization_studies
from lotto_pattern_lab.research_optimization_apply import (
    OPTUNA_TO_RESEARCH_KEYS,
    changed_optimization_settings,
    optimization_origin_record,
)
from lotto_pattern_lab.research_queue import QUEUE_DB, ResearchQueueStore
from lotto_pattern_lab.research_rounds import ResearchRoundStore
from lotto_pattern_lab.research_startup import (
    research_startup_message,
    worker_start_failure_metadata,
)
from lotto_pattern_lab.research_time import (
    checkpoint_status_label,
    format_hhmmss,
    research_time_snapshot,
)
from lotto_pattern_lab.sqlite_support import is_database_locked, log_database_failure

GP_PRESETS = {
    30: {"population": 32, "generations": 10, "max_depth": 5, "controls": 3},
    60: {"population": 48, "generations": 15, "max_depth": 5, "controls": 5},
    120: {"population": 64, "generations": 20, "max_depth": 6, "controls": 5},
    180: {"population": 80, "generations": 25, "max_depth": 6, "controls": 5},
    360: {"population": 96, "generations": 30, "max_depth": 6, "controls": 10},
    480: {"population": 112, "generations": 35, "max_depth": 7, "controls": 10},
}
ENGINE_OPTIONS = ("GP 간헐 패턴 탐색", "PySR 기호 회귀", "GP와 PySR 비교")


def _apply_research_preset() -> None:
    duration_mode = st.session_state.get("new_duration_mode", "recommended")
    duration_value = (
        st.session_state.get(
            "new_custom_duration_minutes",
            st.session_state.get("new_custom_duration_minutes_value", 5),
        )
        if duration_mode == "custom"
        else st.session_state.get("new_duration_minutes", 180)
    )
    minutes = int(duration_value)
    preset = GP_PRESETS[recommended_preset_minutes(minutes, GP_PRESETS)]
    values = {
        "new_population_size": preset["population"],
        "new_generations": preset["generations"],
        "new_crossover_probability": 0.65,
        "new_mutation_probability": 0.25,
        "new_max_depth": preset["max_depth"],
        "new_workers": 4,
        "new_train_ratio": 0.60,
        "new_validation_ratio": 0.20,
        "new_max_lag": 10,
        "new_control_runs": preset["controls"],
        "new_min_activation_rate": 0.10,
        "new_hidden_states": 3,
        "new_seed": 12345,
        "new_elite_seeding_enabled": True,
        "new_elite_seed_ratio": 0.20,
    }
    st.session_state.update(values)
    profile = st.session_state.get("new_profile", "legacy")
    pysr_defaults = PySRConfig(feature_profile=profile)
    is_comparison = st.session_state.get("new_research_engine") == ENGINE_OPTIONS[2]
    share = 0.5 if is_comparison else 0.8
    st.session_state["new_pysr_timeout"] = min(3600.0, max(60.0, minutes * 60 * share))
    st.session_state["new_pysr_iterations"] = min(200, max(20, minutes // 2))
    st.session_state["new_pysr_population"] = pysr_defaults.population_size
    st.session_state["new_pysr_populations"] = pysr_defaults.populations
    st.session_state["new_pysr_maxsize"] = pysr_defaults.maxsize
    st.session_state["new_pysr_maxdepth"] = pysr_defaults.maxdepth
    st.session_state["new_pysr_workers"] = pysr_defaults.workers
    st.session_state["new_pysr_controls"] = pysr_defaults.random_control_replicates
    st.session_state["new_pysr_target"] = "main_positions"
    st.session_state["new_pysr_binary_operators"] = ["+", "-", "*", "/"]
    st.session_state["new_pysr_unary_operators"] = ["square", "abs"]
    st.session_state["new_pysr_train_ratio"] = 0.60
    st.session_state["new_pysr_validation_ratio"] = 0.20
    st.session_state["new_pysr_warm_start"] = False


def _remember_custom_duration() -> None:
    value = st.session_state.get("new_custom_duration_minutes")
    if value is not None:
        try:
            st.session_state["new_custom_duration_minutes_value"] = (
                validate_duration_minutes(value)
            )
        except ValueError:
            return


PAUSE_MESSAGE = (
    "일시정지 요청 완료. 현재 진행 중인 실험의 계산과 저장이 끝난 후 자동으로 일시정지됩니다."
)


def _preflight_fingerprint(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _worker_active(checkpoint_path: Path) -> bool:
    try:
        with interprocess_lock(checkpoint_path, timeout=0, scope="research-worker"):
            return False
    except LockTimeoutError:
        return True


@st.cache_data(ttl=60, show_spinner=False)
def _cached_elite_preview(
    database_path: str,
    experiment_directory: str,
    database_signature: tuple[int, int] | None,
    feature_profile: str,
    max_lag: int,
    max_candidates: int,
    max_data_round: int | None,
    require_temporal_provenance: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    del database_signature
    return PatternLibrary(database_path).select_elite_candidates(
        feature_profile=feature_profile,
        max_lag=max_lag,
        max_candidates=max_candidates,
        max_data_round=max_data_round,
        require_temporal_provenance=require_temporal_provenance,
        experiment_directory=experiment_directory,
    )


def _validate_live_research_start(
    checkpoint_directory: Path, csv_path: Path | None, dataset: Any
) -> None:
    """Recheck dynamic start conditions even when a disk preflight cache is reused."""
    if research_worker_active():
        raise RuntimeError("다른 연구 또는 최적화 작업이 CPU 실행 슬롯을 사용 중입니다.")
    if csv_path is None or not csv_path.is_file() or dataset is None:
        raise ValueError("선택한 당첨번호 데이터 파일을 찾을 수 없습니다.")
    refreshed = load_csv(csv_path)
    if dataset_digest(refreshed) != dataset_digest(dataset):
        raise ValueError("검사 이후 당첨번호 데이터가 변경되었습니다. 재검사가 필요합니다.")
    free_bytes = shutil.disk_usage(PROJECT_ROOT).free
    if free_bytes < 100 * 1024 * 1024:
        raise OSError(f"저장 공간이 부족합니다. 사용 가능 공간: {free_bytes / 1024**2:.1f} MiB")
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".research-write-check-", dir=checkpoint_directory, delete=True
        ):
            pass
    except OSError as error:
        raise OSError(f"체크포인트 저장 경로에 쓸 수 없습니다: {error}") from error
    for checkpoint in checkpoint_directory.glob("research_*.json"):
        try:
            state = json.loads(checkpoint.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if state.get("status") in {"running", "pause_requested"} and _worker_active(checkpoint):
            raise RuntimeError(f"다른 연구가 이미 실행 중입니다: {state.get('research_id')}")


def _dataset_for_checkpoint(state: dict[str, Any]) -> tuple[Path | None, str | None]:
    candidates = []
    if state.get("dataset_path"):
        recorded_dataset = project_path(state["dataset_path"])
        if recorded_dataset.is_relative_to(PROJECT_ROOT):
            candidates.append(recorded_dataset)
    candidates.extend(
        [
            project_path("data/official/draws.csv"),
            project_path("data/raw/lotto_draws_753_1246_with_bonus.csv"),
            project_path("data/raw/lotto_draws_753_1246.csv"),
        ]
    )
    seen: set[Path] = set()
    expected = state.get("dataset_sha256")
    for candidate in candidates:
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        try:
            dataset = load_csv(candidate)
            if dataset_digest(dataset) != expected:
                continue
            profile = state.get("config", {}).get("feature_profile", "legacy")
            integrity = dataset_integrity_report(dataset)
            if profile == "calendar_round" and not integrity["all_draws_dated"]:
                return (
                    candidate,
                    "저장된 calendar_round 프로파일에 필요한 추첨일이 데이터에 없습니다.",
                )
            if profile == "bonus_lag_v1" and not integrity["all_draws_have_bonus"]:
                return candidate, "저장된 bonus_lag_v1 프로파일에 필요한 보너스 번호가 없습니다."
            if not integrity["rounds_contiguous"]:
                return candidate, "저장된 연구 데이터의 회차가 연속적이지 않습니다."
            recorded_schema = (state.get("elite_seeding") or {}).get("feature_schema_version")
            if recorded_schema and recorded_schema != feature_schema_version(profile):
                return candidate, (
                    "저장된 특징 스키마 버전과 현재 구현이 다릅니다: "
                    f"{recorded_schema} != {feature_schema_version(profile)}"
                )
            return candidate, None
        except (OSError, ValueError):
            continue
    return None, f"체크포인트 데이터 해시 {expected or '미기록'}와 일치하는 CSV를 찾지 못했습니다."


def _resume_records(rounds: ResearchRoundStore) -> list[dict[str, Any]]:
    allowed = {"paused", "failed", "interrupted", "running", "pause_requested"}
    by_id = {item["research_id"]: item for item in rounds.list()}
    for checkpoint_path in project_path("checkpoints").glob("research*.json"):
        try:
            state = load_checkpoint(checkpoint_path)
        except (OSError, ValueError):
            continue
        research_id = state.get("research_id")
        if research_id and research_id not in by_id:
            by_id[research_id] = {
                **state,
                "research_id": research_id,
                "checkpoint_path": str(checkpoint_path.resolve()),
            }
    records = []
    for record in by_id.values():
        if record.get("status") not in allowed:
            continue
        raw_path = record.get("checkpoint_path")
        if not raw_path:
            continue
        path, alternatives = locate_checkpoint(raw_path)
        if not path.is_file() and alternatives:
            path = alternatives[0]
        if not path.is_relative_to(PROJECT_ROOT):
            continue
        if not path.is_file():
            records.append({**record, "_path": path, "_error": f"체크포인트 없음: {path}"})
            continue
        try:
            state = load_checkpoint(path)
        except (OSError, ValueError) as error:
            records.append({**record, "_path": path, "_error": f"체크포인트 읽기 실패: {error}"})
            continue
        if pause_request_path(path).is_file() and state.get("status") == "running":
            state["status"] = "pause_requested"
        if state.get("research_id") != record.get("research_id"):
            records.append(
                {**record, **state, "_path": path, "_error": "연구 ID가 레지스트리와 다릅니다."}
            )
            continue
        recorded_path = state.get("checkpoint_path")
        if recorded_path and project_path(recorded_path) != path.resolve():
            records.append(
                {
                    **record,
                    **state,
                    "_path": path,
                    "_error": "체크포인트 경로와 내부 경로가 다릅니다.",
                }
            )
            continue
        data_path, data_error = _dataset_for_checkpoint(state)
        records.append(
            {
                **record,
                **state,
                "_path": path,
                "_data_path": data_path,
                "_worker_active": _worker_active(path),
                "_error": data_error,
            }
        )
    return sorted(records, key=lambda item: int(item.get("research_round", 0)), reverse=True)


def _launch_worker(
    command: list[str], checkpoint_path: Path, log_path: Path
) -> subprocess.Popen[bytes]:
    checkpoint_path = checkpoint_path.resolve()
    launch_path = Path(f"{checkpoint_path}.launch")
    with interprocess_lock(checkpoint_path, timeout=0, scope="research-launch"):
        if _worker_active(checkpoint_path):
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


def _launch_checkpoint_state(checkpoint_path: Path, research_id: str) -> dict[str, Any] | None:
    """Return a checkpoint only when it is valid and belongs to this launch."""
    if not checkpoint_path.is_file():
        return None
    try:
        state = load_checkpoint(checkpoint_path)
    except (OSError, ValueError):
        return None
    return state if state.get("research_id") == research_id else None


def _mark_worker_start_failed(
    rounds: ResearchRoundStore,
    research_id: str,
    worker_pid: int,
    exit_code: int,
    log_path: Path,
) -> None:
    """Record a launch failure only after the child exited before writing its checkpoint."""
    rounds.update(
        research_id,
        worker_start_failure_metadata(worker_pid, exit_code, log_path),
    )


def _research_command(
    *,
    csv_path: Path,
    checkpoint_path: Path,
    config: IntermittentConfig,
    duration_minutes: float,
    engines: tuple[str, ...] = ("intermittent_gp",),
    pysr_config: PySRConfig | None = None,
    resume: bool = False,
    research_id: str | None = None,
    candidate_replay_threshold: int = 5,
    research_mode: str = "legacy",
    optimization_origin: dict[str, Any] | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "lotto_pattern_lab",
        "research",
        "--csv",
        str(csv_path.resolve()),
        "--duration-minutes",
        str(max(0.01, duration_minutes)),
        "--checkpoint",
        str(checkpoint_path.resolve()),
        "--population",
        str(config.population_size),
        "--generations",
        str(config.generations),
        "--workers",
        str(config.workers),
        "--crossover",
        str(config.crossover_probability),
        "--mutation",
        str(config.mutation_probability),
        "--max-depth",
        str(config.max_depth),
        "--train-ratio",
        str(config.train_ratio),
        "--validation-ratio",
        str(config.validation_ratio),
        "--max-lag",
        str(config.max_lag),
        "--min-activation",
        str(config.min_activation_rate),
        "--hidden-states",
        str(config.hidden_states),
        "--controls",
        str(config.control_runs),
        "--seed",
        str(config.seed),
        "--feature-profile",
        config.feature_profile,
        "--elite-seed-ratio",
        str(config.elite_seed_ratio),
        "--candidate-replay-threshold",
        str(candidate_replay_threshold),
        "--research-mode",
        research_mode,
    ]
    if optimization_origin:
        command.extend(("--optimization-origin-json", json.dumps(optimization_origin)))
    if research_id:
        command.extend(("--research-id", research_id))
    for engine in engines:
        command.extend(("--engine", engine))
    if config.elite_seeding_enabled:
        command.append("--elite-seeding")
    if pysr_config is not None:
        command.extend(
            (
                "--pysr-target",
                pysr_config.target,
                "--pysr-timeout",
                str(pysr_config.timeout_seconds),
                "--pysr-iterations",
                str(pysr_config.niterations),
                "--pysr-population",
                str(pysr_config.population_size),
                "--pysr-populations",
                str(pysr_config.populations),
                "--pysr-maxsize",
                str(pysr_config.maxsize),
                "--pysr-maxdepth",
                str(pysr_config.maxdepth),
                "--pysr-workers",
                str(pysr_config.workers),
                "--pysr-train-ratio",
                str(pysr_config.train_ratio),
                "--pysr-validation-ratio",
                str(pysr_config.validation_ratio),
                "--pysr-controls",
                str(pysr_config.random_control_replicates),
                "--pysr-guesses-json",
                json.dumps(list(pysr_config.guesses)),
            )
        )
        for operator in pysr_config.binary_operators:
            command.extend(("--pysr-binary-operator", operator))
        for operator in pysr_config.unary_operators:
            command.extend(("--pysr-unary-operator", operator))
        if pysr_config.warm_start:
            command.append("--pysr-warm-start")
    if resume:
        command.append("--resume")
    return command


st.title("지속적 패턴 연구")
if st.session_state.pop("optimization_reset_notice", False):
    st.info("기본 연구 설정으로 되돌렸습니다.")
st.caption(
    "새 연구는 별도 체크포인트로 시작하고, 저장된 연구는 당시 설정을 유지해 이어서 실행합니다."
)

checkpoint_dir = project_path("checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)
try:
    rounds = ResearchRoundStore()
except sqlite3.OperationalError as error:
    log_database_failure(
        "continuous research initialization", "data/database/patterns.sqlite", error
    )
    if is_database_locked(error):
        st.error("연구 기록 데이터베이스가 다른 저장 작업에서 사용 중입니다.")
    else:
        st.error("연구 기록 데이터베이스를 열지 못했습니다.")
    st.caption("연구 상태는 유지됩니다. 잠시 뒤 다시 시도하거나 상세 오류를 확인하세요.")
    st.code(str(error), language="text")
    if st.button("데이터베이스 연결 다시 시도", key="retry_continuous_database"):
        st.rerun()
    st.stop()


monitor_path_text = st.session_state.get("monitor_research_checkpoint")
resume_records = _resume_records(rounds)
if not monitor_path_text:
    discovered = next((item for item in resume_records if item.get("_worker_active")), None)
    if discovered:
        monitor_path_text = str(discovered["_path"])
        st.session_state["monitor_research_checkpoint"] = monitor_path_text


@st.fragment(run_every="5s")
def render_top_research_status() -> None:
    st.subheader("연구 진행 현황")
    st.caption("이 영역만 5초마다 자동 갱신됩니다.")
    monitor_text = st.session_state.get("monitor_research_checkpoint")
    phase = st.session_state.get("research_launch_status")
    if not monitor_text:
        phase_labels = {
            "environment_checking": "연구 환경 검사 중",
            "environment_passed": "연구 환경 검사 통과 · 실행 대기 중",
            "background_starting": "백그라운드 연구 시작 중",
            "process_verifying": "연구 프로세스 실행 확인 중",
            "start_failed": "연구 프로세스 시작 실패",
        }
        label = phase_labels.get(phase, "실행 중인 연구가 없습니다.")
        st.info(label)
        if phase in {"background_starting", "process_verifying"}:
            st.caption("프로세스와 첫 체크포인트를 확인하고 있습니다.")
        if st.button("상태 새로고침", key="refresh_research_status_empty"):
            st.rerun(scope="fragment")
        return

    monitor_path = project_path(monitor_text)
    if not monitor_path.is_relative_to(PROJECT_ROOT):
        st.error("공식 프로젝트 밖의 체크포인트 경로는 표시하지 않습니다.")
        return
    checkpoint_saved = monitor_path.is_file()
    worker_active = _worker_active(monitor_path)
    global_worker_active = research_worker_active()
    if checkpoint_saved:
        try:
            monitor = load_checkpoint(monitor_path)
        except (OSError, ValueError) as error:
            st.error(f"체크포인트를 읽지 못했습니다: {error}")
            return
        elapsed, remaining = research_time_snapshot(monitor, worker_active=worker_active)
        target = float(monitor.get("target_duration_seconds", 0) or 0)
        progress = min(1.0, elapsed / target) if target > 0 else 0.0
        current = monitor.get("in_progress") or {}
        summary = monitor.get("dashboard_summary") or {}
        strategy = monitor.get("research_strategy") or {}
        failed = summary.get(
            "failed_experiment_count",
            sum(
                not item.get("resolved_by_experiment_id")
                for item in monitor.get("failed_runs", [])
            ),
        )
        completed = int(monitor.get("completed_runs", 0))
        engine_names = {
            "intermittent_gp": "GP",
            "pysr_symbolic_regression": "PySR",
        }
        engine = (
            current.get("engine")
            or monitor.get("last_run_summary", {}).get("engine")
            or (monitor.get("engines") or [None])[0]
        )
        engine_label = engine_names.get(engine, engine or "확인 중")
        state_labels = {
            "queued": "대기 중",
            "running": "연구 실행 중" if worker_active else "프로세스 확인 필요",
            "pause_requested": "일시 정지 요청 중",
            "paused": "일시 정지",
            "completed": "완료 (실패 실험 포함)" if failed else "완료",
            "failed": "실패",
            "interrupted": "중단",
        }
        status = state_labels.get(monitor.get("status"), str(monitor.get("status", "확인 중")))
        if monitor.get("status") == "running" and not worker_active:
            status = "중단 가능성 · 연구 프로세스가 확인되지 않음"
        seed = current.get("seed")
        if seed is None and monitor.get("status") in {"running", "pause_requested"}:
            seed = monitor.get("rng_resume", {}).get("next_seed")
        seed = seed if seed is not None else "—"
        event = monitor.get("current_progress", {}).get("event", {})
        stage = (
            event.get("stage") or event.get("run")
            if isinstance(event, dict)
            else None
        ) or current.get("stage") or "실험 경계 대기"
        progress_time = f"시간 제한 진행률 · {progress:.1%}" if target > 0 else "시간 제한 없음"
        research_round_number = monitor.get("research_round", "확인 중")
        source_research_count = len(strategy.get("source_research_ids", ()))
        reused_formula_count = int(summary.get("inherited_pattern_count", 0) or 0)
        new_formula_count = summary.get("new_pattern_count")
        research_mode_label = (
            "누적 진화 연구"
            if monitor.get("research_mode") == CUMULATIVE_MODE
            else "독립 탐색"
            if monitor.get("research_mode") == INDEPENDENT_MODE
            else "기존 방식"
        )
    else:
        launch_state = {
            "environment_checking": "연구 환경 검사 중",
            "environment_passed": "연구 환경 검사 통과 · 실행 대기 중",
            "background_starting": "백그라운드 연구 시작 중",
            "process_verifying": "연구 프로세스 실행 확인 중",
            "first_checkpoint_waiting": "연구 프로세스 실행 확인 · 첫 체크포인트 저장 대기 중",
            "start_unconfirmed": "시작 요청 전달 · 프로세스 확인 대기 중",
            "start_failed": "연구 프로세스 시작 실패 · 체크포인트 미저장",
        }
        status = launch_state.get(phase, "첫 체크포인트 저장 대기 중")
        launch_started = st.session_state.get("research_launch_started_at")
        launch_target = st.session_state.get("research_target_duration_seconds")
        elapsed, remaining = research_time_snapshot(
            {
                "status": "running",
                "session_started_at": launch_started,
                "session_elapsed_base_seconds": 0,
                "target_duration_seconds": launch_target,
            }
        )
        target = float(launch_target or 0)
        progress = min(1.0, elapsed / target) if target > 0 else 0.0
        completed = 0
        failed = 0
        engine_label = st.session_state.get("research_launch_engine", "확인 중")
        seed = st.session_state.get("research_launch_seed", "—")
        stage = status
        progress_time = f"시간 제한 진행률 · {progress:.1%}" if target > 0 else "시간 제한 없음"
        monitor = {}
        research_round_number = "등록 대기"
        source_research_count = int(
            st.session_state.get("research_launch_source_research_count", 0)
        )
        reused_formula_count = int(
            st.session_state.get("research_launch_reuse_candidate_count", 0)
        )
        new_formula_count = None
        research_mode_label = st.session_state.get("research_launch_mode", "확인 중")

    st.markdown(f"### {status}")
    stats = st.columns(4)
    stats[0].metric("완료 실험", format_count(completed))
    stats[1].metric("실패 실험", format_count(failed))
    stats[2].metric("실행 엔진", engine_label)
    stats[3].metric(
        "현재 시드", format_count(seed) if isinstance(seed, (int, float)) else str(seed)
    )
    times = st.columns(2)
    times[0].metric("누적 실행 시간", format_hhmmss(elapsed))
    times[1].metric(
        "남은 연구 시간",
        format_hhmmss(remaining) if remaining is not None else "시간 제한 없음",
    )
    st.progress(progress)
    st.caption(progress_time)
    st.caption(
        f"연구 회차: {research_round_number} · {research_mode_label} · "
        f"이전 연구: {format_count(source_research_count)}개 · "
        f"실제 재사용 수식: {format_count(reused_formula_count)}개 · "
        f"신규 유효 수식: {format_count(new_formula_count)}개"
    )

    details = st.columns(2)
    details[0].write(f"현재 단계: {stage}")
    details[1].write(
        f"체크포인트: {checkpoint_status_label(checkpoint_saved, monitor.get('last_updated_at'))}"
    )

    research_id = monitor.get("research_id") or st.session_state.get("research_launch_id")
    pid = monitor.get("worker_pid") or st.session_state.get("research_launch_pid")
    with st.expander("연구 실행 상세 정보", expanded=False):
        st.write(f"연구 ID: {research_id or '확인 중'}")
        st.write(f"PID: {pid or '확인 중'}")
        st.write(f"체크포인트 경로: {monitor_path}")
        st.write(f"프로세스 실행 확인: {'예' if worker_active else '아니오'}")
        st.write(f"연구 실행 슬롯: {'점유 중' if global_worker_active else '사용 가능'}")
        if checkpoint_saved and monitor.get("report_path"):
            st.write(f"최종 연구 결과: {monitor['report_path']}")
        if checkpoint_saved and monitor.get("status") in {"completed", "failed", "paused"}:
            st.write(f"최종 상태: {status}")
            st.write(f"최종 실행 시간: {format_hhmmss(elapsed)}")
            st.write(f"완료 실험: {format_count(completed)} · 실패 실험: {format_count(failed)}")
    if checkpoint_saved and monitor.get("status") in {"completed", "failed"}:
        with st.expander("연구 결과 보기", expanded=False):
            st.write(f"최종 상태: {status}")
            st.write(f"실제 총 실행 시간: {format_hhmmss(elapsed)}")
            st.write(f"완료 실험: {format_count(completed)} · 실패 실험: {format_count(failed)}")
            experiments = monitor.get("completed_experiments", [])
            if experiments:
                st.dataframe(
                    [
                        {
                            "시드": item.get("seed"),
                            "상태": item.get("status", "completed"),
                            "실험 ID": item.get("experiment_id"),
                        }
                        for item in experiments
                    ],
                    hide_index=True,
                    width="stretch",
                )
            if monitor.get("report_path"):
                st.write(f"연구 보고서: {monitor['report_path']}")
    actions = st.columns(2)
    if actions[0].button("상태 새로고침", key="refresh_research_status"):
        st.rerun(scope="fragment")
    pause_pending = checkpoint_saved and pause_request_path(monitor_path).is_file()
    if (
        checkpoint_saved
        and monitor.get("status") == "running"
        and worker_active
        and not pause_pending
    ):
        if actions[1].button("연구 일시 정지", key="request_research_pause"):
            try:
                request_pause(monitor_path)
                st.info(PAUSE_MESSAGE)
                st.rerun(scope="fragment")
            except (OSError, ValueError, LockTimeoutError) as error:
                st.error(f"일시 정지 요청을 저장하지 못했습니다: {error}")
    elif pause_pending or (checkpoint_saved and monitor.get("status") == "pause_requested"):
        actions[1].info(PAUSE_MESSAGE)


render_top_research_status()

new_tab, resume_tab, queue_tab = st.tabs(
    (
        "\uc0c8 \uc5f0\uad6c \uc2dc\uc791",
        "\uae30\uc874 \uc5f0\uad6c \uc774\uc5b4\uc11c \ud558\uae30",
        "\uc5f0\uad6c \uc608\uc57d \uc2e4\ud589",
    )
)

with new_tab:
    data_label = st.radio(
        "기준 데이터",
        ("보너스 번호 포함 CSV", "기존 당첨번호 CSV"),
        key="new_research_data_source",
    )
    csv_path = project_path(
        "data/raw/lotto_draws_753_1246_with_bonus.csv"
        if data_label == "보너스 번호 포함 CSV"
        else "data/official/draws.csv"
    )
    try:
        new_dataset = load_csv(csv_path)
        new_integrity = dataset_integrity_report(new_dataset)
        st.caption(
            f"기준 데이터 상태: {new_dataset.verification_status}; "
            f"회차 {new_dataset.records[0].round_number}~{new_dataset.records[-1].round_number}; "
            f"보너스 번호 {'포함' if new_integrity['all_draws_have_bonus'] else '미포함'}; "
            f"날짜 {'포함' if new_integrity['all_draws_dated'] else '미포함'}"
        )
    except (OSError, ValueError, IndexError) as error:
        new_dataset = None
        new_integrity = None
        st.error(f"기준 데이터를 읽지 못했습니다: {error}")

    st.session_state.setdefault("new_research_engine", ENGINE_OPTIONS[0])
    st.session_state.setdefault("new_profile", "legacy")
    st.session_state.setdefault("new_duration_minutes", 180)
    st.session_state.setdefault("new_duration_mode", "recommended")
    st.session_state.setdefault("new_custom_duration_minutes", 5)
    st.session_state.setdefault("new_custom_duration_minutes_value", 5)
    # Keep the previous UI's Elite Seeding-on default while making the mode explicit.
    st.session_state.setdefault("new_research_mode", CUMULATIVE_MODE)
    if "new_population_size" not in st.session_state:
        _apply_research_preset()
    selector_columns = st.columns(4)
    engine_choice = selector_columns[0].selectbox(
        "탐색 엔진",
        ENGINE_OPTIONS,
        key="new_research_engine",
        help=(
            "GP는 유전 프로그래밍, PySR은 기호 회귀입니다. 비교 연구는 두 엔진을 번갈아 실행합니다."
        ),
        on_change=_apply_research_preset,
    )
    engines = {
        ENGINE_OPTIONS[0]: ("intermittent_gp",),
        ENGINE_OPTIONS[1]: ("pysr_symbolic_regression",),
        ENGINE_OPTIONS[2]: ("intermittent_gp", "pysr_symbolic_regression"),
    }[engine_choice]
    profile_names = {
        "legacy": "기본 특징",
        "calendar_round": "회차·날짜 특징",
        "bonus_lag_v1": "보너스 번호 포함 특징",
    }
    profile = selector_columns[1].selectbox(
        "특징 프로파일",
        ("legacy", "calendar_round", "bonus_lag_v1"),
        format_func=lambda value: profile_names[value],
        key="new_profile",
        help="날짜나 보너스 번호 특징을 쓰려면 데이터에 해당 정보가 있어야 합니다.",
        on_change=_apply_research_preset,
    )
    duration_mode = selector_columns[2].selectbox(
        "연구 시간 설정",
        ("recommended", "custom"),
        format_func=lambda value: "추천 시간" if value == "recommended" else "직접 입력",
        key="new_duration_mode",
    )
    if duration_mode == "recommended":
        duration_input = selector_columns[2].selectbox(
            "연구 시간 프리셋",
            (30, 60, 120, 180, 360, 480),
            format_func=lambda value: f"{value}분",
            key="new_duration_minutes",
            help=(
                "전체 연구 시간 예산입니다. GP와 PySR 비교도 같은 예산 안에서 "
                "두 엔진을 번갈아 실행합니다."
            ),
            on_change=_apply_research_preset,
        )
    else:
        st.session_state["new_custom_duration_minutes"] = st.session_state[
            "new_custom_duration_minutes_value"
        ]
        duration_input = selector_columns[2].number_input(
            "연구 시간(분)",
            min_value=1,
            max_value=1440,
            step=1,
            key="new_custom_duration_minutes",
            help="1분부터 1440분(24시간)까지 정수 분으로 직접 설정합니다.",
            on_change=_remember_custom_duration,
        )
    duration_validation_error = None
    try:
        duration_minutes = duration_input
        duration_seconds = duration_seconds_from_minutes(duration_input)
    except (TypeError, ValueError):
        duration_validation_error = (
            "연구 시간은 1분부터 1440분까지 정수로 입력해야 합니다."
        )
        duration_minutes = int(st.session_state.get("new_duration_minutes", 180))
        duration_seconds = None
    research_mode = selector_columns[3].selectbox(
        "연구 방식",
        (INDEPENDENT_MODE, CUMULATIVE_MODE),
        format_func=lambda value: (
            "독립 탐색" if value == INDEPENDENT_MODE else "누적 진화 연구"
        ),
        key="new_research_mode",
        help=(
            "독립 탐색은 과거 수식을 주입하지 않습니다. 누적 진화 연구는 호환되는 GP 수식을 "
            "검증 점수와 다양성 기준으로 일부 초기 개체군에 재사용합니다."
        ),
    )
    current_data_range = (
        [new_dataset.records[0].round_number, new_dataset.records[-1].round_number]
        if new_dataset
        else None
    )
    st.session_state["research_current_engine_ids"] = list(engines)
    st.session_state["research_current_profile"] = profile
    st.session_state["research_current_duration_seconds"] = duration_seconds
    st.session_state["research_current_dataset_range"] = current_data_range
    st.session_state["research_current_dataset_integrity"] = new_integrity
    current_dataset_hashes = (
        dataset_hashes(new_dataset, max_lag=int(st.session_state.get("new_max_lag", 10)))
        if new_dataset is not None
        else None
    )
    current_dataset_snapshot = (
        current_dataset_hashes["bonus_inclusive_sha256"] if current_dataset_hashes else None
    )
    st.session_state["research_current_dataset_sha256"] = current_dataset_snapshot
    current_research_settings = {
        **{key: st.session_state.get(key) for key in OPTUNA_TO_RESEARCH_KEYS.values()},
        "new_profile": profile,
        "new_research_engine": engine_choice,
        "new_duration_minutes": int(duration_minutes),
        "new_research_mode": research_mode,
        "dataset_range": current_data_range,
    }
    st.session_state["research_current_settings"] = current_research_settings
    optimization_context = st.session_state.get("optimization_applied_context")
    if optimization_context:
        applied_settings = optimization_context.get("research_settings", {})
        modified = changed_optimization_settings(applied_settings, current_research_settings)
        optimization_context["modified_by_user"] = modified
        expected_params = optimization_context.get("parameters", {})
        if not optimization_context.get("application_verified"):
            expected_controls = optimization_context.get("expected_controls", {})
            actual_controls = {
                key: st.session_state.get(key) for key in expected_controls
            }
            values_match = (
                bool(expected_controls)
                and actual_controls == expected_controls
                and profile == optimization_context.get("feature_profile")
            )
            optimization_context["application_verified"] = values_match
            if not values_match:
                optimization_context["application_error"] = (
                    "일부 Optuna 값이 연구 입력란에 반영되지 않았습니다."
                )
        else:
            values_match = bool(optimization_context.get("application_verified"))
        st.session_state["optimization_applied_context"] = optimization_context
        started_label = (
            "연구 시작됨"
            if optimization_context.get("started_research_ids")
            else "아직 연구 시작 전"
        )
        if not values_match:
            st.error(
                "최적화 설정 적용에 실패했습니다. "
                + optimization_context.get("application_error", "입력값을 다시 확인하세요.")
            )
        elif modified:
            st.warning("Optuna 최적화 설정을 기반으로 사용자가 일부 값을 변경했습니다.")
        elif not optimization_context.get("started_research_ids"):
            st.success("최적화 설정이 적용되었습니다. 연구 조건을 확인한 후 연구를 시작하세요.")
        else:
            st.success("최적화 설정이 적용된 연구 실행 기록이 있습니다.")
        if values_match:
            st.markdown(
                f"**Optuna 최적화 설정 적용 완료** · "
                f"Study ID `{optimization_context.get('optuna_study_id')}` · "
                f"`{optimization_context.get('study_name')}` / "
                f"Trial #{optimization_context.get('trial_number')} · "
                f"{len(expected_params)}개 파라미터 · "
                f"프로파일 `{optimization_context.get('feature_profile')}` · "
                f"데이터 {optimization_context.get('dataset_range')}회차 · "
                f"적용 시각 {optimization_context.get('applied_at')} · {started_label}"
            )
        with st.expander("최적화 설정 상세 보기", expanded=False):
            st.write(f"특징 프로파일: `{optimization_context.get('feature_profile')}`")
            st.write(f"데이터 회차 범위: `{optimization_context.get('dataset_range')}`")
            st.write("적용된 Optuna 파라미터")
            st.json(expected_params)
            st.write("적용 당시 연구 설정")
            st.json(applied_settings)
        if st.button("기본 설정으로 되돌리기", key="reset_optimization_applied_settings"):
            st.session_state["optimization_reset_pending"] = True
            st.rerun()

    selected_preset = GP_PRESETS[recommended_preset_minutes(duration_minutes, GP_PRESETS)]
    strategy_recommendation = recommend_optuna_settings(
        list_optimization_studies(),
        feature_profile=profile,
        dataset_range=current_data_range,
        dataset_sha256=current_dataset_snapshot,
        dataset_integrity=new_integrity,
    )
    st.caption(
        f"현재 조건: {engine_choice} · {profile_names[profile]} · "
        f"{int(duration_minutes)}분 · "
        f"{'누적 진화 연구' if research_mode == CUMULATIVE_MODE else '독립 탐색'} · "
        + (
            f"Optuna Trial #{optimization_context.get('trial_number')} 적용"
            if optimization_context
            else "기본 연구 설정"
        )
    )
    if duration_minutes <= 10:
        st.info(
            "짧은 테스트 연구는 워커 실행, 체크포인트 저장, 상태 갱신 및 결과 연결을 "
            "확인하는 용도입니다. GP/PySR 계산은 진행 중인 실험 단위가 마무리된 뒤 "
            "시간 제한을 다시 확인하므로 설정한 시간을 넘길 수 있습니다."
        )
    action_columns = st.columns([1, 1, 3])
    preflight_submitted = action_columns[0].button(
        "연구 환경 사전 검사",
        key="run_new_research_preflight",
        help="현재 데이터와 설정으로 실행 환경을 검사합니다.",
    )
    submitted = action_columns[1].button(
        "새 연구 시작",
        type="primary",
        key="start_new_research_now",
        disabled=research_worker_active() or duration_validation_error is not None,
        help="시작을 누르면 유효한 검사가 없을 때 자동 검사 후 새 연구를 시작합니다.",
    )
    if research_worker_active():
        action_columns[2].warning("다른 연구 작업이 실행 중이어서 시작할 수 없습니다.")
    elif new_dataset is None:
        action_columns[2].error("기준 데이터 파일을 읽지 못해 연구를 시작할 수 없습니다.")

    if duration_validation_error and new_dataset is not None and not research_worker_active():
        action_columns[2].error(duration_validation_error)

    startup_status_container = st.container()
    applied_settings_container = st.container()

    with st.expander("자동 연구 기본값 안내", expanded=False):
        st.caption("자동 추천값은 시작점이며 최적 성능이 입증된 설정은 아닙니다.")
        st.write(
            f"GP 권장값: 개체군 {selected_preset['population']}, "
            f"세대 {selected_preset['generations']}, 최대 깊이 {selected_preset['max_depth']}, "
            f"무작위 대조 {selected_preset['controls']}회. "
            "비교 모드에서는 아래 별도 PySR 설정이 적용됩니다."
        )
    with st.container():
        with st.expander("고급 설정", expanded=False):
            candidate_replay_threshold = st.selectbox(
                "자동 전체 회차 재검증 후보 기준",
                (3, 4, 5, 6),
                index=2,
                format_func=lambda value: f"{value}개 이상 일치",
                help=(
                    "연구 중 기준을 충족한 6개 집합 공식은 "
                    "별도 백그라운드 재검증 작업에 등록됩니다."
                ),
                key="candidate_replay_threshold",
            )
            population = st.number_input(
                "개체군",
                min_value=8,
                max_value=512,
                step=8,
                help="한 세대에서 평가할 후보 수이며 값이 클수록 계산량이 늘어납니다.",
                key="new_population_size",
            )
            generations = st.number_input(
                "세대 수",
                min_value=1,
                max_value=500,
                help="진화 탐색 반복 횟수입니다.",
                key="new_generations",
            )
            col1, col2, col3 = st.columns(3)
            crossover = col1.slider(
                "교차 확률",
                0.0,
                1.0,
                help="세대 간 후보를 섞어 새 후보를 만드는 비율입니다.",
                key="new_crossover_probability",
            )
            mutation = col2.slider(
                "돌연변이 확률",
                0.0,
                1.0,
                help="후보를 무작위로 변형해 탐색하는 비율입니다.",
                key="new_mutation_probability",
            )
            max_depth = col3.number_input(
                "최대 수식 깊이", 2, 20, help="GP 수식 트리의 최대 깊이입니다.", key="new_max_depth"
            )
            workers = st.number_input(
                "CPU 워커", 1, 4, help="동시에 사용할 CPU 프로세스 수입니다.", key="new_workers"
            )
            col1, col2, col3 = st.columns(3)
            train_ratio = col1.slider(
                "학습 비율",
                0.1,
                0.9,
                help="후보 생성과 학습에 사용하는 과거 회차 비율입니다.",
                key="new_train_ratio",
            )
            validation_ratio = col2.slider(
                "검증 비율",
                0.05,
                0.8,
                help="후보 비교와 선택에 사용하는 별도 과거 회차 비율입니다.",
                key="new_validation_ratio",
            )
            max_lag = col3.number_input(
                "최대 과거 참조",
                1,
                10,
                help="예측 입력에서 참조할 과거 회차의 최대 수입니다.",
                key="new_max_lag",
            )
            st.session_state.setdefault("new_pysr_profile", profile)
            st.session_state.setdefault("new_pysr_max_lag", int(max_lag))
            if len(engines) > 1:
                st.caption(
                    "비교 모드: Optuna 값은 GP 설정에만 적용됩니다. "
                    "PySR은 아래 별도 특징 설정을 사용합니다."
                )
                pysr_profile = st.selectbox(
                    "PySR 특징 프로파일",
                    ("legacy", "calendar_round", "bonus_lag_v1"),
                    format_func=lambda value: profile_names[value],
                    key="new_pysr_profile",
                    help="비교 모드에서 PySR에만 전달되는 특징 프로파일입니다.",
                )
                pysr_max_lag = st.number_input(
                    "PySR 최대 과거 참조",
                    min_value=1,
                    max_value=10,
                    key="new_pysr_max_lag",
                    help="비교 모드에서 PySR이 참조할 과거 회차 수입니다.",
                )
            else:
                pysr_profile = profile
                pysr_max_lag = int(max_lag)
            col1, col2, col3 = st.columns(3)
            controls = col1.number_input(
                "무작위 대조 반복",
                0,
                100,
                help="무작위 기준선 비교를 반복하는 횟수입니다.",
                key="new_control_runs",
            )
            min_activation = col2.slider(
                "최소 활성화 비율",
                0.05,
                1.0,
                help="이 비율보다 적게 활성화되는 규칙을 후보에서 제외합니다.",
                key="new_min_activation_rate",
            )
            hidden_states = col3.number_input(
                "숨겨진 상태 수",
                2,
                5,
                help="GP 조건식에서 구분하는 내부 상태 수입니다.",
                key="new_hidden_states",
            )
            seed = st.number_input(
                "초기 난수 시드",
                0,
                2**31 - 1,
                help="같은 환경에서 탐색을 재현하기 위한 난수 시드입니다.",
                key="new_seed",
            )
            elite_toggle = st.checkbox(
                "GP Elite Seeding 사용",
                help="호환되는 저장 패턴을 초기 후보에 포함합니다.",
                key="new_elite_seeding_enabled",
                disabled=(
                    research_mode != CUMULATIVE_MODE or "intermittent_gp" not in engines
                ),
            )
            elite_enabled = bool(
                research_mode == CUMULATIVE_MODE
                and "intermittent_gp" in engines
                and elite_toggle
            )
            elite_ratio = st.slider(
                "기존 패턴 계승 비율",
                0.0,
                0.8,
                step=0.05,
                help="초기 개체군 중 저장 패턴에서 가져오는 비율입니다.",
                key="new_elite_seed_ratio",
            )
            pysr_targets = ("main_positions", "one_number") + (
                ("bonus",)
                if pysr_profile in {"bonus_lag_v1", "calendar_round"}
                else ()
            )
            pysr_target = st.selectbox(
                "PySR 탐색 목표",
                pysr_targets,
                disabled="pysr_symbolic_regression" not in engines,
                help="PySR이 수식으로 예측할 번호 대상입니다.",
                key="new_pysr_target",
            )
            pysr_binary_operators = st.multiselect(
                "PySR 이항 연산자",
                ("+", "-", "*", "/"),
                disabled="pysr_symbolic_regression" not in engines,
                help="수식에 허용할 이항 연산자입니다.",
                key="new_pysr_binary_operators",
            )
            pysr_unary_operators = st.multiselect(
                "PySR 단항 연산자",
                ("square", "abs"),
                disabled="pysr_symbolic_regression" not in engines,
                help="수식에 허용할 단항 연산자입니다.",
                key="new_pysr_unary_operators",
            )
            pysr_guesses_text = st.text_area(
                "초기 추정 수식 (한 줄에 하나, 변수 x0, x1, ...)",
                disabled="pysr_symbolic_regression" not in engines,
                help=(
                    "초기 후보 수식입니다. 한 줄에 하나씩 입력하고 입력 변수는 "
                    "실제 특징 스키마와 맞아야 합니다."
                ),
                key="new_pysr_guesses",
            )
            pysr_warm_start = st.checkbox(
                "실패 시 동일 난수 시드의 PySR 저장 상태에서 재개",
                value=False,
                disabled="pysr_symbolic_regression" not in engines,
                help=(
                    "PySR의 저장된 hall-of-fame 상태를 from_file로 불러옵니다. "
                    "재개에는 해당 실행 폴더의 정상 PySR 체크포인트가 필요합니다."
                ),
                key="new_pysr_warm_start",
            )
            pysr_timeout = st.number_input(
                "PySR 탐색 시간(초, 위치별 상한)",
                1.0,
                3600.0,
                step=1.0,
                disabled="pysr_symbolic_regression" not in engines,
                help="PySR 한 번의 fit에 허용할 전체 초입니다. 대상 위치별로 나누어 적용합니다.",
                key="new_pysr_timeout",
            )
            pysr_train_ratio = st.slider(
                "PySR 학습 비율",
                0.1,
                0.8,
                step=0.05,
                disabled="pysr_symbolic_regression" not in engines,
                help="PySR 수식 생성에 사용할 과거 회차 비율입니다.",
                key="new_pysr_train_ratio",
            )
            pysr_validation_ratio = st.slider(
                "PySR 검증 비율",
                0.05,
                0.7,
                step=0.05,
                disabled="pysr_symbolic_regression" not in engines,
                help="PySR 후보를 선택하는 검증 회차 비율입니다.",
                key="new_pysr_validation_ratio",
            )
            col1, col2, col3 = st.columns(3)
            pysr_iterations = col1.number_input(
                "PySR 반복",
                1,
                10000,
                disabled="pysr_symbolic_regression" not in engines,
                help="PySR 진화 탐색 반복 횟수입니다.",
                key="new_pysr_iterations",
            )
            pysr_population = col2.number_input(
                "PySR 모집단",
                8,
                512,
                disabled="pysr_symbolic_regression" not in engines,
                help="PySR 각 모집단의 수식 후보 수입니다.",
                key="new_pysr_population",
            )
            pysr_populations = col3.number_input(
                "PySR 모집단 수",
                1,
                32,
                disabled="pysr_symbolic_regression" not in engines,
                help="PySR이 동시에 유지하는 모집단 수입니다.",
                key="new_pysr_populations",
            )
            col1, col2, col3 = st.columns(3)
            pysr_maxsize = col1.number_input(
                "PySR 최대 복잡도",
                4,
                100,
                disabled="pysr_symbolic_regression" not in engines,
                help="PySR 수식의 최대 복잡도입니다.",
                key="new_pysr_maxsize",
            )
            pysr_maxdepth = col2.number_input(
                "PySR 최대 깊이",
                1,
                100,
                disabled="pysr_symbolic_regression" not in engines,
                help="PySR 수식의 최대 중첩 깊이입니다.",
                key="new_pysr_maxdepth",
            )
            pysr_workers = col3.number_input(
                "PySR CPU 워커",
                1,
                4,
                disabled="pysr_symbolic_regression" not in engines,
                help="PySR에서 사용할 CPU 워커 수입니다.",
                key="new_pysr_workers",
            )
            pysr_controls = st.number_input(
                "PySR 무작위 대조 반복",
                0,
                100,
                disabled="pysr_symbolic_regression" not in engines,
                help="PySR 후보와 비교할 무작위 대조 횟수입니다.",
                key="new_pysr_controls",
            )
        pattern_db = project_path("data/database/patterns.sqlite")
        pattern_db_stat = (
            (pattern_db.stat().st_mtime_ns, pattern_db.stat().st_size)
            if pattern_db.is_file()
            else None
        )
        preflight_input_key = _preflight_fingerprint(
            {
                "dataset_sha256": dataset_digest(new_dataset) if new_dataset else None,
                "dataset_file_sha256": (
                    hashlib.sha256(csv_path.read_bytes()).hexdigest()
                    if csv_path is not None and csv_path.is_file()
                    else None
                ),
                "engines": engines,
                "feature_profile": profile,
                "gp": {
                    "population": population,
                    "generations": generations,
                    "crossover": crossover,
                    "mutation": mutation,
                    "max_depth": max_depth,
                    "workers": workers,
                    "train_ratio": train_ratio,
                    "validation_ratio": validation_ratio,
                    "max_lag": max_lag,
                    "control_runs": controls,
                    "min_activation": min_activation,
                    "hidden_states": hidden_states,
                    "seed": seed,
                    "elite_enabled": elite_enabled,
                    "elite_ratio": elite_ratio,
                },
                "pysr": {
                    "feature_profile": pysr_profile,
                    "max_lag": pysr_max_lag,
                    "target": pysr_target,
                    "binary_operators": pysr_binary_operators,
                    "unary_operators": pysr_unary_operators,
                    "guesses": pysr_guesses_text,
                    "warm_start": pysr_warm_start,
                    "timeout": pysr_timeout,
                    "train_ratio": pysr_train_ratio,
                    "validation_ratio": pysr_validation_ratio,
                    "iterations": pysr_iterations,
                    "population": pysr_population,
                    "populations": pysr_populations,
                    "maxsize": pysr_maxsize,
                    "maxdepth": pysr_maxdepth,
                    "workers": pysr_workers,
                    "controls": pysr_controls,
                },
                "pattern_library_stat": pattern_db_stat,
            }
        )
        preflight_disk_key = preflight_cache_key({"ui_settings_fingerprint": preflight_input_key})
        cache_loaded_from_disk = False
        if st.session_state.get("research_preflight_input_key") != preflight_disk_key:
            disk_cache = load_preflight_cache(preflight_disk_key)
            if disk_cache is not None:
                st.session_state["research_preflight_report"] = disk_cache["report"]
                st.session_state["research_preflight_input_key"] = preflight_disk_key
                st.session_state["research_preflight_checked_at"] = disk_cache["checked_at"]
                cache_loaded_from_disk = True
        saved_preflight = st.session_state.get("research_preflight_report", {})
        preflight_ready = (
            st.session_state.get("research_preflight_input_key") == preflight_disk_key
            and saved_preflight.get("status") == "passed"
        )

    effective_config = None
    config_error = None
    try:
        effective_config = IntermittentConfig(
            population_size=int(population),
            generations=int(generations),
            crossover_probability=float(crossover),
            mutation_probability=float(mutation),
            max_depth=int(max_depth),
            workers=int(workers),
            train_ratio=float(train_ratio),
            validation_ratio=float(validation_ratio),
            max_lag=int(max_lag),
            control_runs=int(controls),
            min_activation_rate=float(min_activation),
            hidden_states=int(hidden_states),
            seed=int(seed),
            feature_profile=profile,
            elite_seeding_enabled=bool(elite_enabled),
            elite_seed_ratio=float(elite_ratio),
        )
    except ValueError as error:
        config_error = str(error)
        st.error(f"GP 설정을 확인하세요: {error}")
    pysr_config = None
    pysr_status = optional_backend_status()
    if "pysr_symbolic_regression" in engines:
        try:
            pysr_config = PySRConfig(
                target=pysr_target,
                feature_profile=pysr_profile,
                train_ratio=float(pysr_train_ratio),
                validation_ratio=float(pysr_validation_ratio),
                max_lag=int(pysr_max_lag),
                timeout_seconds=float(pysr_timeout),
                niterations=int(pysr_iterations),
                population_size=int(pysr_population),
                populations=int(pysr_populations),
                maxsize=int(pysr_maxsize),
                maxdepth=int(pysr_maxdepth),
                workers=int(pysr_workers),
                seed=int(seed),
                binary_operators=tuple(pysr_binary_operators),
                unary_operators=tuple(pysr_unary_operators),
                guesses=tuple(
                    line.strip() for line in pysr_guesses_text.splitlines() if line.strip()
                ),
                warm_start=bool(pysr_warm_start),
                random_control_replicates=int(pysr_controls),
            )
            if not pysr_status["installed"]:
                st.warning(
                    "PySR is not installed in this Streamlit Python environment. "
                    f"Install with: {pysr_status['install_command']}"
                )
        except ValueError as error:
            config_error = str(error)
            st.error(f"PySR 설정을 확인하세요: {error}")
    if effective_config is not None:
        expected = GP_PRESETS[recommended_preset_minutes(duration_minutes, GP_PRESETS)]
        is_custom = (
            effective_config.population_size != expected["population"]
            or effective_config.generations != expected["generations"]
            or effective_config.max_depth != expected["max_depth"]
            or effective_config.control_runs != expected["controls"]
            or effective_config.crossover_probability != 0.65
            or effective_config.mutation_probability != 0.25
            or effective_config.train_ratio != 0.60
            or effective_config.validation_ratio != 0.20
            or effective_config.max_lag != 10
            or effective_config.min_activation_rate != 0.10
            or effective_config.hidden_states != 3
            or effective_config.workers != 4
            or not effective_config.elite_seeding_enabled
            or effective_config.elite_seed_ratio != 0.20
            or effective_config.seed != 12345
        )
        st.caption("설정 상태: 직접 설정" if is_custom else "설정 상태: 사전 구성")

    if new_dataset is not None and effective_config is not None:
        invalid_profile = (
            (profile == "calendar_round" and not new_integrity["all_draws_dated"])
            or (profile == "bonus_lag_v1" and not new_integrity["all_draws_have_bonus"])
            or not new_integrity["rounds_contiguous"]
        )
        if invalid_profile:
            st.error("선택한 특징 프로파일에 필요한 데이터 필드 또는 연속 회차가 없습니다.")
        applied_settings_container.subheader("시작 전 적용 설정")
        applied_settings_container.caption(
            f"{', '.join(engines)} · {profile_names[profile]} · "
            f"{int(duration_minutes)}분 · "
            f"{'누적 진화 연구' if research_mode == CUMULATIVE_MODE else '독립 탐색'}"
        )
        if optimization_context:
            applied_settings_container.caption(
                f"Optuna {optimization_context.get('study_name')} "
                f"Trial #{optimization_context.get('trial_number')} 설정 적용"
            )
        settings_details = applied_settings_container.expander(
            "세부 적용 설정 보기", expanded=False
        )
        setting_rows = [
            {"설정": "탐색 엔진", "값": ", ".join(engines)},
            {"설정": "특징 프로파일", "값": profile},
            {
                "설정": "연구 방식",
                "값": "누적 진화 연구" if research_mode == CUMULATIVE_MODE else "독립 탐색",
            },
            {"설정": "프로파일 스키마", "값": feature_schema_version(profile)},
            {"설정": "연구 시간(분)", "값": duration_minutes},
            {"설정": "개체군 / 세대", "값": f"{population} / {generations}"},
            {"설정": "교차 / 돌연변이", "값": f"{crossover:.3f} / {mutation:.3f}"},
            {"설정": "최대 깊이 / 워커", "값": f"{max_depth} / {workers}"},
            {
                "설정": "학습 / 검증 / 테스트",
                "값": (
                    f"{train_ratio:.0%} / {validation_ratio:.0%} / "
                    f"{1 - train_ratio - validation_ratio:.0%}"
                ),
            },
            {"설정": "최대 과거 참조", "값": max_lag},
            {
                "설정": "최소 활성화 / 숨겨진 상태",
                "값": f"{min_activation:.0%} / {hidden_states}",
            },
            {"설정": "대조 반복 / 난수 시드", "값": f"{controls} / {seed}"},
            {
                "설정": "Elite Seeding / 계승 비율",
                "값": f"{'사용' if elite_enabled else '사용 안 함'} / {elite_ratio:.0%}",
            },
            {
                "설정": "CSV / 검증 상태",
                "값": f"{csv_path.name} / {new_dataset.verification_status}",
            },
        ]
        if pysr_config is not None:
            setting_rows.extend(
                [
                    {"설정": "PySR 목표", "값": pysr_config.target},
                    {
                        "설정": "PySR 반복·모집단·복잡도·깊이",
                        "값": (
                            f"{pysr_config.niterations} / {pysr_config.population_size} × "
                            f"{pysr_config.populations} / {pysr_config.maxsize} / "
                            f"{pysr_config.maxdepth}"
                        ),
                    },
                    {
                        "설정": "PySR 연산자 / 워커 / 시간",
                        "값": (
                            f"{pysr_config.binary_operators}, {pysr_config.unary_operators} / "
                            f"{pysr_config.workers} / {pysr_config.timeout_seconds}s per target"
                        ),
                    },
                    {
                        "설정": "PySR 검증 분할",
                        "값": (
                            f"train {pysr_config.train_ratio:.0%} / "
                            f"validation {pysr_config.validation_ratio:.0%} / "
                            f"test {1 - pysr_config.train_ratio - pysr_config.validation_ratio:.0%}"
                        ),
                    },
                    {
                        "설정": "PySR 무작위 대조 반복",
                        "값": pysr_config.random_control_replicates,
                    },
                    {
                        "설정": "초기 수식 수 / Julia 상태",
                        "값": (
                            f"{len(pysr_config.guesses)} / "
                            f"{'설치됨' if pysr_status['installed'] else '미설치'}"
                        ),
                    },
                ]
            )
        if len(engines) > 1:
            settings_details.info(
                "비교 연구는 공통 난수 시드 흐름에서 엔진별 실험을 번갈아 실행합니다. "
                "동일 후보의 paired 실험으로 해석하지 말고 각 엔진의 실행 시간을 별도로 확인하세요."
            )
        settings_details.dataframe(
            [{"설정": row["설정"], "값": str(row["값"])} for row in setting_rows],
            hide_index=True,
            width="stretch",
        )
        try:
            library = PatternLibrary()
        except sqlite3.OperationalError as error:
            log_database_failure("research pattern preview", "data/database/patterns.sqlite", error)
            applied_settings_container.error(
                "수식 라이브러리 데이터베이스가 사용 중이어서 미리보기를 불러오지 못했습니다."
            )
            applied_settings_container.code(str(error), language="text")
            if applied_settings_container.button("미리보기 다시 시도", key="retry_pattern_preview"):
                st.rerun()
            st.stop()
        preview: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        if elite_enabled:
            preview, skipped = _cached_elite_preview(
                str(library.path),
                str(project_path("experiments")),
                pattern_db_stat,
                feature_profile=profile,
                max_lag=int(max_lag),
                max_candidates=2 * ceil(int(population) * float(elite_ratio)),
                max_data_round=new_dataset.records[-1].round_number,
                require_temporal_provenance=True,
            )
            applied_settings_container.caption(
                f"호환 가능한 이전 수식 {len(preview)}개를 확인했고, "
                f"중복·호환성 문제로 제외한 후보는 {len(skipped)}개입니다. "
                f"최대 예상 주입 수는 초기 개체군의 {elite_ratio:.0%}입니다."
            )
            source_rounds = {
                str(item.get("origin_research_id") or item.get("research_id"))
                for item in preview
                if item.get("origin_research_id") or item.get("research_id")
            }
            compatible_profiles = sorted(
                {str(item.get("feature_profile", "legacy")) for item in preview}
            )
            duplicate_exclusions = sum(
                "duplicate" in str(item.get("reason", "")).casefold() for item in skipped
            )
            applied_settings_container.caption(
                f"재사용 가능한 이전 연구: {len(source_rounds)}개 · "
                "호환 프로파일: "
                f"{', '.join(compatible_profiles) if compatible_profiles else '없음'} · "
                f"중복 제외: {duplicate_exclusions}개"
            )
        elif research_mode == CUMULATIVE_MODE and "intermittent_gp" in engines:
            applied_settings_container.info(
                "호환되는 이전 GP 수식이 없어 이번 초기 개체군에는 주입하지 않습니다."
            )
        if research_mode == CUMULATIVE_MODE and "pysr_symbolic_regression" in engines:
            applied_settings_container.caption(
                "PySR에는 다른 연구의 수식을 자동 주입하지 않습니다. "
                "PySR 후보는 별도 라이브러리에 기록되며, "
                "현재 환경에서는 Julia 런타임이 확인되지 않았습니다."
            )
        if strategy_recommendation["available"]:
            applied_settings_container.info(
                f"Optuna 설정 참고: {strategy_recommendation['study_name']} "
                f"Trial #{strategy_recommendation['trial_number']} · "
                f"{strategy_recommendation['parameters']} · "
                "현재 입력에는 자동 적용되지 않습니다."
            )
        else:
            applied_settings_container.caption(strategy_recommendation["reason"])
        elite_seeds = tuple(preview)
    else:
        invalid_profile = True
        elite_seeds = ()

    with startup_status_container:
        if preflight_submitted or (submitted and not preflight_ready):
            st.session_state["research_launch_status"] = "environment_checking"
            if new_dataset is None or effective_config is None or invalid_profile:
                report = {
                    "status": "failed",
                    "engines": {
                        engine: {
                            "status": "failed",
                            "engine": engine,
                            "feature_profile": profile,
                            "stage": "input_validation",
                            "error": "선택한 데이터 또는 특징 프로파일 검증에 실패했습니다.",
                        }
                        for engine in engines
                    },
                }
            else:
                with st.status(
                    "연구 환경을 검사하고 있습니다.", expanded=True
                ) as progress:
                    st.write(
                        "데이터 무결성, 특징 설정, GP 실행 및 저장 환경을 확인합니다."
                    )
                    report = run_research_preflight(
                        new_dataset,
                        effective_config,
                        engines,
                        pysr_config,
                        elite_seeds=elite_seeds,
                    )
                    progress.update(
                        label=(
                            "연구 환경 검사 통과"
                            if report.get("status") == "passed"
                            else "연구 환경 검사 실패"
                        ),
                        state="complete",
                    )
            st.session_state["research_preflight_report"] = report
            st.session_state["research_preflight_input_key"] = preflight_disk_key
            st.session_state["research_preflight_checked_at"] = datetime.now(UTC).isoformat()
            st.session_state["research_launch_status"] = (
                "environment_passed" if report.get("status") == "passed" else "environment_failed"
            )
            try:
                save_preflight_cache(preflight_disk_key, report)
            except OSError as error:
                st.warning(
                    f"검사 결과를 디스크에 저장하지 못했습니다. 이번 세션에는 유지합니다: {error}"
                )

        current_preflight = st.session_state.get("research_preflight_report")
        current_preflight_matches = (
            st.session_state.get("research_preflight_input_key") == preflight_disk_key
        )
        if current_preflight and current_preflight_matches:
            phase = st.session_state.get("research_launch_status")
            if phase in {None, "ready", "environment_checking"}:
                phase = (
                    "environment_passed"
                    if current_preflight.get("status") == "passed"
                    else "environment_failed"
                )
            severity, message = research_startup_message(phase)
            getattr(st, severity)(message)
            if current_preflight.get("status") == "passed":
                origin = (
                    "\uC800\uC7A5\uB41C \uAC80\uC0AC \uACB0\uACFC \uC0AC\uC6A9"
                    if cache_loaded_from_disk
                    else "\uD604\uC7AC \uC138\uC158 \uAC80\uC0AC \uACB0\uACFC \uC0AC\uC6A9"
                )
                checked_at = st.session_state.get(
                    "research_preflight_checked_at", "\uC2DC\uAC01 \uBBF8\uAE30\uB85D"
                )
                st.caption(
                    f"{origin} \u00B7 \uAC80\uC0AC \uC2DC\uAC01 {checked_at}"
                )
            for engine_id, engine_report in current_preflight.get("engines", {}).items():
                with st.expander(
                    f"{engine_id}: {engine_report.get('status', 'failed')}"
                ):
                    st.json(
                        {key: value for key, value in engine_report.items() if key != "traceback"}
                    )
                    if engine_report.get("traceback"):
                        st.code(engine_report["traceback"], language="text")
                    if engine_id == "pysr_symbolic_regression":
                        if engine_report.get("julia_search") is True:
                            st.caption(
                                "PySR/Julia의 짧은 smoke fit까지 확인했습니다."
                            )
                        else:
                            st.warning(
                                "이 사전 검사에서는 Julia 런타임 실행을 확인하지 않았습니다."
                            )
        else:
            severity, message = research_startup_message(
                st.session_state.get("research_launch_status", "ready")
            )
            getattr(st, severity)(message)

        if submitted:
            if (
                not current_preflight_matches
                or not current_preflight
                or current_preflight.get("status") != "passed"
            ):
                st.error(
                    "현재 엔진·설정 조합의 사전 검사가 완료되지 않아 연구를 시작하지 않았습니다."
                )
            elif new_dataset is None or effective_config is None or invalid_profile:
                st.error(
                    "데이터 또는 설정 검증을 통과하지 못해 연구를 시작하지 않았습니다."
                )
            elif optimization_context and (
                not optimization_context.get("application_verified")
                or "intermittent_gp" not in engines
                or profile != optimization_context.get("feature_profile")
                or current_data_range != optimization_context.get("dataset_range")
            ):
                st.error(
                    "적용한 Optuna 설정과 현재 연구 조건이 호환되지 않습니다. "
                    "GP 포함 엔진, 동일한 특징 프로파일과 데이터 범위를 선택한 뒤 다시 확인하세요."
                )
            else:
                checkpoint_path = checkpoint_dir / (
                    f"research_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}.json"
                )
                st.session_state["research_launch_status"] = "environment_checking"
                try:
                    _validate_live_research_start(checkpoint_dir, csv_path, new_dataset)
                except (OSError, RuntimeError, ValueError) as error:
                    st.session_state["research_launch_status"] = "start_failed"
                    st.error(f"실행 직전 환경 확인 실패: {error}")
                    st.stop()
                st.session_state["research_launch_status"] = "environment_passed"
                research_id = str(uuid.uuid4())
                st.session_state["research_launch_id"] = research_id
                st.session_state["research_launch_engine"] = engine_choice
                st.session_state["research_launch_mode"] = research_mode
                st.session_state["research_launch_source_research_count"] = len(
                    {
                        str(item.get("origin_research_id") or item.get("research_id"))
                        for item in preview
                        if item.get("origin_research_id") or item.get("research_id")
                    }
                )
                st.session_state["research_launch_reuse_candidate_count"] = len(preview)
                st.session_state["research_launch_seed"] = effective_config.seed
                st.session_state["research_launch_started_at"] = datetime.now(UTC).isoformat()
                assert duration_seconds is not None
                st.session_state["research_target_duration_seconds"] = float(duration_seconds)
                created_at = datetime.now(UTC).isoformat()
                optimization_origin = None
                if optimization_context:
                    optimization_origin = optimization_origin_record(
                        study={
                                    "optuna_study_id": optimization_context.get(
                                        "optuna_study_id"
                                    ),
                            "study_name": optimization_context["study_name"],
                            "best_trial_number": optimization_context["trial_number"],
                            "feature_profile": optimization_context["feature_profile"],
                            "dataset_range": optimization_context["dataset_range"],
                        },
                        parameters=optimization_context["parameters"],
                        research_settings=current_research_settings,
                        applied_at=optimization_context["applied_at"],
                        modified=optimization_context.get("modified_by_user", False),
                    )
                research_round = rounds.ensure_round(
                    research_id,
                    {
                        "created_at": created_at,
                        "status": "queued",
                        "checkpoint_path": str(checkpoint_path.resolve()),
                        "dataset_sha256": dataset_digest(new_dataset),
                        "dataset_snapshot_sha256": current_dataset_snapshot,
                        "dataset_hash_algorithm": (
                            current_dataset_hashes["algorithm_version"]
                            if current_dataset_hashes
                            else None
                        ),
                        "dataset_source_kind": new_dataset.source_kind,
                        "dataset_verification_status": new_dataset.verification_status,
                        "dataset_round_range": [
                            new_dataset.records[0].round_number,
                            new_dataset.records[-1].round_number,
                        ],
                        "config": asdict(effective_config),
                        "engines": list(engines),
                        "research_mode": research_mode,
                        "research_strategy": {
                            "mode": research_mode,
                            "elite_seeding_enabled": effective_config.elite_seeding_enabled,
                            "elite_seed_ratio": effective_config.elite_seed_ratio,
                            "selected_pattern_ids": [
                                str(item["pattern_id"]) for item in preview
                            ],
                            "source_research_ids": sorted(
                                {
                                    str(item.get("origin_research_id") or item.get("research_id"))
                                    for item in preview
                                    if item.get("origin_research_id") or item.get("research_id")
                                }
                            ),
                            "candidate_count": len(preview),
                            "seed_source": (
                                "pattern_library_validation_ranked"
                                if effective_config.elite_seeding_enabled
                                else "random_initialization"
                            ),
                            "optuna_origin": optimization_origin,
                        },
                        "pysr_config": asdict(pysr_config) if pysr_config is not None else None,
                        "optimization_origin": optimization_origin | {"state": "queued"}
                        if optimization_origin
                        else None,
                        "completed_runs": 0,
                        "target_duration_seconds": float(duration_seconds),
                        "completed_experiment_ids": [],
                    },
                    checkpoint_dir,
                    project_path("experiments"),
                )
                command = _research_command(
                    csv_path=csv_path,
                    checkpoint_path=checkpoint_path,
                    config=effective_config,
                    duration_minutes=float(duration_minutes),
                    engines=engines,
                    pysr_config=pysr_config,
                    research_id=research_id,
                    candidate_replay_threshold=int(candidate_replay_threshold),
                    research_mode=research_mode,
                    optimization_origin=optimization_origin,
                )
                try:
                    st.session_state["research_launch_status"] = "background_starting"
                    with st.status(
                        "연구 작업을 시작하고 있습니다.", expanded=True
                    ) as start_status:
                        st.write("연구 설정과 최적화 출처를 저장했습니다.")
                        st.write("백그라운드 연구 프로세스를 시작하는 중입니다.")
                        worker = _launch_worker(
                            command, checkpoint_path, checkpoint_dir / "research_worker.log"
                        )
                        pid = int(worker.pid)
                        st.session_state["monitor_research_checkpoint"] = str(checkpoint_path)
                        st.session_state["research_launch_pid"] = pid
                        st.session_state["research_launch_status"] = "process_verifying"
                        started = False
                        checkpoint_state = None
                        for _ in range(20):
                            if _worker_active(checkpoint_path):
                                started = True
                                break
                            checkpoint_state = _launch_checkpoint_state(
                                checkpoint_path, research_id
                            )
                            if checkpoint_state is not None:
                                started = True
                                break
                            if worker.poll() is not None:
                                break
                            time.sleep(0.1)
                        if started:
                            persisted_status = (
                                checkpoint_state.get("status") if checkpoint_state else None
                            )
                            if checkpoint_state:
                                rounds.update(research_id, _round_metadata(checkpoint_state))
                                if persisted_status in {
                                    "running",
                                    "completed",
                                    "failed",
                                    "paused",
                                    "interrupted",
                                }:
                                    st.session_state["research_launch_status"] = persisted_status
                            else:
                                st.session_state["research_launch_status"] = (
                                    "first_checkpoint_waiting"
                                )
                                started_at = datetime.now(UTC).isoformat()
                                origin = None
                                if optimization_context:
                                    origin = optimization_origin_record(
                                        study={
                                            "optuna_study_id": optimization_context.get(
                                                "optuna_study_id"
                                            ),
                                            "study_name": optimization_context["study_name"],
                                            "best_trial_number": optimization_context[
                                                "trial_number"
                                            ],
                                            "feature_profile": optimization_context[
                                                "feature_profile"
                                            ],
                                            "dataset_range": optimization_context[
                                                "dataset_range"
                                            ],
                                        },
                                        parameters=optimization_context["parameters"],
                                        research_settings=current_research_settings,
                                        applied_at=optimization_context["applied_at"],
                                        modified=optimization_context.get(
                                            "modified_by_user", False
                                        ),
                                    )
                                    origin.update(
                                        {
                                            "state": "running",
                                            "started_at": started_at,
                                            "research_id": research_id,
                                        }
                                    )
                                rounds.update(
                                    research_id,
                                    {
                                        "status": "running",
                                        "worker_pid": pid,
                                        "started_at": started_at,
                                        "optimization_origin": origin,
                                    },
                                )
                                persisted_status = "running"
                            if optimization_context and persisted_status == "running":
                                started_ids = optimization_context.setdefault(
                                    "started_research_ids", []
                                )
                                if research_id not in started_ids:
                                    started_ids.append(research_id)
                                st.session_state["optimization_applied_context"] = (
                                    optimization_context
                                )
                            if persisted_status in {
                                "completed",
                                "failed",
                                "paused",
                                "interrupted",
                            }:
                                st.session_state["research_launch_status"] = persisted_status
                                start_status.update(
                                    label=f"워커 종료 상태 확인: {persisted_status}",
                                    state="complete",
                                )
                                st.info(
                                    f"연구 ID `{research_id}`의 체크포인트에서 "
                                    f"최종 상태 {persisted_status}를 확인했습니다."
                                )
                            else:
                                start_status.update(label="연구 실행 중", state="complete")
                                st.success(
                                    f"연구가 실행 중입니다 · ID `{research_id}` · PID {pid} · "
                                    f"{engine_choice} · {profile_names[profile]} · "
                                    f"{duration_minutes:,}분"
                                )
                        else:
                            exit_code = worker.poll()
                            if exit_code is not None:
                                _mark_worker_start_failed(
                                    rounds,
                                    research_id,
                                    pid,
                                    int(exit_code),
                                    checkpoint_dir / "research_worker.log",
                                )
                                st.session_state["research_launch_status"] = "start_failed"
                                start_status.update(
                                    label="연구 워커가 체크포인트 생성 전에 종료됨",
                                    state="error",
                                )
                                st.error(
                                    f"연구 워커가 체크포인트를 쓰기 전에 종료했습니다 "
                                    f"(exit code {exit_code}). worker 로그를 확인하세요."
                                )
                            else:
                                st.session_state["research_launch_status"] = "start_unconfirmed"
                                start_status.update(
                                    label="실행 요청 전달됨 · 시작 확인 중", state="running"
                                )
                                st.info(
                                    f"실행 요청을 전달했습니다 "
                                    f"(PID {pid}, 연구 ID `{research_id}`). "
                                    "작업 프로세스의 실제 시작은 아직 확인되지 않았습니다. "
                                    "잠시 후 상태를 확인하세요."
                                )
                except (OSError, RuntimeError) as error:
                    st.session_state["research_launch_status"] = "start_failed"
                    rounds.update(
                        research_id,
                        {
                            "status": "failed",
                            "failed_run_summary": [
                                {
                                    "error_type": type(error).__name__,
                                    "error": str(error),
                                    "failed_at": datetime.now(UTC).isoformat(),
                                }
                            ],
                        },
                    )
                    st.error(f"연구 워커를 시작하지 못했습니다: {error}")

with resume_tab:
    resume_items = resume_records
    if not resume_items:
        st.info("이어 실행할 수 있는 paused, failed, interrupted 연구가 없습니다.")
        selected_resume = None
        st.selectbox("재개할 연구", ["재개 가능한 연구 없음"], disabled=True)
        st.button("이어서 하기", type="primary", disabled=True, key="resume_research_empty")
    else:
        labels = {
            item["research_id"]: (
                f"{item.get('research_round', '?')}회차 · {item['research_id']} · "
                f"{item.get('status')} · 완료 {item.get('completed_runs', 0)}"
            )
            for item in resume_items
        }
        selected_id = st.selectbox(
            "재개할 연구",
            [item["research_id"] for item in resume_items],
            format_func=lambda value: labels[value],
            key="resume_research_id",
        )
        selected_resume = next(item for item in resume_items if item["research_id"] == selected_id)
        state = selected_resume
        path = selected_resume["_path"]
        active = bool(selected_resume.get("_worker_active"))
        status = state.get("status")
        display_status = (
            "interrupted (워커 미실행)"
            if status in {"running", "pause_requested"} and not active
            else status
        )
        elapsed = float(state.get("total_elapsed_seconds", state.get("elapsed_seconds", 0)))
        target = float(state.get("target_duration_seconds", 0))
        remaining = max(0.0, target - elapsed) if target else None
        failed_seeds = list(
            dict.fromkeys(
                int(item["seed"])
                for item in state.get("failed_runs", [])
                if item.get("seed") is not None
            )
        )
        st.subheader(f"연구 {state.get('research_round', '?')}회차 · {display_status}")
        col1, col2, col3 = st.columns(3)
        col1.metric("완료 실험", int(state.get("completed_runs", 0)))
        col2.metric("실패 기록", len(state.get("failed_runs", [])))
        col3.metric("누적 시간", f"{elapsed / 60:.1f}분")
        st.write(f"연구 ID: `{state['research_id']}`")
        st.write(f"특징 프로파일: `{state.get('config', {}).get('feature_profile', 'legacy')}`")
        st.write(f"복구 대기 시드: {failed_seeds or '없음'}")
        st.write(
            f"남은 시간: {remaining / 60:.1f}분" if remaining is not None else "남은 시간: 미수집"
        )
        st.write(f"체크포인트: `{path}`")
        st.write(f"마지막 저장: {state.get('last_updated_at', '미수집')}")
        st.write(f"마지막 오류: {(state.get('failed_runs') or [{}])[-1].get('error', '없음')}")
        st.write(f"적용 설정: `{json.dumps(state.get('config', {}), ensure_ascii=False)}`")
        if selected_resume.get("_error"):
            st.error(f"재개 검증 실패: {selected_resume['_error']}")
        config_payload = {**state.get("config", {})}
        try:
            resume_config = IntermittentConfig(**config_payload)
        except (TypeError, ValueError):
            resume_config = None
        saved_pysr_config = state.get("pysr_config")
        if isinstance(saved_pysr_config, dict):
            saved_pysr_config = dict(saved_pysr_config)
            for key in ("binary_operators", "unary_operators", "guesses"):
                if key in saved_pysr_config:
                    saved_pysr_config[key] = tuple(saved_pysr_config[key])
            try:
                resume_pysr_config = PySRConfig(**saved_pysr_config)
            except (TypeError, ValueError):
                resume_pysr_config = None
        else:
            resume_pysr_config = None
        if "pysr_symbolic_regression" in state.get("engines", []) and resume_pysr_config is None:
            st.error("저장된 PySR 설정을 복원할 수 없습니다.")
        if active:
            st.warning("같은 연구의 워커가 실행 중이므로 중복 재개를 막았습니다.")
        launch_path = Path(f"{path}.launch")
        launch_pending = launch_path.is_file()
        can_resume = (
            not active
            and not launch_pending
            and not selected_resume.get("_error")
            and selected_resume.get("_data_path") is not None
            and resume_config is not None
            and (
                "pysr_symbolic_regression" not in state.get("engines", [])
                or resume_pysr_config is not None
            )
            and status != "completed"
            and remaining is not None
            and remaining > 0
        )
        if launch_pending:
            st.info("연구 워커 시작을 기다리는 중입니다. 잠시 후 새로고침하세요.")
        if st.button("이어서 하기", type="primary", disabled=not can_resume, key="resume_research"):
            if status in {"running", "pause_requested"}:
                state["status"] = "interrupted"
                state["interruption_reason"] = (
                    "No active worker lease found during explicit resume."
                )
                state["last_updated_at"] = datetime.now(UTC).isoformat()
                from lotto_pattern_lab.research import _atomic_json

                _atomic_json(path, state)
                rounds.update(state["research_id"], _round_metadata(state))
            command = _research_command(
                csv_path=selected_resume["_data_path"],
                checkpoint_path=path,
                config=resume_config,
                duration_minutes=remaining / 60,
                engines=tuple(state.get("engines", ("intermittent_gp",))),
                pysr_config=resume_pysr_config,
                resume=True,
            )
            try:
                worker = _launch_worker(command, path, checkpoint_dir / "research_worker.log")
                pid = int(worker.pid)
                st.session_state["monitor_research_checkpoint"] = str(path)
                st.success(f"연구 {state.get('research_round')}회차를 재개했습니다 (PID {pid}).")
                st.rerun()
            except (OSError, RuntimeError) as error:
                st.error(f"연구를 재개하지 못했습니다: {error}")

st.divider()
st.subheader("패턴 라이브러리")
try:
    patterns = PatternLibrary().list_patterns()
except sqlite3.OperationalError as error:
    log_database_failure(
        "continuous research pattern listing", "data/database/patterns.sqlite", error
    )
    if is_database_locked(error):
        st.error("수식 라이브러리가 다른 저장 작업에서 사용 중입니다. 잠시 후 다시 불러오세요.")
    else:
        st.error("수식 라이브러리를 불러오지 못했습니다.")
    st.code(str(error), language="text")
    if st.button("수식 라이브러리 다시 시도", key="retry_pattern_library_listing"):
        st.rerun()
    st.stop()
if patterns:
    selected_profile = st.selectbox(
        "프로파일 필터",
        ("전체", "legacy", "calendar_round", "bonus_lag_v1"),
        key="research_pattern_profile_filter",
    )
    visible = [
        item
        for item in patterns
        if selected_profile == "전체" or item.get("feature_profile", "legacy") == selected_profile
    ]
    st.dataframe(
        [
            {
                "pattern_id": str(item.get("pattern_id", "")),
                "model_type": str(item.get("model_type", "")),
                "feature_profile": str(item.get("feature_profile", "legacy")),
                "formulas": "; ".join(item.get("formulas", [])),
                "validation_score": item.get("performance", {}).get("validation_score"),
            }
            for item in visible
        ],
        hide_index=True,
        width="stretch",
    )
else:
    st.info("패턴 라이브러리에 등록된 패턴이 없습니다.")


@st.fragment(run_every="5s")
def render_queue_summary(queue_id: str) -> None:
    store = ResearchQueueStore(QUEUE_DB)
    queue = store.queue(queue_id)
    items = store.list_items(queue_id)
    current = store.current_item(queue_id)
    waiting_items = [item for item in items if item["status"] == "waiting"]
    completed_count = sum(item["status"] == "completed" for item in items)
    total_seconds = sum(
        item["duration_seconds"] for item in items if item["status"] not in {"cancelled", "skipped"}
    )
    elapsed = 0.0
    if current and current.get("checkpoint_path"):
        try:
            checkpoint = load_checkpoint(current["checkpoint_path"])
            elapsed = float(checkpoint.get("total_elapsed_seconds", 0))
            round_label = current.get("research_round") or "\ud68c\ucc28 \ud560\ub2f9 \uc911"
            status_label = queue_status_labels.get(
                checkpoint.get("status", current["status"]), current["status"]
            )
            st.caption(
                f"\uc5f0\uad6c {round_label}\ud68c\ucc28 \u00b7 "
                f"\uc0c1\ud0dc {status_label} \u00b7 "
                f"\uc644\ub8cc \uc2e4\ud5d8 "
                f"{len(checkpoint.get('completed_experiments', []))} \u00b7 "
                f"\uc2e4\ud328 {len(checkpoint.get('failed_experiments', []))} \u00b7 "
                f"\ub204\uc801 {format_duration(elapsed)} \u00b7 "
                f"{current['checkpoint_path']}"
            )
        except (OSError, ValueError, KeyError) as error:
            st.warning(
                f"\uccb4\ud06c\ud3ec\uc778\ud2b8 \uc0c1\ud0dc \ud655\uc778 \ubd88\uac00: {error}"
            )
    active_remaining = 0
    if current and current["status"] == "running":
        active_remaining = max(0, int(current["duration_seconds"] - elapsed))
    expected_remaining = sum(item["duration_seconds"] for item in waiting_items) + active_remaining
    columns = st.columns(7)
    columns[0].metric(
        "\uc608\uc57d \uc0c1\ud0dc", queue_status_labels.get(queue["status"], queue["status"])
    )
    columns[1].metric(
        "\ud604\uc7ac \uc608\uc57d", current["position"] + 1 if current else "\uc5c6\uc74c"
    )
    columns[2].metric("\ub300\uae30 \uc608\uc57d", len(waiting_items))
    columns[3].metric("\uc804\uccb4 \uc608\uc57d", len(items))
    columns[4].metric("\uc644\ub8cc \uc608\uc57d", completed_count)
    columns[5].metric(
        "\uc608\uc57d\ub41c \ucd1d \uc5f0\uad6c \uc2dc\uac04", format_duration(total_seconds)
    )
    columns[6].metric("\uc608\uc0c1 \ub0a8\uc740 \uc2dc\uac04", format_duration(expected_remaining))
    if queue.get("last_error"):
        st.error(f"\ub9c8\uc9c0\ub9c9 \uc624\ub958: {queue['last_error']}")
    st.caption(
        "\uc608\uc57d\uc740 SQLite\uc5d0 \uc800\uc7a5\ub418\uace0 "
        "\ubc31\uadf8\ub77c\uc6b4\ub4dc \uc2a4\ucf00\uc904\ub7ec\uac00 \ud558\ub098\uc529 "
        "\uc2e4\ud589\ud569\ub2c8\ub2e4. PC\uac00 \uc808\uc804 \ub610\ub294 "
        "\uc885\ub8cc\ub41c \ub3d9\uc548\uc5d0\ub294 \uc5f0\uad6c\uac00 "
        "\uc9c4\ud589\ub418\uc9c0 \uc54a\uc2b5\ub2c8\ub2e4."
    )


with queue_tab:
    queue_status_labels = {
        "waiting": "\ub300\uae30 \uc911",
        "running": "\uc2e4\ud589 \uc911",
        "pause_requested": "\uc77c\uc2dc\uc815\uc9c0 \uc694\uccad \uc911",
        "paused": "\uc77c\uc2dc\uc815\uc9c0",
        "completed": "\uc644\ub8cc",
        "failed": "\uc2e4\ud328",
        "interrupted": "\uc911\ub2e8\ub428",
    }
    st.subheader("연구 예약 대기열")
    queue_store = ResearchQueueStore(QUEUE_DB)
    queue_rows = queue_store.list_queues()
    if not queue_rows:
        active_queue_id = queue_store.create_queue()
        queue_rows = queue_store.list_queues()
    else:
        active_queue_id = st.selectbox(
            "예약 대기열",
            [item["queue_id"] for item in queue_rows],
            format_func=lambda value: next(
                f"{item['name']} · {queue_status_labels.get(item['status'], item['status'])} · "
                f"{item['created_at'][:10]}"
                for item in queue_rows
                if item["queue_id"] == value
            ),
            key="research_queue_selected",
        )
    if st.button("새 예약 대기열 만들기", key="create_research_queue"):
        queue_store.create_queue()
        st.rerun()
    queue_state = queue_store.queue(active_queue_id)
    queue_items = queue_store.list_items(active_queue_id)
    current_queue_item = queue_store.current_item(active_queue_id)
    render_queue_summary(active_queue_id)

    with st.form("queue_add_form"):
        st.markdown("#### 예약 추가")
        engine_choice = st.selectbox("탐색 엔진", ["GP", "PySR", "GP + PySR"], key="queue_engine")
        profile_choice = st.selectbox(
            "특징 프로파일",
            ["legacy", "calendar_round", "bonus_lag_v1"],
            format_func=lambda value: {
                "legacy": "기본 특징",
                "calendar_round": "달력·회차 특징",
                "bonus_lag_v1": "보너 번호 포함 특징",
            }[value],
            key="queue_profile",
        )
        duration_minutes = st.selectbox(
            "연구 시간",
            [30, 60, 120, 180, 360, 480],
            format_func=lambda value: f"{value}분",
            key="queue_duration",
        )
        pysr_preset = {
            30: (20, 24),
            60: (30, 32),
            120: (50, 40),
            180: (65, 48),
            360: (100, 64),
            480: (140, 80),
        }[int(duration_minutes)]
        advanced = st.checkbox("고급 설정", key="queue_advanced")
        continue_after_error = st.checkbox(
            "\uace0\uae09: \uc2e4\ud328 \uc2dc \ub2e4\uc74c \uc608\uc57d \uc9c4\ud589",
            value=False,
            help=(
                "\uc2dc\uc2a4\ud15c \ubb34\uacb0\uc131 \uc624\ub958\ub294 "
                "\uc790\ub3d9 \uc9c4\ud589\ud558\uc9c0 \uc54a\uc2b5\ub2c8\ub2e4."
            ),
            disabled=not advanced,
            key="queue_continue_after_error",
        )
        seed = st.number_input("난수 시드", 0, 2**31 - 1, 12345, key="queue_seed")
        elite = st.checkbox("Elite Seeding 사용", value=True, key="queue_elite")
        workers = st.number_input("CPU 워커", 1, 4, 2, key="queue_workers")
        controls = st.number_input("무작위 대조 반복", 0, 100, 3, key="queue_controls")
        population = st.number_input(
            "GP 개체군",
            8,
            512,
            GP_PRESETS[duration_minutes]["population"],
            disabled=not advanced,
            key="queue_population",
        )
        generations = st.number_input(
            "GP 세대",
            1,
            500,
            GP_PRESETS[duration_minutes]["generations"],
            disabled=not advanced,
            key="queue_generations",
        )
        max_depth = st.number_input(
            "GP 최대 깊이",
            2,
            20,
            GP_PRESETS[duration_minutes]["max_depth"],
            disabled=not advanced,
            key="queue_depth",
        )
        pysr_target = st.selectbox(
            "PySR 타겟",
            ["main_positions", "one_number", "bonus"],
            disabled=engine_choice == "GP",
            key="queue_pysr_target",
        )
        pysr_iterations = st.number_input(
            "PySR \uc138\ub300 \uc218",
            1,
            1000,
            pysr_preset[0],
            disabled=engine_choice == "GP" or not advanced,
            key="queue_pysr_iterations",
        )
        pysr_population = st.number_input(
            "PySR \uac1c\uccb4\uad70",
            8,
            512,
            pysr_preset[1],
            disabled=engine_choice == "GP" or not advanced,
            key="queue_pysr_population",
        )
        pysr_maxsize = st.number_input(
            "PySR \uc218\uc2dd \ud06c\uae30 \ud55c\ub3c4",
            4,
            64,
            24,
            disabled=engine_choice == "GP" or not advanced,
            key="queue_pysr_maxsize",
        )
        queue_add = st.form_submit_button("\uc608\uc57d \ucd94\uac00")
    if queue_add:
        engines = {
            "GP": ["intermittent_gp"],
            "PySR": ["pysr_symbolic_regression"],
            "GP + PySR": ["intermittent_gp", "pysr_symbolic_regression"],
        }[engine_choice]
        config_snapshot = {
            "engines": engines,
            "seed": int(seed),
            "elite_seeding": bool(elite),
            "elite_seed_ratio": 0.2,
            "workers": min(int(workers), 3) if engine_choice == "GP + PySR" else int(workers),
            "control_runs": int(controls),
            "population_size": int(population),
            "generations": int(generations),
            "max_depth": int(max_depth),
            "crossover_probability": 0.65,
            "mutation_probability": 0.25,
            "train_ratio": 0.6,
            "validation_ratio": 0.2,
            "max_lag": 10,
            "min_activation_rate": 0.1,
            "hidden_states": 3,
            "pysr": {
                "target": pysr_target,
                "workers": 1,
                "niterations": int(pysr_iterations),
                "population_size": int(pysr_population),
                "maxsize": int(pysr_maxsize),
                "timeout_seconds": 60.0,
            },
        }
        try:
            if (
                "pysr_symbolic_regression" in engines
                and pysr_target == "bonus"
                and profile_choice == "legacy"
            ):
                raise ValueError(
                    "\ubcf4\ub108 \ub300\uc0c1 PySR\uc740 calendar_round \ub610\ub294 "
                    "bonus_lag_v1 \ud2b9\uc9d5\uc774 \ud544\uc694\ud569\ub2c8\ub2e4."
                )
            queue_dataset = load_csv(project_path("data/official/draws.csv"))
            gp_preflight_config = IntermittentConfig(
                population_size=int(population),
                generations=int(generations),
                max_depth=int(max_depth),
                workers=int(config_snapshot["workers"]),
                control_runs=int(controls),
                feature_profile=profile_choice,
                seed=int(seed),
            )
            pysr_preflight_config = PySRConfig(
                target=pysr_target,
                feature_profile=profile_choice,
                workers=1,
                niterations=int(pysr_iterations),
                population_size=int(pysr_population),
                maxsize=int(pysr_maxsize),
                seed=int(seed),
                random_control_replicates=int(controls),
            )
            preflight = run_research_preflight(
                queue_dataset,
                gp_preflight_config,
                tuple(engines),
                pysr_preflight_config if "pysr_symbolic_regression" in engines else None,
            )
            if preflight["status"] != "passed":
                raise ValueError(f"\uc0ac\uc804 \uac80\uc0ac \uc2e4\ud328: {preflight['engines']}")
            queue_store.add_item(
                active_queue_id,
                engine={
                    "GP": "intermittent_gp",
                    "PySR": "pysr_symbolic_regression",
                    "GP + PySR": "comparison",
                }[engine_choice],
                feature_profile=profile_choice,
                duration_seconds=int(duration_minutes) * 60,
                config=config_snapshot,
            )
            queue_store.set_error_policy(
                active_queue_id, "continue" if continue_after_error else "stop"
            )
            st.toast("연구 예약을 대기열에 추가했습니다.", icon="✅")
            st.rerun()
        except (OSError, ValueError) as error:
            st.error(f"예약을 추가하지 못했습니다: {error}")

    if queue_items:
        rows = []
        for item in queue_items:
            rows.append(
                {
                    "\uc21c\uc11c": item["position"],
                    "\uc608\uc57d ID": item["item_id"][:8],
                    "\uc5d4\uc9c4": {
                        "intermittent_gp": "GP",
                        "pysr_symbolic_regression": "PySR",
                        "comparison": "GP + PySR",
                    }[item["engine"]],
                    "\ud2b9\uc9d5 \ud504\ub85c\ud30c\uc77c": {
                        "legacy": "\uae30\ubcf8 \ud2b9\uc9d5",
                        "calendar_round": "\ub2ec\ub825\u00b7\ud68c\ucc28 \ud2b9\uc9d5",
                        "bonus_lag_v1": "\ubcf4\ub108 \ubc88\ud638 \ud3ec\ud568 \ud2b9\uc9d5",
                    }[item["feature_profile"]],
                    "\uc5f0\uad6c \uc2dc\uac04": format_duration(item["duration_seconds"]),
                    "\uc2dc\ub4dc": item["config"].get("seed"),
                    "\uc0c1\ud0dc": {
                        "waiting": "\ub300\uae30 \uc911",
                        "running": "\uc2e4\ud589 \uc911",
                        "paused": "\uc77c\uc2dc\uc815\uc9c0",
                        "completed": "\uc644\ub8cc",
                        "failed": "\uc2e4\ud328",
                        "cancelled": "\ucde8\uc18c",
                        "interrupted": "\uc911\ub2e8\ub428",
                        "skipped": "\uac74\ub108\ub700",
                    }[item["status"]],
                    "\uc5f0\uad6c \ud68c\ucc28": item["research_round"] or "\ubbf8\ubc30\uc815",
                    "\uc644\ub8cc \uc2e4\ud5d8": (
                        len(
                            load_checkpoint(item["checkpoint_path"]).get(
                                "completed_experiments", []
                            )
                        )
                        if item.get("checkpoint_path") and Path(item["checkpoint_path"]).is_file()
                        else 0
                    ),
                    "\uc624\ub958": item["last_error"] or "",
                }
            )
        st.dataframe(rows, hide_index=True, width="stretch")
        manageable = [item for item in queue_items if item["status"] == "waiting"]
        if manageable:
            selected_item_id = st.selectbox(
                "대기 예약 선택",
                [item["item_id"] for item in manageable],
                format_func=lambda value: next(
                    f"{item['position']}번 · {item['engine']} · {item['duration_seconds'] // 60}분"
                    for item in manageable
                    if item["item_id"] == value
                ),
                key="queue_manage_item",
            )
            manage_columns = st.columns(5)
            actions = [
                ("맨 앞으로", "first"),
                ("위로", "up"),
                ("아래로", "down"),
                ("맨 뒤로", "last"),
            ]
            for column, (label, direction) in zip(manage_columns[:4], actions, strict=True):
                if column.button(label, key=f"queue_move_{direction}"):
                    queue_store.move(selected_item_id, direction)
                    st.rerun()
            if manage_columns[4].button(
                "예약 취소",
                key="queue_cancel_selected",
                help="대기 중인 예약을 취소합니다. 실행 중인 연구에는 영향을 주지 않습니다.",
            ):
                queue_store.cancel_waiting(selected_item_id)
                st.rerun()
            with st.expander("선택한 대기 예약 수정"):
                edit_item = next(item for item in manageable if item["item_id"] == selected_item_id)
                engine_values = ["intermittent_gp", "pysr_symbolic_regression", "comparison"]
                edit_engine = st.selectbox(
                    "\uc218\uc815 \ud0d0\uc0c9 \uc5d4\uc9c4",
                    engine_values,
                    index=engine_values.index(edit_item["engine"]),
                    format_func=lambda value: {
                        "intermittent_gp": "GP",
                        "pysr_symbolic_regression": "PySR",
                        "comparison": "GP + PySR",
                    }[value],
                    key="queue_edit_engine",
                )
                profile_values = ["legacy", "calendar_round", "bonus_lag_v1"]
                edit_profile = st.selectbox(
                    "\uc218\uc815 \ud2b9\uc9d5 \ud504\ub85c\ud30c\uc77c",
                    profile_values,
                    index=profile_values.index(edit_item["feature_profile"]),
                    key="queue_edit_profile",
                )
                edit_duration = st.selectbox(
                    "수정 연구 시간",
                    [30, 60, 120, 180, 360, 480],
                    index=[30, 60, 120, 180, 360, 480].index(edit_item["duration_seconds"] // 60),
                    key="queue_edit_duration",
                )
                edit_seed = st.number_input(
                    "수정 난수 시드",
                    0,
                    2**31 - 1,
                    int(edit_item["config"].get("seed", 12345)),
                    key="queue_edit_seed",
                )
                edit_workers = st.number_input(
                    "CPU \uc6cc\ucee4",
                    1,
                    4,
                    int(edit_item["config"].get("workers", 2)),
                    key="queue_edit_workers",
                )
                edit_controls = st.number_input(
                    "\ubb34\uc791\uc704 \ub300\uc870 \ubc18\ubcf5",
                    0,
                    100,
                    int(edit_item["config"].get("control_runs", 3)),
                    key="queue_edit_controls",
                )
                edit_elite = st.checkbox(
                    "Elite Seeding \uc0ac\uc6a9",
                    value=bool(edit_item["config"].get("elite_seeding", True)),
                    key="queue_edit_elite",
                )
                edit_population = st.number_input(
                    "GP \uac1c\uccb4\uad70",
                    8,
                    512,
                    int(edit_item["config"].get("population_size", 48)),
                    key="queue_edit_population",
                )
                edit_generations = st.number_input(
                    "GP \uc138\ub300",
                    1,
                    500,
                    int(edit_item["config"].get("generations", 15)),
                    key="queue_edit_generations",
                )
                edit_depth = st.number_input(
                    "GP \ucd5c\ub300 \uae4a\uc774",
                    2,
                    20,
                    int(edit_item["config"].get("max_depth", 5)),
                    key="queue_edit_depth",
                )
                saved_pysr = edit_item["config"].get("pysr", {})
                edit_pysr_target = st.selectbox(
                    "PySR \ud0c0\uac9f",
                    ["main_positions", "one_number", "bonus"],
                    index=["main_positions", "one_number", "bonus"].index(
                        saved_pysr.get("target", "main_positions")
                    ),
                    disabled=edit_engine == "intermittent_gp",
                    key="queue_edit_pysr_target",
                )
                edit_pysr_iterations = st.number_input(
                    "PySR \uc138\ub300 \uc218",
                    1,
                    1000,
                    int(saved_pysr.get("niterations", 20)),
                    disabled=edit_engine == "intermittent_gp",
                    key="queue_edit_pysr_iterations",
                )
                edit_pysr_population = st.number_input(
                    "PySR \uac1c\uccb4\uad70",
                    8,
                    512,
                    int(saved_pysr.get("population_size", 32)),
                    disabled=edit_engine == "intermittent_gp",
                    key="queue_edit_pysr_population",
                )
                if st.button("대기 예약 설정 저장", key="queue_save_edit"):
                    new_engines = {
                        "intermittent_gp": ["intermittent_gp"],
                        "pysr_symbolic_regression": ["pysr_symbolic_regression"],
                        "comparison": ["intermittent_gp", "pysr_symbolic_regression"],
                    }[edit_engine]
                    new_config = {
                        **edit_item["config"],
                        "seed": int(edit_seed),
                        "engines": new_engines,
                        "workers": min(int(edit_workers), 3)
                        if edit_engine == "comparison"
                        else int(edit_workers),
                        "control_runs": int(edit_controls),
                        "elite_seeding": bool(edit_elite),
                        "population_size": int(edit_population),
                        "generations": int(edit_generations),
                        "max_depth": int(edit_depth),
                        "pysr": {
                            **edit_item["config"].get("pysr", {}),
                            "target": edit_pysr_target,
                            "niterations": int(edit_pysr_iterations),
                            "population_size": int(edit_pysr_population),
                            "workers": 1,
                        },
                    }
                    try:
                        edit_dataset = load_csv(project_path("data/official/draws.csv"))
                        edit_gp_config = IntermittentConfig(
                            population_size=int(new_config.get("population_size", 48)),
                            generations=int(new_config.get("generations", 15)),
                            max_depth=int(new_config.get("max_depth", 5)),
                            workers=int(new_config["workers"]),
                            control_runs=int(new_config.get("control_runs", 3)),
                            elite_seeding_enabled=bool(new_config.get("elite_seeding", True)),
                            feature_profile=edit_profile,
                            seed=int(edit_seed),
                        )
                        pysr_values = new_config.get("pysr", {})
                        edit_pysr_config = PySRConfig(
                            target=pysr_values.get("target", "main_positions"),
                            feature_profile=edit_profile,
                            workers=1,
                            niterations=int(pysr_values.get("niterations", 20)),
                            population_size=int(pysr_values.get("population_size", 32)),
                            maxsize=int(pysr_values.get("maxsize", 24)),
                            seed=int(edit_seed),
                        )
                        edit_preflight = run_research_preflight(
                            edit_dataset,
                            edit_gp_config,
                            tuple(new_engines),
                            edit_pysr_config if "pysr_symbolic_regression" in new_engines else None,
                        )
                        if edit_preflight["status"] != "passed":
                            raise ValueError(
                                f"\uc0ac\uc804 \uac80\uc0ac \uc2e4\ud328: "
                                f"{edit_preflight['engines']}"
                            )
                        queue_store.edit_waiting(
                            selected_item_id,
                            engine=edit_engine,
                            feature_profile=edit_profile,
                            duration_seconds=int(edit_duration) * 60,
                            config=new_config,
                        )
                        st.toast("대기 예약 설정을 저장했습니다.", icon="✅")
                        st.rerun()
                    except ValueError as error:
                        st.error(str(error))

    if current_queue_item and current_queue_item.get("status") == "running":
        checkpoint_path = Path(current_queue_item.get("checkpoint_path") or "")
        lease_live = queue_store.lease_is_live(active_queue_id)
        if not lease_live and checkpoint_path.is_file() and not _worker_active(checkpoint_path):
            try:
                state = load_checkpoint(checkpoint_path)
                if state.get("status") in {"running", "pause_requested", "paused", "failed"}:
                    st.warning(
                        "예약 실행이 중단되었고 연구 작업자는 실행 중이 아닙니다. "
                        "체크포인트를 확인한 뒤 이어서 실행할 수 있습니다."
                    )
                    if st.button(
                        "체크포인트 확인 후 이어서 실행",
                        key="queue_recover_interrupted",
                    ):
                        checkpoint_status = state.get("status")
                        if checkpoint_status == "completed":
                            queue_store.finish_item(current_queue_item["item_id"], "completed")
                            queue_store.set_queue_status(active_queue_id, "paused")
                        elif checkpoint_status == "failed":
                            queue_store.finish_item(
                                current_queue_item["item_id"],
                                "failed",
                                error=state.get(
                                    "failure_reason", "\uc5f0\uad6c \uc6cc\ucee4 \uc2e4\ud328"
                                ),
                            )
                            queue_store.set_queue_status(active_queue_id, "failed")
                        else:
                            queue_store.mark_interrupted(current_queue_item["item_id"])
                            queue_store.set_queue_status(active_queue_id, "paused")
                        st.rerun()
            except (OSError, ValueError, KeyError) as error:
                st.error(f"체크포인트 확인에 실패했습니다: {error}")
        elif not lease_live and checkpoint_path.is_file() and _worker_active(checkpoint_path):
            st.info("연구 작업자는 실행 중이지만 예약 실행기의 응답이 지연되고 있습니다.")
            if st.button("예약 실행기 다시 연결", key="queue_scheduler_recover"):
                try:
                    start_detached_scheduler(queue_store, active_queue_id)
                    st.toast(
                        "예약 실행기가 실행 중인 연구 작업자에 다시 연결되었습니다.",
                        icon="\u2705",
                    )
                    st.rerun()
                except (OSError, RuntimeError) as error:
                    st.error(f"예약 실행기 복구에 실패했습니다: {error}")
    failed_items = [item for item in queue_items if item["status"] == "failed"]
    if failed_items:
        failed_id = failed_items[-1]["item_id"]
        failure_cols = st.columns(2)
        if failure_cols[0].button("실패한 예약 건너뛰기", key="queue_skip_failed"):
            queue_store.skip_failed(failed_id)
            st.rerun()
        failed_checkpoint = Path(failed_items[-1].get("checkpoint_path") or "")
        can_recover = failed_checkpoint.is_file() and not _worker_active(failed_checkpoint)
        if failure_cols[1].button(
            "실패한 예약 확인 후 이어서 실행",
            disabled=not can_recover,
            key="queue_retry_failed",
        ):
            try:
                checkpoint_state = load_checkpoint(failed_checkpoint)
                if not checkpoint_state.get("resumable", True):
                    raise ValueError("저장된 체크포인트에서 이어서 실행할 수 없습니다.")
                queue_store.retry_failed(failed_id)
                st.rerun()
            except (OSError, ValueError, KeyError) as error:
                st.error(f"연구를 복구할 수 없습니다: {error}")

    control_columns = st.columns(4)
    if control_columns[0].button(
        "예약 실행 시작",
        type="primary",
        disabled=queue_state["status"] not in {"waiting", "paused"},
        help="백그라운드 스케줄러 heartbeat를 확인한 뒤 시작합니다.",
    ):
        active_workers = [item for item in _resume_records(rounds) if item.get("_worker_active")]
        if active_workers:
            st.error("현재 연구 작업자가 실행 중이므로 다른 작업을 시작할 수 없습니다.")
        else:
            try:
                queue_store.set_queue_status(active_queue_id, "running")
                start_detached_scheduler(queue_store, active_queue_id)
                st.toast(
                    "예약 실행기의 응답을 확인했습니다.",
                    icon="\u2705",
                )
                st.rerun()
            except (OSError, RuntimeError) as error:
                queue_store.set_queue_status(active_queue_id, "failed")
                st.error(f"예약 실행기를 시작하지 못했습니다: {error}")
    if control_columns[1].button(
        "대기열 일시정지 요청",
        disabled=queue_state["status"] != "running",
        help="현재 실험을 안전 경계에서 저장한 뒤 다음 예약 시작을 멈춘니다.",
    ):
        queue_store.set_queue_status(active_queue_id, "pause_requested")
        st.toast("실험이 저장 경계에 도달하면 대기열이 중지됩니다.", icon="⏸️")
        st.rerun()
    if control_columns[2].button(
        "예약 대기열 이어서 실행",
        disabled=queue_state["status"] != "paused",
    ):
        try:
            queue_store.set_queue_status(active_queue_id, "running")
            start_detached_scheduler(queue_store, active_queue_id)
            st.toast("예약 대기열을 다시 실행합니다.", icon="\u2705")
            st.rerun()
        except (OSError, RuntimeError) as error:
            queue_store.set_queue_status(active_queue_id, "paused")
            st.error(f"예약 대기열을 복구할 수 없습니다: {error}")
    if control_columns[3].button(
        "다음 예약 자동 실행 중지",
        disabled=queue_state["status"] != "running",
    ):
        queue_store.set_queue_status(active_queue_id, "pause_requested")
        st.toast("현재 연구 완료 후 다음 예약 자동 실행을 중지합니다.", icon="⏹️")
        st.rerun()
