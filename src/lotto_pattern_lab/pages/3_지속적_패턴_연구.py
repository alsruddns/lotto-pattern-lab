"""Persistent research controls with separate new-run and resume workflows."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from typing import Any

import streamlit as st

from lotto_pattern_lab.atomic_io import LockTimeoutError, interprocess_lock
from lotto_pattern_lab.dataset import dataset_integrity_report, load_csv
from lotto_pattern_lab.feature_schema import feature_schema_version
from lotto_pattern_lab.gp_config import GPConfig
from lotto_pattern_lab.intermittent import IntermittentConfig
from lotto_pattern_lab.paths import PROJECT_ROOT, locate_checkpoint, project_path
from lotto_pattern_lab.pattern_library import PatternLibrary
from lotto_pattern_lab.research import (
    _round_metadata,
    dataset_digest,
    load_checkpoint,
    pause_request_path,
    request_pause,
)
from lotto_pattern_lab.research_rounds import ResearchRoundStore

PAUSE_MESSAGE = (
    "일시정지 요청 완료. 현재 진행 중인 실험의 계산과 저장이 끝난 후 자동으로 일시정지됩니다."
)


def _worker_active(checkpoint_path: Path) -> bool:
    try:
        with interprocess_lock(checkpoint_path, timeout=0, scope="research-worker"):
            return False
    except LockTimeoutError:
        return True


def _dataset_for_checkpoint(state: dict[str, Any]) -> tuple[Path | None, str | None]:
    candidates = []
    if state.get("dataset_path"):
        candidates.append(project_path(state["dataset_path"]))
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


def _launch_worker(command: list[str], checkpoint_path: Path, log_path: Path) -> int:
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
        return int(process.pid)


def _research_command(
    *,
    csv_path: Path,
    checkpoint_path: Path,
    config: IntermittentConfig,
    duration_minutes: float,
    engines: tuple[str, ...] = ("intermittent_gp",),
    resume: bool = False,
    research_id: str | None = None,
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
    ]
    if research_id:
        command.extend(("--research-id", research_id))
    for engine in engines:
        command.extend(("--engine", engine))
    if config.elite_seeding_enabled:
        command.append("--elite-seeding")
    if resume:
        command.append("--resume")
    return command


st.set_page_config(page_title="지속적 패턴 연구", layout="wide")
st.title("지속적 패턴 연구")
st.caption(
    "새 연구는 별도 체크포인트로 시작하고, 저장된 연구는 당시 설정을 유지해 이어서 실행합니다."
)

checkpoint_dir = project_path("checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)
rounds = ResearchRoundStore()
new_tab, resume_tab = st.tabs(("새 연구 시작", "기존 연구 이어서 하기"))

with new_tab:
    defaults = GPConfig()
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

    with st.form("new_research_form"):
        profile = st.selectbox(
            "특징 프로파일", ("legacy", "calendar_round", "bonus_lag_v1"), key="new_profile"
        )
        duration_minutes = st.number_input(
            "총 연구 시간(분)",
            min_value=1,
            max_value=10080,
            value=180,
            step=30,
            key="new_duration_minutes",
        )
        population = st.number_input(
            "개체군",
            min_value=8,
            max_value=512,
            value=defaults.population_size,
            step=8,
            key="new_population_size",
        )
        generations = st.number_input(
            "세대 수", min_value=1, max_value=500, value=defaults.generations, key="new_generations"
        )
        col1, col2, col3 = st.columns(3)
        crossover = col1.slider(
            "교차 확률", 0.0, 1.0, defaults.crossover_probability, key="new_crossover_probability"
        )
        mutation = col2.slider(
            "돌연변이 확률", 0.0, 1.0, defaults.mutation_probability, key="new_mutation_probability"
        )
        max_depth = col3.number_input(
            "최대 수식 깊이", 2, 20, defaults.max_depth, key="new_max_depth"
        )
        workers = st.number_input("CPU 워커", 1, 4, defaults.workers, key="new_workers")
        col1, col2, col3 = st.columns(3)
        train_ratio = col1.slider(
            "학습 비율", 0.1, 0.9, defaults.train_ratio, key="new_train_ratio"
        )
        validation_ratio = col2.slider(
            "검증 비율", 0.05, 0.8, defaults.validation_ratio, key="new_validation_ratio"
        )
        max_lag = col3.number_input("최대 과거 참조", 1, 10, defaults.max_lag, key="new_max_lag")
        col1, col2, col3 = st.columns(3)
        controls = col1.number_input("무작위 대조 반복", 0, 100, 3, key="new_control_runs")
        min_activation = col2.slider(
            "최소 활성화 비율", 0.05, 1.0, 0.1, key="new_min_activation_rate"
        )
        hidden_states = col3.number_input("숨겨진 상태 수", 2, 5, 3, key="new_hidden_states")
        seed = st.number_input("초기 난수 시드", 0, 2**31 - 1, defaults.seed, key="new_seed")
        elite_enabled = st.checkbox(
            "Elite Seeding 사용", value=True, key="new_elite_seeding_enabled"
        )
        elite_ratio = st.slider(
            "기존 패턴 계승 비율", 0.0, 0.8, 0.2, 0.05, key="new_elite_seed_ratio"
        )
        submitted = st.form_submit_button("새 연구 시작", type="primary")

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
    if new_dataset is not None and effective_config is not None:
        invalid_profile = (
            (profile == "calendar_round" and not new_integrity["all_draws_dated"])
            or (profile == "bonus_lag_v1" and not new_integrity["all_draws_have_bonus"])
            or not new_integrity["rounds_contiguous"]
        )
        if invalid_profile:
            st.error("선택한 특징 프로파일에 필요한 데이터 필드 또는 연속 회차가 없습니다.")
        st.subheader("시작 전 적용 설정")
        setting_rows = [
            {"설정": "탐색 엔진", "값": "intermittent_gp"},
            {"설정": "특징 프로파일", "값": profile},
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
        st.dataframe(
            [{"설정": row["설정"], "값": str(row["값"])} for row in setting_rows],
            hide_index=True,
            width="stretch",
        )
        library = PatternLibrary()
        if elite_enabled:
            preview, skipped = library.select_elite_candidates(
                feature_profile=profile,
                max_lag=int(max_lag),
                max_candidates=2 * ceil(int(population) * float(elite_ratio)),
            )
            st.caption(
                f"호환 계승 후보 {len(preview)}개, 호환되지 않아 제외된 후보 {len(skipped)}개"
            )
    else:
        invalid_profile = True

    if submitted:
        if new_dataset is None or effective_config is None or invalid_profile:
            st.error("데이터 또는 설정 검증을 통과하지 못해 연구를 시작하지 않았습니다.")
        else:
            checkpoint_path = checkpoint_dir / (
                f"research_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}.json"
            )
            research_id = str(uuid.uuid4())
            created_at = datetime.now(UTC).isoformat()
            research_round = rounds.ensure_round(
                research_id,
                {
                    "created_at": created_at,
                    "status": "queued",
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "dataset_sha256": dataset_digest(new_dataset),
                    "dataset_source_kind": new_dataset.source_kind,
                    "dataset_verification_status": new_dataset.verification_status,
                    "dataset_round_range": [
                        new_dataset.records[0].round_number,
                        new_dataset.records[-1].round_number,
                    ],
                    "config": asdict(effective_config),
                    "engines": ["intermittent_gp"],
                    "completed_runs": 0,
                    "target_duration_seconds": float(duration_minutes) * 60,
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
                research_id=research_id,
            )
            try:
                pid = _launch_worker(
                    command, checkpoint_path, checkpoint_dir / "research_worker.log"
                )
                st.session_state["monitor_research_checkpoint"] = str(checkpoint_path)
                st.success(
                    f"연구 {research_round}회차를 시작했습니다 (PID {pid}; ID {research_id})."
                )
            except (OSError, RuntimeError) as error:
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
    resume_items = _resume_records(rounds)
    if not resume_items:
        st.info("이어 실행할 수 있는 paused, failed, interrupted 연구가 없습니다.")
        selected_resume = None
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
        except (TypeError, ValueError) as error:
            resume_config = None
            st.error(f"저장된 연구 설정을 복원할 수 없습니다: {error}")
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
                resume=True,
            )
            try:
                pid = _launch_worker(command, path, checkpoint_dir / "research_worker.log")
                st.session_state["monitor_research_checkpoint"] = str(path)
                st.success(f"연구 {state.get('research_round')}회차를 재개했습니다 (PID {pid}).")
                st.rerun()
            except (OSError, RuntimeError) as error:
                st.error(f"연구를 재개하지 못했습니다: {error}")

monitor_path_text = st.session_state.get("monitor_research_checkpoint")
if not monitor_path_text:
    active_item = next((item for item in resume_items if item.get("_worker_active")), None)
    if active_item:
        monitor_path_text = str(active_item["_path"])
if monitor_path_text:
    monitor_path = project_path(monitor_path_text)
    st.divider()
    st.subheader("연구 진행 상태")
    if monitor_path.is_file():
        try:
            monitor = load_checkpoint(monitor_path)
        except (OSError, ValueError) as error:
            st.error(f"체크포인트를 읽지 못했습니다: {error}")
        else:
            elapsed = float(monitor.get("total_elapsed_seconds", monitor.get("elapsed_seconds", 0)))
            target = float(monitor.get("target_duration_seconds", 0))
            remaining = max(0.0, target - elapsed) if target else None
            completed = int(monitor.get("completed_runs", 0))
            attempts = int(monitor.get("attempts", 0))
            cols = st.columns(4)
            cols[0].metric(
                "회차 / 상태", f"{monitor.get('research_round', '?')} / {monitor.get('status')}"
            )
            cols[1].metric(
                "현재 시드",
                monitor.get("in_progress", {}).get(
                    "seed", monitor.get("rng_resume", {}).get("next_seed", "—")
                ),
            )
            cols[2].metric("완료 / 실패", f"{completed} / {len(monitor.get('failed_runs', []))}")
            cols[3].metric(
                "누적 / 남은 시간",
                f"{elapsed / 60:.1f} / {remaining / 60:.1f}분"
                if remaining is not None
                else f"{elapsed / 60:.1f}분",
            )
            progress_event = monitor.get("current_progress", {}).get("event", {})
            st.write(f"현재 단계: {progress_event or '실험 경계 대기'}")
            st.progress(min(1.0, elapsed / target) if target else 0.0)
            st.caption(f"완료 실험 {completed}; 시도 {attempts}")
            st.caption(f"마지막 저장: {monitor.get('last_updated_at', '미수집')}")
            last_error = monitor.get("failed_runs", [])
            st.caption(f"마지막 오류: {last_error[-1].get('error') if last_error else '없음'}")
            pause_pending = pause_request_path(monitor_path).is_file()
            if monitor.get("status") == "running" and not pause_pending:
                if st.button("일시정지", key="request_research_pause"):
                    try:
                        request_pause(monitor_path)
                        st.info(PAUSE_MESSAGE)
                        st.rerun()
                    except (OSError, ValueError, LockTimeoutError) as error:
                        st.error(f"일시정지 요청을 저장하지 못했습니다: {error}")
            elif (
                monitor.get("status") == "pause_requested"
                or pause_request_path(monitor_path).exists()
            ):
                st.info(PAUSE_MESSAGE)
            elif monitor.get("status") == "paused":
                st.success("일시정지되었습니다. ‘기존 연구 이어서 하기’에서 재개할 수 있습니다.")
    else:
        st.info(f"첫 체크포인트 저장을 기다리는 중: {monitor_path}")
    if st.button("상태 새로고침", key="refresh_research_status"):
        st.rerun()

st.divider()
st.subheader("패턴 라이브러리")
patterns = PatternLibrary().list_patterns()
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
