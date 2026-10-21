"""Continuous research and automatic optimization workspace."""

from __future__ import annotations

import csv
import io
import json
import re
import runpy
import sqlite3
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st

from lotto_pattern_lab.optimization_process_state import launch_optimization_worker
from lotto_pattern_lab.paths import PROJECT_ROOT, project_path
from lotto_pattern_lab.research import research_worker_active
from lotto_pattern_lab.research_optimization import (
    estimate_optimization_remaining_seconds,
    format_optimization_duration,
    list_optimization_run_states,
    list_optimization_studies,
    optimization_monitor_snapshot,
    optuna_status,
    request_optimization_stop,
)
from lotto_pattern_lab.research_optimization_apply import (
    OPTUNA_TO_RESEARCH_KEYS,
    validate_optimization_compatibility,
)
from lotto_pattern_lab.sqlite_support import is_database_locked, log_database_failure

st.set_page_config(page_title="지속적 패턴 연구", layout="wide")
_pending_optimization = st.session_state.pop("optimization_apply_pending", None)
if _pending_optimization:
    for _key, _value in _pending_optimization["values"].items():
        st.session_state[_key] = _value
    st.session_state["optimization_applied_context"] = _pending_optimization["context"]
    st.session_state["optimization_applied_notice"] = True
    st.session_state["research_workspace_tab"] = "지속적 패턴 연구"
if st.session_state.pop("optimization_reset_pending", False):
    for _key in (
        "new_research_engine",
        "new_profile",
        "new_duration_minutes",
        "new_duration_mode",
        "new_custom_duration_minutes",
        "new_custom_duration_minutes_value",
        *OPTUNA_TO_RESEARCH_KEYS.values(),
    ):
        st.session_state.pop(_key, None)
    st.session_state.pop("optimization_applied_context", None)
    st.session_state["optimization_reset_notice"] = True
    st.session_state["research_workspace_tab"] = "지속적 패턴 연구"

research_tab, optimization_tab = st.tabs(
    ["지속적 패턴 연구", "연구 조건 자동 최적화"],
    key="research_workspace_tab",
    default=st.session_state.get("research_workspace_tab", "지속적 패턴 연구"),
    on_change="rerun",
)
with research_tab:
    try:
        _continuous_runtime = runpy.run_path(
            str(PROJECT_ROOT / "src/lotto_pattern_lab/continuous_research_view.py"),
            run_name="lotto_pattern_lab_continuous_research_view",
        )
    except sqlite3.OperationalError as error:
        log_database_failure(
            "continuous research page render", "data/database/patterns.sqlite", error
        )
        if is_database_locked(error):
            st.error("연구 데이터베이스가 다른 작업에서 잠겨 있습니다.")
        else:
            st.error("연구 데이터베이스 처리 중 오류가 발생했습니다.")
        st.caption("저장된 연구 상태는 유지됩니다. 상세 오류를 확인한 뒤 다시 시도하세요.")
        st.code(str(error), language="text")
        if st.button("연구 화면 다시 불러오기", key="retry_continuous_page_database"):
            st.rerun()
        st.stop()
globals().update(
    {
        name: _continuous_runtime[name]
        for name in (
            "_worker_active",
            "_launch_worker",
            "format_duration",
            "PROJECT_ROOT",
            "project_path",
        )
    }
)


def _worker_active(checkpoint_path: Path) -> bool:
    return _continuous_runtime["_worker_active"](checkpoint_path)


def _launch_worker(
    command: list[str], checkpoint_path: Path, log_path: Path
) -> subprocess.Popen[bytes]:
    return _continuous_runtime["_launch_worker"](command, checkpoint_path, log_path)


def format_duration(seconds: object) -> str:
    return _continuous_runtime["format_duration"](seconds)


def render_optimization() -> None:
    st.title("연구 조건 자동 최적화")
    apply_failure = st.session_state.pop("optimization_apply_failure", None)
    if apply_failure:
        st.error(f"최적화 설정을 적용하지 못했습니다: {apply_failure}")
    st.caption(
        "여러 연구 조건을 자동으로 시험하여 검증 성능과 수식 복잡도를 고려한 설정을 찾습니다. "
        "점수는 내부 평가값이며 실제 당첨 확률을 뜻하지 않습니다."
    )
    status = optuna_status()
    records = list_optimization_studies()

    monitor_states = list_optimization_run_states()
    monitor_refresh = st.session_state.get("optimization_monitor_start_pending", False) or any(
        _worker_active(
            Path(
                item.get("state_path")
                or project_path(f"data/optimizations/{item['study_name']}.state.json")
            )
        )
        for item in monitor_states
    )

    @st.fragment(run_every="5s" if monitor_refresh else None)
    def optimization_monitor() -> None:
        fresh_records = list_optimization_run_states()
        active = next(
            (
                item
                for item in fresh_records
                if _worker_active(
                    Path(
                        item.get("state_path")
                        or project_path(f"data/optimizations/{item['study_name']}.state.json")
                    )
                )
            ),
            None,
        )
        labels = {
            "running": "최적화 진행 중",
            "stopping": "중지 요청 중 · 현재 Trial 종료 대기",
            "completed": "최적화 완료",
            "stopped": "안전하게 중지됨",
            "failed": "최적화 실패",
            "failed_no_completed_trials": "완료 Trial 없음",
        }
        if active:
            st.session_state["optimization_monitor_was_active"] = True
            if st.session_state.get("optimization_monitor_pending_run_id") == active.get("run_id"):
                st.session_state.pop("optimization_monitor_start_pending", None)
                st.session_state.pop("optimization_monitor_pending_since", None)
                st.session_state.pop("optimization_monitor_pending_run_id", None)
            state_path = Path(
                active.get("state_path")
                or project_path(f"data/optimizations/{active['study_name']}.state.json")
            )
            try:
                process_info = json.loads(
                    Path(f"{state_path}.process.json").read_text(encoding="utf-8")
                )
            except (OSError, ValueError, TypeError):
                process_info = {}
            state = active
            run_id = state.get("run_id")
            status_value = str(state.get("status", "preparing"))
            started_at = process_info.get("started_at") or state.get("started_at")
            elapsed_seconds = None
            if started_at:
                try:
                    elapsed_seconds = max(
                        0,
                        int(
                            (datetime.now(UTC) - datetime.fromisoformat(started_at)).total_seconds()
                        ),
                    )
                except (TypeError, ValueError):
                    pass
            snapshot = optimization_monitor_snapshot(active, recent_limit=20)
            counts = snapshot.get("counts", {})
            complete_total = int(counts.get("complete", state.get("completed_trials", 0)) or 0)
            failed_total = int(counts.get("fail", state.get("failed_trials", 0)) or 0)
            pruned_total = int(counts.get("pruned", state.get("pruned_trials", 0)) or 0)
            running_trial = next(
                (trial for trial in snapshot.get("trials", []) if trial["state"] == "RUNNING"), None
            )
            start_count = int(state.get("starting_trial_count", 0) or 0)
            prior = state.get("starting_state_counts") or {}
            current_run_complete = max(0, complete_total - int(prior.get("complete", 0) or 0))
            requested = int(state.get("requested_additional_trials", 0) or 0)
            terminal_in_run = max(
                0,
                sum(int(counts.get(key, 0) or 0) for key in ("complete", "fail", "pruned"))
                - sum(int(prior.get(key, 0) or 0) for key in ("complete", "failed", "pruned")),
            )
            remaining_trials = max(0, requested - terminal_in_run)
            mean_trial_seconds = snapshot.get("mean_trial_seconds")
            running_elapsed_seconds = 0
            if running_trial and running_trial.get("started_at"):
                try:
                    trial_started = datetime.fromisoformat(running_trial["started_at"])
                    if trial_started.tzinfo is None:
                        trial_started = trial_started.replace(tzinfo=UTC)
                    running_elapsed_seconds = max(
                        0, int((datetime.now(UTC) - trial_started).total_seconds())
                    )
                except (TypeError, ValueError):
                    running_elapsed_seconds = 0
            eta_seconds = estimate_optimization_remaining_seconds(
                remaining_trials,
                mean_trial_seconds,
                int(snapshot.get("sampled_trial_durations", 0)),
                running_elapsed_seconds,
            )
            st.markdown("### 자동 최적화 실시간 현황")
            display_status = labels.get(status_value, "실행 준비 또는 상태 확인 중")
            st.info(f"**{display_status}** · Study {active['study_name']}")
            cols = st.columns(5)
            cols[0].metric("실행 ID", str(run_id or "확인 중")[:12])
            cols[1].metric("워커 PID", process_info.get("pid", "확인 중"))
            cols[2].metric("목표 Trial", f"{requested:,}")
            cols[3].metric("이번 실행 완료", f"{current_run_complete:,} / {requested:,}")
            cols[4].metric("경과 시간", format_optimization_duration(elapsed_seconds))
            progress = min(1.0, current_run_complete / requested) if requested else 0.0
            st.progress(
                progress,
                text=f"Trial 처리 진행률 {progress:.0%} · 과거 Study 전체 Trial {start_count:,}개",
            )
            failed_this_run = max(0, failed_total - int(prior.get("failed", 0) or 0))
            pruned_this_run = max(0, pruned_total - int(prior.get("pruned", 0) or 0))
            st.caption(
                f"이번 실행 완료 {current_run_complete:,} · 실패 {failed_this_run:,} · "
                f"가지치기 {pruned_this_run:,}"
            )
            trial_cols = st.columns(5)
            trial_cols[0].metric("현재 Trial", running_trial["number"] if running_trial else "대기")
            trial_cols[1].metric("완료", f"{complete_total:,}")
            trial_cols[2].metric("실패", f"{failed_total:,}")
            trial_cols[3].metric("가지치기", f"{pruned_total:,}")
            trial_cols[4].metric(
                "예상 남은 시간",
                format_optimization_duration(eta_seconds) if eta_seconds is not None else "예측 중",
            )
            if running_trial and running_trial.get("started_at"):
                st.caption(
                    f"현재 Trial #{running_trial['number']} 경과: "
                    f"{format_optimization_duration(running_elapsed_seconds)}"
                )
            if eta_seconds is not None and started_at:
                from datetime import timedelta

                eta_at = datetime.now(UTC) + timedelta(seconds=eta_seconds)
                st.caption(
                    f"예상 종료 시각: {eta_at.astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}"
                )
            st.caption(
                f"마지막 상태 갱신: {state.get('updated_at') or '기록 대기'} · "
                "Objective는 내부 검증 점수이며 당첨 확률이 아닙니다."
            )
            if state.get("best_value") is not None:
                st.metric("현재 최고 Objective 점수", f"{float(state['best_value']):.6f}")
                st.caption(
                    f"최고 Trial #{state.get('best_trial_number', '—')} · "
                    f"최적 파라미터: {state.get('best_parameters') or 'Study 조회 중'}"
                )
            if snapshot.get("available") and snapshot.get("trials"):
                rows = []
                for trial in reversed(snapshot["trials"]):
                    rows.append(
                        {
                            "Trial": trial["number"],
                            "상태": trial["state"],
                            "시작": trial.get("started_at"),
                            "종료": trial.get("finished_at"),
                            "소요 시간(초)": trial.get("duration_seconds"),
                            "Objective": trial.get("value"),
                            "실패 사유": trial.get("failure_reason"),
                            "주요 GP 파라미터": ", ".join(
                                f"{key}={value:g}"
                                for key, value in trial.get("parameters", {}).items()
                            ),
                        }
                    )
                st.dataframe(rows, hide_index=True, use_container_width=True)
                scores = [
                    {"Trial": trial["number"], "Objective": trial["value"]}
                    for trial in reversed(snapshot["trials"])
                    if trial["state"] == "COMPLETE" and trial.get("value") is not None
                ]
                if scores:
                    st.line_chart(scores, x="Trial", y="Objective")
            elif not snapshot.get("available"):
                st.caption("Optuna Study를 읽을 수 없습니다. 워커 로그와 상태 파일을 확인하세요.")
            log_path = project_path(f"logs/{active['study_name']}_optimization.log")
            if log_path.is_file():
                try:
                    st.code(
                        "\n".join(
                            log_path.read_text(encoding="utf-8", errors="replace").splitlines()[
                                -12:
                            ]
                        ),
                        language="text",
                    )
                except OSError:
                    st.caption("최적화 로그를 읽지 못했습니다.")
            if status_value in {"running", "stopping"} and run_id:
                if status_value == "running" and st.button(
                    "최적화 중지", key="optimization_stop_request"
                ):
                    st.session_state["optimization_stop_confirm"] = True
                if st.session_state.get("optimization_stop_confirm"):
                    st.warning("현재 Trial을 마친 뒤 새 Trial을 시작하지 않고 안전하게 중지합니다.")
                    left, right = st.columns(2)
                    if left.button("중지 요청 확인", key="confirm_optimization_stop"):
                        try:
                            request_optimization_stop(state_path, run_id=str(run_id))
                            st.session_state["optimization_stop_confirm"] = False
                            st.success("안전한 중지 요청을 전달했습니다.")
                        except RuntimeError as error:
                            st.error(str(error))
                    if right.button("계속 실행", key="cancel_optimization_stop"):
                        st.session_state["optimization_stop_confirm"] = False
        else:
            if st.session_state.pop("optimization_monitor_was_active", False):
                st.session_state.pop("optimization_monitor_start_pending", None)
                st.session_state.pop("optimization_monitor_pending_since", None)
                st.rerun()
            if st.session_state.get("optimization_monitor_start_pending"):
                pending_study = st.session_state.get("optimization_monitor_pending_study")
                pending_record = next(
                    (item for item in fresh_records if item["study_name"] == pending_study),
                    None,
                )
                if pending_record and pending_record.get("status") in {
                    "failed",
                    "failed_no_completed_trials",
                    "completed",
                    "stopped",
                }:
                    st.session_state.pop("optimization_monitor_start_pending", None)
                    st.session_state.pop("optimization_monitor_pending_since", None)
                    st.session_state.pop("optimization_monitor_pending_run_id", None)
                    st.session_state.pop("optimization_monitor_pending_study", None)
                    st.rerun()
                pending_since = st.session_state.get("optimization_monitor_pending_since")
                if pending_since and (datetime.now(UTC) - pending_since).total_seconds() > 120:
                    st.session_state.pop("optimization_monitor_start_pending", None)
                    st.session_state.pop("optimization_monitor_pending_since", None)
                    st.session_state.pop("optimization_monitor_pending_run_id", None)
                    st.session_state.pop("optimization_monitor_pending_study", None)
                    st.rerun()
                st.info("최적화 워커 실행을 확인 중이며 전역 CPU 실행 슬롯을 검사하고 있습니다.")
            stale = next(
                (
                    item
                    for item in fresh_records
                    if item.get("status") in {"running", "stopping"}
                    and not _worker_active(
                        Path(
                            item.get("state_path")
                            or project_path(f"data/optimizations/{item['study_name']}.state.json")
                        )
                    )
                ),
                None,
            )
            if stale:
                st.warning(
                    f"Study {stale['study_name']} 상태 파일은 {stale.get('status')}이지만 "
                    "실행 잠금이 없습니다. 워커가 종료됐거나 시작에 실패했을 수 있습니다. "
                    "상태를 완료로 추정하지 않았으며 로그를 확인해 주세요."
                )
                stale_snapshot = optimization_monitor_snapshot(stale, recent_limit=5)
                unfinished = int(stale_snapshot.get("counts", {}).get("running", 0) or 0)
                if unfinished:
                    st.caption(
                        f"Optuna DB에 RUNNING 상태로 남은 Trial {unfinished}개가 있습니다. "
                        "워커 잠금은 해제됐지만 해당 Trial 완료 여부는 확정하지 않았습니다."
                    )
            elif research_worker_active() and not st.session_state.get(
                "optimization_monitor_start_pending"
            ):
                st.warning(
                    _execution_conflict(fresh_records)
                    or "다른 작업이 공용 연구 실행 자원을 사용 중입니다."
                )
            else:
                terminal = next(
                    (
                        item
                        for item in sorted(
                            fresh_records,
                            key=lambda row: str(row.get("updated_at", row.get("finished_at", ""))),
                            reverse=True,
                        )
                        if item.get("status")
                        in {"completed", "stopped", "failed", "failed_no_completed_trials"}
                    ),
                    None,
                )
                if terminal:
                    st.info(
                        f"최근 실행: {terminal['study_name']} · "
                        f"{labels.get(terminal.get('status'), terminal.get('status'))} · "
                        f"완료 Trial {int(terminal.get('completed_trials', 0) or 0):,}"
                    )
                    if terminal.get("error"):
                        st.error(str(terminal["error"]))
                else:
                    st.caption("현재 자동 최적화 실행은 없습니다.")

    def _display_score(value: Any) -> str:
        return "기록 없음" if value is None else f"{float(value):.6f}"

    if not status["available"]:
        st.warning(
            "Optuna 선택 기능이 설치되지 않았습니다. "
            "기존 연구 결과 조회는 계속 사용할 수 있습니다. " + status["install_extra"]
        )

    create_tab, resume_tab, history_tab = st.tabs(
        ["새 자동 최적화", "기존 연구 이어하기", "최적화 이력"],
        key="optimization_workspace_tab",
        on_change="rerun",
    )
    with create_tab:
        st.markdown(
            "**새 자동 최적화 시작** · 새로운 이름으로 별도 기록을 만듭니다. "
            "기존 이름과 겹치면 시작하지 않습니다."
        )
        available_data = [
            path
            for path in (
                project_path("data/official/draws.csv"),
                project_path("data/raw/lotto_draws_753_1246_with_bonus.csv"),
                project_path("data/raw/lotto_draws_753_1246.csv"),
            )
            if path.is_file()
        ]
        if not available_data:
            st.info("자동 최적화에 사용할 당첨번호 데이터가 없습니다.")
        else:
            with st.form("optimization_new_form"):
                new_name = st.text_input(
                    "새 최적화 연구 이름",
                    value="gp-tuning-new",
                    help=(
                        "이름은 기존 SQLite 및 JSON 기록과 비교됩니다. "
                        "이미 있는 이름은 덮어쓰지 않습니다."
                    ),
                )
                new_data = st.selectbox(
                    "연구 데이터",
                    available_data,
                    format_func=lambda value: str(value.relative_to(PROJECT_ROOT)),
                    key="optimization_new_csv",
                )
                new_profile = st.selectbox(
                    "특징 프로파일",
                    ("legacy", "bonus_lag_v1", "calendar_round"),
                    format_func=lambda value: {
                        "legacy": "기본 당첨번호 특징",
                        "bonus_lag_v1": "보너스 번호 포함 특징",
                        "calendar_round": "회차 및 날짜 특징",
                    }[value],
                    help=(
                        "프로파일은 이름으로 추측하지 않습니다. 선택한 데이터에 "
                        "필수 정보가 있는지 시작 전에 검사합니다."
                    ),
                )
                new_mode = st.selectbox(
                    "최적화 모드",
                    ("quick", "standard", "deep"),
                    format_func=lambda value: {
                        "quick": "빠른 점검",
                        "standard": "표준 연구 · 권장",
                        "deep": "심층 연구 · 계산 부담 큼",
                    }[value],
                    help="탐색 범위와 계산량을 정합니다.",
                )
                new_trials = st.number_input(
                    "Trial 수",
                    min_value=1,
                    max_value=500,
                    value={"quick": 3, "standard": 10, "deep": 30}[new_mode],
                    help="Trial은 한 가지 연구 설정을 평가하는 실행 단위입니다.",
                )
                new_timeout = st.number_input(
                    "시간 제한(분)",
                    min_value=1,
                    max_value=1440,
                    value={"quick": 30, "standard": 180, "deep": 480}[new_mode],
                    help="제한 시간에 도달하면 완료된 Trial을 보존하고 실행을 마칩니다.",
                )
                start_new = st.form_submit_button(
                    "새 자동 최적화 시작", type="primary", disabled=not status["available"]
                )
            if start_new:
                _start_optimization(
                    new_name,
                    new_data,
                    new_profile,
                    new_mode,
                    int(new_trials),
                    float(new_timeout),
                    "new",
                    records,
                )

    with resume_tab:
        resumable = [row for row in records if row.get("database_available")]
        if not resumable:
            st.info("이어서 실행할 Optuna SQLite 연구가 없습니다.")
        else:
            choices = {row["study_name"]: row for row in resumable}
            chosen_name = st.selectbox(
                "기존 자동 최적화 연구", list(choices), key="optimization_resume_choice"
            )
            chosen = choices[chosen_name]
            st.write(
                f"상태: **{_status_label(chosen)}** · "
                f"완료 Trial: **{int(chosen.get('completed_trials', 0)):,}** · "
                f"최고 Trial: **{chosen.get('best_trial_number', '—')}** · "
                f"최고 내부 점수: **{_display_score(chosen.get('best_value'))}**"
            )
            st.caption(
                f"SQLite 기록: `{chosen['database_path']}` · "
                f"프로파일: `{chosen.get('feature_profile', '기록 없음')}` · "
                f"데이터 범위: `{chosen.get('dataset_range', '기록 없음')}`"
            )
            trial_add = st.number_input(
                "추가 Trial 수",
                min_value=1,
                max_value=500,
                value=3,
                key="optimization_resume_trials",
                help="기존 Trial을 유지하고 지정한 수만큼 추가로 평가합니다.",
            )
            timeout_add = st.number_input(
                "추가 실행 시간 제한(분)",
                min_value=1,
                max_value=1440,
                value=30,
                key="optimization_resume_timeout",
            )
            c1, c2 = st.columns(2)
            if c1.button(
                "기존 자동 최적화 이어서 하기",
                type="primary",
                disabled=not status["available"] or _any_optimization_active(records),
                help="기존 SQLite Study와 Trial을 그대로 불러와 이어서 실행합니다.",
            ):
                _start_existing_optimization(chosen, int(trial_add), float(timeout_add), "resume")
            if c2.button(
                "완료 연구에 추가 Trial 실행",
                disabled=not status["available"]
                or _any_optimization_active(records)
                or chosen.get("status") != "completed",
                help="완료 결과를 보존한 채 같은 Study에 Trial을 추가합니다.",
            ):
                _start_existing_optimization(
                    chosen, int(trial_add), float(timeout_add), "add_trials"
                )

    with history_tab:
        if not records:
            st.info("저장된 자동 최적화 이력이 없습니다.")
        else:
            rows = [
                {
                    "연구 이름": r["study_name"],
                    "모드": r.get("mode", "기록 없음"),
                    "데이터": r.get("dataset_range", "기록 없음"),
                    "상태": _status_label(r),
                    "완료 Trial": r.get("completed_trials", 0),
                    "최고 Trial": r.get("best_trial_number", "—"),
                    "최고 점수": r.get("best_value", "—"),
                    "종료 시각": r.get("finished_at", "—"),
                    "DB": "사용 가능" if r.get("database_available") else "없음",
                }
                for r in records
            ]
            st.dataframe(rows, width="stretch", hide_index=True)
            csv_buffer = io.StringIO(newline="")
            writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            st.download_button(
                "최적화 이력 CSV 다운로드",
                "\ufeff" + csv_buffer.getvalue(),
                file_name="optimization_history.csv",
                mime="text/csv; charset=utf-8",
            )
            selected_name = st.selectbox(
                "상세 조회할 연구",
                [r["study_name"] for r in records],
                key="optimization_history_choice",
            )
            selected = next(r for r in records if r["study_name"] == selected_name)
            st.json({key: value for key, value in selected.items() if key != "best_parameters"})
            st.download_button(
                "선택한 연구 결과 JSON 다운로드",
                json.dumps(selected, ensure_ascii=False, indent=2),
                file_name=f"{selected_name}_optimization.json",
                mime="application/json",
            )
            parameters = selected.get("best_parameters") or {}
            if parameters:
                st.subheader("최적 파라미터")
                st.json(parameters)
                st.download_button(
                    "최적 설정 JSON 다운로드",
                    json.dumps(
                        {"study_name": selected_name, "parameters": parameters},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    file_name=f"{selected_name}_best_parameters.json",
                    mime="application/json",
                )
                profile = selected.get("feature_profile")
                normalized, compatibility_error = validate_optimization_compatibility(
                    selected,
                    current_engines=tuple(st.session_state.get("research_current_engine_ids", ())),
                    current_dataset_range=st.session_state.get("research_current_dataset_range"),
                    dataset_integrity=st.session_state.get("research_current_dataset_integrity"),
                    current_dataset_sha256=st.session_state.get("research_current_dataset_sha256"),
                )
                if (
                    selected.get("status") != "completed"
                    or selected.get("best_trial_number") is None
                ):
                    compatibility_error = "최고 Trial이 완료된 최적화 연구만 적용할 수 있습니다."
                if compatibility_error:
                    st.warning(f"설정을 적용할 수 없습니다: {compatibility_error}")
                else:
                    current_settings = st.session_state.get("research_current_settings", {})
                    preview_rows = [
                        {
                            "GP 파라미터": source,
                            "현재 값": current_settings.get(target),
                            "적용 값": normalized[source],
                        }
                        for source, target in OPTUNA_TO_RESEARCH_KEYS.items()
                    ]
                    st.write("적용 전후 비교")
                    st.dataframe(preview_rows, hide_index=True, width="stretch")
                    if "pysr_symbolic_regression" in st.session_state.get(
                        "research_current_engine_ids", ()
                    ):
                        st.info(
                            "비교 모드에서는 7개 최적화 값이 GP에만 적용됩니다. "
                            "PySR 전용 특징과 탐색 설정은 별도로 유지됩니다."
                        )
                    if st.button(
                        "이 설정으로 지속적 패턴 연구 시작",
                        type="primary",
                        key="apply_optimization_settings",
                    ):
                        st.session_state["optimization_apply_confirmation"] = {
                            "study_name": selected_name,
                            "parameters": normalized,
                        }
                    if (
                        st.session_state.get("optimization_apply_confirmation", {}).get(
                            "study_name"
                        )
                        == selected_name
                    ):
                        st.warning("최종 확인 후 적용됩니다. 연구는 자동으로 시작하지 않습니다.")
                        if st.button("설정 적용 확인", key="confirm_optimization_apply"):
                            still_compatible, apply_error = validate_optimization_compatibility(
                                selected,
                                current_engines=tuple(
                                    st.session_state.get("research_current_engine_ids", ())
                                ),
                                current_dataset_range=st.session_state.get(
                                    "research_current_dataset_range"
                                ),
                                dataset_integrity=st.session_state.get(
                                    "research_current_dataset_integrity"
                                ),
                                current_dataset_sha256=st.session_state.get(
                                    "research_current_dataset_sha256"
                                ),
                            )
                            if apply_error or still_compatible != normalized:
                                st.session_state["optimization_apply_failure"] = (
                                    apply_error or "확인 단계에서 최고 Trial의 값이 달라졌습니다."
                                )
                            else:
                                current_settings = dict(
                                    st.session_state.get("research_current_settings", {})
                                )
                                control_values = {
                                    OPTUNA_TO_RESEARCH_KEYS[key]: value
                                    for key, value in normalized.items()
                                }
                                control_values["new_profile"] = profile
                                settings_snapshot = {
                                    **current_settings,
                                    **control_values,
                                }
                                context = {
                                    "study_name": selected_name,
                                    "optuna_study_id": selected.get("optuna_study_id"),
                                    "trial_number": int(selected["best_trial_number"]),
                                    "parameters": normalized,
                                    "research_settings": settings_snapshot,
                                    "expected_controls": control_values,
                                    "feature_profile": profile,
                                    "dataset_range": list(selected["dataset_range"]),
                                    "applied_at": datetime.now(UTC).isoformat(),
                                    "modified_by_user": False,
                                    "started_research_ids": [],
                                    "application_verified": False,
                                }
                                st.session_state["optimization_apply_pending"] = {
                                    "values": control_values,
                                    "context": context,
                                }
                                st.session_state["research_workspace_tab"] = "지속적 패턴 연구"
                            st.session_state.pop("optimization_apply_confirmation", None)
                            st.rerun()
            else:
                st.info("최적 파라미터가 저장되지 않았거나 SQLite 데이터베이스를 읽을 수 없습니다.")

    optimization_monitor()


def _display_score(value: Any) -> str:
    return "기록 없음" if value is None else f"{float(value):.6f}"


def _status_label(record: dict[str, Any]) -> str:
    if _worker_active(
        Path(
            record.get("state_path")
            or project_path(f"data/optimizations/{record['study_name']}.state.json")
        )
    ):
        return "실행 중"
    labels = {
        "completed": "완료",
        "stopped": "안전 중지",
        "failed": "실패",
        "preparing": "실행 준비 중",
        "failed_no_completed_trials": "완료 Trial 없음",
        "running": "중단 또는 상태 확인 필요",
        "state_unreadable": "상태 파일 손상",
    }
    status_value = str(record.get("status", "기록 확인 필요"))
    if status_value == "completed" and not record.get("database_available"):
        return "SQLite 기록 없음 · 결과 확인 필요"
    return labels.get(status_value, status_value)


def _any_optimization_active(records: list[dict[str, Any]]) -> bool:
    return research_worker_active() or any(
        _worker_active(
            Path(
                row.get("state_path")
                or project_path(f"data/optimizations/{row['study_name']}.state.json")
            )
        )
        for row in records
    )


def _execution_conflict(records: list[dict[str, Any]]) -> str | None:
    for record in records:
        state_path = Path(
            record.get("state_path")
            or project_path(f"data/optimizations/{record['study_name']}.state.json")
        )
        if _worker_active(state_path):
            return (
                f"현재 연구조건 자동 최적화 {record['study_name']} (실행 ID "
                f"{record.get('run_id', '확인 중')})가 CPU 실행 슬롯을 사용 중입니다."
            )
    if not research_worker_active():
        return None
    for checkpoint in project_path("checkpoints").rglob("research*.json"):
        try:
            state = json.loads(checkpoint.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if state.get("status") not in {"running", "pause_requested"}:
            continue
        if not _worker_active(checkpoint):
            continue
        research_id = state.get("research_id", "확인 중")
        started_at = state.get("started_at", "시각 미기록")
        job_type = "예약 연구" if "research_queue" in str(checkpoint) else "일반 연구"
        return (
            f"현재 {job_type} (연구 ID {research_id}, 시작 {started_at})가 "
            "CPU 실행 슬롯을 사용 중입니다. "
            "실행 종료 후 자동 최적화를 시작해 주세요."
        )
    return "다른 연구 워커 또는 예약 작업이 공용 CPU 실행 슬롯을 사용 중입니다."


def _start_optimization(
    name: str,
    data_path: Path,
    profile: str,
    mode: str,
    trials: int,
    timeout_minutes: float,
    action: str,
    records: list[dict[str, Any]],
) -> None:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-")[:80]
    if not slug:
        st.error("최적화 연구 이름을 입력하세요.")
        return
    if (
        slug in {row["study_name"] for row in records}
        or project_path(f"data/optimizations/{slug}.state.json").exists()
        or project_path(f"data/optimizations/{slug}.sqlite").exists()
    ):
        st.error(
            "같은 이름의 최적화 기록이 있습니다. "
            "기존 연구 이어하기 탭을 사용하세요. 자료는 변경하지 않았습니다."
        )
        return
    started = False
    try:
        from lotto_pattern_lab.dataset import dataset_integrity_report, load_csv

        dataset = load_csv(data_path)
        integrity = dataset_integrity_report(dataset)
        if profile == "bonus_lag_v1" and not integrity["all_draws_have_bonus"]:
            raise ValueError("선택한 데이터에 보너스 번호가 없습니다.")
        if profile == "calendar_round" and not integrity["all_draws_dated"]:
            raise ValueError("선택한 데이터에 추첨일이 없습니다.")
        conflict = _execution_conflict(records)
        if conflict:
            raise RuntimeError(conflict)
        run_id = str(uuid.uuid4())
        command = [
            sys.executable,
            "-m",
            "lotto_pattern_lab",
            "optimize",
            "--action",
            action,
            "--csv",
            str(data_path.resolve()),
            "--study-name",
            slug,
            "--run-id",
            run_id,
            "--mode",
            mode,
            "--trials",
            str(trials),
            "--timeout-seconds",
            str(timeout_minutes * 60),
            "--feature-profile",
            profile,
        ]
        with st.status("최적화 실행을 준비합니다", expanded=True) as status_box:
            st.write("연구 설정과 입력 데이터를 확인했습니다.")
            st.write("기존 이름과 SQLite 기록을 확인했습니다.")
            st.write("실행 환경과 저장 경로를 준비했습니다.")
            st.write("백그라운드 프로세스를 시작합니다.")
            process_state = launch_optimization_worker(
                _launch_worker,
                command,
                project_path(f"data/optimizations/{slug}.state.json"),
                project_path(f"logs/{slug}_optimization.log"),
                project_path(f"data/optimizations/{slug}.state.json.process.json"),
                study_name=slug,
                run_id=run_id,
            )
            pid = process_state.pid
            st.write(f"실행 상태를 확인했습니다. PID {pid}")
            status_box.update(label="자동 최적화 실행 중", state="complete", expanded=False)
        st.success(f"자동 최적화를 시작했습니다: {slug}")
        st.session_state["optimization_monitor_start_pending"] = True
        st.session_state["optimization_monitor_pending_since"] = datetime.now(UTC)
        st.session_state["optimization_monitor_pending_run_id"] = run_id
        st.session_state["optimization_monitor_pending_study"] = slug
        started = True
    except Exception as error:
        st.error(f"최적화 실행 준비 실패: {error}")
    if started:
        st.rerun()


def _start_existing_optimization(
    record: dict[str, Any], trials: int, timeout_minutes: float, action: str
) -> None:
    if not Path(record["database_path"]).is_file():
        st.error("기존 SQLite 연구 기록을 찾지 못해 이어서 실행할 수 없습니다.")
        return
    profile = record.get("feature_profile")
    if profile not in {"legacy", "bonus_lag_v1", "calendar_round"}:
        st.error("특징 프로파일이 저장 기록에 없어 이름으로 추정하지 않고 실행을 막았습니다.")
        return
    expected_range = record.get("dataset_range")
    candidates = [
        path
        for path in (
            project_path("data/official/draws.csv"),
            project_path("data/raw/lotto_draws_753_1246_with_bonus.csv"),
            project_path("data/raw/lotto_draws_753_1246.csv"),
        )
        if path.is_file()
    ]
    selected = None
    for candidate in candidates:
        try:
            from lotto_pattern_lab.dataset import load_csv

            ds = load_csv(candidate)
            if expected_range == [ds.records[0].round_number, ds.records[-1].round_number]:
                selected = candidate
                break
        except (OSError, ValueError, IndexError):
            continue
    if selected is None:
        st.error("저장된 데이터 범위와 일치하는 파일이 없어 실행하지 않았습니다.")
        return
    run_id = str(uuid.uuid4())
    command = [
        sys.executable,
        "-m",
        "lotto_pattern_lab",
        "optimize",
        "--action",
        action,
        "--csv",
        str(selected.resolve()),
        "--study-name",
        record["study_name"],
        "--run-id",
        run_id,
        "--mode",
        record.get("mode", "standard"),
        "--trials",
        str(trials),
        "--timeout-seconds",
        str(timeout_minutes * 60),
        "--feature-profile",
        profile,
    ]
    started = False
    try:
        conflict = _execution_conflict(list_optimization_studies())
        if conflict:
            raise RuntimeError(conflict)
        with st.status(
            "기존 Optuna 연구를 확인하고 이어서 실행합니다", expanded=True
        ) as status_box:
            st.write(f"기존 완료 Trial {record.get('completed_trials', 0)}개를 보존합니다.")
            process_state = launch_optimization_worker(
                _launch_worker,
                command,
                project_path(f"data/optimizations/{record['study_name']}.state.json"),
                project_path(f"logs/{record['study_name']}_optimization.log"),
                project_path(f"data/optimizations/{record['study_name']}.state.json.process.json"),
                study_name=record["study_name"],
                run_id=run_id,
            )
            pid = process_state.pid
            status_box.update(label="기존 연구 이어서 실행 중", state="complete", expanded=False)
        st.success(f"기존 연구를 실행했습니다. PID {pid}")
        st.session_state["optimization_monitor_start_pending"] = True
        st.session_state["optimization_monitor_pending_since"] = datetime.now(UTC)
        st.session_state["optimization_monitor_pending_run_id"] = run_id
        st.session_state["optimization_monitor_pending_study"] = record["study_name"]
        started = True
    except Exception as error:
        st.error(f"기존 연구를 시작하지 못했습니다: {error}")
    if started:
        st.rerun()


with optimization_tab:
    render_optimization()
