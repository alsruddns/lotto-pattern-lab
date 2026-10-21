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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st

from lotto_pattern_lab.optimization_process_state import launch_optimization_worker
from lotto_pattern_lab.paths import PROJECT_ROOT, project_path
from lotto_pattern_lab.research import research_worker_active
from lotto_pattern_lab.research_optimization import list_optimization_studies, optuna_status
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

    @st.fragment(run_every="5s")
    def optimization_monitor() -> None:
        active = next(
            (
                item
                for item in records
                if _worker_active(
                    Path(
                        item.get("state_path")
                        or project_path(f"data/optimizations/{item['study_name']}.state.json")
                    )
                )
            ),
            None,
        )
        if active:
            state_path = Path(
                active.get("state_path")
                or project_path(f"data/optimizations/{active['study_name']}.state.json")
            )
            process_path = Path(f"{state_path}.process.json")
            process_info = {}
            try:
                process_info = json.loads(process_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
            fresh = next(
                (
                    row
                    for row in list_optimization_studies()
                    if row["study_name"] == active["study_name"]
                ),
                active,
            )
            started = process_info.get("started_at") or fresh.get("started_at")
            elapsed = "기록 없음"
            if started:
                try:
                    elapsed = format_duration(
                        max(
                            0,
                            int(
                                (
                                    datetime.now(UTC) - datetime.fromisoformat(started)
                                ).total_seconds()
                            ),
                        )
                    )
                except (TypeError, ValueError):
                    pass
            target_trials = int(fresh.get("starting_trial_count", 0)) + int(
                fresh.get("requested_additional_trials", 0)
            )
            st.info("자동 최적화 실행 중 · 5초마다 상태를 확인합니다.")
            cols = st.columns(6)
            cols[0].metric("연구 이름", fresh["study_name"])
            cols[1].metric("PID", process_info.get("pid", "확인 중"))
            cols[2].metric(
                "완료 Trial", f"{fresh.get('completed_trials', 0):,} / {target_trials:,}"
            )
            cols[3].metric("최고 Trial", fresh.get("best_trial_number", "—"))
            cols[4].metric("최고 내부 평가 점수", _display_score(fresh.get("best_value")))
            cols[5].metric("경과 시간", elapsed)
            if fresh.get("last_trial_number") is not None:
                st.caption(
                    f"마지막 처리 Trial: {fresh['last_trial_number']} · "
                    f"상태: {fresh.get('last_trial_state', '확인 중')}"
                )
            log_path = project_path(f"logs/{fresh['study_name']}_optimization.log")
            if log_path.is_file():
                try:
                    st.code(
                        "\n".join(
                            log_path.read_text(encoding="utf-8", errors="replace").splitlines()[
                                -18:
                            ]
                        ),
                        language="text",
                    )
                except OSError:
                    st.caption("실행 로그를 읽지 못했습니다.")
        else:
            st.caption("현재 실행 중인 자동 최적화가 없습니다.")

    def _display_score(value: Any) -> str:
        return "기록 없음" if value is None else f"{float(value):.6f}"

    optimization_monitor()
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
                        st.warning(
                            "최종 확인 후 적용됩니다. 연구는 자동으로 시작하지 않습니다."
                        )
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
                                    apply_error
                                    or "확인 단계에서 최고 Trial의 값이 달라졌습니다."
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
        "failed": "실패",
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
    try:
        from lotto_pattern_lab.dataset import dataset_integrity_report, load_csv

        dataset = load_csv(data_path)
        integrity = dataset_integrity_report(dataset)
        if profile == "bonus_lag_v1" and not integrity["all_draws_have_bonus"]:
            raise ValueError("선택한 데이터에 보너스 번호가 없습니다.")
        if profile == "calendar_round" and not integrity["all_draws_dated"]:
            raise ValueError("선택한 데이터에 추첨일이 없습니다.")
        if _any_optimization_active(records):
            raise RuntimeError("다른 연구 또는 최적화 작업이 실행 중입니다.")
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
            )
            pid = process_state.pid
            st.write(f"실행 상태를 확인했습니다. PID {pid}")
            status_box.update(label="자동 최적화 실행 중", state="complete", expanded=False)
        st.success(f"자동 최적화를 시작했습니다: {slug}")
    except Exception as error:
        st.error(f"최적화 실행 준비 실패: {error}")


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
        "--mode",
        record.get("mode", "standard"),
        "--trials",
        str(trials),
        "--timeout-seconds",
        str(timeout_minutes * 60),
        "--feature-profile",
        profile,
    ]
    try:
        if _any_optimization_active(list_optimization_studies()):
            raise RuntimeError("다른 연구 또는 최적화 작업이 실행 중입니다.")
        with st.status(
            "기존 Optuna 연구를 확인하고 이어서 실행합니다", expanded=True
        ) as status_box:
            st.write(f"기존 완료 Trial {record.get('completed_trials', 0)}개를 보존합니다.")
            process_state = launch_optimization_worker(
                _launch_worker,
                command,
                project_path(f"data/optimizations/{record['study_name']}.state.json"),
                project_path(f"logs/{record['study_name']}_optimization.log"),
                project_path(
                    f"data/optimizations/{record['study_name']}.state.json.process.json"
                ),
                study_name=record["study_name"],
            )
            pid = process_state.pid
            status_box.update(label="기존 연구 이어서 실행 중", state="complete", expanded=False)
        st.success(f"기존 연구를 실행했습니다. PID {pid}")
    except Exception as error:
        st.error(f"기존 연구를 시작하지 못했습니다: {error}")


with optimization_tab:
    render_optimization()
