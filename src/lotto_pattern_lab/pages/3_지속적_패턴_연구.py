"""Persistent multi-pattern research dashboard and local worker controls."""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

from lotto_pattern_lab.dataset import dataset_integrity_report, load_csv
from lotto_pattern_lab.pattern_library import PatternLibrary, import_historical_experiment
from lotto_pattern_lab.periodicity import periodicity_report
from lotto_pattern_lab.pysr_backend import optional_backend_status
from lotto_pattern_lab.research import load_checkpoint, request_stop

st.set_page_config(page_title="지속적 패턴 연구", layout="wide")
st.title("지속적 다중 패턴 연구")
st.caption(
    "현재 실행 엔진은 기존 DEAP 간헐적 GP입니다. "
    "새 엔진은 실제 구현 전까지 선택지에 표시하지 않습니다."
)

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)
checkpoint_paths = sorted(
    checkpoint_dir.glob("research*.json"), key=lambda p: p.stat().st_mtime, reverse=True
)
with st.sidebar:
    data_source = st.radio("연구 데이터", ("공식 검증 데이터", "로컬 검증 미완료 CSV"))
    csv_path = (
        Path("data/official/draws.csv")
        if data_source == "공식 검증 데이터"
        else Path("data/raw/lotto_draws_753_1246.csv")
    )
    workflow = st.radio("작업", ("새 장시간 연구", "저장된 연구 재개"))
    selected_checkpoint = None
    if workflow == "저장된 연구 재개":
        if checkpoint_paths:
            selected_checkpoint = st.selectbox(
                "체크포인트", checkpoint_paths, format_func=lambda p: p.name
            )
        else:
            st.warning("재개할 체크포인트가 없습니다.")
    engine = st.selectbox("탐색 엔진", ("intermittent_gp",))
    feature_profile = st.selectbox("특징 프로파일", ("legacy", "calendar_round"))
    duration_option = st.selectbox("실행 시간", ("30분", "1시간", "3시간", "6시간", "사용자 지정"))
    duration_minutes = {"30분": 30, "1시간": 60, "3시간": 180, "6시간": 360}.get(duration_option)
    if duration_option == "사용자 지정":
        duration_minutes = st.number_input("분", min_value=1, max_value=1440, value=30)
    population = st.number_input("개체군 크기", min_value=8, max_value=1000, value=32, step=8)
    generations = st.number_input("세대 수", min_value=1, max_value=1000, value=10)
    workers = st.slider("CPU 워커", 1, 4, 4)
    controls = st.number_input("무작위 대조 반복", min_value=0, max_value=1000, value=3)
    seed = st.number_input("기본 난수 시드", min_value=0, max_value=2**31 - 1, value=12345)

dataset = None
integrity = None
checkpoint = None
active_checkpoint_path = (
    Path(st.session_state["active_research_checkpoint"])
    if st.session_state.get("active_research_checkpoint")
    else (selected_checkpoint or (checkpoint_paths[0] if checkpoint_paths else None))
)
if active_checkpoint_path is not None:
    try:
        checkpoint = load_checkpoint(active_checkpoint_path)
    except Exception as error:
        st.error(f"체크포인트를 읽을 수 없습니다: {error}")

if not csv_path.is_file():
    st.error(f"데이터 파일이 없습니다: {csv_path}")
else:
    try:
        dataset = load_csv(csv_path)
        st.write(
            f"데이터: {dataset.verification_status} · "
            f"{dataset.records[0].round_number}~{dataset.records[-1].round_number}회"
        )
        integrity = dataset_integrity_report(dataset)
        if not integrity["rounds_contiguous"]:
            st.error(f"누락 회차가 있어 탐색을 실행할 수 없습니다: {integrity['missing_rounds']}")
        st.json(integrity)
        if integrity["rounds_contiguous"]:
            totals = [sum(record.draw.numbers) for record in dataset.records]
            with st.expander("회차 합계의 자기상관 및 FFT 주기 진단"):
                st.json(periodicity_report(totals))
        if feature_profile == "calendar_round" and any(
            record.draw_date is None for record in dataset.records
        ):
            st.warning("날짜 특징은 공식 추첨일이 모두 있는 데이터에서만 실행할 수 있습니다.")
    except Exception as error:
        dataset = None
        st.error(f"데이터 검증 실패: {error}")

if checkpoint:
    state = checkpoint
    state_col, run_col, match_col = st.columns(3)
    state_col.metric("연구 상태", state.get("status", "unknown"))
    run_col.metric("완료 실험", state.get("completed_runs", 0))
    match_col.metric("완전 일치 발견", len(state.get("discovered_exact_matches", [])))
    st.caption(
        f"엔진: {', '.join(state.get('engines', []))} · "
        f"마지막 업데이트: {state.get('last_updated_at', '없음')} · "
        f"재개 방식: 실험 경계에서 다음 시드로 계속"
    )
    if state.get("current_progress"):
        st.json(state["current_progress"])
    if state.get("last_run_summary"):
        st.json(state["last_run_summary"])
    if state.get("failed_runs"):
        st.warning(f"실패 또는 중단된 실험: {len(state['failed_runs'])}")
else:
    st.info("연구를 시작하면 진행 상황과 재개용 체크포인트가 로컬에 저장됩니다.")

active_worker = False
if checkpoint and checkpoint.get("status") == "running":
    pid = checkpoint.get("worker_pid")
    if pid:
        try:
            os.kill(int(pid), 0)
            active_worker = True
        except (OSError, ValueError):
            active_worker = False
if checkpoint and checkpoint.get("status") == "running" and not active_worker:
    st.warning("기록은 실행 중이지만 작업 프로세스가 없습니다. 체크포인트에서 재개할 수 있습니다.")

if st.button("기존 완료 실험을 라이브러리에 가져오기", disabled=dataset is None):
    imported = []
    for experiment_path in Path("experiments").glob("*.json"):
        with experiment_path.open(encoding="utf-8") as source_file:
            experiment = json.load(source_file)
        if (
            experiment.get("status") == "completed"
            and experiment.get("experiment_type") == "intermittent"
        ):
            imported.append(import_historical_experiment(experiment, dataset, PatternLibrary()))
    st.session_state["last_import_summary"] = imported
    st.rerun()
if st.session_state.get("last_import_summary"):
    st.success(f"완료된 기존 실험 {len(st.session_state['last_import_summary'])}개를 확인했습니다.")
    st.json(st.session_state["last_import_summary"])

button_col, stop_col = st.columns(2)
with button_col:
    if st.button(
        "연구 시작 또는 재개",
        type="primary",
        disabled=dataset is None
        or integrity is None
        or not integrity["rounds_contiguous"]
        or (feature_profile == "calendar_round" and not integrity["all_draws_dated"])
        or active_worker
        or (workflow == "저장된 연구 재개" and selected_checkpoint is None),
    ):
        if workflow == "새 장시간 연구":
            checkpoint_path = (
                checkpoint_dir / f"research_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
            )
            resume = False
        else:
            checkpoint_path = Path(selected_checkpoint)
            resume = True
        log_path = checkpoint_dir / "research_worker.log"
        log_file = log_path.open("a", encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "lotto_pattern_lab",
            "research",
            "--csv",
            str(csv_path),
            "--duration-minutes",
            str(int(duration_minutes)),
            "--checkpoint",
            str(checkpoint_path),
            "--population",
            str(int(population)),
            "--generations",
            str(int(generations)),
            "--workers",
            str(int(workers)),
            "--controls",
            str(int(controls)),
            "--seed",
            str(int(seed)),
            "--feature-profile",
            feature_profile,
            "--engine",
            engine,
        ]
        if resume:
            command.append("--resume")
        creation_flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
            subprocess, "CREATE_NO_WINDOW", 0
        )
        subprocess.Popen(
            command,
            cwd=Path.cwd(),
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
            close_fds=True,
        )
        log_file.close()
        st.session_state["active_research_checkpoint"] = str(checkpoint_path)
        st.success(f"연구 작업을 시작했습니다. 체크포인트: {checkpoint_path}")
        st.rerun()
with stop_col:
    if st.button(
        "실험 경계에서 중단 요청",
        disabled=not active_worker or active_checkpoint_path is None,
    ):
        request_stop(active_checkpoint_path)
        st.info("현재 GP 탐색은 완료한 뒤 저장하고 중단합니다.")

st.subheader("패턴 라이브러리")
library = PatternLibrary()
patterns = library.list_patterns()
if patterns:
    st.dataframe(
        [
            {
                "패턴 ID": item["pattern_id"],
                "엔진": item.get("engine"),
                "상태": item.get("current_status"),
                "발견 방식": item.get("discovery_mode"),
                "조건": item.get("activation_condition"),
                "수식": item.get("formulas"),
                "실험": item.get("experiment_id"),
            }
            for item in patterns
        ],
        width="stretch",
        hide_index=True,
    )
else:
    st.info("저장된 패턴이 없습니다.")

st.subheader("과거 6개 완전 일치 발견")
exact_matches = library.list_exact_matches()
if exact_matches:
    st.warning("과거 탐색 중의 일치입니다. 사전 예측 성능이나 생성 원리를 입증하지 않습니다.")
    st.dataframe(
        [
            {
                "회차": item["match"].get("round_number"),
                "예측 번호": item["match"].get("predicted_numbers"),
                "실제 번호": item["match"].get("winning_numbers"),
                "패턴 ID": item["pattern"].get("pattern_id"),
                "발견 방식": item["pattern"].get("discovery_mode"),
            }
            for item in exact_matches
        ],
        width="stretch",
        hide_index=True,
    )
else:
    st.info("아직 기록된 6개 완전 일치 후보가 없습니다.")

st.subheader("연구 기능 구현 상태")
st.dataframe(
    [
        {"기능": "간헐적 수학 규칙", "상태": "구현됨 · DEAP GP"},
        {"기능": "날짜·회차 특징", "상태": "구현됨 · calendar_round 프로파일"},
        {"기능": "비트·자릿수·소수·약수·피보나치 연산", "상태": "구현됨 · 정수 범위 제한"},
        {"기능": "완전 일치 및 패턴 라이브러리", "상태": "구현됨 · SQLite, 과거/사전 구분"},
        {"기능": "무작위 대조군", "상태": "구현됨 · 기존 동일 GP 조건 대조"},
        {"기능": "장시간 실행과 재개", "상태": "구현됨 · 실험 경계 체크포인트"},
        {"기능": "HMM·변화점", "상태": "기존 구현 유지 · 인과 필터링"},
        {"기능": "자기상관·FFT 주기 진단", "상태": "구현됨 · 탐색적 진단"},
        {"기능": "PySR/Julia 백엔드", "상태": "미구현"},
        {"기능": "세대 중간 GP 상태 복원", "상태": "미구현 · 다음 실험부터 재개"},
        {"기능": "제한 재귀식·완전한 비정형 문법", "상태": "미구현"},
    ],
    width="stretch",
    hide_index=True,
)
st.subheader("선택형 PySR 설치 상태")
st.json(optional_backend_status())
