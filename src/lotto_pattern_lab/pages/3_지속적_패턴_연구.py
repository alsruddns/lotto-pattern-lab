"""Persistent multi-pattern research dashboard and local worker controls."""

import os
import subprocess
import sys
from datetime import UTC, datetime
from math import ceil

import streamlit as st

from lotto_pattern_lab.dataset import dataset_integrity_report, load_csv
from lotto_pattern_lab.paths import PROJECT_ROOT, locate_checkpoint, project_path
from lotto_pattern_lab.pattern_library import PatternLibrary, import_historical_experiment
from lotto_pattern_lab.periodicity import periodicity_report
from lotto_pattern_lab.pysr_backend import optional_backend_status
from lotto_pattern_lab.research import load_checkpoint, request_stop
from lotto_pattern_lab.storage import ExperimentStore

st.set_page_config(page_title="지속적 패턴 연구", layout="wide")
st.title("지속적 다중 패턴 연구")
st.caption(
    "현재 실행 엔진은 기존 DEAP 간헐적 GP입니다. "
    "새 엔진은 실제 구현 전까지 선택지에 표시하지 않습니다."
)

checkpoint_dir = project_path("checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)
checkpoint_paths = sorted(
    checkpoint_dir.glob("research*.json"), key=lambda p: p.stat().st_mtime, reverse=True
)
resume_config = {}
with st.sidebar:
    data_source = st.radio("연구 데이터", ("공식 검증 데이터", "로컬 검증 미완료 CSV"))
    csv_path = (
        project_path("data/official/draws.csv")
        if data_source == "공식 검증 데이터"
        else project_path("data/raw/lotto_draws_753_1246.csv")
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
        if selected_checkpoint is not None:
            try:
                resume_config = load_checkpoint(selected_checkpoint).get("config", {})
            except (OSError, ValueError):
                resume_config = {}
    engine = st.selectbox("탐색 엔진", ("intermittent_gp",))
    feature_profile = st.selectbox(
        "특징 프로파일",
        ("legacy", "calendar_round"),
        index=("legacy", "calendar_round").index(
            resume_config.get("feature_profile", "legacy")
            if resume_config.get("feature_profile", "legacy") in {"legacy", "calendar_round"}
            else "legacy"
        ),
        disabled=workflow == "저장된 연구 재개",
    )
    elite_enabled = st.checkbox(
        "Elite Seeding 사용",
        value=bool(
            resume_config.get("elite_seeding_enabled", workflow != "저장된 연구 재개")
        ),
        disabled=workflow == "저장된 연구 재개",
        help="검증 성능과 호환성을 통과한 기존 수식을 초기 GP 개체군에 넣습니다.",
    )
    elite_ratio = st.slider(
        "기존 수식 계승 비율",
        min_value=0.0,
        max_value=0.8,
        value=float(resume_config.get("elite_seed_ratio", 0.2)),
        step=0.05,
        disabled=workflow == "저장된 연구 재개" or not elite_enabled,
        help="나머지 개체는 무작위로 새로 생성합니다.",
    )
    duration_option = st.selectbox("실행 시간", ("30분", "1시간", "3시간", "6시간", "사용자 지정"))
    duration_minutes = {"30분": 30, "1시간": 60, "3시간": 180, "6시간": 360}.get(duration_option)
    if duration_option == "사용자 지정":
        duration_minutes = st.number_input("분", min_value=1, max_value=1440, value=30)
    population = st.number_input(
        "개체군 크기",
        min_value=8,
        max_value=1000,
        value=int(resume_config.get("population_size", 32)),
        step=8,
        disabled=workflow == "저장된 연구 재개",
    )
    generations = st.number_input(
        "세대 수",
        min_value=1,
        max_value=1000,
        value=int(resume_config.get("generations", 10)),
        disabled=workflow == "저장된 연구 재개",
    )
    workers = st.slider(
        "CPU 워커",
        1,
        4,
        int(resume_config.get("workers", 4)),
        disabled=workflow == "저장된 연구 재개",
    )
    controls = st.number_input(
        "무작위 대조 반복",
        min_value=0,
        max_value=1000,
        value=int(resume_config.get("control_runs", 3)),
        disabled=workflow == "저장된 연구 재개",
    )
    seed = st.number_input(
        "기본 난수 시드",
        min_value=0,
        max_value=2**31 - 1,
        value=int(resume_config.get("seed", 12345)),
        disabled=workflow == "저장된 연구 재개",
    )

dataset = None
integrity = None
checkpoint = None
pending_checkpoint = st.session_state.get("pending_research_checkpoint")
stored_checkpoint = st.session_state.get("active_research_checkpoint")
active_checkpoint_path = (
    project_path(
        pending_checkpoint or stored_checkpoint or selected_checkpoint or checkpoint_paths[0]
    )
    if (pending_checkpoint or stored_checkpoint or selected_checkpoint or checkpoint_paths)
    else None
)
checkpoint_relocated = None
if active_checkpoint_path is not None:
    resolved_path, alternate_paths = locate_checkpoint(active_checkpoint_path)
    if resolved_path.is_file():
        active_checkpoint_path = resolved_path
    elif alternate_paths:
        checkpoint_relocated = alternate_paths[0]
        active_checkpoint_path = checkpoint_relocated
        st.session_state["active_research_checkpoint"] = str(checkpoint_relocated)
        st.session_state.pop("pending_research_checkpoint", None)
        st.warning(
            f"기록된 경로는 찾지 못했지만 같은 이름의 체크포인트를 발견했습니다: "
            f"{checkpoint_relocated}"
        )
    elif pending_checkpoint:
        pending_pid = st.session_state.get("pending_research_worker_pid")
        try:
            os.kill(int(pending_pid), 0)
            pending_worker_alive = True
        except (OSError, TypeError, ValueError):
            pending_worker_alive = False
        if pending_worker_alive:
            st.info("연구 프로세스가 시작되었습니다. 첫 체크포인트 저장을 기다리는 중입니다.")
        else:
            st.error(
                f"첫 체크포인트가 생성되기 전에 연구 프로세스가 종료되었습니다. "
                f"실행 로그를 확인하세요: {checkpoint_dir / 'research_worker.log'}"
            )
            st.session_state.pop("pending_research_checkpoint", None)
        active_checkpoint_path = None
    else:
        st.error(
            f"체크포인트 파일이 실제로 없습니다: {resolved_path}. "
            "프로젝트의 checkpoints 폴더도 확인했지만 같은 파일을 찾지 못했습니다. "
            "아래 저장된 체크포인트 목록에서 복구 가능한 실행 기록을 선택할 수 있습니다."
        )
        active_checkpoint_path = None
    if active_checkpoint_path is not None:
        try:
            checkpoint = load_checkpoint(active_checkpoint_path)
            if not checkpoint.get("research_id"):
                st.error(
                    "체크포인트에 연구 ID가 없습니다. 손상 가능성이 있어 자동 수정하지 않았습니다."
                )
                checkpoint = None
            elif (
                checkpoint.get("checkpoint_path")
                and project_path(checkpoint["checkpoint_path"]) != active_checkpoint_path
            ):
                st.error(
                    f"연구 ID {checkpoint['research_id']}가 다른 체크포인트 경로를 가리킵니다: "
                    f"{checkpoint['checkpoint_path']}"
                )
                checkpoint = None
            elif pending_checkpoint and active_checkpoint_path == project_path(pending_checkpoint):
                st.session_state["active_research_checkpoint"] = str(active_checkpoint_path)
                st.session_state.pop("pending_research_checkpoint", None)
                st.session_state.pop("pending_research_worker_pid", None)
        except (OSError, ValueError) as error:
            st.error(f"체크포인트 JSON을 읽을 수 없습니다: {error}")

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
        f"연구 ID: {state.get('research_id')} · 엔진: {', '.join(state.get('engines', []))} · "
        f"마지막 업데이트: {state.get('last_updated_at', '없음')} · "
        f"재개 방식: 실험 경계에서 다음 시드로 계속"
    )
    if state.get("completed_experiments"):
        st.caption(
            f"복원된 난수 시드 진행: 다음 시드 {state.get('rng_resume', {}).get('next_seed')} · "
            f"완료 실험 기록 {len(state['completed_experiments'])}건"
        )
    if state.get("current_progress"):
        st.json(state["current_progress"])
    if state.get("last_run_summary"):
        st.json(state["last_run_summary"])
    if state.get("failed_runs"):
        st.warning(f"실패 또는 중단된 실험: {len(state['failed_runs'])}")
    current_experiment_id = state.get("current_progress", {}).get("event", {}).get("experiment_id")
    if (
        current_experiment_id
        and not (project_path("experiments") / f"{current_experiment_id}.json").is_file()
    ):
        st.info(
            f"현재 진행 기록의 실험 {current_experiment_id}는 "
            "완료 실험 JSON으로 저장되지 않았습니다. "
            "기존 파일은 보존했으며, 재개 시 체크포인트의 다음 시드부터 계속합니다."
        )
else:
    st.info("연구를 시작하면 진행 상황과 재개용 체크포인트가 로컬에 저장됩니다.")

active_worker = False
if not checkpoint and st.session_state.get("pending_research_worker_pid"):
    try:
        os.kill(int(st.session_state["pending_research_worker_pid"]), 0)
        active_worker = True
    except (OSError, TypeError, ValueError):
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
    for experiment_path in project_path("experiments").glob("*.json"):
        experiment = ExperimentStore(project_path("experiments")).load(experiment_path.stem)
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
            checkpoint_path = project_path(selected_checkpoint)
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
            "--elite-seed-ratio",
            str(float(elite_ratio)),
            "--engine",
            engine,
        ]
        if elite_enabled:
            command.append("--elite-seeding")
        if resume:
            command.append("--resume")
        creation_flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
            subprocess, "CREATE_NO_WINDOW", 0
        )
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
            close_fds=True,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        log_file.close()
        st.session_state["pending_research_checkpoint"] = str(checkpoint_path)
        st.session_state["pending_research_worker_pid"] = process.pid
        st.success(f"연구 프로세스를 시작했습니다. 체크포인트 저장을 기다립니다: {checkpoint_path}")
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
if workflow == "새 장시간 연구" and elite_enabled:
    seed_preview, preview_skipped = library.select_elite_candidates(
        feature_profile=feature_profile,
        max_lag=int(resume_config.get("max_lag", 10)),
        max_candidates=2 * ceil(float(population) * float(elite_ratio)),
    )
    st.caption(
        f"현재 프로파일과 검증 기준에 맞는 계승 후보 {len(seed_preview)}개 · "
        f"제외된 후보 {len(preview_skipped)}개 · 새 무작위 개체 비율 {1 - elite_ratio:.0%}"
    )
    if seed_preview:
        st.dataframe(
            [
                {
                    "패턴 ID": item["pattern_id"],
                    "모델": item.get("model_type"),
                    "프로파일": item.get("feature_profile"),
                    "검증 점수": item.get("performance", {}).get("validation_score"),
                    "활성화 비율": item.get("performance", {}).get("validation_activation_rate"),
                    "선정 점수": item.get("elite_selection_score"),
                    "수식": item.get("formulas"),
                }
                for item in seed_preview
            ],
            width="stretch",
            hide_index=True,
        )
    if preview_skipped:
        with st.expander("계승 후보 제외 사유"):
            st.dataframe(preview_skipped, width="stretch", hide_index=True)
if patterns:
    profile_filter = st.selectbox(
        "프로파일 필터", ("전체", "legacy", "calendar_round"), key="pattern_profile_filter"
    )
    visible_patterns = [
        item
        for item in patterns
        if profile_filter == "전체" or item.get("feature_profile", "legacy") == profile_filter
    ]
    st.dataframe(
        [
            {
                "패턴 ID": item["pattern_id"],
                "모델": item.get("model_type"),
                "프로파일": item.get("feature_profile", "legacy"),
                "엔진": item.get("engine"),
                "상태": item.get("current_status"),
                "발견 방식": item.get("discovery_mode"),
                "계승 횟수": item.get("inheritance_count", 0),
                "자식 수": item.get("child_count", 0),
                "검증 점수": item.get("performance", {}).get("validation_score"),
                "조건": item.get("activation_condition"),
                "수식": item.get("formulas"),
                "실험": item.get("experiment_id"),
            }
            for item in visible_patterns
        ],
        width="stretch",
        hide_index=True,
    )
    lineage = library.list_lineage()
    if lineage:
        with st.expander("수식 계보와 변형 이력"):
            st.dataframe(
                [
                    {
                        "부모 패턴": item["parent_pattern_id"],
                        "자식 패턴": item["child_pattern_id"],
                        "연구 ID": item.get("research_id"),
                        "세대": item["generation"],
                        "생성 방식": item["method"],
                        "사용 연산자": item["operators"],
                        "부모 수식": item.get("parent_formula"),
                        "자식 수식": item.get("child_formula"),
                    }
                    for item in lineage
                ],
                width="stretch",
                hide_index=True,
            )
else:
    st.info("저장된 패턴이 없습니다.")

st.subheader("누적 연구 전략 비교")
comparison_rows = []
for experiment in ExperimentStore(project_path("experiments")).list():
    if (
        experiment.get("status") != "completed"
        or experiment.get("experiment_type") != "intermittent"
    ):
        continue
    elite = experiment.get("elite_seeding", {})
    one = experiment.get("one_number_model", {}).get("selected", {})
    six = experiment.get("six_number_set_model", {}).get("selected", {})
    comparison_rows.append(
        {
            "실험 ID": experiment.get("experiment_id"),
            "전략": "Elite Seeding" if elite.get("enabled") else "신규 무작위 초기화",
            "계승 비율": elite.get("ratio", 0),
            "실제 계승 개체 수": sum(
                int(elite.get(key, {}).get("seeded_count", 0))
                for key in ("one_number_search", "six_number_set_search")
            ),
            "실행 시간(초)": experiment.get("runtime_seconds"),
            "신규 등록 패턴": len(experiment.get("pattern_library_candidates", [])),
            "1개 번호 검증": one.get("validation_score"),
            "6개 조합 검증": six.get("validation_score"),
            "독립 테스트 점수": {
                "1개 번호": one.get("test_score"),
                "6개 조합": six.get("test_score"),
            },
            "테스트 완전 일치": int(one.get("test_full_set_matches", 0) or 0)
            + int(six.get("test_full_set_matches", 0) or 0),
            "독립 대조 해석": elite.get("control_comparability", "기존 실험 기록 없음"),
        }
    )
    paired = experiment.get("paired_unseeded_comparison")
    if paired:
        paired_one = paired["one_number_model"]["selected"]
        paired_six = paired["six_number_set_model"]["selected"]
        comparison_rows.append(
            {
                "실험 ID": experiment.get("experiment_id"),
                "전략": "같은 자료 무작위 초기화 대조",
                "계승 비율": 0,
                "실제 계승 개체 수": 0,
                "실행 시간(초)": paired.get("runtime_seconds"),
                "신규 등록 패턴": 0,
                "1개 번호 검증": paired_one.get("validation_score"),
                "6개 조합 검증": paired_six.get("validation_score"),
                "독립 테스트 점수": {
                    "1개 번호": paired_one.get("test_score"),
                    "6개 조합": paired_six.get("test_score"),
                },
                "테스트 완전 일치": int(paired_one.get("test_full_set_matches", 0) or 0)
                + int(paired_six.get("test_full_set_matches", 0) or 0),
                "독립 대조 해석": "동일한 자료 분할·개체군·세대 수",
            }
        )
if comparison_rows:
    st.dataframe(comparison_rows, width="stretch", hide_index=True)
    if any(row["전략"] == "Elite Seeding" for row in comparison_rows):
        st.caption(
            "실제 자료에서 후보를 선택한 실험은 무작위 대조군이 누적 이력을 재현하지 않습니다. "
            "따라서 대조 해석을 제한적으로 표시하며, 검증 점수와 독립 테스트 점수를 구분합니다."
        )
else:
    st.info("비교할 완료 연구 JSON이 아직 없습니다.")

if checkpoint and checkpoint.get("elite_seeding"):
    elite_state = checkpoint["elite_seeding"]
    st.subheader("이 연구의 계승 후보")
    st.caption(
        f"Elite Seeding: {'사용' if elite_state.get('enabled') else '사용 안 함'} · "
        f"계승 비율 {elite_state.get('ratio', 0):.0%} · "
        f"선정 후보 {len(elite_state.get('source_pattern_ids', []))}개"
    )
    if elite_state.get("source_pattern_ids"):
        st.write("계승 원본 패턴 ID", elite_state["source_pattern_ids"])
    if elite_state.get("skipped"):
        st.warning("호환성 또는 품질 기준에서 제외된 후보")
        st.dataframe(elite_state["skipped"], width="stretch", hide_index=True)

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
