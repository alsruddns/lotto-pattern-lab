"""Saved formula historical replay and background candidate queue UI."""

from __future__ import annotations

import csv
import hashlib
import io
import json

import streamlit as st

from lotto_pattern_lab.dataset import load_csv
from lotto_pattern_lab.formula_replay_queue import (
    discover_high_hit_candidates,
    enqueue_formula_replay,
    list_replay_jobs,
    recover_replay_jobs,
    resume_replay_job,
)
from lotto_pattern_lab.paths import project_path
from lotto_pattern_lab.storage import ExperimentStore

st.set_page_config(page_title="전체 회차 재검증", layout="wide")
st.title("전체 회차 재검증")
job_status_labels = {
    "queued": "대기 중",
    "running": "실행 중",
    "completed": "완료",
    "failed": "실패",
    "interrupted": "중단됨",
    "corrupt": "기록 손상",
}
row_status_labels = {
    "OK": "평가 완료",
    "INACTIVE": "게이트 비활성",
    "INPUT_UNAVAILABLE": "입력 부족",
    "EXECUTION_ERROR": "실행 오류",
}
st.caption(
    "저장된 6개 집합 수식을 수정하거나 재학습하지 않고 과거 회차에 다시 적용합니다. "
    "이 결과는 전체 과거 재적용이며 미래 독립 검증이 아닙니다."
)

experiments_directory = project_path("experiments")
experiment_paths = sorted(
    experiments_directory.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True
)
if not experiment_paths:
    st.info("저장된 실험이 없습니다.")
    st.stop()

store = ExperimentStore(experiments_directory)
selected_path = st.selectbox("원본 실험", experiment_paths, format_func=lambda path: path.stem)
try:
    experiment = store.load(selected_path.stem)
except (OSError, ValueError, KeyError) as error:
    st.error(f"실험 기록을 읽지 못했습니다: {error}")
    st.stop()

model = experiment.get("six_number_set_model") or {}
candidates = model.get("candidates") or []
if not candidates:
    st.info("선택한 실험에는 재검증 가능한 6개 집합 후보 수식이 없습니다.")
    st.stop()

candidate_index = st.selectbox(
    "후보 수식",
    range(len(candidates)),
    format_func=lambda index: (
        f"후보 {index + 1} · 원본 테스트에서 "
        f"3개 이상 "
        f"{candidates[index].get('at_least_3_hits_by_split', {}).get('independent_test', 0)}회 "
        f"· 복잡도 {candidates[index].get('complexity', '미상')}"
    ),
)
candidate = candidates[candidate_index]
profile = (experiment.get("feature_schema") or {}).get(
    "feature_profile", experiment.get("config", {}).get("feature_profile", "legacy")
)
dataset_path = project_path(
    "data/raw/lotto_draws_753_1246_with_bonus.csv"
    if profile == "bonus_lag_v1"
    else "data/official/draws.csv"
)
if not dataset_path.is_file():
    st.error(f"현재 수식에 필요한 데이터 파일을 찾을 수 없습니다: {dataset_path.name}")
    st.stop()
try:
    dataset = load_csv(dataset_path)
    current_data_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
except (OSError, ValueError) as error:
    st.error(f"당첨번호 데이터를 읽지 못했습니다: {error}")
    st.stop()
st.caption(f"재검증 입력 데이터: {dataset_path.name} · SHA-256 `{current_data_hash}`")
formula_payload = json.dumps(
    {
        "gate": candidate.get("activation_condition"),
        "formulas": candidate.get("formulas"),
        "profile": profile,
        "experiment_id": experiment.get("experiment_id"),
    },
    sort_keys=True,
    ensure_ascii=False,
    separators=(",", ":"),
)
formula_id = hashlib.sha256(formula_payload.encode("utf-8")).hexdigest()[:16]
st.code(candidate.get("activation_condition", ""), language="text")
st.caption(f"수식 ID: {formula_id} · 원본 실험 ID: {experiment.get('experiment_id')}")

left, right = st.columns(2)
minimum_round = min((record.round_number for record in dataset.records), default=1)
maximum_round = max((record.round_number for record in dataset.records), default=1)
start_round = int(left.number_input("시작 회차", min_value=1, value=753, step=1))
end_round = int(
    right.number_input("종료 회차", min_value=1, value=max(1246, maximum_round), step=1)
)
if start_round < minimum_round:
    st.info(
        f"데이터에서 가장 이른 회차는 {minimum_round}회입니다. "
        "이전 회차는 입력 부족으로 기록됩니다."
    )
if start_round > end_round:
    st.warning("시작 회차는 종료 회차보다 클 수 없습니다.")

if st.button("재검증 작업 등록", type="primary", disabled=start_round > end_round):
    try:
        job = enqueue_formula_replay(
            dataset_path=dataset_path,
            dataset_sha256=current_data_hash,
            experiment=experiment,
            model_key="six_number_set_model",
            candidate=candidate,
            formula_id=formula_id,
            start_round=start_round,
            end_round=end_round,
        )
        st.session_state["last_replay_job_id"] = job["job_id"]
        if job.get("deduplicated"):
            st.info(f"같은 수식과 데이터의 작업이 이미 있습니다: {job['job_id']}")
        else:
            st.success(f"백그라운드 재검증 작업을 등록했습니다: {job['job_id']}")
    except (OSError, ValueError) as error:
        st.error(f"작업 등록에 실패했습니다: {error}")

st.subheader("고적중 후보 일괄 재검증")
threshold = st.selectbox(
    "후보 등록 기준", (5, 4, 3, 6), format_func=lambda value: f"{value}개 이상"
)
try:
    discovered = discover_high_hit_candidates(minimum_hits=threshold)
except (OSError, ValueError) as error:
    discovered = []
    st.error(f"후보 수집에 실패했습니다: {error}")
st.caption(f"원본 실험에서 기준을 충족한 서로 다른 공식: {len(discovered)}개")
if discovered:
    st.dataframe(
        [
            {
                "수식 ID": item["formula_id"],
                "특징 프로파일": item["profile"],
                "원본 실험 수": len(set(item["origin_experiment_ids"])),
                "반복 적중 이벤트": item["repeated_hit_events"],
                "고유 적중 회차": len(item["original_hit_rounds"]),
                "고유 회차 목록": ", ".join(map(str, item["original_hit_rounds"])),
            }
            for item in discovered
        ],
        hide_index=True,
        width="stretch",
    )
    if st.button("후보 전체를 현재 범위로 작업 등록", disabled=start_round > end_round):
        registered = 0
        deduplicated = 0
        failures = []
        for item in discovered:
            try:
                job = enqueue_formula_replay(
                    dataset_path=dataset_path,
                    dataset_sha256=current_data_hash,
                    experiment=item["experiment"],
                    model_key="six_number_set_model",
                    candidate=item["candidate"],
                    formula_id=item["formula_id"],
                    start_round=start_round,
                    end_round=end_round,
                )
                deduplicated += int(bool(job.get("deduplicated")))
                registered += 1
            except (OSError, ValueError) as error:
                failures.append(f"{item['formula_id']}: {error}")
        st.success(f"{registered}개 후보를 처리했습니다. 기존 작업과 중복된 항목 {deduplicated}개.")
        if failures:
            st.error("일부 후보를 등록하지 못했습니다: " + "; ".join(failures[:5]))

st.subheader("재검증 작업 상태")
if st.button("작업 상태 새로고침"):
    st.rerun()
try:
    recovery = recover_replay_jobs()
    jobs = list_replay_jobs()
except (OSError, ValueError) as error:
    recovery = {"interrupted": 0, "queued": 0}
    jobs = []
    st.error(f"작업 상태를 읽지 못했습니다: {error}")
if recovery["interrupted"]:
    st.warning(
        f"재시작으로 중단된 작업 {recovery['interrupted']}개를 보존했습니다. "
        "다시 시작할 수 있습니다."
    )
if not jobs:
    st.info("등록된 재검증 작업이 없습니다.")
if jobs:
    st.dataframe(
        [
            {
                "수식 ID": job.get("formula_id"),
                "상태": job_status_labels.get(job.get("status"), "상태 미상"),
                "진행": f"{job.get('processed_rounds', 0)} / {job.get('total_rounds', 0)}",
                "범위": f"{job.get('start_round')}~{job.get('end_round')}",
                "경과 초": round(float(job.get("elapsed_seconds", 0) or 0), 1),
                "중복 요청": job.get("duplicate_request_count", 0),
                "작업 ID": job.get("job_id"),
            }
            for job in jobs
        ],
        hide_index=True,
        width="stretch",
    )
    selected_job_id = st.selectbox(
        "상세 결과를 확인할 작업",
        [job["job_id"] for job in jobs],
        format_func=lambda job_id: next(
            f"{job['formula_id']} · "
            f"{job_status_labels.get(job['status'], '상태 미상')} · {job['job_id']}"
            for job in jobs
            if job["job_id"] == job_id
        ),
    )
    job = next(item for item in jobs if item["job_id"] == selected_job_id)
    total = int(job.get("total_rounds", 0) or 0)
    processed = int(job.get("processed_rounds", 0) or 0)
    if total:
        st.progress(min(processed / total, 1.0), text=f"처리 회차 {processed} / {total}")
    st.caption(
        f"경과 {float(job.get('elapsed_seconds', 0) or 0):.1f}초 · "
        f"중복 요청 {job.get('duplicate_request_count', 0)}회"
    )
    remaining = job.get("estimated_remaining_seconds")
    if remaining is not None and job.get("status") == "running":
        st.caption(f"측정 기반 예상 잔여 시간: {float(remaining):.1f}초")
    if job.get("error"):
        st.error(job["error"])
    if job.get("status") in {"failed", "interrupted"}:
        if st.button("이 작업 다시 시작", key=f"resume-{job['job_id']}"):
            try:
                resume_replay_job(job["job_id"])
                st.rerun()
            except (OSError, ValueError) as error:
                st.error(f"작업을 다시 시작하지 못했습니다: {error}")
    if job.get("status") == "completed" and job.get("result_path"):
        summary = job.get("result_summary") or {}
        st.write(
            f"대상 {summary.get('target_round_count', 0)}회 · "
            f"평가 {summary.get('evaluable_round_count', 0)}회 · "
            f"3개 이상 {summary.get('at_least_3', 0)}회 · "
            f"4개 이상 {summary.get('at_least_4', 0)}회 · "
            f"5개 이상 {summary.get('at_least_5', 0)}회 · "
            f"6개 {summary.get('at_least_6', 0)}회"
        )
        result_path = project_path(job["result_path"])
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            rows = result.get("rows", [])
            st.caption(f"실행 ID: {result.get('run_id')} · 결과 파일: {result_path}")
            csv_buffer = io.StringIO(newline="")
            if rows:
                display_rows = [
                    {
                        **row,
                        "validation_status": row_status_labels.get(
                            row.get("validation_status"), "상태 미상"
                        ),
                        "predicted_numbers": ", ".join(map(str, row.get("predicted_numbers", []))),
                        "winning_numbers": ", ".join(map(str, row.get("winning_numbers", []))),
                        "matched_numbers": ", ".join(map(str, row.get("matched_numbers", []))),
                    }
                    for row in rows
                ]
                writer = csv.DictWriter(csv_buffer, fieldnames=list(display_rows[0]))
                writer.writeheader()
                writer.writerows(display_rows)
                threshold_filter = st.selectbox(
                    "회차 결과 필터", ("전체", "3개 이상", "4개 이상", "5개 이상", "6개")
                )
                if threshold_filter != "전체":
                    minimum_hits = int(threshold_filter[0])
                    display_rows = [
                        row
                        for row in display_rows
                        if row.get("match_count") is not None and row["match_count"] >= minimum_hits
                    ]
                st.dataframe(display_rows, hide_index=True, width="stretch")
                st.download_button(
                    "회차별 결과 CSV 다운로드",
                    data=csv_buffer.getvalue().encode("utf-8-sig"),
                    file_name=f"formula_replay_{result.get('run_id')}.csv",
                    mime="text/csv",
                    key=f"download-{job['job_id']}",
                )
            st.download_button(
                "재검증 결과 JSON 다운로드",
                data=result_path.read_bytes(),
                file_name=result_path.name,
                mime="application/json",
                key=f"download-json-{job['job_id']}",
            )

st.subheader("평가 구간 구분")
st.markdown(
    "학습 · 후보 선택 검증 · 원본 실험 테스트 · 연구에서 확인한 과거 회차를 구분합니다. "
    "과거 전체 재적용은 후보 선택과 학습에 관여한 회차를 포함하므로 독립 미래 검증이 아닙니다."
)
