"""Saved formula historical replay and background candidate queue UI."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3

import plotly.graph_objects as go
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
from lotto_pattern_lab.revalidation_catalog import (
    build_experiment_catalog,
    experiment_choice_label,
    filter_experiment_catalog,
    sort_experiment_catalog,
)
from lotto_pattern_lab.storage import ExperimentStore
from lotto_pattern_lab.temporal_analysis import analyze_temporal_performance

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
research_database = project_path("data/database/patterns.sqlite")

LABEL = {
    "refresh": "\uc2e4\ud5d8 \ubaa9\ub85d \uc0c8\ub85c\uace0\uce68",
    "all": "\uc804\uccb4",
    "engine": "\ud0d0\uc0c9 \uc5d4\uc9c4",
    "profile": "\ud2b9\uc9d5 \ud504\ub85c\ud30c\uc77c",
    "validation": "\uac80\uc99d \uc0c1\ud0dc",
    "independent": "\ub3c5\ub9bd \ud14c\uc2a4\ud2b8",
    "has_result": "\uacb0\uacfc \uc788\uc74c",
    "missing": "\ubbf8\uc218\uc9d1",
    "date_range": "\uc5f0\uad6c \ub0a0\uc9dc \ubc94\uc704",
    "use_date_filter": "\ub0a0\uc9dc \ubc94\uc704 \ud544\ud130 \uc801\uc6a9",
    "sort": "\uc815\ub82c",
    "recent": "\ucd5c\uadfc \uc5f0\uad6c\uc21c",
    "rate4": "\ub3c5\ub9bd \ud14c\uc2a4\ud2b8 4\uac1c+ \ube44\uc728",
    "rate5": "\ub3c5\ub9bd \ud14c\uc2a4\ud2b8 5\uac1c+ \ube44\uc728",
    "rate2": "2\ub4f1 \ube44\uc728",
    "rate6": "6\uac1c \uc644\uc804 \uc77c\uce58 \ube44\uc728",
    "random_p": "\ubb34\uc791\uc704 \ube44\uad50 p\uac12 \ub0ae\uc740 \uc21c",
    "no_matches": (
        "\uc120\ud0dd\ud55c \ud544\ud130\uc5d0 \ub9de\ub294 "
        "\uc2e4\ud5d8\uc774 \uc5c6\uc2b5\ub2c8\ub2e4.",
    ),
    "no_experiments": (
        "\uc7ac\uac80\uc99d\ud560 \uc218 \uc788\ub294 "
        "\uc800\uc7a5 \uc2e4\ud5d8\uc774 \uc5c6\uc2b5\ub2c8\ub2e4.",
    ),
}


@st.cache_data(ttl=20, show_spinner=False)
def _cached_experiment_catalog(database: str, directory: str) -> list[dict]:
    return build_experiment_catalog(database, directory)


if st.button(LABEL["refresh"], key="refresh_revalidation_catalog"):
    _cached_experiment_catalog.clear()
try:
    catalog = _cached_experiment_catalog(str(research_database), str(experiments_directory))
except (OSError, ValueError, sqlite3.Error) as error:
    st.error(
        f"\uc2e4\ud5d8 \ubaa9\ub85d\uc744 \uc77d\uc9c0 \ubabb\ud588\uc2b5\ub2c8\ub2e4: {error}"
    )
    st.stop()
if not catalog:
    st.info(LABEL["no_experiments"])
    st.stop()

engine_values = sorted({str(item["engine"]) for item in catalog if item.get("engine")})
profile_values = sorted(
    {str(item["feature_profile"]) for item in catalog if item.get("feature_profile")}
)
validation_values = sorted(
    {str(item["validation_status"]) for item in catalog if item.get("validation_status")}
)
filter_cols = st.columns(4)
engine_filter = filter_cols[0].selectbox(LABEL["engine"], [LABEL["all"], *engine_values])
profile_filter = filter_cols[1].selectbox(LABEL["profile"], [LABEL["all"], *profile_values])
validation_filter = filter_cols[2].selectbox(
    LABEL["validation"], [LABEL["all"], *validation_values]
)
independent_filter = filter_cols[3].selectbox(
    LABEL["independent"], [LABEL["all"], LABEL["has_result"], LABEL["missing"]]
)
hit_filter = st.selectbox(
    "\uc801\uc911 \uc218\uc900",
    [
        LABEL["all"],
        "3\uac1c \uc774\uc0c1 \uae30\ub85d",
        "4\uac1c \uc774\uc0c1 \uae30\ub85d",
        "5\uac1c \uc774\uc0c1 \uae30\ub85d",
        "2\ub4f1 \uae30\ub85d",
        "6\uac1c \uc644\uc804 \uc77c\uce58",
    ],
)

available_dates = sorted(
    str(item["research_date"]) for item in catalog if item.get("research_date")
)
if available_dates and st.checkbox(LABEL["use_date_filter"], value=False):
    from datetime import date

    date_range = st.date_input(
        LABEL["date_range"],
        value=(date.fromisoformat(available_dates[0]), date.fromisoformat(available_dates[-1])),
        key="revalidation_research_date_range",
    )
else:
    date_range = ()

sort_options = [LABEL[key] for key in ("recent", "rate4", "rate5", "rate2", "rate6", "random_p")]
sort_label = st.selectbox(LABEL["sort"], sort_options)
sort_keys = {
    LABEL["recent"]: ("research_date", True),
    LABEL["rate4"]: ("rate_4_plus", True),
    LABEL["rate5"]: ("rate_5_plus", True),
    LABEL["rate2"]: ("second_prize_rate", True),
    LABEL["rate6"]: ("rate_6_exact", True),
    LABEL["random_p"]: ("random_comparison_p", False),
}
hit_key = {
    "3\uac1c \uc774\uc0c1 \uae30\ub85d": "hits_3_plus",
    "4\uac1c \uc774\uc0c1 \uae30\ub85d": "hits_4_plus",
    "5\uac1c \uc774\uc0c1 \uae30\ub85d": "hits_5_plus",
    "2\ub4f1 \uae30\ub85d": "second_prize",
    "6\uac1c \uc644\uc804 \uc77c\uce58": "hits_6_exact",
}.get(hit_filter)
if date_range and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = None
filtered = filter_experiment_catalog(
    catalog,
    engine=engine_filter if engine_filter != LABEL["all"] else None,
    feature_profile=profile_filter if profile_filter != LABEL["all"] else None,
    validation_status=validation_filter if validation_filter != LABEL["all"] else None,
    independent_test=True
    if independent_filter == LABEL["has_result"]
    else False
    if independent_filter == LABEL["missing"]
    else None,
    hit_key=hit_key,
    start_date=start_date,
    end_date=end_date,
)
sort_key, descending = sort_keys[sort_label]
filtered = sort_experiment_catalog(filtered, key=sort_key, descending=descending)
if not filtered:
    st.info(LABEL["no_matches"])
    st.stop()

selected_by_id = {item["experiment_id"]: item for item in filtered}
selected_experiment_id = st.selectbox(
    "\uc6d0\ubcf8 \uc2e4\ud5d8",
    list(selected_by_id),
    format_func=lambda experiment_id: experiment_choice_label(selected_by_id[experiment_id]),
    key="revalidation_source_experiment_id",
)
source_summary = selected_by_id[selected_experiment_id]
st.dataframe(
    [
        {
            "\uc5f0\uad6c ID": source_summary.get("research_id") or "\uc815\ubcf4 \uc5c6\uc74c",
            "\uc5f0\uad6c \ud68c\ucc28": source_summary.get("research_round"),
            "\ud0d0\uc0c9 \uc5d4\uc9c4": source_summary.get("engine"),
            "\ud2b9\uc9d5 \ud504\ub85c\ud30c\uc77c": source_summary.get("feature_profile"),
            "\uc2e4\ud589 \uc0c1\ud0dc": source_summary.get("status"),
            "\uac80\uc99d \uc0c1\ud0dc": source_summary.get("validation_status"),
            "\ud3c9\uac00 \uc870\ud569 \uc218": source_summary.get("evaluation_count"),
            "3\uac1c+": source_summary.get("hits_3_plus"),
            "4\uac1c+": source_summary.get("hits_4_plus"),
            "5\uac1c+": source_summary.get("hits_5_plus"),
            "2\ub4f1": source_summary.get("second_prize"),
            "2\ub4f1 \ud310\uc815 \uac00\ub2a5 \uc870\ud569": source_summary.get(
                "second_prize_known"
            ),
            "6\uac1c \uc644\uc804 \uc77c\uce58": source_summary.get("hits_6_exact"),
            "4\uac1c+ 독립 비율": source_summary.get("rate_4_plus"),
            "5\uac1c+ 독립 비율": source_summary.get("rate_5_plus"),
            "2등 판정 비율": source_summary.get("second_prize_rate"),
            "6개 독립 비율": source_summary.get("rate_6_exact"),
            "\ub3c5\ub9bd \ud14c\uc2a4\ud2b8 \uc810\uc218": source_summary.get(
                "independent_test_score"
            ),
            "\ubb34\uc791\uc704 \ube44\uad50 p\uac12": source_summary.get("random_comparison_p"),
        }
    ],
    hide_index=True,
    width="stretch",
)
st.caption(f"\uc2e4\ud5d8 ID: {selected_experiment_id}")

store = ExperimentStore(experiments_directory)
try:
    experiment = store.load(selected_experiment_id)
except (OSError, ValueError, KeyError) as error:
    st.error(
        f"\uc120\ud0dd\ud55c \uc2e4\ud5d8 \uc0c1\uc138\ub97c "
        f"\uc77d\uc9c0 \ubabb\ud588\uc2b5\ub2c8\ub2e4: {error}"
    )
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
    if profile in {"bonus_lag_v1", "integrated_v1"}
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
    discoveries = job.get("discovery_events", [])
    if discoveries:
        source_experiments = ", ".join(job.get("origin_experiment_ids", [])) or "기록 없음"
        st.caption(
            f"연구 중 후보 발견 {len(discoveries)}건 · 원본 실험 {source_experiments} · "
            f"발견 당시 고유 적중 회차 {len(job.get('original_hit_rounds', []))}회"
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
        second_prize_display = (
            summary.get("second_prize_exact")
            if summary.get("second_prize_evaluable_count")
            else "미수집"
        )
        st.write(
            f"대상 {summary.get('target_round_count', 0)}회 · "
            f"평가 {summary.get('evaluable_round_count', 0)}회 · "
            f"3개 이상 {summary.get('at_least_3', 0)}회 · "
            f"4개 이상 {summary.get('at_least_4', 0)}회 · "
            f"5개 이상 {summary.get('at_least_5', 0)}회 · "
            f"2등 {second_prize_display}회 · "
            f"본번호 6개 {summary.get('at_least_6', 0)}회"
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
                    "회차 결과 필터", ("전체", "3개 이상", "4개 이상", "5개 이상", "2등", "6개")
                )
                if threshold_filter == "2등":
                    display_rows = [row for row in display_rows if row.get("prize_tier") == 2]
                elif threshold_filter != "전체":
                    minimum_hits = int(threshold_filter[0])
                    display_rows = [
                        row
                        for row in display_rows
                        if row.get("match_count") is not None and row["match_count"] >= minimum_hits
                    ]
                st.dataframe(display_rows, hide_index=True, width="stretch")
                with st.expander("구간별 성능 변화 분석", expanded=False):
                    window_size = st.selectbox(
                        "이동 구간 크기",
                        (30, 50, 100),
                        index=1,
                        help="각 점은 선택한 수의 연속 평가 회차를 요약합니다.",
                        key=f"temporal-window:{job['job_id']}",
                    )
                    temporal = analyze_temporal_performance(rows, window_size=int(window_size))
                    st.caption(
                        f"평가 가능 {temporal['evaluable_round_count']}회 · "
                        f"구간 {temporal['window_count']}개 · "
                        "전체 회차 재적용의 사후 분석"
                    )
                    if temporal["windows"]:
                        temporal_frame = temporal["windows"]
                        chart = go.Figure()
                        chart.add_trace(
                            go.Scatter(
                                x=[item["end_round"] for item in temporal_frame],
                                y=[item["average_hits_per_target"] for item in temporal_frame],
                                mode="lines+markers",
                                name="대상 회차당 평균 적중",
                            )
                        )
                        chart.add_trace(
                            go.Scatter(
                                x=[item["end_round"] for item in temporal_frame],
                                y=[item["gate_activation_rate"] for item in temporal_frame],
                                mode="lines+markers",
                                name="게이트 활성 비율",
                                yaxis="y2",
                            )
                        )
                        chart.update_layout(
                            xaxis_title="회차",
                            yaxis_title="대상 회차당 평균 적중 수",
                            yaxis2={
                                "title": "게이트 활성 비율",
                                "overlaying": "y",
                                "side": "right",
                                "range": [0, 1],
                            },
                            legend={"orientation": "h"},
                        )
                        st.plotly_chart(chart, width="stretch")
                        st.dataframe(temporal_frame, hide_index=True, width="stretch")
                        if temporal["descriptive_change_points"]:
                            st.dataframe(
                                temporal["descriptive_change_points"],
                                hide_index=True,
                                width="stretch",
                            )
                    else:
                        st.info("선택한 구간 크기만큼 평가 가능한 과거 회차가 없습니다.")
                    st.caption(temporal["interpretation"])
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
