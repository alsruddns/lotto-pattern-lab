"""Run an immutable historical replay of a saved six-number GP candidate."""

from __future__ import annotations

import csv
import hashlib
import io
import json

import streamlit as st

from lotto_pattern_lab.dataset import load_csv
from lotto_pattern_lab.formula_replay import replay_formula
from lotto_pattern_lab.paths import project_path
from lotto_pattern_lab.storage import ExperimentStore

st.set_page_config(page_title="전체 회차 재검증", layout="wide")
st.title("전체 회차 재검증")
st.caption(
    "저장된 6개 집합 GP 수식을 수정하거나 재학습하지 않고 과거 회차에 다시 적용합니다. "
    "이 화면의 결과는 Full Historical Replay이며 미래 독립 검증이 아닙니다."
)

experiment_paths = sorted(
    project_path("experiments").glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True
)
if not experiment_paths:
    st.info("저장된 실험이 없습니다.")
    st.stop()

selected_path = st.selectbox("원본 실험", experiment_paths, format_func=lambda path: path.stem)
store = ExperimentStore(project_path("experiments"))
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
        f"후보 {index + 1} · 적중 "
        f"{candidates[index].get('at_least_3_hits_by_split', {}).get('independent_test', 0)}회 "
        f"(원본 테스트) · 복잡도 {candidates[index].get('complexity', '미상')}"
    ),
)
candidate = candidates[candidate_index]
formula_payload = json.dumps(
    {
        "gate": candidate.get("activation_condition"),
        "formulas": candidate.get("formulas"),
        "profile": (experiment.get("feature_schema") or {}).get("feature_profile"),
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
start_round = int(left.number_input("시작 회차", min_value=1, value=753, step=1))
end_round = int(right.number_input("종료 회차", min_value=1, value=1246, step=1))
dataset_path = project_path("data/official/draws.csv")
if not dataset_path.exists():
    st.error("공식 데이터 파일을 찾을 수 없습니다.")
    st.stop()

if st.button("전체 회차 재검증 실행", type="primary", disabled=start_round > end_round):
    dataset = load_csv(dataset_path)
    with st.spinner("저장된 수식을 회차별로 적용하고 별도 결과 파일에 저장합니다…"):
        try:
            result = replay_formula(
                dataset=dataset,
                experiment=experiment,
                model_key="six_number_set_model",
                candidate=candidate,
                formula_id=formula_id,
                start_round=start_round,
                end_round=end_round,
            )
            st.session_state["last_formula_replay"] = result
        except (ArithmeticError, OSError, ValueError, TypeError) as error:
            st.error(f"재검증에 실패했습니다: {error}")

result = st.session_state.get("last_formula_replay")
if result and result.get("formula_id") == formula_id:
    st.success(f"결과 저장 완료 · 실행 ID {result['run_id']} · {result['result_path']}")
    metrics = st.columns(6)
    metrics[0].metric("대상 회차", result["target_round_count"])
    metrics[1].metric("계산 가능", result["evaluable_round_count"])
    metrics[2].metric("활성 예측", result["active_prediction_count"])
    metrics[3].metric("게이트 비활성", result["inactive_round_count"])
    metrics[4].metric("입력 부족", result["input_unavailable_count"])
    metrics[5].metric("실행 오류", result["execution_error_count"])
    st.metric(
        "평균 일치",
        "—"
        if result["mean_hits_when_active"] is None
        else f"{result['mean_hits_when_active']:.3f}",
    )
    st.caption(
        "정확히 일치 분포: "
        + " · ".join(
            f"{hits}개 {count}회" for hits, count in result["match_distribution_exact"].items()
        )
        + f" · 3개 이상 {result['at_least_3']}회 · 4개 이상 {result['at_least_4']}회"
        + f" · 5개 이상 {result['at_least_5']}회 · 6개 {result['at_least_6']}회"
    )
    rows = result["rows"]
    threshold = st.selectbox("일치 결과 필터", ("전체", "3개 이상", "4개 이상", "5개 이상", "6개"))
    if threshold != "전체":
        limit = int(threshold[0])
        rows = [
            row
            for row in rows
            if row.get("match_count") is not None and row["match_count"] >= limit
        ]
    display_rows = [
        {
            **row,
            "predicted_numbers": ", ".join(map(str, row["predicted_numbers"])),
            "winning_numbers": ", ".join(map(str, row["winning_numbers"])),
            "matched_numbers": ", ".join(map(str, row["matched_numbers"])),
        }
        for row in rows
    ]
    st.dataframe(display_rows, use_container_width=True, hide_index=True)
    csv_buffer = io.StringIO(newline="")
    if display_rows:
        writer = csv.DictWriter(csv_buffer, fieldnames=list(display_rows[0]))
        writer.writeheader()
        writer.writerows(display_rows)
    st.download_button(
        "회차별 CSV 다운로드",
        data=csv_buffer.getvalue().encode("utf-8-sig"),
        file_name=f"formula_replay_{result['run_id']}.csv",
        mime="text/csv",
    )

st.subheader("평가 구간 구분")
st.markdown(
    "학습 · 후보 선택 검증 · 원본 실험 테스트 · 연구에서 확인한 과거 회차를 구분해 표시합니다. "
    "이 전체 재적용은 후보 선택과 학습에 이미 관여한 과거 구간을 포함하므로 "
    "독립 미래 검증으로 해석하지 않습니다."
)
