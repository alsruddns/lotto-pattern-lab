"""다음 회차 추천 연구 화면."""

import hashlib
import tempfile
from datetime import date
from pathlib import Path

import streamlit as st

from lotto_pattern_lab.dataset import load_csv
from lotto_pattern_lab.official_data import _fetch_latest_round
from lotto_pattern_lab.paths import project_path
from lotto_pattern_lab.pattern_library import PatternLibrary, register_prospective_comparison
from lotto_pattern_lab.prediction import (
    compare_recommendation,
    create_recommendation,
    load_prediction_history,
    save_comparison,
    save_recommendation,
    summarize_prediction_history,
)
from lotto_pattern_lab.storage import ExperimentStore

st.set_page_config(page_title="AI 다음 회차 번호 추천", layout="wide")
st.title("AI 다음 회차 번호 추천")
st.warning("저장된 GP 공식을 실행하는 연구용 후보입니다. 당첨 확률이나 당첨을 보장하지 않습니다.")

experiments = project_path("experiments")
prediction_dir = project_path("predictions")
experiment_paths = sorted(experiments.glob("*.json"))
experiment_ids = [path.stem for path in experiment_paths]
default_id = "56f27190-1f2a-4573-9542-b30a2bbc60b4"
default_index = experiment_ids.index(default_id) if default_id in experiment_ids else 0

with st.sidebar:
    st.header("예측 입력")
    source = st.radio(
        "데이터 선택", ("로컬 데이터 753~1246회", "공식 데이터 파일", "CSV 파일 업로드")
    )
    uploaded = st.file_uploader("CSV 파일", type=("csv",)) if source == "CSV 파일 업로드" else None
    target = st.number_input("추천 대상 회차", min_value=2, value=1247, step=1)
    target_date = st.date_input(
        "\uc608\uc815 \ucd94\ucc28\uc77c \u0028\ub0a0\uc9dc \ud2b9\uc9d5 \ubaa8\ub378\uc6a9\u0029",
        value=date.today(),
    )
    experiment_id = (
        st.selectbox("기존 완료 실험", experiment_ids, index=default_index)
        if experiment_ids
        else ""
    )
    if not experiment_ids:
        st.error("experiments/에 완료된 실험 JSON이 없습니다.")
    model_label = st.radio("모델 종류", ("한 번호 모델", "6개 번호 집합 모델"))
    model_type = "one_number" if model_label == "한 번호 모델" else "six_number_set"
    seed = st.number_input("무작위 비교 기준 시드", min_value=0, max_value=2**31 - 1, value=1247)

dataset = None
data_bytes = None
try:
    if source == "로컬 데이터 753~1246회":
        csv_path = project_path("data/raw/lotto_draws_753_1246.csv")
        if csv_path.is_file():
            data_bytes = csv_path.read_bytes()
            dataset = load_csv(csv_path)
    elif source == "공식 데이터 파일":
        csv_path = project_path("data/official/draws.csv")
        if csv_path.is_file():
            data_bytes = csv_path.read_bytes()
            dataset = load_csv(csv_path)
    elif uploaded is not None:
        data_bytes = uploaded.getvalue()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temporary:
            temporary.write(data_bytes)
            temporary_path = Path(temporary.name)
        try:
            dataset = load_csv(temporary_path)
        finally:
            temporary_path.unlink(missing_ok=True)
except Exception as error:
    st.error(str(error))

if dataset is None:
    st.info("선택한 CSV를 사용할 수 없습니다. 파일 경로 또는 업로드를 확인하세요.")
else:
    rounds = [record.round_number for record in dataset.records]
    st.write(
        f"검증 상태: {dataset.verification_status} · "
        f"회차 {rounds[0]}~{rounds[-1]} ({len(rounds)}회)"
    )
    st.caption(
        "대상 회차 이전의 연속 10회만 입력 특징에 사용합니다. 이후 결과는 예측 입력에서 제외합니다."
    )

if st.button(
    "공식 실행 및 추천 저장", type="primary", disabled=dataset is None or not experiment_id
):
    try:
        experiment = ExperimentStore(experiments).load(experiment_id)
        try:
            official_latest = _fetch_latest_round()
        except Exception as error:
            official_latest = None
            st.warning(f"공식 최신 회차 확인 실패: {error}")
        recommendation = create_recommendation(
            experiment,
            dataset,
            int(target),
            model_type,
            data_file_sha256=hashlib.sha256(data_bytes).hexdigest() if data_bytes else None,
            random_seed=int(seed),
            official_latest_round=official_latest,
            target_date=target_date,
        )
        saved_path = save_recommendation(recommendation, prediction_dir)
        st.session_state["active_recommendation_id"] = recommendation["recommendation_id"]
        st.success(f"추천 결과를 저장했습니다: {saved_path}")
    except Exception as error:
        st.exception(error)

history = load_prediction_history(prediction_dir)
active_id = st.session_state.get("active_recommendation_id")
recommendation = next((item for item in history if item["recommendation_id"] == active_id), None)
if recommendation:
    st.subheader(f"추천 기록: {recommendation['recommendation_id']}")
    st.write(
        f"대상 {recommendation['target_round']}회 · 실험 {recommendation['experiment_id']} · "
        f"입력 데이터 {recommendation['input_through_round']}회까지 · "
        f"상태 {recommendation['recommendation_status']}"
    )
    if recommendation["recommendation_status"] == "historical_reproduction":
        st.warning("추첨 이후 생성한 과거 재현 기록입니다. 사전 예측이 아닙니다.")
    elif recommendation["recommendation_status"] == "timing_unverified":
        st.warning("공식 최신 회차를 확인하지 못했습니다. 사전 예측 여부는 미검증입니다.")
    else:
        st.info("생성 시점의 공식 최신 회차 기준으로 사전 추천 상태를 기록했습니다.")
    pattern_context = recommendation.get("pattern_context", {})
    if pattern_context:
        st.caption(
            f"계산 수식 패턴 ID: {pattern_context.get('pattern_id') or '라이브러리 ID 없음'} · "
            f"원본 연구: {pattern_context.get('origin_research_id') or '기존 연구'} · "
            f"계승 횟수: {pattern_context.get('inheritance_count', 0)} · "
            f"이번 회차 활성화: {'예' if pattern_context.get('active_for_target') else '아니요'} · "
            f"신규 회차 결과: {pattern_context.get('prospective_round_result', '미평가')}"
        )
        st.info("계승 횟수는 계산 기록이며 추천 신뢰도를 뜻하지 않습니다.")
    if recommendation["gate_result"]["branch"] == "inactive":
        st.warning("패턴 비활성: 게이트에 따라 번호를 생성하지 않았습니다.")
    else:
        st.success(f"패턴 활성: 규칙 {recommendation['selected_rule']}")
        balls = "".join(
            f'<span style="display:inline-flex;align-items:center;justify-content:center;'
            f"width:54px;height:54px;margin:5px;border-radius:50%;background:#f5c542;"
            f'color:#172033;font-weight:700;font-size:20px">{number}</span>'
            for number in recommendation["recommended_numbers"]
        )
        st.markdown(balls, unsafe_allow_html=True)
        if recommendation["model_type"] == "six_number_set":
            st.caption("번호는 순서 없는 조합이며 실제 추첨 순서를 뜻하지 않습니다.")
        st.dataframe(recommendation["formula_calculations"], width="stretch")
    with st.expander("활성화 공식과 변수 의미"):
        st.write("게이트 수식", recommendation["activation_condition"])
        st.json(recommendation["gate_calculation"])
        st.json(
            {
                key: {"feature": recommendation["feature_mapping"][key], "value": value}
                for key, value in recommendation["feature_values"].items()
                if key in recommendation["gate_calculation"]["variable_values"]
            }
        )
        st.write("검증 지표", recommendation["model_validation"])
        st.write("별도 무작위 기준", recommendation["random_baseline"])
    actual = (
        next(
            (
                record
                for record in dataset.records
                if record.round_number == recommendation["target_round"]
            ),
            None,
        )
        if dataset is not None
        else None
    )
    if actual is None:
        st.info("대상 회차 결과가 아직 데이터에 없습니다.")
    else:
        comparison = compare_recommendation(recommendation, actual.draw.numbers)
        st.subheader("실제 결과 비교")
        st.json(comparison)
        if st.button("비교 결과를 별도 기록으로 저장"):
            st.success(f"비교 기록: {save_comparison(comparison)}")
            pattern_id = register_prospective_comparison(
                recommendation, comparison, PatternLibrary()
            )
            if pattern_id:
                st.info(f"고정된 후보 패턴의 사전 검증 이력을 추가했습니다: {pattern_id}")

st.subheader("저장된 추천 이력")
if history:
    st.dataframe(
        [
            {
                "추천 ID": item["recommendation_id"],
                "회차": item["target_round"],
                "모델": item["model_type"],
                "상태": item["recommendation_status"],
                "활성 분기": item["gate_result"]["branch"],
                "번호": item["recommended_numbers"],
                "생성 시각": item["created_at"],
            }
            for item in history
        ],
        width="stretch",
        hide_index=True,
    )
else:
    st.info("저장된 추천이 없습니다. 실행 전에 결과를 미리 만들지 않습니다.")
if dataset is not None:
    st.subheader("실제 결과가 연결된 사전 추천 누적 현황")
    totals = summarize_prediction_history(history, dataset)
    st.json(totals) if totals else st.info("실제 결과까지 확인된 사전 추천이 없습니다.")
