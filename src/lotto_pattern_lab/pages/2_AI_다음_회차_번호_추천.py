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
    audit_candidate_history,
    compare_recommendation,
    create_recommendation,
    load_prediction_history,
    save_comparison,
    save_recommendation,
    summarize_prediction_history,
)
from lotto_pattern_lab.presentation import format_count, format_decimal, pretty_formula
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
    target = st.number_input(
        "\ucd94\ucc9c \ub300\uc0c1 \ud68c\ucc28",
        min_value=2,
        value=1247,
        step=1,
        key="prediction_target_round",
    )
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
    selected_experiment = None
    if experiment_id:
        try:
            selected_experiment = ExperimentStore(experiments).load(experiment_id)
        except (OSError, ValueError, KeyError) as error:
            st.error(f"실험 결과를 읽지 못했습니다: {error}")
    is_pysr = bool(
        selected_experiment and selected_experiment.get("engine") == "pysr_symbolic_regression"
    )
    model_label = st.radio(
        "\ubaa8\ub378 \uc885\ub958",
        (
            ("추천 번호 6개 집합",)
            if is_pysr and selected_experiment.get("config", {}).get("target") == "main_positions"
            else ("한 번호",)
            if is_pysr
            else ("추천 번호 6개 집합", "한 번호")
        ),
    )
    model_type = (
        "six_number_set"
        if model_label == "\ucd94\ucc9c \ubc88\ud638 6\uac1c \uc9d1\ud569"
        else "one_number"
    )
    selected_candidate_rank = 1
    if experiment_id:
        try:
            if is_pysr:
                target_name = selected_experiment.get("config", {}).get("target")
                target_names = (
                    {f"main_position_{index}" for index in range(1, 7)}
                    if target_name == "main_positions"
                    else {target_name}
                )
                candidate_rows = [
                    row
                    for row in selected_experiment.get("candidates", [])
                    if row.get("target") == sorted(target_names)[0]
                ]
            else:
                model_key = (
                    "one_number_model" if model_type == "one_number" else "six_number_set_model"
                )
                model_record = selected_experiment.get(model_key, {})
                candidate_rows = model_record.get("candidates") or [
                    model_record.get("selected", {})
                ]
            if candidate_rows and candidate_rows[0]:
                selected_candidate_rank = st.selectbox(
                    "\uc5f0\uad6c \ud6c4\ubcf4",
                    range(1, len(candidate_rows) + 1),
                    format_func=lambda rank: (
                        f"\ud6c4\ubcf4 {rank} \u00b7 \uac80\uc99d \uc810\uc218 "
                        f"{format_decimal(candidate_rows[rank - 1].get('validation_score'))}\uc810"
                    ),
                    help=(
                        "\uac80\uc99d \uc810\uc218\uc640 "
                        "\uc218\uc2dd\uc744 \ud655\uc778\ud558\uace0 "
                        "\ud6c4\ubcf4\ub97c \uace0\ub985\ub2c8\ub2e4."
                    ),
                )
            else:
                st.warning(
                    "\uc120\ud0dd\ud55c \ubaa8\ub378\uc5d0 "
                    "\ud6c4\ubcf4\uac00 \uc5c6\uc2b5\ub2c8\ub2e4."
                )
        except (OSError, ValueError, KeyError) as error:
            st.error(
                f"\uc2e4\ud5d8 \ud6c4\ubcf4\ub97c \uc77d\uc9c0 "
                f"\ubabb\ud588\uc2b5\ub2c8\ub2e4: {error}"
            )
    seed = st.number_input("무작위 비교 기준 시드", min_value=0, max_value=2**31 - 1, value=1247)

dataset = None
data_bytes = None
try:
    if source == "로컬 데이터 753~1246회":
        csv_path = project_path("data/raw/lotto_draws_753_1246_with_bonus.csv")
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

with st.expander("\uae30\uc874 \uace0\uc801\uc911 \ud328\ud134 \uc0ac\ub840", expanded=False):
    st.write("\ud328\ud134 ID: `b5f4b83e-20e3-47ea-9ac1-9d613e0e28f0`")
    st.write("\uc81c\uacf5\ub41c \uc0ac\ub840: 1118\ud68c 5\uac1c, 1207\ud68c 4\uac1c")
    st.caption(
        "\ubc18\ubcf5 \ud3c9\uac00\uc5d0\uc11c \ub098\uc628 "
        "\ud68c\uace0 \uc0ac\ub840\uc785\ub2c8\ub2e4. "
        "\ubbf8\ub798 \uc608\uce21 \uc131\ub2a5\uc774 \uc785\uc99d\ub41c "
        "\ud328\ud134\uc740 \uc544\ub2d9\ub2c8\ub2e4."
    )

if st.button(
    "공식 실행 및 추천 저장", type="primary", disabled=dataset is None or not experiment_id
):
    try:
        experiment = ExperimentStore(experiments).load(experiment_id)
        is_pysr = experiment.get("engine") == "pysr_symbolic_regression"
        try:
            official_latest = _fetch_latest_round()
        except Exception as error:
            official_latest = None
            st.warning(f"공식 최신 회차 확인 실패: {error}")
        if official_latest is None:
            raise ValueError("?? ?? ??? ???? ?? ?? ??? ???? ?????.")
        if int(target) != official_latest + 1:
            raise ValueError(f"?? ?? ?? ??? {official_latest + 1}????. ??? ??? ? ?? ?????.")
        if is_pysr:
            from lotto_pattern_lab.prediction import create_pysr_recommendation

            recommendation = create_pysr_recommendation(
                experiment,
                dataset,
                int(target),
                candidate_rank=selected_candidate_rank,
                data_file_sha256=hashlib.sha256(data_bytes).hexdigest() if data_bytes else None,
                official_latest_round=official_latest,
                target_date=target_date,
            )
        else:
            model_key = "one_number_model" if model_type == "one_number" else "six_number_set_model"
            model_record = experiment.get(model_key, {})
            candidate_rows = model_record.get("candidates") or [model_record.get("selected", {})]
            if not 1 <= selected_candidate_rank <= len(candidate_rows):
                raise ValueError("선택한 후보 순위가 실험 결과에 없습니다.")
            experiment[model_key]["selected"] = candidate_rows[selected_candidate_rank - 1]
            experiment["selected_candidate_rank"] = selected_candidate_rank
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
        if recommendation["recommendation_status"] != "pre_draw_prediction":
            raise ValueError("?? ? ?? ?? ??? ???? ?? ???? ?????.")
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
        with st.expander("\uc6d0\ubcf8 \uc218\uc2dd \ubc0f \uacc4\uc0b0 \ub0b4\uc5ed"):
            for formula in recommendation.get("used_formulas", []):
                st.markdown(f"\uc77d\uae30 \uc26c\uc6b4 \ud45c\uae30: `{pretty_formula(formula)}`")
                st.code(formula, language="text", wrap_lines=True)
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
if dataset is not None and selected_experiment is not None:
    if st.button(
        "\uc120\ud0dd \ud6c4\ubcf4\ub97c \uace0\uc720 \ud68c\ucc28\ub85c \ud68c\uace0 \uac80\uc99d"
    ):
        try:
            st.session_state["candidate_history_audit"] = audit_candidate_history(
                selected_experiment, dataset, model_type, selected_candidate_rank
            )
        except (OSError, ValueError, KeyError) as error:
            st.error(f"\ud68c\uace0 \uac80\uc99d \uc2e4\ud328: {error}")
audit_result = st.session_state.get("candidate_history_audit")
if audit_result:
    st.subheader("\uace0\uc720 \ud68c\ucc28 \ud68c\uace0 \uac80\uc99d")
    st.caption(
        "\uc0ac\uc804 \uc800\uc7a5 \uc2e4\uc801\uacfc \ubd84\ub9ac\ud55c "
        "\ud68c\uace0 \ud3c9\uac00\uc785\ub2c8\ub2e4. "
        "\uac01 \ud68c\ucc28\ub97c \ud55c \ubc88\uc529 "
        "\uacc4\uc0b0\ud569\ub2c8\ub2e4."
    )
    audit_metrics = st.columns(4)
    audit_metrics[0].metric(
        "\ud3c9\uac00 \ud68c\ucc28", format_count(audit_result["evaluation_rounds"])
    )
    audit_metrics[1].metric(
        "\ud65c\uc131 \ud68c\ucc28", format_count(audit_result["active_rounds"])
    )
    audit_metrics[2].metric("3\uac1c \uc774\uc0c1", format_count(audit_result["three_or_more"]))
    audit_metrics[3].metric(
        "6\uac1c \uc644\uc804 \uc77c\uce58", format_count(audit_result["six_exact"])
    )
    st.dataframe(
        [
            {"\uc801\uc911 \uc218": key, "\ud68c\ucc28 \uc218": format_count(value)}
            for key, value in audit_result["hit_distribution"].items()
        ],
        hide_index=True,
        width="stretch",
    )
    st.dataframe(audit_result["events"], hide_index=True, width="stretch")

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
    if totals:
        st.json(totals)
    else:
        st.info("아직 추첨 결과까지 확인된 AI 사전 추천 기록이 없습니다.")
