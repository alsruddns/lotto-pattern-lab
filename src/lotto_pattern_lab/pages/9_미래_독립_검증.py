"""Freeze formulas and register prospective predictions with a local integrity ledger."""

from __future__ import annotations

import hashlib
from datetime import date

import plotly.graph_objects as go
import streamlit as st

from lotto_pattern_lab.dataset import load_csv
from lotto_pattern_lab.future_validation import (
    audit_future_ledger,
    build_future_scorecard,
    freeze_formula,
    load_frozen_formula,
    read_future_ledger_events,
    record_future_comparison,
    save_frozen_prediction,
)
from lotto_pattern_lab.paths import project_path
from lotto_pattern_lab.prediction import MODEL_VERSION, load_prediction_history
from lotto_pattern_lab.storage import ExperimentStore

st.set_page_config(page_title="미래 독립 검증", layout="wide")
st.title("미래 독립 검증")
verification_labels = {
    "official_verified": "공식 대조 완료",
    "manual_unverified": "수동 입력·미검증",
    "verification_failed": "공식 대조 불일치",
    "locally_validated_unreconciled": "형식 검사 완료·공식 미대조",
    "unverified": "미검증",
}
prediction_status_labels = {
    "pre_draw_prediction": "추첨 전 사전 등록",
    "historical_reproduction": "과거 데이터 재현",
    "timing_unverified": "등록 시점 미확인",
}
result_status_labels = {
    "official_verified": "공식 대조 완료",
    "manual_unverified": "수동 입력·임시 평가",
    "verification_failed": "공식 대조 불일치",
}
st.caption(
    "후보 선택에 사용한 과거 성능과 분리하여, 공식을 동결한 뒤 "
    "새 회차 전에 기록한 예측만 평가합니다. "
    "로컬 시각과 해시 체인은 파일 변경 탐지용이며 외부에서 증명된 타임스탬프는 아닙니다."
)

ledger = audit_future_ledger()
if ledger["valid"]:
    st.success(f"예측 이력 무결성 확인 · 기록 {ledger['event_count']}건")
else:
    st.error(f"예측 이력 무결성 오류 · {ledger['error']} · 새 기록을 중단했습니다.")

experiment_dir = project_path("experiments")
experiment_paths = sorted(
    experiment_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True
)
experiment_ids = [item.stem for item in experiment_paths]
csv_options = {
    "공식 날짜 데이터": project_path("data/official/draws.csv"),
    "당첨번호 및 보너스 데이터": project_path(
        "data/raw/lotto_draws_753_1246_with_bonus.csv"
    ),
}
available_sources = {label: path for label, path in csv_options.items() if path.is_file()}
dataset = None
data_path = None
data_bytes = None
if available_sources:
    source_label = st.selectbox("동결 기준 데이터", list(available_sources))
    data_path = available_sources[source_label]
    data_bytes = data_path.read_bytes()
    try:
        dataset = load_csv(data_path)
    except (OSError, ValueError) as error:
        st.error(f"기준 데이터를 읽을 수 없습니다: {error}")
else:
    st.error("사용 가능한 당첨번호 CSV가 없습니다.")

if dataset is not None:
    last_round = dataset.records[-1].round_number
    latest_record = dataset.records[-1]
    latest_verification_status = (
        latest_record.verification_status or dataset.verification_status
    )
    latest_draw_verified = (
        dataset.source_kind == "official_verified"
        or latest_verification_status == "official_verified"
    )
    st.write(
        f"기준 회차 {dataset.records[0].round_number}~{last_round} · "
        f"최신 회차 {last_round} · 검증 상태 "
        f"{verification_labels.get(dataset.verification_status, '확인 필요')}"
    )
    if not latest_draw_verified:
        st.warning(
            "최신 회차가 공식 대조되지 않아 공식 동결과 미래 예측 등록을 사용할 수 없습니다."
        )
    st.caption(f"기준 데이터: {source_label} · SHA-256 `{hashlib.sha256(data_bytes).hexdigest()}`")
else:
    latest_draw_verified = False

st.subheader("후보 공식 동결")
if not experiment_ids:
    st.info("동결 가능한 완료 실험이 없습니다.")
else:
    experiment_id = st.selectbox("원본 실험", experiment_ids, key="freeze_experiment_id")
    try:
        experiment = ExperimentStore(experiment_dir).load(experiment_id)
        if experiment.get("engine") not in {None, "deap_gp", "intermittent_gp"}:
            st.warning("현재 동결 예측 실행기는 저장된 GP 수식만 지원합니다.")
            experiment = None
    except (OSError, ValueError, KeyError) as error:
        st.error(f"실험 기록을 읽을 수 없습니다: {error}")
        experiment = None

    if experiment:
        model_label = st.radio("동결할 모델", ("6개 집합 모델", "한 번호 모델"), horizontal=True)
        model_type = "six_number_set" if model_label == "6개 집합 모델" else "one_number"
        model_key = (
            "six_number_set_model" if model_type == "six_number_set" else "one_number_model"
        )
        model = experiment.get(model_key, {})
        candidates = model.get("candidates") or (
            [model["selected"]] if model.get("selected") else []
        )
        if candidates:
            rank = st.selectbox("동결 후보 순위", range(1, len(candidates) + 1))
            candidate = candidates[rank - 1]
            st.code(candidate.get("activation_condition", ""), language="text", wrap_lines=True)
            with st.expander("원본 분기 수식"):
                for formula in candidate.get("formulas", []):
                    st.code(formula, language="text", wrap_lines=True)
            st.caption(
                f"특징 프로파일: {experiment.get('config', {}).get('feature_profile', 'legacy')} · "
                f"원본 연구 ID: {experiment.get('research_id') or '기록 없음'}"
            )
            if st.button(
                "이 후보를 새 공식 버전으로 동결",
                type="primary",
                disabled=not ledger["valid"] or dataset is None or not latest_draw_verified,
            ):
                try:
                    from lotto_pattern_lab.official_data import _fetch_latest_round

                    official_latest = _fetch_latest_round()
                    if official_latest != dataset.records[-1].round_number:
                        raise ValueError(
                            "동결 기준 데이터의 최신 회차가 공식 최신 확정 회차와 다릅니다."
                        )
                    frozen = freeze_formula(
                        experiment,
                        model_type,
                        rank,
                        latest_confirmed_round=official_latest,
                        dataset_sha256=hashlib.sha256(data_bytes).hexdigest(),
                        code_version=MODEL_VERSION,
                    )
                    st.session_state["selected_freeze_id"] = frozen["freeze_id"]
                    st.success(f"공식 동결 완료 · 공식 ID {frozen['freeze_id']}")
                except (OSError, ValueError, KeyError) as error:
                    st.error(f"공식 동결 실패: {error}")
        else:
            st.info("선택한 실험에 동결할 후보가 없습니다.")

freeze_directory = project_path("predictions/future_validation/freezes")
freeze_paths = sorted(freeze_directory.glob("*.json")) if freeze_directory.is_dir() else []
freeze_ids = [path.stem for path in freeze_paths]
st.subheader("동결 공식으로 다음 회차 예측 사전 등록")
if not freeze_ids:
    st.info("먼저 후보 공식을 동결하세요.")
elif dataset is None:
    st.warning("예측에 사용할 데이터를 불러오지 못했습니다.")
else:
    preferred = st.session_state.get("selected_freeze_id")
    default_index = freeze_ids.index(preferred) if preferred in freeze_ids else len(freeze_ids) - 1
    freeze_id = st.selectbox("동결 공식", freeze_ids, index=default_index)
    try:
        frozen = load_frozen_formula(freeze_id)
        st.caption(
            f"공식 콘텐츠 해시 `{frozen['formula_content_sha256']}` · "
            f"동결 시각 {frozen['frozen_at']} · 동결 당시 최신 회차 "
            f"{frozen['latest_confirmed_round']}회"
        )
        target_round = dataset.records[-1].round_number + 1
        st.write(
            f"예측 대상: {target_round}회 · "
            f"최신 확정 회차: {dataset.records[-1].round_number}회"
        )
        target_date = st.date_input("예정 추첨일", value=date.today(), key="future_target_date")
        if st.button(
            "다음 회차 예측을 변경 불가 기록으로 저장",
            type="primary",
            disabled=not ledger["valid"],
        ):
            try:
                from lotto_pattern_lab.official_data import _fetch_latest_round

                with st.spinner("공식 최신 확정 회차를 확인하고 예측을 기록합니다."):
                    official_latest = _fetch_latest_round()
                    if official_latest != dataset.records[-1].round_number:
                        raise ValueError(
                            f"선택 데이터 최신 회차({dataset.records[-1].round_number})와 "
                            f"공식 최신 회차({official_latest})가 다릅니다."
                        )
                    recommendation = save_frozen_prediction(
                        freeze_id,
                        dataset,
                        target_round,
                        data_file_sha256=hashlib.sha256(data_bytes).hexdigest(),
                        official_latest_round=official_latest,
                        target_date=target_date,
                    )
                st.success(
                    f"{target_round}회 사전 예측 기록 저장 완료 · "
                    f"예측 ID {recommendation['recommendation_id']}"
                )
                if recommendation["gate_result"]["branch"] == "inactive":
                    st.info("동결 공식의 활성화 조건이 충족되지 않아 번호를 생성하지 않았습니다.")
                else:
                    st.write(f"예측 번호: {recommendation['recommended_numbers']}")
            except Exception as error:
                st.error(f"사전 예측 저장 실패: {error}")
    except (OSError, ValueError, KeyError) as error:
        st.error(f"동결 공식 검증 실패: {error}")

st.subheader("기록된 미래 예측과 당첨 결과 대조")
history = load_prediction_history("predictions")
prospective = [item for item in history if item.get("freeze_id")]
if not prospective:
    st.info("공식 동결 후 저장한 사전 예측이 없습니다.")
elif dataset is not None:
    st.dataframe(
        [
            {
                "예측 ID": item["recommendation_id"],
                "공식 ID": item.get("freeze_id"),
                "대상 회차": item["target_round"],
                "예측 번호": item["recommended_numbers"],
                "생성 시각": item["created_at"],
                "상태": prediction_status_labels.get(
                    item["recommendation_status"], "상태 미상"
                ),
            }
            for item in prospective
        ],
        hide_index=True,
        width="stretch",
    )
    if st.button("현재 데이터에 등록된 회차와 대조 기록", disabled=not ledger["valid"]):
        registered = []
        for item in prospective:
            result = record_future_comparison(item, dataset)
            if result is not None:
                registered.append(result)
        if registered:
            st.success(f"{len(registered)}개 회차 비교를 별도 이력으로 추가했습니다.")
            st.dataframe(
                [
                    {
                        **item,
                        "result_status": result_status_labels.get(
                            item.get("result_status"), "검증 상태 확인 필요"
                        ),
                    }
                    for item in registered
                ],
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("대조할 새 추첨 결과가 없습니다.")

st.subheader("공식별 미래 검증 성적표")
if not prospective:
    st.info("추첨 전에 저장된 예측이 없어 성적표가 비어 있습니다.")
elif not ledger["valid"]:
    st.warning("기록 무결성 확인이 통과된 뒤 미래 검증 성적표를 표시할 수 있습니다.")
else:
    try:
        frozen_records = [load_frozen_formula(freeze_id) for freeze_id in freeze_ids]
        scorecard = build_future_scorecard(
            prospective, read_future_ledger_events(), frozen_records
        )
        if scorecard:
            st.dataframe(
                [
                    {
                        "공식 ID": row["freeze_id"],
                        "동결 시각": row["frozen_at"],
                        "모델": "6개 집합" if row["model_type"] == "six_number_set" else "한 번호",
                        "예측 대상 회차": row["target_rounds"],
                        "사전 예측 수": row["prediction_count"],
                        "공식 대조 회차": row["official_verified_round_count"],
                        "임시 결과": row["provisional_round_count"],
                        "대기 회차": row["pending_round_count"],
                        "공식 대조 평균 적중": row["mean_hits_official"],
                        "2등 일치": row["second_prize_hits_official"],
                        "2등 판정 회차": row["second_prize_evaluable_round_count"],
                        "2등 통계 상태": row["second_prize_status"],
                        "무작위 기준 평균": row["mean_random_hits_official"],
                        "최근 공식 검증 평균": row["recent_mean_hits_official"],
                        "활성화율": row["official_gate_activation_rate"],
                        "검증 상태": row["validation_status"],
                    }
                    for row in scorecard
                ],
                hide_index=True,
                width="stretch",
            )
            chart_rows = [row for row in scorecard if row["mean_hits_official"] is not None]
            if chart_rows:
                chart = go.Figure()
                chart.add_trace(
                    go.Bar(
                        x=[row["freeze_id"][:8] for row in chart_rows],
                        y=[row["mean_hits_official"] for row in chart_rows],
                        name="동결 공식",
                        text=[f"n={row['official_verified_round_count']}" for row in chart_rows],
                        textposition="auto",
                    )
                )
                chart.add_trace(
                    go.Scatter(
                        x=[row["freeze_id"][:8] for row in chart_rows],
                        y=[row["mean_random_hits_official"] for row in chart_rows],
                        mode="markers",
                        name="동일 회차 무작위 기준 평균",
                    )
                )
                chart.update_layout(
                    xaxis_title="공식 동결 ID",
                    yaxis_title="공식 대조 회차당 평균 적중 수",
                    legend={"orientation": "h"},
                )
                st.plotly_chart(chart, width="stretch")
            st.caption(
                "순위는 표본 수가 적어 표시하지 않습니다. 공식 검증 결과만 평균에 포함하며, "
                "수동 입력 결과는 임시 결과로 따로 셉니다."
            )
        else:
            st.info("연결된 동결 공식과 사전 예측 기록이 없습니다.")
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.error(f"미래 검증 성적표를 구성하지 못했습니다: {error}")

with st.expander("로컬 기록의 한계"):
    st.write(
        "기록은 append-only JSONL과 SHA-256 해시 체인으로 변경 여부를 감지합니다. "
        "로컬 파일과 PC 시각만으로 기록 시점의 외부 증명이나 서버 서명은 제공되지 않습니다."
    )
