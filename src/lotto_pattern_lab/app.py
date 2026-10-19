"""Korean Streamlit research interface."""

import json
import tempfile
from pathlib import Path

import streamlit as st

from .dataset import load_csv, make_random_dataset
from .experiment import run_experiment
from .gp_engine import SearchConfig
from .storage import ExperimentStore

st.set_page_config(page_title="Lotto Pattern Lab", layout="wide")
st.title("Lotto Pattern Lab 연구실")
st.caption(
    "순서 없는 6개 번호 집합의 과거 패턴을 GP로 탐색합니다. 결과는 당첨 예측이나 보장이 아닙니다."
)

store = ExperimentStore("experiments")
with st.sidebar:
    st.header("데이터")
    data_mode = st.radio("데이터 유형", ("합성 무작위 데이터", "CSV 파일"))
    uploaded = st.file_uploader("회차 CSV", type=("csv",)) if data_mode == "CSV 파일" else None
    source_url = st.text_input("출처 URL (사용자 입력, 자동 검증되지 않음)")
    synthetic_count = st.number_input(
        "합성 데이터 회차 수", min_value=20, max_value=10000, value=120, step=10
    )
    st.header("탐색 설정")
    population = st.number_input("개체군 크기", min_value=4, max_value=1000, value=80, step=4)
    generations = st.number_input("세대 수", min_value=1, max_value=1000, value=25)
    crossover = st.slider("교차 확률", 0.0, 1.0, 0.65)
    mutation = st.slider("돌연변이 확률", 0.0, 1.0, 0.25)
    max_depth = st.slider("최대 수식 깊이", 2, 10, 6)
    seed = st.number_input("난수 시드", min_value=0, max_value=2**31 - 1, value=12345)
    workers = st.slider("CPU 워커 (최대 4)", 1, 4, 4)
    train_ratio = st.slider("학습 비율", 0.4, 0.8, 0.6, 0.05)
    validation_ratio = st.slider("검증 비율", 0.1, 0.4, 0.2, 0.05)
    controls = st.number_input("무작위 대조 탐색 횟수", min_value=0, max_value=30, value=3)
    run_clicked = st.button(
        "탐색 실행", type="primary", disabled=(data_mode == "CSV 파일" and uploaded is None)
    )

if run_clicked:
    try:
        if data_mode == "CSV 파일":
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temporary:
                temporary.write(uploaded.getvalue())
                temporary_path = Path(temporary.name)
            try:
                dataset = load_csv(temporary_path, source_url.strip() or None)
            finally:
                temporary_path.unlink(missing_ok=True)
        else:
            dataset = make_random_dataset(int(synthetic_count), int(seed))
        config = SearchConfig(
            population_size=int(population),
            generations=int(generations),
            crossover_probability=crossover,
            mutation_probability=mutation,
            max_depth=max_depth,
            seed=int(seed),
            workers=workers,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            control_runs=int(controls),
        )
        progress_bar = st.progress(0.0, text="실험 준비 중")
        detail = st.empty()

        def update(event: dict[str, object]) -> None:
            total_runs = int(controls) + 1
            run_index = 0 if event["run"] == "dataset" else int(str(event["run"]).rsplit("_", 1)[1])
            fraction = (
                run_index + int(event["generation"]) / max(int(generations), 1)
            ) / total_runs
            progress_bar.progress(
                min(fraction, 1.0), text=f"{event['run']}: 세대 {event['generation']}/{generations}"
            )
            detail.write(
                f"현재 최고 후보 `{event['formula']}` · 학습 적합률 "
                f"{float(event['train_accuracy']):.3f} · 검증 적합률 "
                f"{float(event['validation_accuracy']):.3f}"
            )

        with st.spinner("탐색 중입니다. 진행 중 결과는 experiments 폴더에 저장됩니다."):
            result = run_experiment(dataset, config, store, update)
        progress_bar.progress(1.0, text="완료")
        st.success(f"실험 완료: {result['experiment_id']}")
    except Exception as error:  # surfaced in the UI for data/configuration errors
        st.exception(error)

st.header("실험 기록")
experiments = store.list()
if not experiments:
    st.info("저장된 실험이 없습니다. 왼쪽 설정에서 탐색을 실행하세요.")
else:
    st.write(f"저장된 실험: {len(experiments)}건 · 경로: `{store.directory.resolve()}`")
    summary_rows = []
    for item in experiments:
        candidate = item.get("candidate", {})
        summary_rows.append(
            {
                "실험 ID": item.get("experiment_id"),
                "상태": item.get("status"),
                "데이터": item.get("dataset", {}).get("source_kind"),
                "학습 적합률": candidate.get("training_accuracy"),
                "검증 적합률": candidate.get("validation_accuracy"),
                "독립 테스트": candidate.get("test_accuracy"),
                "실행 시간(초)": item.get("runtime_seconds"),
            }
        )
    st.dataframe(summary_rows, use_container_width=True, hide_index=True)
    selected_id = st.selectbox(
        "실험 상세 조회", [str(item.get("experiment_id")) for item in experiments]
    )
    selected = next(item for item in experiments if item.get("experiment_id") == selected_id)
    candidate = selected.get("candidate", {})
    st.subheader("선택된 수식")
    st.code(candidate.get("formula", "탐색 진행 중"), language="text")
    st.json(
        {
            "후보 수식 목록": candidate.get("candidate_formulas"),
            "무작위 대조 최고 테스트 적합률": selected.get("random_control_max_test_accuracy"),
            "무작위 대조 경험적 p 값": selected.get("random_control_empirical_p_value"),
            "무작위 기준 적합률": selected.get("random_baseline_accuracy", 6 / 45),
            "대조 탐색": selected.get("random_controls"),
        }
    )
    with st.expander("수식 변수 이름 보기"):
        st.json(selected.get("feature_variables", {}))
    st.download_button(
        "선택 실험 JSON 다운로드",
        json.dumps(selected, ensure_ascii=False, indent=2),
        file_name=f"{selected_id}.json",
        mime="application/json",
    )
