"""Korean Streamlit interface for historical and intermittent rule research."""

import json
import tempfile
from pathlib import Path

import streamlit as st

from lotto_pattern_lab.bonus_research import (
    evaluate_paired_prize_backtest,
    run_bonus_experiment,
)
from lotto_pattern_lab.dataset import (
    dataset_integrity_report,
    filter_dataset,
    load_csv,
    make_random_dataset,
    save_official_csv,
)
from lotto_pattern_lab.experiment import run_experiment
from lotto_pattern_lab.gp_config import GPConfig
from lotto_pattern_lab.gp_engine import SearchConfig
from lotto_pattern_lab.intermittent import IntermittentConfig
from lotto_pattern_lab.intermittent_experiment import run_intermittent_experiment
from lotto_pattern_lab.official_data import (
    FIRST_SUPPORTED_DATE,
    FIRST_SUPPORTED_ROUND,
    fetch_official_dataset,
)
from lotto_pattern_lab.paths import project_path
from lotto_pattern_lab.pattern_library import PatternLibrary
from lotto_pattern_lab.storage import ExperimentStore
from lotto_pattern_lab.storage_management import (
    load_storage_settings,
    storage_inventory,
    storage_over_quota,
)

st.set_page_config(page_title="Lotto Pattern Lab", layout="wide")
gp_defaults = GPConfig()
storage_settings = load_storage_settings()
storage_state = storage_inventory()
if storage_over_quota(storage_state, storage_settings["quota_bytes"]):
    st.warning(
        f"저장 용량이 설정 한도를 넘었습니다: {storage_state['total_bytes'] / 1024**3:.2f} / "
        f"{storage_settings['quota_bytes'] / 1024**3:.2f} GiB. "
        "시스템 설정에서 정리 항목을 검토하세요."
    )
st.title("Lotto Pattern Lab 연구실")
st.caption(
    "간헐적으로 활성화되는 규칙 가능성을 무작위 대조군과 비교합니다. "
    "결과만으로 조작이나 개입을 입증할 수 없습니다."
)

store = ExperimentStore(project_path("experiments"))
official_path = project_path("data/official/draws.csv")
local_draws_path = project_path("data/raw/lotto_draws_753_1246_with_bonus.csv")

with st.sidebar:
    st.header("데이터")
    if st.button(
        "동행복권 공식 데이터 최신화",
        help="753회부터 최신 발표 회차까지 공식 결과를 다시 수집합니다.",
    ):
        download_progress = st.progress(0.0, text="공식 데이터 수집 준비")
        try:
            official_dataset = fetch_official_dataset(
                start_round=FIRST_SUPPORTED_ROUND,
                delay_seconds=0.08,
                progress=lambda done, total: download_progress.progress(
                    done / total, text=f"공식 회차 수집 {done}/{total}"
                ),
            )
            save_official_csv(official_dataset, official_path)
            st.success(
                f"공식 데이터 확인 완료: {official_dataset.records[0].round_number}~"
                f"{official_dataset.records[-1].round_number}회"
            )
            st.rerun()
        except Exception as error:  # show network/source failures in the UI
            st.exception(error)

    source_mode = st.radio(
        "분석 데이터",
        (
            "로컬 실데이터 753~1246회 (공식 대조 전)",
            "동행복권 공식 데이터",
            "CSV 업로드",
            "합성 무작위 데이터",
        ),
        help="실제 데이터와 합성 대조 데이터는 분리된 선택지입니다.",
    )
    uploaded = st.file_uploader("CSV 파일", type=("csv",)) if source_mode == "CSV 업로드" else None
    source_url = st.text_input("CSV 출처 URL (자동 인증하지 않음)")
    synthetic_count = st.number_input(
        "합성 회차 수", min_value=20, max_value=10000, value=160, step=10
    )
    seed = st.number_input(
        "데이터 난수 시드",
        min_value=0,
        max_value=2**31 - 1,
        value=gp_defaults.seed,
        key="general_seed",
    )

    st.header("탐색 설정")
    research_mode = st.radio("탐색 방식", ("간헐적 패턴 탐색", "기존 전체 회차 GP"))
    population = st.number_input(
        "개체군 크기", min_value=4, max_value=1000, value=gp_defaults.population_size, step=4,
        key="general_population_size",
    )
    generations = st.number_input(
        "세대 수",
        min_value=1,
        max_value=1000,
        value=gp_defaults.generations,
        key="general_generations",
    )
    crossover = st.slider(
        "교차 확률", 0.0, 1.0, gp_defaults.crossover_probability,
        key="general_crossover_probability"
    )
    mutation = st.slider(
        "돌연변이 확률", 0.0, 1.0, gp_defaults.mutation_probability,
        key="general_mutation_probability"
    )
    max_depth = st.slider(
        "최대 수식 깊이", 2, 10, gp_defaults.max_depth, key="general_max_depth"
    )
    workers = st.slider("CPU 워커 (최대 4)", 1, 4, gp_defaults.workers, key="general_workers")
    train_ratio = st.slider(
        "학습 비율", 0.1, 0.8, gp_defaults.train_ratio, 0.05,
        key="general_train_ratio"
    )
    validation_ratio = st.slider(
        "검증 비율", 0.05, 0.4, gp_defaults.validation_ratio, 0.05,
        key="general_validation_ratio"
    )
    max_lag = st.slider(
        "최대 과거 참조 회차", 1, 10, gp_defaults.max_lag, key="general_max_lag"
    )
    controls = st.number_input(
        "무작위 대조 탐색 횟수", min_value=0, max_value=99, value=20,
        key="general_control_runs",
    )
    min_activation = st.slider(
        "최소 활성 비율", 0.05, 1.0, 0.1, 0.05, key="general_min_activation_rate"
    )
    hidden_states = st.slider("가설 상태 수", 2, 5, 3, key="general_hidden_states")

base_dataset = None
data_error = None
try:
    if source_mode == "로컬 실데이터 753~1246회 (공식 대조 전)":
        if local_draws_path.exists():
            base_dataset = load_csv(local_draws_path)
            rounds = [record.round_number for record in base_dataset.records]
            if rounds != list(range(753, 1247)):
                raise ValueError("로컬 실데이터는 753~1246회 전체가 연속되어야 합니다.")
        else:
            data_error = f"실데이터 파일을 찾을 수 없습니다: {local_draws_path}"
    elif source_mode == "동행복권 공식 데이터":
        if official_path.exists():
            base_dataset = load_csv(official_path)
        else:
            data_error = "공식 데이터가 없습니다. 먼저 왼쪽의 공식 데이터 최신화 버튼을 누르세요."
    elif source_mode == "CSV 업로드":
        if uploaded is not None:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temporary:
                temporary.write(uploaded.getvalue())
                temporary_path = Path(temporary.name)
            try:
                base_dataset = load_csv(temporary_path, source_url.strip() or None)
            finally:
                temporary_path.unlink(missing_ok=True)
    else:
        base_dataset = make_random_dataset(int(synthetic_count), int(seed))
except Exception as error:
    data_error = str(error)

selected_dataset = None
if base_dataset is not None:
    st.subheader("분석 구간")
    low_round = base_dataset.records[0].round_number
    high_round = base_dataset.records[-1].round_number
    status_labels = {
        "locally_validated_unreconciled": "내부 형식·회차 검증 완료 · 공식 원본 대조 전",
        "official_verified": "동행복권 공식 원본 확인 완료",
        "unverified": "출처 미검증 CSV",
        "synthetic": "합성 무작위 데이터",
    }
    status_text = status_labels.get(
        base_dataset.verification_status, base_dataset.verification_status
    )
    feature_profile = st.selectbox(
        "\ud2b9\uc9d5 \ud504\ub85c\ud30c\uc77c",
        ("legacy", "calendar_round", "bonus_lag_v1"),
        help="\uae30\uc874\u0020\ud2b9\uc9d5\u0020\ud504\ub85c\ud30c\uc77c\uc785\ub2c8\ub2e4\u002e\u0020\ub0a0\uc9dc\u0020\ud2b9\uc9d5\uc740\u0020\ubaa8\ub4e0\u0020\ud68c\ucc28\uc5d0\u0020\uacf5\uc2dd\u0020\ucd94\ucc28\uc77c\uc774\u0020\uc788\uc744\u0020\ub54c\ub9cc\u0020\uc0ac\uc6a9\ud569\ub2c8\ub2e4\u002e",
        key="general_feature_profile",
    )
    st.caption(
        f"데이터 상태: {status_text} · "
        f"기본 연구 범위: 2017-05-06(753회) 이후 · 현재 {low_round}~{high_round}회"
    )
    integrity = dataset_integrity_report(base_dataset)
    dataset_label = (
        local_draws_path.name
        if source_mode.startswith("로컬")
        else base_dataset.source_description
    )
    st.info(
        f"기준 데이터: {dataset_label} · "
        f"보너스 포함 {integrity['bonus_draw_count']}/{integrity['draw_count']}회 · "
        f"날짜 포함 {integrity['dated_draw_count']}/{integrity['draw_count']}회 · "
        f"검증 상태: {integrity['verification_status']}"
    )
    date_values = [
        record.draw_date for record in base_dataset.records if record.draw_date is not None
    ]
    range_columns = st.columns(4)
    with range_columns[0]:
        start_round = st.number_input(
            "시작 회차", min_value=low_round, max_value=high_round, value=low_round
        )
    with range_columns[1]:
        end_round = st.number_input(
            "종료 회차", min_value=low_round, max_value=high_round, value=high_round
        )
    start_date = end_date = None
    if date_values:
        first_date = min(date_values)
        latest_date = max(date_values)
        with range_columns[2]:
            start_date = st.date_input(
                "시작일",
                value=max(FIRST_SUPPORTED_DATE, first_date),
                min_value=first_date,
                max_value=latest_date,
            )
        with range_columns[3]:
            end_date = st.date_input(
                "종료일", value=latest_date, min_value=first_date, max_value=latest_date
            )
    if feature_profile == "calendar_round" and any(
        record.draw_date is None for record in base_dataset.records
    ):
        st.warning("Calendar features need dates on every draw. Use the official verified dataset.")

    try:
        selected_dataset = filter_dataset(
            base_dataset,
            start_date=start_date,
            end_date=end_date,
            start_round=int(start_round),
            end_round=int(end_round),
        )
        st.caption(
            f"선택 범위 {selected_dataset.records[0].round_number}~"
            f"{selected_dataset.records[-1].round_number}회 · "
            f"{len(selected_dataset.records)}회차 · 보너스 번호 제외"
        )
        bonus_research_enabled = st.checkbox(
            "보너스 번호를 별도 GP 연구 대상으로 실행",
            value=False,
            disabled=not all(record.draw.bonus is not None for record in selected_dataset.records),
            help="1등 6개 번호 모델은 유지하고, 별도 실행으로 보너스 번호 정확 일치를 평가합니다.",
        )
    except ValueError as error:
        st.error(str(error))

run_clicked = st.button(
    "탐색 실행",
    type="primary",
    disabled=(
        selected_dataset is None
        or (source_mode == "CSV 업로드" and uploaded is None)
        or (
            feature_profile == "bonus_lag_v1"
            and not all(record.draw.bonus is not None for record in selected_dataset.records)
        )
        or (
            feature_profile == "calendar_round"
            and any(record.draw_date is None for record in selected_dataset.records)
        )
    ),
)
if data_error:
    st.warning(data_error)

if run_clicked and selected_dataset is not None:
    try:
        progress_bar = st.progress(0.0, text="실험 준비 중")
        detail = st.empty()

        def update(event: dict[str, object]) -> None:
            generation = int(event["generation"])
            generations_in_run = max(int(event["generations"]), 1)
            if "validation_score" in event:
                run_label = str(event["run"])
                if run_label == "actual_one_number":
                    run_index = 0
                elif run_label == "actual_six_number_set":
                    run_index = 1
                else:
                    control_index = int(run_label.split("_")[1]) - 1
                    run_index = 2 + control_index * 2 + int(run_label.endswith("six_number_set"))
                total_runs = 2 * (int(controls) + 1)
                fraction = (run_index + generation / generations_in_run) / total_runs
            else:
                total_runs = int(controls) + 1
                run_index = (
                    0 if event["run"] == "dataset" else int(str(event["run"]).rsplit("_", 1)[1])
                )
                fraction = (run_index + generation / generations_in_run) / total_runs
            progress_bar.progress(
                min(fraction, 0.999),
                text=f"{event['run']}: 세대 {generation}/{generations}",
            )
            if "validation_score" in event:
                detail.write(
                    f"현재 후보 IF `{event['gate']}` THEN `{event['formula']}` · "
                    f"검증 점수 {float(event['validation_score']):.4f} · "
                    f"활성률 {float(event['validation_coverage']):.1%}"
                )
            else:
                detail.write(
                    f"현재 후보 `{event['formula']}` · 학습 적합률 "
                    f"{float(event['train_accuracy']):.3f} · 검증 적합률 "
                    f"{float(event['validation_accuracy']):.3f}"
                )

        if research_mode == "간헐적 패턴 탐색":
            config = IntermittentConfig(
                population_size=max(int(population), 8),
                generations=int(generations),
                crossover_probability=crossover,
                mutation_probability=mutation,
                max_depth=max_depth,
                seed=int(seed),
                workers=workers,
                train_ratio=train_ratio,
                validation_ratio=validation_ratio,
                max_lag=max_lag,
                control_runs=int(controls),
                min_activation_rate=min_activation,
                hidden_states=hidden_states,
                feature_profile=feature_profile,
            )
            with st.spinner("간헐적 후보와 동일 조건의 무작위 대조군을 탐색합니다."):
                result = run_intermittent_experiment(
                    selected_dataset, config, store, update, PatternLibrary()
                )
            progress_bar.progress(1.0, text="완료")
            st.success(f"간헐적 탐색 완료: {result['experiment_id']}")
        else:
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
                max_lag=max_lag,
                control_runs=int(controls),
            )
            with st.spinner("탐색 중입니다. 진행 결과는 experiments 폴더에 저장됩니다."):
                result = run_experiment(selected_dataset, config, store, update)
            progress_bar.progress(1.0, text="완료")
            st.success(f"실험 완료: {result['experiment_id']}")
        if bonus_research_enabled:
            bonus_config = SearchConfig(
                population_size=int(population),
                generations=int(generations),
                crossover_probability=crossover,
                mutation_probability=mutation,
                max_depth=max_depth,
                seed=int(seed) + 500_009,
                workers=workers,
                train_ratio=train_ratio,
                validation_ratio=validation_ratio,
                max_lag=max_lag,
                control_runs=int(controls),
            )
            with st.spinner("별도 보너스 번호 GP 연구와 무작위 대조군을 실행합니다."):
                bonus_result = run_bonus_experiment(selected_dataset, bonus_config, store)
            if result.get("experiment_type") == "intermittent":
                bonus_result["paired_prize_evaluation"] = evaluate_paired_prize_backtest(
                    selected_dataset, result, bonus_result
                )
                store.save(bonus_result)
                prizes = bonus_result["paired_prize_evaluation"]
                st.info(
                    "공유 독립 테스트 구간 · "
                    f"1등 6개 완전 일치 {prizes['first_prize_exact_six']['hits']}/"
                    f"{prizes['first_prize_exact_six']['denominator']} · "
                    "2등 5개+보너스 일치 "
                    f"{prizes['second_prize_five_plus_bonus']['hits']}/"
                    f"{prizes['second_prize_five_plus_bonus']['denominator']}"
                )
            else:
                bonus_result["paired_prize_evaluation"] = None
                store.save(bonus_result)
                st.caption(
                    "보너스 번호는 별도 GP로 평가했습니다. 1등/2등 조합 집계에는 "
                    "같은 회차 범위의 6개 집합 모델이 필요합니다."
                )
            st.success(
                f"보너스 번호 연구 완료: {bonus_result['experiment_id']} · "
                f"검증 적중률 {bonus_result['candidate']['validation_accuracy']:.3f} · "
                f"독립 테스트 {bonus_result['candidate']['test_accuracy']:.3f} · "
                f"무작위 기준 1/45"
            )
    except Exception as error:  # surfaced in the UI for data/configuration errors
        st.exception(error)

st.header("실험 기록")
experiments = store.list()
if not experiments:
    st.info("저장된 실험이 없습니다. 위 설정으로 탐색을 실행하세요.")
else:
    st.write(f"저장된 실험: {len(experiments)}건 · 경로: `{store.directory.resolve()}`")
    summary_rows = []
    for item in experiments:
        if item.get("experiment_type") == "intermittent":
            candidate = item.get("one_number_model", {}).get("selected", {})
            summary_rows.append(
                {
                    "실험 ID": item.get("experiment_id"),
                    "상태": item.get("status"),
                    "데이터 상태": item.get("dataset", {}).get("verification_status"),
                    "학습 적합률": candidate.get("training_hit_rate"),
                    "검증 적합률": candidate.get("validation_hit_rate"),
                    "독립 테스트": candidate.get("test_hit_rate"),
                    "실행 시간(초)": item.get("runtime_seconds"),
                }
            )
            continue
        if item.get("experiment_type") == "bonus_target":
            candidate = item.get("candidate", {})
            summary_rows.append(
                {
                    "실험 ID": item.get("experiment_id"),
                    "대상": "보너스 번호 별도 모델",
                    "상태": item.get("status"),
                    "데이터 상태": item.get("dataset", {}).get("verification_status"),
                    "학습 정확도": candidate.get("training_accuracy"),
                    "검증 정확도": candidate.get("validation_accuracy"),
                    "독립 테스트 정확도": candidate.get("test_accuracy"),
                    "실행 시간(초)": item.get("runtime_seconds"),
                }
            )
            continue
        candidate = item.get("candidate", {})
        summary_rows.append(
            {
                "실험 ID": item.get("experiment_id"),
                "상태": item.get("status"),
                "데이터 상태": item.get("dataset", {}).get("verification_status"),
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
    selected = store.load(selected_id)
    if selected.get("status") != "completed":
        st.info("이 실험은 아직 완료되지 않았습니다. 마지막 저장 진행 상태를 표시합니다.")
        st.json(selected.get("progress", []))
    elif selected.get("experiment_type") == "intermittent":
        pattern_tab, activation_tab, state_tab, validation_tab, control_tab = st.tabs(
            (
                "간헐적 패턴 탐색",
                "패턴 활성 회차",
                "숨겨진 상태 분석",
                "규칙별 검증",
                "실제 데이터 vs 무작위 데이터",
            )
        )
        model_items = (
            ("한 번호 모델", "one_number_model"),
            ("6개 집합 모델", "six_number_set_model"),
        )
        with pattern_tab:
            st.subheader("검증 구간에서 선택된 두 규칙")
            st.caption("게이트 결과 0은 비활성, 1과 2는 각각 규칙 1과 규칙 2를 선택합니다.")
            for label, key in model_items:
                model = selected[key]
                candidate = model["selected"]
                st.markdown(f"#### {label}")
                formulas = candidate["formulas"]
                rule_count = int(candidate["rule_count"])
                outputs_per_rule = int(candidate["outputs_per_event"])
                rule_lines = []
                for rule_index in range(rule_count):
                    start = rule_index * outputs_per_rule
                    formula_text = "; ".join(formulas[start : start + outputs_per_rule])
                    rule_lines.append(
                        f"IF gate % {rule_count + 1} == {rule_index + 1} THEN {formula_text}"
                    )
                st.code(f"GATE: {candidate['activation_condition']}\n" + "\n".join(rule_lines))

        with activation_tab:
            st.subheader("활성 회차와 불규칙 간격")
            for label, key in model_items:
                candidate = selected[key]["selected"]
                st.markdown(f"#### {label}")
                active_events = [
                    event for event in candidate["activation_events"] if event["active"]
                ]
                st.write("활성 회차", [event["round_number"] for event in active_events])
                st.write("활성 회차 간격", candidate["active_gaps"])
                st.write("규칙별 활성 회차", candidate["rule_active_rounds"])
                st.write("규칙별 회차 간격", candidate["rule_active_gaps"])
                st.dataframe(active_events, use_container_width=True, hide_index=True)

        with state_tab:
            st.subheader("가설적 상태 및 온라인 변화점")
            st.caption("상태 A~E는 통계적 요약 라벨이며 추첨기 내부 상태나 개입의 증거가 아닙니다.")
            st.dataframe(selected["hypothetical_states"]["state_rows"], use_container_width=True)
            st.json(selected["hypothetical_states"]["test_metrics"])

        with validation_tab:
            st.subheader("학습·검증·독립 테스트")
            st.json(selected["splits"])
            for label, key in model_items:
                st.markdown(f"#### {label}")
                candidate_rows = selected[key]["candidates"]
                st.dataframe(candidate_rows, use_container_width=True, hide_index=True)

        with control_tab:
            st.subheader("실제 후보와 공정한 무작위 대조군")
            st.json(
                {
                    "무작위 대조 최고 테스트 점수": selected["random_control_max_test_score"],
                    "경험적 p 값": selected["random_control_empirical_p_value"],
                    "대조 반복 횟수": selected["random_control_replicates"],
                    "p 값 해상도": selected["p_value_resolution"],
                    "판정 상태": selected["inference_status"],
                }
            )
            st.dataframe(selected["random_controls"], use_container_width=True, hide_index=True)
    elif selected.get("experiment_type") == "bonus_target":
        candidate = selected["candidate"]
        st.subheader("보너스 번호 단일 대상 GP 평가")
        st.caption(
            "현재 회차의 보너스는 정답 레이블이며 입력 특징에는 "
            "이전 회차 보너스만 포함됩니다."
        )
        st.code(candidate.get("formula", "미수집"), language="text")
        st.json(
            {
                "학습 정확도": candidate.get("training_accuracy"),
                "검증 정확도": candidate.get("validation_accuracy"),
                "독립 테스트 정확도": candidate.get("test_accuracy"),
                "무작위 기준 정확도": selected.get("random_baseline_accuracy"),
                "무작위 대조 반복": len(selected.get("random_controls", [])),
                "경험적 p값": selected.get("random_control_empirical_p_value"),
                "1등/2등 공유 테스트 평가": selected.get("paired_prize_evaluation"),
            }
        )
        st.json(selected.get("feature_variables", {}))
    else:
        candidate = selected.get("candidate", {})
        st.subheader("검증 구간에서 선택된 공식")
        st.code(candidate.get("formula", "탐색 진행 중"), language="text")
        st.json(
            {
                "후보 공식 목록": candidate.get("candidate_formulas"),
                "무작위 대조 최고 테스트 적합률": selected.get("random_control_max_test_accuracy"),
                "무작위 대조 경험적 p 값": selected.get("random_control_empirical_p_value"),
                "무작위 기준 적합률": selected.get("random_baseline_accuracy", 6 / 45),
                "대조 탐색": selected.get("random_controls"),
            }
        )
    with st.expander("공식 변수 이름 보기"):
        st.json(selected.get("feature_variables", {}))
    st.download_button(
        "선택 실험 JSON 다운로드",
        json.dumps(selected, ensure_ascii=False, indent=2),
        file_name=f"{selected_id}.json",
        mime="application/json",
    )
