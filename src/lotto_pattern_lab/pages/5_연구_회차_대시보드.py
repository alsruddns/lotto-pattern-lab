"""Research-round list, detailed research results, and ChatGPT report export."""

from pathlib import Path
from statistics import median

import streamlit as st

from lotto_pattern_lab.paths import project_path
from lotto_pattern_lab.pattern_library import PatternLibrary
from lotto_pattern_lab.report_clipboard import report_copy_button
from lotto_pattern_lab.research import load_checkpoint
from lotto_pattern_lab.research_rounds import (
    SUMMARY_SCHEMA_VERSION,
    ResearchRoundStore,
    build_chatgpt_report,
    collect_prediction_candidates,
    compare_research_rounds,
    model_metric_stats,
    paired_comparison_summary,
    rebuild_dashboard_summary_from_store,
)

st.set_page_config(page_title="연구 회차 대시보드", layout="wide")
st.title("연구 회차 대시보드")
st.caption("연구 ID는 UUID로 유지하고, 연구 회차는 생성 순서에 따라 별도로 관리합니다.")

checkpoint_directory = project_path("checkpoints")
experiment_directory = project_path("experiments")
rounds_store = ResearchRoundStore()
pattern_library = PatternLibrary()
rounds_store.migrate_legacy(checkpoint_directory, experiment_directory)
rounds = rounds_store.list()


def _hit_total(summary: dict, metric: str) -> int | str:
    totals = summary.get("hit_totals", {})
    if not totals:
        return "미수집"
    return sum(int(item.get(metric, 0)) for item in totals.values())


def _test_summary(summary: dict) -> str:
    metrics = summary.get("model_metrics", {})
    values = []
    for model, label in (("one_number", "한 번호"), ("six_number_set", "6개 집합")):
        scores = [
            item.get("independent_test_score")
            for item in metrics.get(model, [])
            if item.get("independent_test_score") is not None
        ]
        values.append(
            f"{label}: 평균 {sum(scores) / len(scores):.3f}" if scores else f"{label}: 미수집"
        )
    return " · ".join(values)


def _duration(seconds: int | float | None) -> str:
    if seconds is None:
        return "미수집"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}시간 {minutes}분 {seconds}초"


def hydrate_round(
    record: dict, *, reaggregate: bool = False, library: PatternLibrary = pattern_library
) -> dict:
    hydrated = dict(record)
    checkpoint_path = record.get("checkpoint_path")
    if checkpoint_path and Path(checkpoint_path).is_file():
        try:
            checkpoint = load_checkpoint(checkpoint_path)
            hydrated.update(checkpoint)
            hydrated["research_round"] = record["research_round"]
        except (OSError, ValueError):
            pass
    summary = hydrated.get("dashboard_summary")
    checkpoint_summary = hydrated.get("dashboard_summary")
    registry_summary = record.get("dashboard_summary")
    if (
        isinstance(registry_summary, dict)
        and registry_summary.get("summary_schema_version") == SUMMARY_SCHEMA_VERSION
        and (
            not isinstance(checkpoint_summary, dict)
            or checkpoint_summary.get("summary_schema_version") != SUMMARY_SCHEMA_VERSION
        )
    ):
        hydrated["dashboard_summary"] = registry_summary
    summary = hydrated.get("dashboard_summary")
    needs_reaggregation = (
        not isinstance(summary, dict)
        or summary.get("completed_experiment_count") is None
        or summary.get("summary_schema_version") != SUMMARY_SCHEMA_VERSION
    )
    if needs_reaggregation and reaggregate:
        with st.spinner("실험 상세 기록에서 회차 통계를 다시 계산하고 있습니다..."):
            summary = rebuild_dashboard_summary_from_store(hydrated, experiment_directory)
        hydrated["dashboard_summary"] = summary
    pattern_counts = library.research_pattern_counts(str(hydrated["research_id"]))
    summary.update(pattern_counts)
    summary["registration_metrics_available"] = True
    summary["registration_count_source"] = "pattern_library"
    if not summary.get("exact_match_count") and hydrated.get("discovered_exact_matches"):
        summary["exact_match_count"] = len(hydrated.get("discovered_exact_matches", []))
    if reaggregate:
        summary["summary_schema_version"] = SUMMARY_SCHEMA_VERSION
        rounds_store.update(str(hydrated["research_id"]), {"dashboard_summary": summary})
    hydrated["experiment_summaries"] = summary.get("experiment_summaries", [])
    return hydrated


if not rounds:
    st.info("기록된 연구 회차가 없습니다. 지속적 패턴 연구를 시작하면 회차가 생성됩니다.")
else:
    rounds = [hydrate_round(item) for item in rounds]
    list_rows = []
    for item in rounds:
        summary = item.get("dashboard_summary", {})
        config = item.get("config", {})
        list_rows.append(
            {
                "연구 회차": f"연구 {item['research_round']}회차",
                "연구 ID": item["research_id"],
                "실행 일시": item.get("started_at") or item.get("created_at"),
                "연구 시간(초)": item.get("total_elapsed_seconds"),
                "상태": item.get("status", "미수집"),
                "탐색 엔진": ", ".join(item.get("engines", [])) or "미수집",
                "특징 프로파일": config.get("feature_profile", "미수집"),
                "Elite Seeding": (
                    "활성화"
                    if item.get("elite_seeding", {}).get("enabled") is True
                    else "비활성화"
                    if item.get("elite_seeding", {}).get("enabled") is False
                    else "미수집"
                ),
                "완료 실험": summary.get(
                    "completed_experiment_count", item.get("completed_runs", 0)
                ),
                "실패 실험": summary.get("failed_experiment_count", item.get("failed_runs", 0)),
                "신규 패턴": summary.get("new_pattern_count", "미수집"),
                "3개 이상 반복 평가": _hit_total(summary, "3_plus"),
                "4개 이상 반복 평가": _hit_total(summary, "4_plus"),
                "5개 이상 반복 평가": _hit_total(summary, "5_plus"),
                "6개 완전 일치 반복 평가": _hit_total(summary, "6_exact"),
                "독립 테스트 요약": _test_summary(summary),
            }
        )
    st.dataframe(list_rows, hide_index=True, width="stretch")

    choices = {
        f"연구 {item['research_round']}회차 · {item['research_id'][:8]}": item for item in rounds
    }
    selected_label = st.selectbox("상세 연구 회차", list(choices))
    selected = hydrate_round(choices[selected_label], reaggregate=True)
    summary = selected.get("dashboard_summary", {})
    config = selected.get("config", {})
    selected["prediction_candidates"] = collect_prediction_candidates(
        selected.get("completed_experiment_ids", [])
    )

    st.divider()
    st.subheader(f"연구 {selected['research_round']}회차 상세")
    model_metrics = model_metric_stats(summary)
    headline = st.columns(5)
    headline[0].metric("연구 시간", _duration(selected.get("total_elapsed_seconds")))
    headline[1].metric("완료 실험", summary.get("completed_experiment_count", 0))
    headline[2].metric("실패·중단", summary.get("failed_experiment_count", 0))
    headline[3].metric("검사 후보 수", summary.get("candidate_count", "미수집"))
    headline[4].metric("신규 패턴", summary.get("new_pattern_count", "미수집"))
    second = st.columns(5)
    second[0].metric("중복 패턴", summary.get("duplicate_pattern_count", "미수집"))
    second[1].metric("계승 수식", summary.get("inherited_pattern_count", "미수집"))
    second[2].metric("자식 패턴", summary.get("child_pattern_count", "미수집"))
    second[3].metric("회고적 6개 일치 후보", summary.get("exact_match_count", "미수집"))
    second[4].metric(
        "다음 난수 시드",
        selected.get("next_seed", selected.get("rng_resume", {}).get("next_seed", "미수집")),
    )
    st.caption(
        f"패턴 등록 시도 {summary.get('registration_attempt_count', '미수집')}회 · "
        f"변형 신규 자식 {summary.get('new_modified_child_pattern_count', '미수집')}개 · "
        f"신규 자식 전체 {summary.get('new_child_pattern_count', '미수집')}개 · "
        f"부모 패턴 {summary.get('parent_pattern_count', '미수집')}개"
    )
    st.caption(
        "검사 후보 수는 세대별 개체 평가 횟수이며, 같은 수식의 반복 평가는 중복 포함입니다."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### 데이터 및 실행 설정")
        st.json(
            {
                "데이터셋 해시": selected.get("dataset_sha256"),
                "데이터 출처": selected.get("dataset_source_kind"),
                "공식 검증 상태": selected.get("dataset_verification_status"),
                "로또 회차 범위": selected.get("dataset_round_range"),
                "엔진": selected.get("engines"),
                "특징 프로파일": config.get("feature_profile"),
                "개체군": config.get("population_size"),
                "세대 수": config.get("generations"),
                "교차 확률": config.get("crossover_probability"),
                "돌연변이 확률": config.get("mutation_probability"),
                "최대 수식 깊이": config.get("max_depth"),
                "CPU 워커": config.get("workers"),
                "학습/검증 비율": [config.get("train_ratio"), config.get("validation_ratio")],
                "최대 과거 참조": config.get("max_lag"),
                "무작위 대조 반복": config.get("control_runs"),
                "Elite Seeding": selected.get("elite_seeding", {}).get("enabled"),
                "계승 비율": config.get("elite_seed_ratio"),
                "계승 원본 패턴 ID": selected.get("elite_seeding", {}).get("source_pattern_ids"),
                "데이터셋 전체 SHA-256": selected.get("dataset_sha256"),
            }
        )
    with right:
        st.markdown("#### 실패 및 중단 기록")
        failures = selected.get("failed_runs", [])
        if isinstance(failures, list) and failures:
            st.dataframe(
                [
                    {
                        "시드": item.get("seed"),
                        "오류 종류": item.get("error_type"),
                        "오류": item.get("error"),
                        "시각": item.get("failed_at"),
                    }
                    for item in failures
                ],
                hide_index=True,
                width="stretch",
            )
        elif failures:
            st.info(f"실패 건수 {failures}, 상세 오류는 미수집입니다.")
        else:
            st.info("실패 기록 없음 또는 미수집")

    st.markdown("#### 모델별 성능 · 학습, 검증, 독립 테스트 분리")
    score_rows = []
    for model, label in (("one_number", "한 번호 모델"), ("six_number_set", "6개 집합 모델")):
        for metric, phase in (
            ("training_score", "학습"),
            ("validation_score", "검증"),
            ("independent_test_score", "독립 테스트"),
        ):
            values = model_metrics.get(model, {}).get(metric, {})
            score_rows.append(
                {
                    "모델": label,
                    "구간": phase,
                    "평균": values.get("mean"),
                    "최고": values.get("best"),
                    "중앙값": values.get("median"),
                    "평가 실험 수": values.get("count"),
                }
            )
    st.dataframe(score_rows, hide_index=True, width="stretch")
    st.caption(
        "한 번호와 6개 집합 모델은 분리 집계합니다. 점수는 해당 모델·구간별 실험 단위 통계입니다."
    )

    st.markdown("#### 적중 분포 및 활성화")
    aggregate_hit_rows = []
    for model, label in (("one_number", "한 번호 모델"), ("six_number_set", "6개 집합 모델")):
        model_rows = summary.get("model_metrics", {}).get(model, [])
        for phase, phase_label in (
            ("training", "학습"),
            ("validation", "검증/탐색 선택"),
            ("independent_test", "독립 테스트"),
        ):
            repeated = summary.get("hit_totals", {}).get(f"{model}:{phase}", {})
            unique = summary.get("unique_hit_totals", {}).get(model, {}).get(phase, {})
            aggregate_hit_rows.append(
                {
                    "모델": label,
                    "평가 구간": phase_label,
                    "활성 예측 / 전체 대상 회차 수": (
                        sum(int(row.get(f"{phase}_activation_count") or 0) for row in model_rows),
                        sum(
                            int(row.get(f"{phase}_target_round_count") or 0)
                            for row in model_rows
                        ),
                    ),
                    "반복 평가 3+ / 4+ / 5+ / 6개": [
                        repeated.get("3_plus", 0),
                        repeated.get("4_plus", 0),
                        repeated.get("5_plus", 0),
                        repeated.get("6_exact", 0),
                    ],
                    "고유 로또 회차 3+ / 4+ / 5+ / 6개": [
                        unique.get("3_plus", 0),
                        unique.get("4_plus", 0),
                        unique.get("5_plus", 0),
                        unique.get("6_exact", 0),
                    ],
                }
            )
    st.dataframe(aggregate_hit_rows, hide_index=True, width="stretch")
    st.caption(
        "반복 평가는 실험별 활성 예측 회차를 합산합니다. 고유 회차는 같은 로또 회차를 "
        "여러 실험이 맞혀도 한 번만 셉니다. 3+/4+/5+/6개 기준의 분모는 활성 평가 회차입니다. "
        "비활성 회차는 번호 예측이 없으므로 적중 분포에서 제외합니다. 검증 구간은 GP 세대 선택에 "
        "사용되어 독립 검증으로 해석할 수 없습니다."
    )
    hit_rows = []
    for item in summary.get("experiment_summaries", []):
        for model, label in (("one_number", "한 번호 모델"), ("six_number_set", "6개 집합 모델")):
            metrics = item.get("models", {}).get(model, {})
            for phase, phase_label in (
                ("training", "학습"),
                ("validation", "검증"),
                ("independent_test", "독립 테스트"),
            ):
                distribution = metrics.get("match_distribution", {}).get(phase)
                if not isinstance(distribution, dict):
                    continue
                hit_rows.append(
                    {
                        "실험 ID": item.get("experiment_id"),
                        "모델": label,
                    "평가 구간": "검증/탐색 선택" if phase == "validation" else phase_label,
                        "활성화 횟수": metrics.get(f"{phase}_activation_count", "미수집"),
                        "적중 번호 수": metrics.get(f"{phase}_hit_count", "미수집"),
                        "0개": distribution.get("0", 0),
                        "1개": distribution.get("1", 0),
                        "2개": distribution.get("2", 0),
                        "3개": distribution.get("3", 0),
                        "4개": distribution.get("4", 0),
                        "5개": distribution.get("5", 0),
                        "6개": distribution.get("6", 0),
                    }
                )
    if hit_rows:
        st.dataframe(hit_rows, hide_index=True, width="stretch")
    else:
        st.info("적중 분포는 과거 실험에서 상세 이벤트가 저장되지 않은 경우 미수집으로 표시됩니다.")

    st.markdown("#### 무작위 대조 및 Elite GP 비교")
    pvalue_rows = []
    control_runs = config.get("control_runs")
    for model, label in (("one_number", "한 번호 모델"), ("six_number_set", "6개 집합 모델")):
        values = summary.get("p_value_distributions", {}).get(model, [])
        pvalue_rows.append(
            {
                "모델": label,
                "경험적 p값 평균": sum(values) / len(values) if values else None,
                "중앙값": median(values) if values else None,
                "범위": [min(values), max(values)] if values else None,
                "계산 실험 수": len(values),
                "대조 반복/실험": control_runs,
                "p값 최소 해상도": 1 / (1 + control_runs)
                if isinstance(control_runs, int)
                else None,
            }
        )
    st.dataframe(pvalue_rows, hide_index=True, width="stretch")
    st.caption(
        "각 실험은 대조 반복이 적어 p값 해상도가 거칩니다. 서로 겹치는 회차와 데이터 분할을 "
        "사용한 실험이므로 p값 69개를 독립 표본처럼 해석하지 않습니다."
    )
    paired_stats = paired_comparison_summary(summary)
    paired_rows = []
    for model, label in (("one_number", "한 번호 모델"), ("six_number_set", "6개 집합 모델")):
        stats = paired_stats.get(model, {})
        paired_rows.append(
            {
                "모델": label,
                "검증 평균 점수 차(Elite-일반)": stats.get("validation_mean_delta"),
                "검증 우세 / 열세 / 동률": [
                    stats.get("validation_wins"),
                    stats.get("validation_losses"),
                    stats.get("validation_ties"),
                ],
                "독립 테스트 평균 점수 차": stats.get("test_mean_delta"),
                "테스트 우세 / 열세 / 동률": [
                    stats.get("test_wins"),
                    stats.get("test_losses"),
                    stats.get("test_ties"),
                ],
                "GP 시간 평균 실제 / 일반(초)": [
                    stats.get("actual_runtime_mean"),
                    stats.get("paired_runtime_mean"),
                ],
                "시간 근거": stats.get("runtime_source", "미수집"),
            }
        )
    st.dataframe(paired_rows, hide_index=True, width="stretch")
    st.caption(
        "점수 차는 같은 실험 시드·데이터 분할 paired 비교입니다. 시간 추정은 진행 로그의 첫/마지막 "
        "세대 기록 차이로 실제 탐색 시작·종료 비용을 모두 포함하지 않습니다."
    )
    compare_rows = []
    for item in summary.get("experiment_summaries", []):
        for model, label in (("one_number", "한 번호 모델"), ("six_number_set", "6개 집합 모델")):
            actual = item.get("models", {}).get(model, {})
            paired = item.get("paired", {}).get(model, {})
            compare_rows.append(
                {
                    "실험 ID": item.get("experiment_id"),
                    "모델": label,
                    "Elite 검증": actual.get("validation_score"),
                    "Elite 독립 테스트": actual.get("independent_test_score"),
                    "일반 GP 검증": paired.get("validation_score", "해당 없음"),
                    "일반 GP 독립 테스트": paired.get("independent_test_score", "해당 없음"),
                    "동일 데이터 분할": paired.get("same_dataset_and_splits", "미수집"),
                    "대조 반복": item.get("control_replicates", "미수집"),
                }
            )
    if compare_rows:
        st.dataframe(compare_rows, hide_index=True, width="stretch")

    st.markdown("#### 완전 일치 패턴 상세")
    matches = selected.get("discovered_exact_matches", [])
    if matches:
        st.warning(
            "아래 항목은 과거 데이터에 대한 회고적 적합입니다. 사전 예측 결과로 해석하지 마세요."
        )
        st.dataframe(
            [
                {
                    "패턴 ID": item.get("pattern_id"),
                    "실험 ID": item.get("experiment_id"),
                    "로또 회차": item.get("round_number"),
                    "번호 6개": item.get("winning_numbers"),
                    "수식": item.get("formulas"),
                    "활성화 조건": item.get("activation_condition"),
                    "프로파일": item.get("feature_profile"),
                    "데이터 검증": item.get("data_verification_status"),
                    "사전 예측": item.get("prediction_created_before_draw", False),
                    "검증 상태": item.get("validation_status", "retrospective_search_fit"),
                }
                for item in matches
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("완전 일치 기록 없음 또는 과거 체크포인트에서 상세 정보 미수집")

    st.markdown("#### 검증 기반 상위 후보")
    candidates = [
        candidate
        for item in summary.get("experiment_summaries", [])
        for candidate in item.get("top_candidates", [])
    ]
    candidates.sort(
        key=lambda item: (
            item.get("validation_score") is not None,
            float(item.get("validation_score") or -1),
            -int(item.get("complexity") or 0),
        ),
        reverse=True,
    )
    if candidates:
        st.dataframe(candidates[:10], hide_index=True, width="stretch")
        st.caption(
            "후보는 검증 점수, 학습-검증 안정성, 복잡도 정보를 기준으로 요약합니다. "
            "독립 테스트 점수만으로 순위를 정하지 않습니다."
        )
    else:
        st.info("후보 상세 미수집")

    st.markdown("#### 저장된 다음 회차 예측")
    if selected["prediction_candidates"]:
        st.dataframe(selected["prediction_candidates"], hide_index=True, width="stretch")
        if any(
            item.get("recommendation_status") == "pre_draw_prediction"
            for item in selected["prediction_candidates"]
        ):
            st.caption(
                "사전 저장된 예측과 실제 추첨 결과가 아직 연결되지 않은 항목은 미검증 상태입니다."
            )
    else:
        st.info("해당 연구의 저장된 예측 후보 없음")

    report = build_chatgpt_report(selected)
    full_report = build_chatgpt_report(selected, detailed=True)
    st.markdown("#### ChatGPT 분석 보고서")
    for key, text, detailed in (
        ("research_report_summary", report, False),
        ("research_report_full", full_report, True),
    ):
        response = report_copy_button(text, key=key, detailed=detailed)
        if response and response.get("request_id") != st.session_state.get(f"{key}_last_request"):
            st.session_state[f"{key}_last_request"] = response.get("request_id")
            if response.get("status") == "copied":
                st.toast("보고서를 클립보드에 복사했습니다.", icon="✅")
            else:
                st.error(f"클립보드 복사 실패: {response.get('message', '브라우저 오류')}")
    with st.expander("복사될 Markdown 미리보기"):
        st.code(report, language="markdown")

    st.markdown("#### 연구 회차 간 비교")
    if len(rounds) >= 2:
        labels = list(choices)
        selected_left = st.selectbox("비교할 연구 A", labels, index=0, key="round_compare_left")
        selected_right = st.selectbox(
            "비교할 연구 B", labels, index=min(1, len(labels) - 1), key="round_compare_right"
        )
        if selected_left != selected_right:
            left_record = hydrate_round(choices[selected_left], reaggregate=True)
            right_record = hydrate_round(choices[selected_right], reaggregate=True)
            comparison = compare_research_rounds(left_record, right_record)
            if comparison["directly_comparable"]:
                st.success(
                    "데이터와 주요 탐색·평가 설정이 일치합니다. 모델별 지표를 비교할 수 있습니다."
                )
            else:
                st.warning("데이터셋 또는 탐색·평가 설정이 달라 직접 비교가 제한됩니다.")
                st.json(
                    {
                        "동일 데이터 해시": comparison["same_dataset"],
                        "동일 평가 설정": comparison["same_evaluation_settings"],
                        "설정 차이": comparison["differences"],
                    }
                )
            st.dataframe(
                [
                    {
                        "연구": f"연구 {record['research_round']}회차",
                        "모델": model,
                        "지표": metric,
                        "평균": stats.get("mean"),
                        "최고": stats.get("best"),
                        "중앙값": stats.get("median"),
                    }
                    for record, side in (
                        (left_record, "left_metrics"),
                        (right_record, "right_metrics"),
                    )
                    for model, model_stats in comparison[side].items()
                    for metric, stats in model_stats.items()
                ],
                hide_index=True,
                width="stretch",
            )
    else:
        st.info("회차 비교는 연구가 2회 이상 저장되면 사용할 수 있습니다.")
