"""Research-round list, detailed research results, and ChatGPT report export."""

import sqlite3
from statistics import median

import plotly.graph_objects as go
import streamlit as st

from lotto_pattern_lab.cumulative_research import (
    CUMULATIVE_MODE,
    INDEPENDENT_MODE,
    compare_research_modes,
    cumulative_growth_summary,
)
from lotto_pattern_lab.feature_schema import feature_mapping
from lotto_pattern_lab.hit_audit import audit_research_round, build_hit_audit_markdown
from lotto_pattern_lab.paths import PROJECT_ROOT, project_path
from lotto_pattern_lab.pattern_library import PatternLibrary
from lotto_pattern_lab.presentation import (
    format_count,
    format_decimal,
    format_duration,
    format_percent,
    format_probability,
    pretty_formula,
)
from lotto_pattern_lab.randomized_controls import (
    control_score_rows,
    summarize_matched_random_controls,
)
from lotto_pattern_lab.report_clipboard import report_copy_button
from lotto_pattern_lab.research import load_checkpoint, research_worker_active
from lotto_pattern_lab.research_dashboard import (
    dashboard_failure_records,
    normalize_dashboard_summary,
    research_aggregation_status,
    stringify_dashboard_table_rows,
    validated_pattern_counts,
)
from lotto_pattern_lab.research_rounds import (
    SUMMARY_SCHEMA_VERSION,
    ResearchRoundStore,
    build_chatgpt_report,
    collect_prediction_candidates,
    compare_research_rounds,
    model_metric_stats,
    paired_comparison_summary,
    rebuild_dashboard_summary_from_store,
    safely_reaggregate_second_prize,
)
from lotto_pattern_lab.sqlite_support import is_database_locked, log_database_failure
from lotto_pattern_lab.storage import ExperimentStore
from lotto_pattern_lab.tabular_models import model_availability


def _stop_on_database_error(operation: str, error: sqlite3.OperationalError) -> None:
    database = project_path("data/database/patterns.sqlite")
    log_database_failure(operation, database, error)
    if is_database_locked(error):
        st.error("연구 데이터베이스가 다른 저장 작업에서 사용 중입니다.")
    else:
        st.error("연구 데이터베이스를 읽거나 저장하지 못했습니다.")
    st.caption("화면 상태와 연구 기록은 유지됩니다. 잠시 뒤 다시 시도할 수 있습니다.")
    st.code(str(error), language="text")
    if st.button("대시보드 데이터 다시 불러오기", key=f"retry_{operation}"):
        st.rerun()
    st.stop()


st.set_page_config(page_title="연구 회차 대시보드", layout="wide")
st.title("연구 회차 대시보드")
st.caption("연구 ID는 UUID로 유지하고, 연구 회차는 생성 순서에 따라 별도로 관리합니다.")

SCORE_COLUMN_CONFIG = {
    "평균 평가 점수": st.column_config.TextColumn(
        "평균 평가 점수",
        help=(
            "실험별 모델 평가 점수의 평균입니다. 적중률(맞힌 번호/활성 예측 수)과 다른 지표입니다."
        ),
    ),
    "최고 평가 점수": st.column_config.TextColumn(
        "최고 평가 점수", help="해당 구간의 실험별 평가 점수 중 최댓값입니다."
    ),
    "중앙값 평가 점수": st.column_config.TextColumn(
        "중앙값 평가 점수", help="해당 구간의 실험별 평가 점수 중앙값입니다."
    ),
    "평가 실험 수": st.column_config.TextColumn(
        "평가 실험 수", help="통계에 포함한 실험 수입니다."
    ),
}

FEATURE_LABELS = {
    "round_scaled": ("회차 정규화", "현재 대상 회차 번호를 정규화한 값입니다."),
    "sum": ("번호 합계", "해당 과거 회차의 6개 번호 합입니다."),
    "squares": ("제곱합", "해당 과거 회차 6개 번호의 제곱 합입니다."),
    "pairs": ("쌍의 곱 합", "6개 번호에서 만들 수 있는 모든 두 수 곱의 합입니다."),
    "product_mod": ("곱의 나머지", "6개 번호 곱을 997로 나눈 나머지입니다."),
    "min": ("최솟값", "해당 과거 회차에서 가장 작은 번호입니다."),
    "max": ("최댓값", "해당 과거 회차에서 가장 큰 번호입니다."),
    "bonus": ("보너스 번호", "해당 과거 회차의 보너스 번호입니다."),
    "draw_year": ("추첨 연도", "검증된 추첨일의 연도입니다."),
    "draw_month": ("추첨 월", "검증된 추첨일의 월입니다."),
    "draw_day": ("추첨 일", "검증된 추첨일의 날짜입니다."),
    "draw_weekday_monday0": ("요일(월=0)", "월요일 0부터 일요일 6까지로 표시한 요일입니다."),
    "draw_day_of_year": ("연중 일수", "해당 날짜가 그해의 몇 번째 날인지 나타냅니다."),
    "draw_iso_week": ("ISO 주차", "ISO 8601 기준의 연중 주 번호입니다."),
    "draw_yyyymmdd": ("날짜 숫자(YYYYMMDD)", "연·월·일을 YYYYMMDD 정수로 이어 붙인 값입니다."),
    "draw_digit_sum": ("날짜 자릿수 합", "YYYYMMDD 숫자의 각 자릿수를 더한 값입니다."),
    "draw_reversed_digits": ("날짜 숫자 역순", "YYYYMMDD 정수의 자릿수 순서를 뒤집은 값입니다."),
    "draw_mod_31": ("날짜의 31 나머지", "YYYYMMDD 정수를 31로 나눈 나머지입니다."),
    "draw_mod_45": ("날짜의 45 나머지", "YYYYMMDD 정수를 45로 나눈 나머지입니다."),
    "draw_mod_97": ("날짜의 97 나머지", "YYYYMMDD 정수를 97로 나눈 나머지입니다."),
    "date_round_concat": ("날짜·회차 연결값", "YYYYMMDD 뒤에 회차 번호를 이어 붙인 정수입니다."),
    "date_round_xor": ("날짜·회차 XOR", "날짜 정수와 회차 번호의 비트 XOR 값입니다."),
    "date_round_mod_997": (
        "날짜·회차 합의 997 나머지",
        "날짜 정수와 회차 합을 997로 나눈 나머지입니다.",
    ),
    "round_digit_sum": ("회차 자릿수 합", "회차 번호를 이루는 각 자릿수의 합입니다."),
    "round_reversed_digits": ("회차 숫자 역순", "회차 번호의 자릿수 순서를 뒤집은 값입니다."),
    "round_mod_3": ("회차의 3 나머지", "회차 번호를 3으로 나눈 나머지입니다."),
    "round_mod_7": ("회차의 7 나머지", "회차 번호를 7로 나눈 나머지입니다."),
    "round_mod_45": ("회차의 45 나머지", "회차 번호를 45로 나눈 나머지입니다."),
}


def _feature_rows(formulas: list[str], profile: str, max_lag: int) -> list[dict[str, str]]:
    """Explain only variables used by the selected formulas."""
    import re

    mapping = feature_mapping(profile, max_lag)
    variables = sorted(
        {
            f"x{index}"
            for formula in formulas
            for index in map(int, re.findall(r"\bx(\d+)\b", formula))
        },
        key=lambda item: int(item[1:]),
    )
    rows = []
    for variable in variables:
        name = mapping.get(variable, {}).get("name", "알 수 없는 변수")
        if name.startswith("lag") and "_" in name:
            lag, feature = name.split("_", 1)
            label, description = FEATURE_LABELS.get(
                feature, (feature, "과거 회차에서 계산한 특징값입니다.")
            )
            name = f"{lag}회 전 {label}"
            description = f"{lag}회 전 추첨 결과의 {description}"
        elif name.startswith("sum_delta_"):
            name = name.replace("sum_delta_", "합계 변화량 ").replace("_", " → ")
            description = "지정된 두 과거 회차 번호 합계의 차이입니다."
        else:
            name, description = FEATURE_LABELS.get(
                name, (name.replace("_", " "), "회차 또는 날짜에서 계산한 특징값입니다.")
            )
        rows.append({"변수": variable, "특징 변수": name, "설명": description})
    return rows


@st.cache_resource
def _pattern_library() -> PatternLibrary:
    return PatternLibrary()


checkpoint_directory = project_path("checkpoints")
experiment_directory = project_path("experiments")
try:
    rounds_store = ResearchRoundStore()
    rounds_store.migrate_legacy(checkpoint_directory, experiment_directory)
    rounds = rounds_store.list_overview()
except sqlite3.OperationalError as error:
    _stop_on_database_error("research_dashboard_initialization", error)


def _hit_total(summary: dict, metric: str) -> int | str:
    totals = summary.get("hit_totals", {})
    if not totals:
        return "미수집"
    return sum(int(item.get(metric, 0)) for item in totals.values())


def _second_prize_total(summary: dict) -> tuple[int | None, str]:
    groups = {
        key: value
        for key, value in summary.get("hit_totals", {}).items()
        if key.startswith("six_number_set:") and isinstance(value, dict)
    }
    known = sum(int(value.get("2_known", 0) or 0) for value in groups.values())
    unknown = sum(int(value.get("2_unknown", 0) or 0) for value in groups.values())
    if known == 0:
        if int(
            summary.get("summary_schema_version", 0) or 0
        ) < SUMMARY_SCHEMA_VERSION and summary.get("completed_experiment_count", 0):
            return None, "재집계 필요"
        return None, "미수집" if unknown == 0 else "판정 불가"
    hits = sum(int(value.get("2_exact", 0) or 0) for value in groups.values())
    return hits, "일부 미수집" if unknown else "판정 완료"


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


def _summary_count(summary: dict, key: str, fallback: object = None) -> object:
    value = summary.get(key)
    if value is not None:
        return value
    return fallback


def hydrate_round(
    record: dict,
    *,
    reaggregate: bool = False,
    load_checkpoint_data: bool = True,
    include_pattern_counts: bool = True,
    library: PatternLibrary | None = None,
) -> dict:
    hydrated = dict(record)
    checkpoint_path = record.get("checkpoint_path")
    resolved_checkpoint = project_path(checkpoint_path) if checkpoint_path else None
    if (
        load_checkpoint_data
        and resolved_checkpoint
        and resolved_checkpoint.is_relative_to(PROJECT_ROOT)
        and resolved_checkpoint.is_file()
    ):
        try:
            checkpoint = load_checkpoint(resolved_checkpoint)
            hydrated.update(checkpoint)
            hydrated["research_round"] = record["research_round"]
        except (OSError, ValueError):
            pass
    summary_value = hydrated.get("dashboard_summary")
    registry_summary_value = record.get("dashboard_summary")
    registry_summary, registry_diagnostic = normalize_dashboard_summary(
        registry_summary_value, missing="dashboard_summary" not in record
    )
    checkpoint_summary, checkpoint_diagnostic = normalize_dashboard_summary(
        summary_value, missing="dashboard_summary" not in hydrated
    )
    if (
        registry_diagnostic is None
        and registry_summary.get("summary_schema_version") == SUMMARY_SCHEMA_VERSION
        and (
            checkpoint_diagnostic is not None
            or checkpoint_summary.get("summary_schema_version") != SUMMARY_SCHEMA_VERSION
        )
    ):
        summary_value = registry_summary_value
        hydrated["dashboard_summary"] = registry_summary_value
    summary, summary_diagnostic = normalize_dashboard_summary(
        summary_value, missing="dashboard_summary" not in hydrated
    )
    needs_reaggregation = (
        summary_diagnostic is not None
        or summary.get("completed_experiment_count") is None
        or summary.get("summary_schema_version") != SUMMARY_SCHEMA_VERSION
    )
    if needs_reaggregation and reaggregate:
        with st.spinner("연구 집계를 다시 계산하고 있습니다..."):
            rebuilt_summary = rebuild_dashboard_summary_from_store(hydrated, experiment_directory)
        summary, summary_diagnostic = normalize_dashboard_summary(rebuilt_summary)
        hydrated["dashboard_summary"] = summary
    if include_pattern_counts:
        library = library or _pattern_library()
        try:
            raw_pattern_counts = library.research_pattern_counts(str(hydrated["research_id"]))
        except sqlite3.OperationalError as error:
            _stop_on_database_error("research_pattern_count_lookup", error)
        pattern_counts, pattern_count_diagnostic = validated_pattern_counts(raw_pattern_counts)
        if pattern_counts is not None:
            summary.update(pattern_counts)
            summary["registration_metrics_available"] = True
            summary["registration_count_source"] = "pattern_library"
        else:
            summary["registration_metrics_available"] = False
            summary["registration_diagnostic"] = pattern_count_diagnostic
    summary["aggregation_status"] = research_aggregation_status(
        hydrated, summary, summary_diagnostic
    )
    if summary_diagnostic:
        summary["summary_state"] = "pending_or_invalid"
        summary["summary_diagnostic"] = summary_diagnostic
    elif include_pattern_counts and pattern_counts is not None:
        summary["summary_state"] = "available"
    else:
        summary["summary_state"] = "available"
    if not summary.get("exact_match_count") and hydrated.get("discovered_exact_matches"):
        summary["exact_match_count"] = len(hydrated.get("discovered_exact_matches", []))
    if reaggregate and summary_diagnostic is None:
        summary["summary_schema_version"] = SUMMARY_SCHEMA_VERSION
        try:
            rounds_store.update(str(hydrated["research_id"]), {"dashboard_summary": summary})
        except sqlite3.OperationalError as error:
            _stop_on_database_error("research_dashboard_summary_update", error)
    hydrated["experiment_summaries"] = summary.get("experiment_summaries", [])
    return hydrated


if not rounds:
    st.info("기록된 연구 회차가 없습니다. 지속적 패턴 연구를 시작하면 회차가 생성됩니다.")
else:
    rounds = [
        hydrate_round(item, load_checkpoint_data=False, include_pattern_counts=False)
        for item in rounds
    ]
    list_rows = []
    for item in rounds:
        summary = item.get("dashboard_summary", {})
        config = item.get("config", {})
        second_prize_hits, second_prize_status = _second_prize_total(summary)
        list_rows.append(
            {
                "연구 회차": f"연구 {item['research_round']}회차",
                "연구 ID": item["research_id"],
                "실행 일시": item.get("started_at") or item.get("created_at"),
                "연구 시간": format_duration(item.get("total_elapsed_seconds")),
                "상태": item.get("status", "미수집"),
                "집계 상태": summary.get(
                    "aggregation_status",
                    research_aggregation_status(item, summary, summary.get("summary_diagnostic")),
                ),
                "탐색 엔진": ", ".join(item.get("engines", [])) or "미수집",
                "특징 프로파일": config.get("feature_profile", "미수집"),
                "Elite Seeding": (
                    "활성화"
                    if item.get("elite_seeding", {}).get("enabled") is True
                    else "비활성화"
                    if item.get("elite_seeding", {}).get("enabled") is False
                    else "미수집"
                ),
                "완료 실험": _summary_count(
                    summary, "completed_experiment_count", item.get("completed_runs")
                ),
                "실패 실험": _summary_count(
                    summary, "failed_experiment_count", item.get("failed_runs")
                ),
                "신규 패턴": format_count(summary.get("new_pattern_count")),
                "3개 이상 반복 평가": format_count(_hit_total(summary, "3_plus")),
                "4개 이상 반복 평가": format_count(_hit_total(summary, "4_plus")),
                "5개 이상 반복 평가": format_count(_hit_total(summary, "5_plus")),
                "2등 일치": second_prize_hits,
                "2등 통계 상태": second_prize_status,
                "6개 완전 일치 반복 평가": format_count(_hit_total(summary, "6_exact")),
                "독립 테스트 요약": _test_summary(summary),
            }
        )
    list_tooltips = {
        "연구 회차": "연구가 시작된 순서입니다.",
        "연구 ID": "연구를 식별하는 UUID입니다.",
        "실행 일시": "연구를 시작한 시각입니다.",
        "연구 시간": "누적 연구 실행 시간입니다. 시·분·초로 표시합니다.",
        "상태": "연구의 마지막 저장 상태입니다.",
        "탐색 엔진": "해당 연구에서 사용한 탐색 엔진입니다.",
        "특징 프로파일": "모델 입력 특징의 구성 버전입니다.",
        "Elite Seeding": "기존 패턴을 초기 후보로 사용했는지 표시합니다.",
        "완료 실험": "완료 상태로 저장된 실험 수입니다.",
        "실패 실험": "실패 또는 중단된 실험 수입니다.",
        "신규 패턴": "패턴 라이브러리에 새로 등록한 패턴 수입니다.",
        "3개 이상 반복 평가": "활성 예측 중 3개 이상 적중한 실험 평가 수입니다.",
        "4개 이상 반복 평가": "활성 예측 중 4개 이상 적중한 실험 평가 수입니다.",
        "5개 이상 반복 평가": "활성 예측 중 5개 이상 적중한 실험 평가 수입니다.",
        "2등 일치": "예측한 6개 조합 중 본번호 정확히 5개와 보너스 번호가 함께 일치한 수입니다.",
        "2등 통계 상태": "보너스 번호가 저장된 활성 6개 조합에 대해서만 판정합니다.",
        "6개 완전 일치 반복 평가": "본번호 6개 일치입니다. 보너스 번호는 판정에 사용하지 않습니다.",
        "독립 테스트 요약": "독립 테스트 평가 점수의 모델별 평균입니다.",
    }
    list_tooltips["집계 상태"] = "집계 정보의 준비 또는 형식 상태입니다."
    numeric_labels = {
        "완료 실험",
        "실패 실험",
    }
    list_column_config = {
        label: (
            st.column_config.NumberColumn(label, help=tooltip, format="localized")
            if label in numeric_labels
            else st.column_config.TextColumn(label, help=tooltip)
        )
        for label, tooltip in list_tooltips.items()
    }
    list_column_config["실행 일시"] = st.column_config.TextColumn(
        "실행 일시", help=list_tooltips["실행 일시"]
    )
    list_column_config["연구 ID"] = st.column_config.TextColumn(
        "연구 ID", help=list_tooltips["연구 ID"], width="small"
    )
    list_column_config["2등 일치"] = st.column_config.NumberColumn(
        "2등 일치",
        help=list_tooltips["2등 일치"],
        format="localized",
    )
    list_column_config["2등 통계 상태"] = st.column_config.TextColumn(
        "2등 통계 상태", help=list_tooltips["2등 통계 상태"]
    )
    list_column_config["독립 테스트 요약"] = st.column_config.TextColumn(
        "독립 테스트 요약", help=list_tooltips["독립 테스트 요약"], width="medium"
    )
    st.dataframe(
        list_rows,
        column_config=list_column_config,
        hide_index=True,
        width="stretch",
    )

    show_growth = st.checkbox(
        (
            "\uacc4\uc18d \uc131\uc7a5 \ubd84\uc11d\uacfc "
            "\uc5f0\uad6c \ubc29\uc2dd \ube44\uad50 \ubd88\ub7ec\uc624\uae30"
        ),
        key="research_dashboard_show_growth",
        help=(
            "\uc120\ud0dd\ud558\uba74 \uc5f0\uad6c\ubcc4 \uc0c1\uc138 \uc694\uc57d\uc744 "
            "\uc870\ud68c\ud569\ub2c8\ub2e4. \uae30\ubcf8 \ud68c\ucc28 \ubaa9\ub85d\uc740 "
            "\uc694\uc57d \ud544\ub4dc\ub9cc \uc77d\uc2b5\ub2c8\ub2e4."
        ),
    )
    if show_growth:
        growth_records = rounds_store.list()
        growth = cumulative_growth_summary(growth_records)
        st.divider()
        st.subheader("누적 연구 성장 분석")
        st.caption(growth["growth_claim"])
        growth_first = st.columns(5)
        growth_first[0].metric("누적 연구 회차", format_count(growth["research_round_count"]))
        growth_first[1].metric("누적 탐색 후보", format_count(growth["candidate_count"]))
        growth_first[2].metric(
            "신규 등록 수식 후보", format_count(growth["new_valid_formula_count"])
        )
        growth_first[3].metric(
            "재사용된 우수 수식", format_count(growth["reused_elite_formula_count"])
        )
        growth_first[4].metric("기존 수식 변형", format_count(growth["modified_formula_count"]))
        growth_second = st.columns(5)
        growth_second[0].metric("중복 제외 수식", format_count(growth["duplicate_formula_count"]))
        growth_second[1].metric("연구 시간", _duration(growth["runtime_seconds"]))
        rate = growth["new_valid_formulas_per_hour"]
        growth_second[2].metric(
            "시간당 신규 수식", f"{rate:.2f}" if rate is not None else "평가 불가"
        )
        growth_second[3].metric(
            "validation 평균",
            format_decimal(growth["mean_validation_scores"]["six_number_set"], suffix="점"),
        )
        growth_second[4].metric(
            "독립 테스트 평균",
            format_decimal(growth["mean_independent_test_scores"]["six_number_set"], suffix="점"),
        )
        efficiency_change = growth["efficiency_change_vs_comparable_previous"]
        st.caption(
            "이전 동일 데이터·설정 연구 대비 시간당 신규 등록 후보 변화: "
            + (
                f"{efficiency_change:+.1%}"
                if efficiency_change is not None
                else "비교 가능한 완료 연구가 부족합니다."
            )
        )
        random_p_values = growth["random_control_median_p_values"]
        one_p_value = (
            random_p_values["one_number"]
            if random_p_values["one_number"] is not None
            else "평가 불가"
        )
        six_p_value = (
            random_p_values["six_number_set"]
            if random_p_values["six_number_set"] is not None
            else "평가 불가"
        )
        st.caption(
            f"무작위 대조 p 중앙값 · 한 번호: {one_p_value} · 6개 집합: {six_p_value}. "
            "p 값은 효과 크기나 미래 추첨 예측력을 나타내지 않습니다."
        )
        mode_comparison = compare_research_modes(growth_records)
        st.markdown("#### 독립 탐색과 누적 진화 연구 비교")
        if mode_comparison["available"]:
            st.caption(mode_comparison["reason"])
            st.dataframe(
                [
                    {
                        "공통 시드 수": item["paired_seed_count"],
                        "validation 중앙값 차이": item["validation_median_delta"],
                        "독립 테스트 중앙값 차이": item["independent_test_median_delta"],
                        "신규 수식 중앙값 차이": item["new_formula_median_delta"],
                        "복잡도 중앙값 차이": item["complexity_median_delta"],
                        "실행 시간 중앙값 차이(초)": item["runtime_median_delta_seconds"],
                        "독립 탐색 무작위 대조 p 중앙값 (6개 집합)": item[
                            "random_control_median_p_values"
                        ][INDEPENDENT_MODE]["six_number_set"],
                        "누적 진화 무작위 대조 p 중앙값 (6개 집합)": item[
                            "random_control_median_p_values"
                        ][CUMULATIVE_MODE]["six_number_set"],
                    }
                    for item in mode_comparison["comparisons"]
                ],
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "무작위 대조 p 값은 대조 반복에 따른 통계 요약입니다. "
                "개선 효과의 크기나 미래 당첨번호 예측력을 뜻하지 않습니다."
            )
        else:
            st.info(mode_comparison["reason"])

    choices = {
        f"연구 {item['research_round']}회차 · {item['research_id'][:8]}": item for item in rounds
    }
    selected_label = st.selectbox(
        "연구 회차 선택",
        list(choices),
        key="research_dashboard_selected_round",
    )
    selected_record = choices[selected_label]
    selected_research_id = str(selected_record["research_id"])
    if st.session_state.get("research_dashboard_detail_round_id") != selected_research_id:
        st.session_state["research_dashboard_detail_round_id"] = selected_research_id
        st.session_state["research_dashboard_detail_loaded"] = False
    if not st.session_state.get("research_dashboard_detail_loaded", False):
        st.info(
            "\ud68c\ucc28 \uc0c1\uc138 \ub370\uc774\ud130\uc640 "
            "\uccb4\ud06c\ud3ec\uc778\ud2b8\u00b7\uc608\uce21 \ud30c\uc77c\uc740 "
            "\uc694\uccad \uc2dc \uc870\ud68c\ud569\ub2c8\ub2e4."
        )
        if st.button(
            "\uc120\ud0dd\ud55c \uc5f0\uad6c \uc0c1\uc138 \uc870\ud68c",
            key="load_research_round_detail",
        ):
            st.session_state["research_dashboard_detail_loaded"] = True
            st.rerun()
        st.stop()
    if st.button("\uc0c1\uc138 \ubcf4\uae30 \ub2eb\uae30", key="close_research_round_detail"):
        st.session_state["research_dashboard_detail_loaded"] = False
        st.rerun()
    detail_cache = st.session_state.setdefault("research_dashboard_detail_cache", {})
    detail_version = (
        selected_research_id,
        selected_record.get("last_updated_at"),
        selected_record.get("status"),
    )
    if st.button("\uc0c1\uc138 \uc0c8\ub85c\uace0\uce68", key="refresh_research_round_detail"):
        detail_cache.pop(detail_version, None)
        st.rerun()
    cached_detail = detail_cache.get(detail_version)
    if cached_detail is not None and selected_record.get("status") != "running":
        selected = cached_detail
    else:
        detail_record = rounds_store.get(selected_research_id) or selected_record
        selected = hydrate_round(detail_record, reaggregate=False)
        selected["prediction_candidates"] = collect_prediction_candidates(
            list(selected.get("completed_experiment_ids", []))
        )
        if selected_record.get("status") != "running":
            if len(detail_cache) >= 3:
                detail_cache.pop(next(iter(detail_cache)))
            detail_cache[detail_version] = selected
    summary = selected.get("dashboard_summary", {})
    config = selected.get("config", {})
    if int(summary.get("summary_schema_version", 0) or 0) < SUMMARY_SCHEMA_VERSION and summary.get(
        "completed_experiment_count", 0
    ):
        with st.expander("기존 연구의 2등 통계 재집계"):
            st.caption(
                "저장된 예측 조합과 정확히 같은 데이터 스냅샷이 확인되는 경우에만 계산합니다. "
                "기존 3+/4+/5+/6개 값은 대조 후 보존하고, DB 온라인 백업을 먼저 생성합니다."
            )
            worker_active = research_worker_active()
            if worker_active:
                st.warning("연구 작업 슬롯이 사용 중이라 재집계를 시작할 수 없습니다.")
            if st.button(
                "선택 회차 2등 통계 재집계",
                key=f"reaggregate-second-prize-{selected['research_id']}",
                disabled=worker_active,
            ):
                try:
                    result = safely_reaggregate_second_prize(
                        choices[selected_label], experiment_directory
                    )
                    st.success(
                        "2등 통계를 추가했습니다. "
                        f"기존 연구 DB 온라인 백업: {result['backup_path']}"
                    )
                    detail_cache.pop(detail_version, None)
                    st.rerun()
                except (OSError, RuntimeError, TimeoutError, ValueError, KeyError) as error:
                    st.error(f"2등 통계를 안전하게 재집계하지 못했습니다: {error}")
    if summary.get("summary_diagnostic"):
        st.warning(
            f"이 연구의 회차 통계는 집계 대기 또는 점검이 필요합니다: "
            f"{summary['summary_diagnostic']}"
        )

    st.divider()
    st.subheader(f"연구 {selected['research_round']}회차 상세")
    model_metrics = model_metric_stats(summary)
    headline = st.columns(5)
    headline[0].metric("연구 시간", _duration(selected.get("total_elapsed_seconds")))
    headline[1].metric(
        "완료 실험", format_count(_summary_count(summary, "completed_experiment_count"))
    )
    headline[2].metric(
        "실패·중단", format_count(_summary_count(summary, "failed_experiment_count"))
    )
    headline[3].metric("검사 후보 수", format_count(summary.get("candidate_count")))
    headline[4].metric("신규 패턴", format_count(summary.get("new_pattern_count")))
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
    st.caption("검사 후보 수는 세대별 개체 평가 횟수이며, 같은 수식의 반복 평가는 중복 포함입니다.")

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
        failures = dashboard_failure_records(selected)
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
                    "평균 평가 점수": format_decimal(values.get("mean"), suffix="점"),
                    "최고 평가 점수": format_decimal(values.get("best"), suffix="점"),
                    "중앙값 평가 점수": format_decimal(values.get("median"), suffix="점"),
                    "평가 실험 수": format_count(values.get("count")),
                }
            )
    st.dataframe(
        score_rows,
        column_config={
            "모델": st.column_config.TextColumn(
                "모델", help="번호 하나를 예측하는 모델과 6개 번호 집합 모델을 구분합니다."
            ),
            "구간": st.column_config.TextColumn(
                "평가 구간", help="학습, 검증, 독립 테스트 중 점수를 계산한 구간입니다."
            ),
            **SCORE_COLUMN_CONFIG,
        },
        hide_index=True,
        width="stretch",
    )
    st.markdown("#### 추가 기계학습 기준 모델")
    availability_rows = []
    for model_id, model_label in (
        ("random_forest", "Random Forest"),
        ("xgboost", "XGBoost"),
        ("torch_mlp", "PyTorch MLP"),
    ):
        model_state = model_availability()[model_id]
        availability_rows.append(
            {
                "모델": model_label,
                "설치 상태": "사용 가능" if model_state["available"] else "선택 의존성 미설치",
                "설치 선택 항목": model_state["install_extra"],
            }
        )
    st.dataframe(availability_rows, hide_index=True, width="stretch")
    st.caption(
        "추가 기준 모델은 45개 번호의 출현 여부 점수를 학습해 상위 6개를 출력합니다. "
        "훈련과 검증으로 설정을 정하고 독립 테스트는 최종 평가에만 사용합니다."
    )
    model_result_paths = sorted(
        project_path("experiments/model_comparisons").glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:100]
    model_result_rows = []
    for path in model_result_paths:
        try:
            import json

            record = json.loads(path.read_text(encoding="utf-8"))
            test = record.get("independent_test", {})
            model_result_rows.append(
                {
                    "실험 ID": record.get("experiment_id"),
                    "모델": record.get("engine"),
                    "평가 회차": test.get("evaluation_round_count"),
                    "평균 적중": test.get("average_hits"),
                    "3개 이상": test.get("at_least_3"),
                    "4개 이상": test.get("at_least_4"),
                    "5개 이상": test.get("at_least_5"),
                    "6개 일치": test.get("exactly_6"),
                    "독립 테스트 사용": "최종 평가만",
                }
            )
        except (OSError, ValueError):
            continue
    if model_result_rows:
        st.dataframe(model_result_rows, hide_index=True, width="stretch")
    else:
        st.info(
            "저장된 추가 모델 비교가 없습니다. 선택형 모델 실행 기록은 기존 GP 기록과 분리됩니다."
        )
    st.markdown("#### 실제 데이터와 무작위 대조 성능 분포")
    completed_ids = list(selected.get("completed_experiment_ids", []))
    if completed_ids:
        comparison_id = st.selectbox(
            "비교할 원본 실험",
            completed_ids,
            index=len(completed_ids) - 1,
            key=f"random-control-experiment:{selected['research_id']}",
        )
        try:
            comparison_experiment = ExperimentStore(experiment_directory).load(comparison_id)
            control_report = summarize_matched_random_controls(comparison_experiment)
            control_rows = []
            for model_key, label in (
                ("one_number", "한 번호 모델"),
                ("six_number_set", "6개 집합 모델"),
            ):
                values = control_report["models"][model_key]
                interval = values.get("control_95_interval")
                mean_ci = values.get("control_mean_95_ci")
                control_rows.append(
                    {
                        "모델": label,
                        "관측 독립 테스트 점수": values.get("observed_test_score"),
                        "무작위 평균 점수": values.get("control_mean"),
                        "대조 분포 95% 구간": interval,
                        "대조 평균 95% 구간": mean_ci,
                        "관측-대조 평균": values.get("score_difference"),
                        "경험적 p값": values.get("empirical_p_value"),
                        "Holm 보정 p값": values.get("holm_adjusted_p_value"),
                        "유효 반복 수": values.get("control_replicates"),
                        "상태": values.get("interpretation_status"),
                    }
                )
            st.dataframe(control_rows, hide_index=True, width="stretch")
            selected_model = st.selectbox(
                "대조 분포 모델",
                ("six_number_set", "one_number"),
                format_func=lambda key: (
                    "6개 집합 모델" if key == "six_number_set" else "한 번호 모델"
                ),
                key=f"random-control-model:{comparison_id}",
            )
            selected_model_data = control_report["models"][selected_model]
            scores = control_score_rows(comparison_experiment, selected_model)
            if scores:
                figure = go.Figure()
                figure.add_trace(
                    go.Histogram(
                        x=[row["독립 테스트 점수"] for row in scores],
                        name="무작위 대조 GP 탐색",
                        marker_color="#6baed6",
                    )
                )
                observed_score = selected_model_data.get("observed_test_score")
                if observed_score is not None:
                    figure.add_vline(
                        x=observed_score,
                        line_dash="dash",
                        line_color="#d62728",
                        annotation_text="실제 데이터 선택 공식",
                    )
                figure.update_layout(
                    xaxis_title="독립 테스트 평가 점수",
                    yaxis_title="대조 반복 수",
                    bargap=0.08,
                )
                st.plotly_chart(figure, width="stretch")
            for limitation in control_report["limitations"]:
                st.caption(limitation)
        except (OSError, ValueError, KeyError, TypeError) as error:
            st.error(f"무작위 대조 결과를 불러오지 못했습니다: {error}")
    else:
        st.info("선택한 연구 회차에 비교할 완료 실험이 아직 없습니다.")
    st.caption(
        "평가 점수는 연구 내부의 선택·비교 지표이며 적중률이 아닙니다. "
        "적중률은 맞힌 번호 수를 활성 예측 수로 나눈 값으로 아래 표에 따로 표시합니다."
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
                        sum(int(row.get(f"{phase}_target_round_count") or 0) for row in model_rows),
                    ),
                    "적중률(번호/활성 예측)": (
                        sum(int(row.get(f"{phase}_hit_count") or 0) for row in model_rows)
                        / max(
                            1,
                            sum(
                                int(row.get(f"{phase}_activation_count") or 0) for row in model_rows
                            ),
                        )
                    ),
                    "반복 평가 3+ / 4+ / 5+ / 2등 / 6개": [
                        format_count(repeated.get("3_plus", 0)),
                        format_count(repeated.get("4_plus", 0)),
                        format_count(repeated.get("5_plus", 0)),
                        (
                            format_count(repeated.get("2_exact"))
                            if repeated.get("2_known", 0)
                            else "미수집"
                        )
                        if model == "six_number_set"
                        else "해당 없음",
                        format_count(repeated.get("6_exact", 0)),
                    ],
                    "고유 로또 회차 3+ / 4+ / 5+ / 2등 / 6개": [
                        format_count(unique.get("3_plus", 0)),
                        format_count(unique.get("4_plus", 0)),
                        format_count(unique.get("5_plus", 0)),
                        format_count(unique.get("2_exact"))
                        if model == "six_number_set" and unique.get("2_exact") is not None
                        else "미수집"
                        if model == "six_number_set"
                        else "해당 없음",
                        format_count(unique.get("6_exact", 0)),
                    ],
                }
            )
    st.dataframe(
        aggregate_hit_rows,
        column_config={
            "모델": st.column_config.TextColumn(
                "모델", help="번호 하나 모델 또는 6개 집합 모델입니다."
            ),
            "평가 구간": st.column_config.TextColumn(
                "평가 구간", help="적중 통계를 계산한 자료 구간입니다."
            ),
            "활성 예측 / 전체 대상 회차 수": st.column_config.TextColumn(
                "활성 예측 / 전체 대상 회차 수",
                help=(
                    "번호를 실제로 예측한 회차 수와 전체 평가 대상 회차 수입니다. "
                    "비활성 회차에는 예측이 없습니다."
                ),
            ),
            "적중률(번호/활성 예측)": st.column_config.NumberColumn(
                "적중률(번호/활성 예측)",
                help=(
                    "활성 예측에서 맞힌 번호 수를 활성 예측 회차 수로 나눈 값입니다. "
                    "6개 집합 모델은 회차당 최대 6입니다."
                ),
                format="%.4f",
            ),
            "반복 평가 3+ / 4+ / 5+ / 2등 / 6개": st.column_config.TextColumn(
                "반복 평가 3+ / 4+ / 5+ / 2등 / 6개",
                help=(
                    "실험별 평가를 합산한 누적 3개 이상·4개 이상·5개 이상, "
                    "정확한 2등, 본번호 6개 일치입니다."
                ),
            ),
            "고유 로또 회차 3+ / 4+ / 5+ / 2등 / 6개": st.column_config.TextColumn(
                "고유 로또 회차 3+ / 4+ / 5+ / 2등 / 6개",
                help="같은 로또 회차가 여러 실험에 포함돼도 한 번만 세어 계산한 횟수입니다.",
            ),
        },
        hide_index=True,
        width="stretch",
    )
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
                        "적중률": (
                            int(metrics.get(f"{phase}_hit_count") or 0)
                            / max(1, int(metrics.get(f"{phase}_activation_count") or 0))
                            if metrics.get(f"{phase}_hit_count") is not None
                            and metrics.get(f"{phase}_activation_count") is not None
                            else None
                        ),
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
        st.dataframe(
            hit_rows,
            column_config={
                "실험 ID": st.column_config.TextColumn(
                    "실험 ID", help="원본 연구 실험 식별자입니다.", width="small"
                ),
                "적중률": st.column_config.NumberColumn(
                    "적중률",
                    help=(
                        "활성 예측에서 맞힌 번호 수를 활성 예측 횟수로 나눈 값입니다. "
                        "점수 지표와 다릅니다."
                    ),
                    format="%.3f",
                ),
                **{
                    f"{count}개": st.column_config.NumberColumn(
                        f"{count}개",
                        help=f"해당 평가 구간에서 {count}개 번호를 맞힌 활성 예측 수입니다.",
                        format="%d",
                    )
                    for count in range(7)
                },
            },
            column_order=[
                "실험 ID",
                "모델",
                "평가 구간",
                "활성화 횟수",
                "적중 번호 수",
                "적중률",
                "0개",
                "1개",
                "2개",
                "3개",
                "4개",
                "5개",
                "6개",
            ],
            hide_index=True,
            width="stretch",
        )
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
                "경험적 p값 평균": format_probability(
                    sum(values) / len(values) if values else None
                ),
                "중앙값": format_probability(median(values) if values else None),
                "범위": (
                    f"{format_probability(min(values))} – {format_probability(max(values))}"
                    if values
                    else "미수집"
                ),
                "계산 실험 수": format_count(len(values)),
                "대조 반복/실험": format_count(control_runs),
                "p값 최소 해상도": format_probability(1 / (1 + control_runs))
                if isinstance(control_runs, int)
                else "미수집",
            }
        )
    st.dataframe(
        pvalue_rows,
        column_config={
            "모델": st.column_config.TextColumn(
                "모델", help="무작위 대조를 수행한 예측 모델입니다."
            ),
            "경험적 p값 평균": st.column_config.TextColumn(
                "경험적 p값 평균",
                help=(
                    "무작위 대조에서 관측한 p값의 평균입니다. 낮은 대조 횟수는 해상도가 거칩니다."
                ),
            ),
            "중앙값": st.column_config.TextColumn(
                "p값 중앙값", help="무작위 대조 p값의 중앙값입니다."
            ),
            "범위": st.column_config.TextColumn(
                "p값 범위", help="계산된 p값의 최솟값과 최댓값입니다."
            ),
            "계산 실험 수": st.column_config.TextColumn(
                "계산 실험 수", help="유효 p값이 계산된 실험 수입니다."
            ),
            "대조 반복/실험": st.column_config.TextColumn(
                "대조 반복/실험",
                help="실험 하나에서 수행한 무작위 대조 횟수입니다.",
            ),
            "p값 최소 해상도": st.column_config.TextColumn(
                "p값 최소 해상도",
                help="대조 반복 수로 제한되는 가장 작은 경험적 p값입니다.",
            ),
        },
        hide_index=True,
        width="stretch",
    )
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
        st.dataframe(stringify_dashboard_table_rows(compare_rows), hide_index=True, width="stretch")

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
        pattern_ids = list(
            dict.fromkeys(str(item.get("pattern_id")) for item in matches if item.get("pattern_id"))
        )
        if pattern_ids:
            selected_pattern_id = st.selectbox(
                "상세 조회할 일치 패턴", pattern_ids, key=f"exact-pattern:{selected['research_id']}"
            )
            try:
                pattern_detail = _pattern_library().get_pattern(selected_pattern_id)
            except sqlite3.OperationalError as error:
                _stop_on_database_error("pattern_detail_lookup", error)
            except KeyError:
                pattern_detail = next(
                    (
                        item
                        for item in matches
                        if str(item.get("pattern_id")) == selected_pattern_id
                    ),
                    {"pattern_id": selected_pattern_id},
                )
            with st.expander("선택 패턴의 조건·수식·특징 변수 보기", expanded=True):
                st.markdown("**활성화 조건**")
                st.code(
                    pattern_detail.get("activation_condition", "미수집"),
                    language="text",
                    wrap_lines=True,
                )
                formulas = pattern_detail.get("formulas", [])
                st.markdown("**출력 수식**")
                for index, formula in enumerate(formulas, start=1):
                    st.caption(f"번호 출력 {index}")
                    st.code(str(formula), language="text", wrap_lines=True)
                st.json(
                    {
                        "패턴 ID": pattern_detail.get("pattern_id"),
                        "현재 상태": pattern_detail.get(
                            "current_status", pattern_detail.get("validation_status", "미수집")
                        ),
                        "모델": pattern_detail.get("model_type", "미수집"),
                        "특징 프로파일": pattern_detail.get(
                            "feature_profile", config.get("feature_profile", "legacy")
                        ),
                        "출처 연구 ID": pattern_detail.get(
                            "origin_research_id", pattern_detail.get("experiment_id")
                        ),
                        "당첨 회차": pattern_detail.get("round_number"),
                    }
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
        model_names = {"one_number": "한 번호 모델", "six_number_set": "6개 집합 모델"}
        candidate_display_rows = [
            {
                "모델": model_names.get(item.get("model"), item.get("model", "미수집")),
                "후보 순위": item.get("rank"),
                "검증 평가 점수": format_decimal(item.get("validation_score"), suffix="점"),
                "독립 테스트 평가 점수": format_decimal(
                    item.get("independent_test_score"), suffix="점"
                ),
                "검증 활성률": format_percent(item.get("validation_activation_rate")),
                "수식 복잡도": item.get("complexity"),
                "활성화 조건": item.get("activation_condition"),
                "출력 수식": [pretty_formula(str(formula)) for formula in item.get("formulas", [])],
            }
            for item in candidates[:10]
        ]
        st.dataframe(
            candidate_display_rows,
            column_config={
                "모델": st.column_config.TextColumn(
                    "모델", help="번호 하나 모델과 6개 번호 집합 모델을 구분합니다."
                ),
                "후보 순위": st.column_config.NumberColumn(
                    "후보 순위", help="검증 결과를 바탕으로 정렬한 후보의 순위입니다.", format="%d"
                ),
                "검증 평가 점수": st.column_config.TextColumn(
                    "검증 평가 점수",
                    help="검증 구간에서 후보를 비교하는 점수입니다. 적중률과 별도 지표입니다.",
                ),
                "독립 테스트 평가 점수": st.column_config.TextColumn(
                    "독립 테스트 평가 점수",
                    help="검증 구간에서 선택을 마친 뒤 독립 테스트 구간에서 계산한 점수입니다.",
                ),
                "검증 활성률": st.column_config.TextColumn(
                    "검증 활성률",
                    help="검증 구간 중 예측 규칙이 활성화된 회차의 비율입니다.",
                ),
                "수식 복잡도": st.column_config.NumberColumn(
                    "수식 복잡도", help="활성화 조건과 출력 공식의 복잡도 지표입니다.", format="%d"
                ),
                "출력 수식": st.column_config.ListColumn(
                    "출력 수식", help="선택된 후보가 번호를 출력할 때 사용하는 수식입니다."
                ),
                "활성화 조건": st.column_config.TextColumn(
                    "활성화 조건",
                    help="이 조건이 선택하는 회차에서 예측 규칙을 활성화합니다.",
                    width="medium",
                ),
            },
            column_order=[
                "모델",
                "후보 순위",
                "검증 평가 점수",
                "독립 테스트 평가 점수",
                "검증 활성률",
                "수식 복잡도",
                "활성화 조건",
                "출력 수식",
            ],
            hide_index=True,
            width="stretch",
        )
        formula_candidates = [item for item in candidates if item.get("formulas")]
        if formula_candidates:
            formula_options = {
                (
                    f"{model_names.get(item.get('model'), item.get('model', '모델'))} · "
                    f"후보 {item.get('rank', '?')} · "
                    f"목록 {index}"
                ): item
                for index, item in enumerate(formula_candidates[:20], start=1)
            }
            formula_label = st.selectbox(
                "수식 상세 조회",
                list(formula_options),
                key=f"formula-detail:{selected['research_id']}",
            )
            chosen = formula_options[formula_label]
            with st.expander("수식과 입력 특징 변수", expanded=False):
                st.markdown("**활성화 조건**")
                st.code(
                    str(chosen.get("activation_condition", "미수집")),
                    language="text",
                    wrap_lines=True,
                )
                st.markdown("**출력 수식**")
                for index, formula in enumerate(chosen.get("formulas", []), start=1):
                    st.caption(f"번호 출력 {index}")
                    st.markdown(f"읽기 쉬운 표기: `{pretty_formula(str(formula))}`")
                    st.code(str(formula), language="text", wrap_lines=True)
                feature_rows = _feature_rows(
                    list(chosen.get("formulas", []))
                    + [str(chosen.get("activation_condition", ""))],
                    str(config.get("feature_profile", "legacy")),
                    int(config.get("max_lag", 10)),
                )
                st.dataframe(
                    feature_rows,
                    column_config={
                        "변수": st.column_config.TextColumn(
                            "변수", help="수식에 표시된 내부 특징 이름입니다.", width="small"
                        ),
                        "특징 변수": st.column_config.TextColumn(
                            "특징 변수", help="입력 변수의 한국어 이름입니다."
                        ),
                        "설명": st.column_config.TextColumn(
                            "설명", help="계산 대상과 기준 회차를 설명합니다."
                        ),
                    },
                    hide_index=True,
                    width="stretch",
                )
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

    st.markdown("#### 고적중 수식과 당첨 회차 직접 재검증")
    st.caption(
        "원본 JSON의 예측을 기준 데이터와 대조하고 GP 공식을 과거 입력 특징으로 다시 실행합니다. "
        "원본 연구 파일은 수정하지 않습니다."
    )
    audit_state_key = f"research_hit_audit:{selected['research_id']}"
    if st.button("고적중 기록 전체 재검증", key=f"audit:{selected['research_id']}"):
        try:
            with st.spinner("연결된 실험의 상세 기록과 GP 공식을 읽어 재계산하고 있습니다..."):
                st.session_state[audit_state_key] = audit_research_round(
                    str(selected["research_id"])
                )
        except (OSError, ValueError, KeyError) as error:
            st.error(f"고적중 기록 재검증 실패: {error}")
    hit_audit = st.session_state.get(audit_state_key)
    if hit_audit:
        six_audit = hit_audit.get("models", {}).get("six_number_set", {})
        audit_counts = []
        for phase, phase_name in (("validation", "검증"), ("independent_test", "독립 테스트")):
            repeated = six_audit.get("repeated_hit_counts", {}).get(phase, {})
            unique = six_audit.get("unique_round_counts", {}).get(phase, {})
            audit_counts.append(
                {
                    "평가 구간": phase_name,
                    "반복 평가 4+ / 5+ / 6개": [
                        repeated.get("4_plus", 0),
                        repeated.get("5_plus", 0),
                        repeated.get("6_exact", 0),
                    ],
                    "고유 회차 4+ / 5+ / 6개": [
                        unique.get("4_plus", 0),
                        unique.get("5_plus", 0),
                        unique.get("6_exact", 0),
                    ],
                }
            )
        st.dataframe(audit_counts, hide_index=True, width="stretch")
        formula_check = six_audit.get("formula_prediction_reproduction", {})
        winner_check = six_audit.get("winning_numbers_match_official", {})
        st.caption(
            f"실험 JSON 상세 로드 {hit_audit.get('loaded_experiment_count')} / "
            f"{hit_audit.get('linked_experiment_count')} · "
            f"공식 재실행 {formula_check.get('reproduced')} / "
            f"{formula_check.get('compared')} 이벤트 일치 · "
            f"당첨번호 기준 데이터 대조 {winner_check.get('verified')} / "
            f"{winner_check.get('compared')} 이벤트 일치"
        )
        st.caption(
            "연구 당시 메인 번호·날짜 해시 대조: "
            f"{'일치' if hit_audit.get('main_draw_date_hash_matches_study') else '불일치'} · "
            f"연구 해시 {hit_audit.get('recorded_dataset_sha256')} · "
            f"보너스 포함 데이터 해시 {hit_audit.get('current_full_dataset_sha256')}"
        )
        feature_map = hit_audit.get("feature_variables", {})
        st.caption(
            f"특징 변수 의미: x0 = {feature_map.get('x0', '미수집')} · "
            f"x57 = {feature_map.get('x57', '미수집')} (10회 전 번호의 쌍별 합)"
        )
        seed_provenance = hit_audit.get("elite_seed_provenance", {})
        st.caption(
            f"Elite 원본 {seed_provenance.get('source_pattern_count', 0)}개 중 동일 데이터 해시 "
            f"{seed_provenance.get('same_dataset_hash_count', 0)}개, 동일 검증·테스트 분할 "
            f"{seed_provenance.get('same_validation_split_count', 0)} / "
            f"{seed_provenance.get('same_test_split_count', 0)}개"
        )
        seed_rows = [
            {
                "원본 패턴 ID": item.get("pattern_id"),
                "원본 실험 ID": item.get("experiment_id"),
                "동일 데이터 해시": item.get("same_dataset_hash"),
                "동일 검증 / 테스트 분할": [
                    item.get("same_validation_split"),
                    item.get("same_test_split"),
                ],
                "테스트로 선택 여부": item.get("test_set_used_for_selection"),
                "원본 수식 검증 5+ 회차": item.get("seed_formula_validation_5_plus_rounds"),
                "원본 실험 테스트 4+ 회차": item.get("test_4_plus_rounds"),
            }
            for item in seed_provenance.get("patterns", [])
            if item.get("seed_formula_validation_5_plus_rounds") or item.get("test_4_plus_rounds")
        ]
        if seed_rows:
            st.dataframe(seed_rows, hide_index=True, width="stretch")
        high_hit_rows = [
            {
                "평가 구간": "검증" if item.get("phase") == "validation" else "독립 테스트",
                "로또 회차": item.get("round_number"),
                "일치 수": item.get("hit_count"),
                "예측 번호": item.get("predicted_numbers"),
                "실제 당첨번호": item.get("winning_numbers"),
                "실험 ID": item.get("experiment_id"),
                "활성화 조건": item.get("activation_condition"),
                "분기": item.get("rule_index"),
                "적중 분기 수식": item.get("active_formulas"),
                "사용 변수 의미": item.get("used_feature_variables"),
            }
            for item in six_audit.get("high_hits", [])
        ]
        if high_hit_rows:
            st.dataframe(high_hit_rows, hide_index=True, width="stretch")
        else:
            st.info("기록된 고적중 이벤트가 없습니다.")
        ranking_rows = [
            {
                "순위": index,
                "독립 테스트 4+ 반복": item.get("independent_test_4_plus"),
                "검증 5+ 반복": item.get("validation_5_plus"),
                "고유 적중 회차": item.get("independent_test_unique_rounds"),
                "검증 4+ 회차": item.get("validation_unique_rounds"),
                "실험 ID": item.get("experiment_ids"),
                "일치 Elite 원본 패턴": item.get("elite_seed_pattern_ids"),
                "활성화 조건": item.get("activation_condition"),
                "적중 분기 수식": item.get("active_formulas"),
                "변수 의미": item.get("variables"),
            }
            for index, item in enumerate(
                [
                    row
                    for row in six_audit.get("hit_formula_rankings", [])
                    if row.get("independent_test_4_plus", 0)
                ][:10],
                start=1,
            )
        ]
        if ranking_rows:
            st.markdown("**독립 테스트 고적중 공식 순위**")
            st.dataframe(ranking_rows, hide_index=True, width="stretch")
        for key, audit_report, detailed in (
            (
                "hit_audit_summary",
                build_hit_audit_markdown(hit_audit),
                False,
            ),
            (
                "hit_audit_detailed",
                build_hit_audit_markdown(hit_audit, detailed=True),
                True,
            ),
        ):
            response = report_copy_button(audit_report, key=key, detailed=detailed)
            if response and response.get("request_id") != st.session_state.get(
                f"{key}_last_request"
            ):
                st.session_state[f"{key}_last_request"] = response.get("request_id")
                if response.get("status") == "copied":
                    st.toast("재검증 보고서를 클립보드에 복사했습니다.", icon="✅")
                else:
                    st.error(f"재검증 보고서 복사 실패: {response.get('message', '브라우저 오류')}")

    report = build_chatgpt_report(selected)
    full_report = build_chatgpt_report(selected, detailed=True)
    saved_report_path = selected.get("report_path")
    if saved_report_path:
        saved_report = project_path(saved_report_path)
        if saved_report.is_file():
            st.download_button(
                "\uc800\uc7a5\ub41c \uc5f0\uad6c \ubcf4\uace0\uc11c \ub2e4\uc6b4\ub85c\ub4dc",
                saved_report.read_bytes(),
                file_name=saved_report.name,
                mime="text/markdown",
                key="saved_research_report_download",
            )
        else:
            report_error = selected.get("report_error")
            if report_error:
                st.warning(
                    "\uc5f0\uad6c\ub294 \uc644\ub8cc\ub410\uc73c\ub098 \ubcf4\uace0\uc11c \uc800\uc7a5\uc740 \uc2e4\ud328\ud588\uc2b5\ub2c8\ub2e4. "  # noqa: E501
                    f"{report_error.get('message', '')}"
                )
            else:
                st.caption(
                    "\uc774 \uc5f0\uad6c\uc758 \uc790\ub3d9 \ubcf4\uace0\uc11c\ub97c \ucc3e\uc9c0 \ubabb\ud588\uc2b5\ub2c8\ub2e4."  # noqa: E501
                )  # noqa: E501
    st.markdown("#### ChatGPT 분석 보고서")
    st.caption(
        "보고서에는 평가 점수와 번호 적중 횟수를 별도 항목으로 담고, "
        "검증 선택과 독립 테스트 결과를 구분합니다."
    )
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
            left_record = hydrate_round(
                rounds_store.get(str(choices[selected_left]["research_id"])), reaggregate=False
            )
            right_record = hydrate_round(
                rounds_store.get(str(choices[selected_right]["research_id"])), reaggregate=False
            )
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
