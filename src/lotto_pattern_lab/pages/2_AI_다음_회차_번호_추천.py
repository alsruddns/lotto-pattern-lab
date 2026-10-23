"""다음 회차 추천 연구 화면."""

import csv
import hashlib
import io
import json
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
    pysr_replay_readiness,
    save_comparison,
    save_recommendation,
    summarize_prediction_history,
)
from lotto_pattern_lab.presentation import format_count, format_decimal, pretty_formula
from lotto_pattern_lab.research_recommendations import (
    generate_research_recommendations,
    list_research_recommendation_candidates,
)
from lotto_pattern_lab.storage import ExperimentStore


def _candidate_validation_value(candidate: dict) -> object:
    score = candidate.get("validation_score")
    if score is not None:
        return score
    return candidate.get("evaluation", {}).get("validation", {}).get("mae")


@st.cache_data(ttl=60, show_spinner=False)
def _cached_research_candidates(
    directory: str, file_manifest: tuple[tuple[str, int, int], ...]
):
    # The manifest is part of the key so newly completed experiments invalidate the cache.
    del file_manifest
    return list_research_recommendation_candidates(directory, limit=200)


st.set_page_config(page_title="다음 회차 번호 추천", layout="wide")
st.title("다음 회차 번호 추천")
st.warning("저장된 GP 공식을 실행하는 연구용 후보입니다. 당첨 확률이나 당첨을 보장하지 않습니다.")

experiments = project_path("experiments")
prediction_dir = project_path("predictions")
experiment_paths = sorted(experiments.glob("*.json"))
experiment_ids = [path.stem for path in experiment_paths]
default_id = "56f27190-1f2a-4573-9542-b30a2bbc60b4"
default_index = experiment_ids.index(default_id) if default_id in experiment_ids else 0
local_data_path = project_path("data/raw/lotto_draws_753_1246_with_bonus.csv")
official_data_path = project_path("data/official/draws.csv")
local_latest = 1246
official_latest_from_file = 1246
for data_path, value_name in (
    (local_data_path, "local_latest"),
    (official_data_path, "official_latest_from_file"),
):
    if data_path.is_file():
        try:
            latest_value = load_csv(data_path).records[-1].round_number
            if value_name == "local_latest":
                local_latest = latest_value
            else:
                official_latest_from_file = latest_value
        except (OSError, ValueError):
            pass

with st.sidebar:
    st.header("예측 입력")
    source = st.radio(
        "데이터 선택", ("수동 관리 데이터", "공식 날짜 데이터", "CSV 파일 업로드")
    )
    uploaded = st.file_uploader("CSV 파일", type=("csv",)) if source == "CSV 파일 업로드" else None
    source_latest = (
        official_latest_from_file if source == "공식 날짜 데이터" else local_latest
    )
    target_key = "prediction_target_round"
    if (
        st.session_state.get(target_key) is None
        or int(st.session_state[target_key]) <= source_latest
    ):
        st.session_state[target_key] = source_latest + 1
    target = st.number_input("추천 대상 회차", min_value=2, step=1, key=target_key)
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
    pysr_readiness = (
        pysr_replay_readiness(selected_experiment)
        if is_pysr and selected_experiment
        else {"status": "verified", "reason": "GP executor"}
    )
    if is_pysr:
        if pysr_readiness["status"] == "verified":
            st.caption(f"PySR 실행 준비 상태: {pysr_readiness['reason']}")
        else:
            st.warning(f"PySR 준비 실패: {pysr_readiness['reason']}")
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
                target_candidates = {
                    target: [
                        row
                        for row in selected_experiment.get("candidates", [])
                        if row.get("target") == target
                    ]
                    for target in target_names
                }
                common_count = min(map(len, target_candidates.values()), default=0)
                candidate_rows = target_candidates[sorted(target_names)[0]][:common_count]
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
                        f"{format_decimal(_candidate_validation_value(candidate_rows[rank - 1]))}"
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
    if source == "수동 관리 데이터":
        csv_path = project_path("data/raw/lotto_draws_753_1246_with_bonus.csv")
        if csv_path.is_file():
            data_bytes = csv_path.read_bytes()
            dataset = load_csv(csv_path)
    elif source == "공식 날짜 데이터":
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

st.subheader("누적 연구 기반 자동 추천")
st.caption(
    "완료된 GP·PySR 연구 중 시간순 검증과 홀드아웃 기록이 있고, "
    "현재 데이터 프로파일과 호환되는 저장 수식만 재실행합니다. "
    "과거 검증 성과는 실제 당첨 확률을 뜻하지 않습니다."
)
st.caption("성능을 위해 최근 저장 실험 최대 200건의 요약만 확인합니다.")
experiment_manifest = tuple(
    (path.name, path.stat().st_mtime_ns, path.stat().st_size)
    for path in sorted(
        experiments.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True
    )[:200]
)
research_candidates = _cached_research_candidates(str(experiments.resolve()), experiment_manifest)
research_profiles = sorted({str(item["feature_profile"]) for item in research_candidates})
if not research_profiles:
    st.info("시간순 검증, 홀드아웃 결과 및 재실행 조건을 갖춘 GP·PySR 후보를 찾지 못했습니다.")
elif dataset is None:
    st.info("연구 기반 추천을 만들려면 먼저 연속된 과거 회차가 포함된 데이터를 선택하세요.")
else:
    profile_index = (
        research_profiles.index("integrated_v1")
        if "integrated_v1" in research_profiles
        else 0
    )
    research_profile = st.selectbox(
        "추천에 사용할 특징 프로파일",
        research_profiles,
        index=profile_index,
        format_func=lambda value: {
            "legacy": "기본 특징",
            "calendar_round": "회차·날짜 특징",
            "bonus_lag_v1": "보너스 번호 포함 특징",
            "integrated_v1": "전체 통합 특징",
        }.get(value, value),
        key="research_prediction_feature_profile",
    )
    st.caption(
        "포함 특징 그룹: "
        + {
            "legacy": "기본",
            "calendar_round": "기본 + 회차·날짜",
            "bonus_lag_v1": "기본 + 과거 보너스",
            "integrated_v1": "기본 + 회차·날짜 + 과거 보너스",
        }.get(research_profile, "저장된 연구 프로파일")
    )
    profile_candidates = [
        item for item in research_candidates if item["feature_profile"] == research_profile
    ]
    st.write(f"재실행 가능 후보: **{len(profile_candidates):,}개**")
    st.caption(
        "후보는 검증 근거와 수식 재실행 가능성으로 거릅니다. "
        "GP와 PySR의 서로 다른 점수는 직접 비교하지 않습니다. "
        "미래 독립 검증 기록은 별도 영역에서 확인할 수 있습니다."
    )
    requested_recommendations = st.selectbox(
        "추천 조합 수", (1, 3, 5, 10), key="research_prediction_recommendation_count"
    )
    if st.button(
        "검증 기반 추천 조합 생성",
        key="generate_research_based_recommendations",
        disabled=not profile_candidates,
        type="primary",
    ):
        try:
            st.session_state["research_recommendation_drafts"] = []
            st.session_state["research_recommendation_failures"] = []
            official_latest = _fetch_latest_round()
            if int(target) != official_latest + 1:
                raise ValueError(
                    "사전 추천 대상은 최신 확정 회차의 다음 회차인 "
                    f"{official_latest + 1}회여야 합니다."
                )
            drafts, failures = generate_research_recommendations(
                profile_candidates,
                dataset,
                int(target),
                int(requested_recommendations),
                feature_profile=research_profile,
                data_file_sha256=hashlib.sha256(data_bytes).hexdigest() if data_bytes else None,
                random_seed=int(seed),
                official_latest_round=official_latest,
                target_date=target_date,
            )
            st.session_state["research_recommendation_drafts"] = drafts
            st.session_state["research_recommendation_failures"] = failures
            if len(drafts) < int(requested_recommendations):
                st.info(
                    f"요청한 {requested_recommendations}조합 중 근거가 확인된 고유 조합 "
                    f"{len(drafts)}개만 생성했습니다. 임의 번호로 채우지 않았습니다."
                )
        except Exception as error:
            st.error(f"추천 생성 실패: {error}")

research_drafts = st.session_state.get("research_recommendation_drafts", [])
if research_drafts:
    st.markdown("#### 생성된 추천 초안")
    st.caption("초안은 자동 저장하지 않습니다. 저장을 눌러야 사전 등록 이력에 추가됩니다.")
    saved_draft_ids = st.session_state.setdefault("saved_research_recommendation_ids", set())
    for draft_index, draft in enumerate(research_drafts, start=1):
        with st.container(border=True):
            st.markdown(
                f"**추천 {draft_index} · {draft['target_round']}회**  "
                + " · ".join(f"`{number}`" for number in draft["recommended_numbers"])
            )
            for source_record in draft.get("research_selection", {}).get(
                "supporting_candidates", []
            ):
                st.caption(
                    f"{source_record['engine']} · 연구 {source_record['experiment_id']} · "
                    f"후보 {source_record['candidate_rank']} · "
                    f"검증 {source_record.get('validation_round_count') or '미수집'}회 / "
                    f"홀드아웃 {source_record.get('test_round_count') or '미수집'}회"
                )
            with st.expander("추천 근거 및 사용 공식"):
                st.json(draft.get("research_selection", {}))
                st.write(
                    "후보별 기록 통계",
                    draft.get("research_selection", {}).get("supporting_candidates", []),
                )
                st.write("사용 공식", draft.get("used_formulas", []))
                st.write("데이터 출처", draft.get("data_provenance", {}))
            already_saved = draft["recommendation_id"] in saved_draft_ids
            if already_saved:
                st.success("이 추천은 이미 저장되었습니다.")
            elif st.button(
                f"추천 {draft_index} 사전 등록",
                key=f"save_research_recommendation_{draft['recommendation_id']}",
            ):
                try:
                    current_official_round = _fetch_latest_round()
                    if int(draft["target_round"]) != int(current_official_round) + 1:
                        raise ValueError(
                            "최신 확정 회차가 바뀌어 이 초안을 사전 추천으로 등록할 수 없습니다. "
                            "현재 회차 기준으로 다시 생성하세요."
                        )
                    saved_path = save_recommendation(draft, prediction_dir)
                    saved_draft_ids.add(draft["recommendation_id"])
                    st.session_state["saved_research_recommendation_ids"] = saved_draft_ids
                    st.session_state["active_recommendation_id"] = draft["recommendation_id"]
                    st.success(f"사전 추천 기록을 저장했습니다: {saved_path.name}")
                except FileExistsError:
                    saved_draft_ids.add(draft["recommendation_id"])
                    st.info("동일 추천 기록이 이미 저장되어 있습니다.")
                except (OSError, ValueError) as error:
                    st.error(f"추천 저장 실패: {error}")
if st.session_state.get("research_recommendation_failures"):
    with st.expander("실행할 수 없었던 후보"):
        st.dataframe(
            st.session_state["research_recommendation_failures"],
            hide_index=True,
            width="stretch",
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
    "공식 실행 및 추천 저장",
    type="primary",
    disabled=(
        dataset is None or not experiment_id or (is_pysr and pysr_readiness["status"] != "verified")
    ),
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
            raise ValueError("최신 확정 회차를 확인할 수 없어 사전 추천을 만들 수 없습니다.")
        if int(target) != official_latest + 1:
            raise ValueError(
                f"예측 대상은 최신 확정 회차의 다음 회차인 {official_latest + 1}회입니다. "
                "회차가 달력상 예정 회차인지 확인해 주세요."
            )
        if is_pysr:
            from lotto_pattern_lab.prediction import create_pysr_recommendation

            recommendation = create_pysr_recommendation(
                experiment,
                dataset,
                int(target),
                candidate_rank=selected_candidate_rank,
                data_file_sha256=hashlib.sha256(data_bytes).hexdigest() if data_bytes else None,
                random_seed=int(seed),
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
            raise ValueError("추첨 완료 회차는 사전 추천으로 저장할 수 없습니다.")
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
        prediction_csv = io.StringIO(newline="")
        csv_writer = csv.writer(prediction_csv, lineterminator="\n")
        csv_writer.writerow(("예측 회차", "예측 번호", "공식 ID", "예측 생성 시각"))
        for number in recommendation["recommended_numbers"]:
            csv_writer.writerow(
                (
                    recommendation["target_round"],
                    number,
                    recommendation.get("formula_fingerprint"),
                    recommendation["created_at"],
                )
            )
        st.download_button(
            "예측 결과 CSV 다운로드",
            prediction_csv.getvalue().encode("utf-8-sig"),
            file_name=f"예측_{recommendation['target_round']}회.csv",
            mime="text/csv",
        )
        st.download_button(
            "예측 결과 JSON 다운로드",
            (json.dumps(recommendation, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
            file_name=f"예측_{recommendation['target_round']}회.json",
            mime="application/json",
        )
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
        comparison = compare_recommendation(
            recommendation, actual.draw.numbers, actual_bonus=actual.draw.bonus
        )
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
        "\uc120\ud0dd \ud6c4\ubcf4\ub97c \uace0\uc720 \ud68c\ucc28\ub85c \ud68c\uace0 \uac80\uc99d",
        disabled=is_pysr,
        help=("PySR 후보별 고유 회차 재검증은 현재 추가 검증이 필요합니다." if is_pysr else None),
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
