"""Read-only view of formulas already recorded in the pattern database."""

import hashlib
import json
import sqlite3

import streamlit as st

from lotto_pattern_lab.dataset import load_csv
from lotto_pattern_lab.formula_similarity import cluster_patterns
from lotto_pattern_lab.formula_simplify import (
    save_verified_simplification,
    simplify_candidate,
    verify_candidate_simplification,
)
from lotto_pattern_lab.paths import project_path
from lotto_pattern_lab.storage import ExperimentStore

st.set_page_config(page_title="수학 공식 라이브러리", layout="wide")
st.title("수학 공식 라이브러리")
st.caption("저장된 패턴을 읽기 전용으로 표시합니다. 원본 연구 ID와 실험 ID를 보존합니다.")

database = project_path("data/database/patterns.sqlite")
if not database.is_file():
    st.info("수학 공식 데이터베이스가 아직 없습니다.")
    st.stop()

connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
connection.row_factory = sqlite3.Row
try:
    columns = {row[1] for row in connection.execute("PRAGMA table_info(patterns)")}
    wanted = [
        "pattern_id",
        "origin_research_id",
        "status",
        "model_type",
        "feature_profile",
        "discovered_at",
        "formula",
        "activation_condition",
        "training_score",
        "validation_score",
        "independent_test_score",
        "complexity",
        "payload",
    ]
    selected = [name for name in wanted if name in columns]
    records = [
        dict(row)
        for row in connection.execute(
            f"SELECT {', '.join(selected)} FROM patterns ORDER BY discovered_at DESC"
        )
    ]
finally:
    connection.close()

if not records:
    st.info("저장된 수식이 없습니다.")
    st.stop()

st.metric("저장된 공식 후보", len(records))
statuses = ["전체", *sorted({str(item.get("status") or "미분류") for item in records})]
status = st.selectbox("후보 상태", statuses)
visible = (
    records
    if status == "전체"
    else [item for item in records if (item.get("status") or "미분류") == status]
)
visible_patterns = []
for item in visible:
    payload = json.loads(item.get("payload") or "{}")
    performance = payload.get("second_prize_by_split", {}).get("independent_test", {})
    model_type = item.get("model_type") or payload.get("model_type")
    known = int(performance.get("known", 0) or 0)
    visible_patterns.append(
        {
            **{key: value for key, value in item.items() if key != "payload"},
            "2등 독립 테스트": performance.get("hits") if known else None,
            "2등 판정 가능 조합": known if performance else None,
            "2등 통계 상태": (
                "판정 완료"
                if known and model_type == "six_number_set"
                else "미수집"
                if model_type == "six_number_set"
                else "해당 없음"
            ),
        }
    )
st.dataframe(
    visible_patterns,
    column_config={
        "2등 독립 테스트": st.column_config.NumberColumn(
            "2등 독립 테스트",
            help=(
                "독립 테스트에서 본번호 정확히 5개와 예측 6개 안의 "
                "실제 보너스가 일치한 횟수입니다."
            ),
            format="localized",
        ),
        "2등 판정 가능 조합": st.column_config.NumberColumn(
            "2등 판정 가능 조합", format="localized"
        ),
        "2등 통계 상태": st.column_config.TextColumn("2등 통계 상태"),
    },
    width="stretch",
    hide_index=True,
)

st.subheader("수식 유사도 및 그룹")
analysis_limit = st.selectbox(
    "비교할 최신 수식 수",
    (100, 250),
    index=1,
    help="수식 수가 많으면 모든 쌍 비교가 매우 커지므로 최근 기록부터 제한하여 계산합니다.",
)
library_patterns = []
for item in records[:analysis_limit]:
    payload = json.loads(item.get("payload") or "{}")
    library_patterns.append({**payload, "pattern_id": item["pattern_id"]})
with st.expander("유사도 기준과 결과 해석"):
    similarity_threshold = st.slider(
        "구조 유사도 기준",
        min_value=0.5,
        max_value=1.0,
        value=0.8,
        step=0.05,
        help=(
            "연산자·변수·상수의 AST 구조가 얼마나 비슷한지 비교합니다. "
            "높은 점수도 수학적 동등성의 증명은 아닙니다."
        ),
    )
    similarity_filter = st.selectbox(
        "수식 종류",
        (
            "전체",
            "완전 동일 수식",
            "구조적 유사 수식",
            "출력 결과 유사 수식",
            "서로 다른 고유 수식",
        ),
    )
try:
    similarity = cluster_patterns(
        library_patterns, structural_threshold=float(similarity_threshold)
    )
except (SyntaxError, ValueError, TypeError) as error:
    similarity = None
    st.error(f"수식 구조를 비교하지 못했습니다: {error}")
if similarity:
    group_by_id = {
        member_id: group
        for group in similarity["groups"]
        for member_id in group["member_ids"]
    }
    pair_by_id = {}
    for pair in similarity["pairwise"]:
        pair_by_id.setdefault(pair["left_id"], []).append(pair)
        pair_by_id.setdefault(pair["right_id"], []).append(pair)
    visible_patterns = []
    for pattern in library_patterns:
        pattern_id = str(pattern["pattern_id"])
        group = group_by_id[pattern_id]
        pairs = pair_by_id.get(pattern_id, [])
        exact = any(item["exact_duplicate"] for item in pairs)
        structural = any(
            item["structural_similarity"] >= similarity_threshold for item in pairs
        )
        output_similar = any(
            item.get("output_similarity") is not None
            and item["output_similarity"] >= similarity_threshold
            for item in pairs
        )
        if similarity_filter == "완전 동일 수식" and not exact:
            continue
        if similarity_filter == "구조적 유사 수식" and not structural:
            continue
        if similarity_filter == "출력 결과 유사 수식" and not output_similar:
            continue
        if similarity_filter == "서로 다른 고유 수식" and len(group["member_ids"]) > 1:
            continue
        visible_patterns.append(
            {
                "수식 ID": pattern_id,
                "그룹 ID": group["group_id"],
                "그룹 수식 수": group["member_count"],
                "대표 수식": group["representative_id"] == pattern_id,
                "그룹 유형": group["group_type"],
                "최고 구조 유사도": max(
                    (item["structural_similarity"] for item in pairs), default=0.0
                ),
                "관측 출력 비교 가능": output_similar,
            }
        )
    st.caption(
        f"최신 {similarity['patterns']:,}개 수식 분석 · "
        f"{len(similarity['groups']):,}개 그룹. "
        "구조가 닮은 결과는 유사 후보일 뿐 수학적으로 같은 공식이라는 뜻은 아닙니다."
    )
    st.dataframe(visible_patterns, hide_index=True, width="stretch")
    selected_group = st.selectbox(
        "그룹 비교",
        [group["group_id"] for group in similarity["groups"]],
        format_func=lambda group_id: next(
            f"{group_id} · {group['member_count']}개 수식"
            for group in similarity["groups"]
            if group["group_id"] == group_id
        ),
    )
    group = next(item for item in similarity["groups"] if item["group_id"] == selected_group)
    st.write(
        {
            "대표 수식": group["representative_id"],
            "구성 수식 ID": group["member_ids"],
            "대표 선정 기준": "복잡도 우선, 동률이면 독립 테스트 점수 순",
        }
    )
    if similarity_filter == "출력 결과 유사 수식" and not any(
        pair.get("output_similarity") is not None for pair in similarity["pairwise"]
    ):
        st.info(
            "같은 회차에서 비교 가능한 예측 출력 기록이 없어 "
            "출력 유사도는 아직 계산되지 않았습니다."
        )

st.subheader("수식 계보")
with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as lineage_connection:
    lineage_connection.row_factory = sqlite3.Row
    lineage_rows = [
        dict(row)
        for row in lineage_connection.execute(
            "SELECT parent_pattern_id, child_pattern_id, research_id, generation, method, "
            "operators, parent_formula, child_formula, created_at "
            "FROM pattern_lineage ORDER BY created_at DESC LIMIT 500"
        )
    ]
if lineage_rows:
    st.caption(
        "저장된 부모·자식 연결만 표시합니다. "
        "연결이 없는 과거 공식의 계보는 미확인 상태입니다."
    )
    st.dataframe(lineage_rows, hide_index=True, width="stretch")
else:
    st.info(
        "확인된 공식 계보 기록이 없습니다. "
        "계보 정보가 없는 과거 기록은 임의로 연결하지 않습니다."
    )
selected_id = st.selectbox("공식 상세 보기", [item["pattern_id"] for item in visible])
selected = next(item for item in visible if item["pattern_id"] == selected_id)
payload = json.loads(selected.get("payload") or "{}")
st.code(json.dumps(payload, ensure_ascii=False, indent=2), language="json")
st.caption("원본 후보와 원본 연구 기록은 이 화면에서 수정하지 않습니다.")
st.download_button(
    "후보 JSON 다운로드",
    data=(json.dumps(selected, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    file_name=f"formula_{selected_id}.json",
    mime="application/json",
)

st.subheader("수식 단순화 및 전체 데이터 동등성 검증")
with st.expander("실험 후보에서 단순화 버전 만들기", expanded=False):
    experiment_paths = sorted(
        project_path("experiments").glob("*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not experiment_paths:
        st.info("불러올 실험 수식이 없습니다.")
    else:
        experiment_ids = [path.stem for path in experiment_paths]
        experiment_id = st.selectbox("원본 GP 실험", experiment_ids, key="simplify_experiment")
        try:
            experiment = ExperimentStore("experiments").load(experiment_id)
            model = experiment.get("six_number_set_model", {})
            candidates = model.get("candidates") or (
                [model["selected"]] if model.get("selected") else []
            )
        except (OSError, ValueError, KeyError) as error:
            experiment = None
            candidates = []
            st.error(f"실험 후보를 읽을 수 없습니다: {error}")
        if candidates:
            rank = st.selectbox(
                "6개 집합 후보 순위", range(1, len(candidates) + 1), key="simplify_candidate_rank"
            )
            source_profile = str(experiment.get("config", {}).get("feature_profile", "legacy"))
            input_path = project_path(
                "data/raw/lotto_draws_753_1246_with_bonus.csv"
                if source_profile == "bonus_lag_v1"
                else "data/official/draws.csv"
            )
            if not input_path.is_file():
                st.error("원본 특징 프로파일에 필요한 전체 데이터 파일이 없습니다.")
            else:
                try:
                    evaluation_dataset = load_csv(input_path)
                    original_candidate = candidates[rank - 1]
                    simplified_candidate, metrics = simplify_candidate(original_candidate)
                    st.caption(
                        f"원본 실험 {experiment_id} · 프로파일 {source_profile} · "
                        f"평가 범위 {evaluation_dataset.records[0].round_number}~"
                        f"{evaluation_dataset.records[-1].round_number}회"
                    )
                    with st.expander("원본 및 단순화 수식 미리보기"):
                        st.code(
                            json.dumps(
                                {
                                    "original_gate": original_candidate.get("activation_condition"),
                                    "simplified_gate": simplified_candidate["activation_condition"],
                                    "original_formulas": original_candidate.get("formulas"),
                                    "simplified_formulas": simplified_candidate["formulas"],
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                            language="json",
                        )
                    file_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
                    state_key = f"simplification:{experiment_id}:{rank}:{file_hash}"
                    if st.button(
                        "전체 평가 가능 회차에서 동등성 검사",
                        key="run_formula_equivalence",
                    ):
                        try:
                            with st.spinner(
                                "게이트, 분기별 중간값 및 최종 번호를 전 회차에서 비교합니다."
                            ):
                                verification = verify_candidate_simplification(
                                    original_candidate,
                                    simplified_candidate,
                                    experiment,
                                    evaluation_dataset,
                                )
                            st.session_state[state_key] = {
                                "simplified_candidate": simplified_candidate,
                                "metrics": metrics,
                                "verification": verification,
                                "dataset_sha256": file_hash,
                            }
                        except (OSError, ValueError, KeyError) as error:
                            st.error(f"동등성 검사 실패: {error}")
                    result = st.session_state.get(state_key)
                    if result:
                        verification = result["verification"]
                        if verification["equivalent"]:
                            st.success(
                                f"전 회차 동등성 확인 · "
                                f"평가 {verification['evaluation_rounds']}회 · "
                                f"연산 수 {metrics['original_operator_count']} → "
                                f"{metrics['simplified_operator_count']}"
                            )
                            if st.button("검증된 단순화 수식을 새 버전으로 저장"):
                                saved = save_verified_simplification(
                                    experiment_id=experiment_id,
                                    research_id=experiment.get("research_id"),
                                    candidate_rank=rank,
                                    original_candidate=original_candidate,
                                    simplified_candidate=result["simplified_candidate"],
                                    metrics=result["metrics"],
                                    verification=verification,
                                    dataset_sha256=result["dataset_sha256"],
                                )
                                st.success(
                                    "원본과 분리된 단순화 버전을 저장했습니다: "
                                    f"{saved['simplification_id']}"
                                )
                        else:
                            st.error(
                                f"동등성 검증 실패 · 차이 {verification['difference_count']}건. "
                                "원본 후보를 유지하고 단순화 버전은 저장하지 않았습니다."
                            )
                            st.dataframe(verification["differences"], hide_index=True)
                except (OSError, ValueError, KeyError) as error:
                    st.error(f"수식 동등성 검증을 준비하지 못했습니다: {error}")
        else:
            st.info("선택한 실험에 6개 집합 GP 후보가 없습니다.")

saved_simplifications = sorted(
    project_path("data/formula_simplifications").glob("*.json")
) if project_path("data/formula_simplifications").is_dir() else []
if saved_simplifications:
    st.caption(f"동등성 검증된 별도 단순화 버전: {len(saved_simplifications)}개")
