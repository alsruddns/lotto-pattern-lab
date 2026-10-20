"""Read-only view of formulas already recorded in the pattern database."""

import json
import sqlite3

import streamlit as st

from lotto_pattern_lab.paths import project_path

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
st.dataframe(
    [{key: value for key, value in item.items() if key != "payload"} for item in visible],
    use_container_width=True,
    hide_index=True,
)
selected_id = st.selectbox("공식 상세 보기", [item["pattern_id"] for item in visible])
selected = next(item for item in visible if item["pattern_id"] == selected_id)
payload = json.loads(selected.get("payload") or "{}")
st.code(json.dumps(payload, ensure_ascii=False, indent=2), language="json")
st.caption("수식 단순화 검증, 공식 동결, 미래 예측 등록은 아직 구현되지 않았습니다.")
st.download_button(
    "후보 JSON 다운로드",
    data=(json.dumps(selected, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    file_name=f"formula_{selected_id}.json",
    mime="application/json",
)
