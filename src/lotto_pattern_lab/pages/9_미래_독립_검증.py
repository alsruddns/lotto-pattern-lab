"""Honest status page for the not-yet-implemented prospective validation flow."""

import streamlit as st

st.set_page_config(page_title="미래 독립 검증", layout="wide")
st.title("미래 독립 검증")
st.warning(
    "미래 회차 예측 사전 등록과 추첨 이후의 불변 대조 기능은 아직 구현되지 않았습니다. "
    "이 화면에는 과거 데이터를 나중에 실행한 결과를 미래 사전 예측으로 표시하지 않습니다."
)
st.markdown(
    "필요한 기능: 수식 동결 버전, 추첨 전 예측 시각과 입력 해시 기록, "
    "실제 결과 공개 후 append-only 평가 기록, 공식 동결 이후 회차만 집계."
)
