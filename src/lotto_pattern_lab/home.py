"""Research laboratory landing page."""

import streamlit as st

st.set_page_config(page_title="AI 패턴 연구실", layout="wide")
st.title("AI 패턴 연구실")
st.write("당첨번호 데이터의 패턴을 탐색하고, 저장된 연구 결과를 재검증하는 로컬 연구 도구입니다.")

columns = st.columns(3)
columns[0].page_link("pages/1_단일 탐색.py", label="단일 탐색", icon="🔎")
columns[0].caption("지정한 설정으로 한 번의 탐색을 실행합니다.")
columns[1].page_link("pages/3_지속적_패턴_연구.py", label="지속적 패턴 연구", icon="🔬")
columns[1].caption("실험을 반복하고 후보와 실행 기록을 누적합니다.")
columns[2].page_link("pages/8_수학_공식_라이브러리.py", label="수학 공식 라이브러리", icon="📚")
columns[2].caption("연구에서 수집된 원본 후보 수식을 찾아봅니다.")

st.subheader("검증 도구")
left, middle, right = st.columns(3)
left.page_link("pages/6_전체_회차_재검증.py", label="전체 회차 재검증", icon="🔁")
left.caption("과거 구간에 저장된 6개 집합 수식을 다시 적용합니다.")
middle.page_link("pages/9_미래_독립_검증.py", label="미래 독립 검증", icon="🧪")
middle.caption("미래 회차 사전 예측 등록과 결과 대조 기능은 아직 구현되지 않았습니다.")
right.page_link("pages/5_연구_회차_대시보드.py", label="연구 회차 대시보드", icon="📊")

st.subheader("데이터와 설정")
left, right = st.columns(2)
left.page_link("pages/4_저장_용량_관리.py", label="데이터 및 저장 공간 관리", icon="🗄️")
right.page_link("pages/10_설정.py", label="설정", icon="⚙️")
st.page_link("pages/2_AI_다음_회차_번호_추천.py", label="다음 회차 번호 추천", icon="🎯")
st.page_link("pages/7_데이터_해시_비교.py", label="데이터 해시 비교", icon="🔍")
st.info("기존 연구 기록은 자동으로 초기화하거나 정리하지 않습니다.")
