"""Streamlit navigation entry point for the research interface."""

import streamlit as st

pages = [
    st.Page("home.py", title="연구실 홈", icon="🏠", default=True),
    st.Page("pages/1_단일 탐색.py", title="단일 탐색", icon="🔎"),
    st.Page("pages/3_지속적_패턴_연구.py", title="지속적 패턴 연구", icon="🔬"),
    st.Page("pages/8_수학_공식_라이브러리.py", title="수학 공식 라이브러리", icon="📚"),
    st.Page("pages/6_전체_회차_재검증.py", title="전체 회차 재검증", icon="🔁"),
    st.Page("pages/9_미래_독립_검증.py", title="미래 독립 검증", icon="🧪"),
    st.Page("pages/5_연구_회차_대시보드.py", title="연구 회차 대시보드", icon="📊"),
    st.Page("pages/4_저장_용량_관리.py", title="데이터 및 저장 공간 관리", icon="🗄️"),
    st.Page("pages/10_설정.py", title="설정", icon="⚙️"),
    st.Page("pages/2_AI_다음_회차_번호_추천.py", title="다음 회차 추천", visibility="hidden"),
    st.Page("pages/7_데이터_해시_비교.py", title="데이터 해시 비교", visibility="hidden"),
]
st.navigation(pages, position="sidebar").run()
