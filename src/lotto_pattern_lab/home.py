"""Research laboratory landing page."""

from pathlib import Path

import streamlit as st

from lotto_pattern_lab.dataset import load_csv
from lotto_pattern_lab.paths import project_path

st.set_page_config(page_title="AI 패턴 연구실", layout="wide")
st.title("AI 패턴 연구실")
st.write("당첨번호 데이터의 패턴을 탐색하고, 저장된 연구 결과를 재검증하는 로컬 연구 도구입니다.")

columns = st.columns(4)
columns[0].page_link("pages/1_단일 탐색.py", label="단일 탐색", icon="🔎")
columns[0].caption("지정한 설정으로 한 번의 탐색을 실행합니다.")
columns[1].page_link("pages/3_지속적_패턴_연구.py", label="지속적 패턴 연구", icon="🔬")
columns[1].caption("실험을 반복하고 후보와 실행 기록을 누적합니다.")
columns[2].page_link("pages/8_수학_공식_라이브러리.py", label="수학 공식 라이브러리", icon="📚")
columns[2].caption("연구에서 수집된 원본 후보 수식을 찾아봅니다.")
columns[3].page_link(
    "pages/11_당첨번호_관리.py", label="당첨번호 관리 열기", icon="🎱"
)
columns[3].caption("새 회차의 당첨번호를 등록하고 기존 회차 데이터를 조회·검증합니다.")


@st.cache_data(show_spinner=False)
def _cached_draw_summary(
    csv_path: str,
    csv_mtime_ns: int,
    csv_size: int,
    metadata_mtime_ns: int,
    metadata_size: int,
    manifest_mtime_ns: int,
    manifest_size: int,
) -> dict[str, int | None]:
    dataset = load_csv(csv_path)
    verified_rounds = []
    pending_count = 0
    for record in dataset.records:
        status = record.verification_status or dataset.verification_status
        if status == "official_verified":
            verified_rounds.append(record.round_number)
        elif status != "verification_failed":
            pending_count += 1
    return {
        "latest_registered_round": dataset.records[-1].round_number,
        "latest_official_round": max(verified_rounds, default=None),
        "registered_round_count": len(dataset.records),
        "verification_pending_count": pending_count,
    }


def _file_signature(path: Path) -> tuple[int, int]:
    try:
        stat = path.stat()
    except OSError:
        return 0, 0
    return stat.st_mtime_ns, stat.st_size


draw_csv_path = project_path("data/raw/lotto_draws_753_1246_with_bonus.csv")
draw_metadata_path = Path(f"{draw_csv_path}.metadata.json")
draw_manifest_path = Path(f"{draw_csv_path}.manifest.json")
draw_summary = None
if draw_csv_path.is_file():
    try:
        draw_summary = _cached_draw_summary(
            str(draw_csv_path),
            *_file_signature(draw_csv_path),
            *_file_signature(draw_metadata_path),
            *_file_signature(draw_manifest_path),
        )
    except (OSError, ValueError):
        draw_summary = None

st.subheader("당첨번호 데이터 현황")
summary_columns = st.columns(4)
summary_values = {
    "최신 등록 회차": (
        f"{draw_summary['latest_registered_round']}회" if draw_summary else "미수집"
    ),
    "최신 공식 검증 완료 회차": (
        f"{draw_summary['latest_official_round']}회"
        if draw_summary and draw_summary["latest_official_round"]
        else "미수집"
    ),
    "총 등록 회차 수": (
        f"{draw_summary['registered_round_count']:,}회" if draw_summary else "미수집"
    ),
    "검증 대기 회차 수": (
        f"{draw_summary['verification_pending_count']:,}회" if draw_summary else "미수집"
    ),
}
for column, (label, value) in zip(summary_columns, summary_values.items(), strict=True):
    column.metric(label, value)

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
