"""Display current local research configuration and resource settings."""

import shutil

import streamlit as st

from lotto_pattern_lab.gp_config import GPConfig
from lotto_pattern_lab.paths import PROJECT_ROOT
from lotto_pattern_lab.storage_management import load_storage_settings

st.set_page_config(page_title="설정", layout="wide")
st.title("설정")
defaults = GPConfig()
settings = load_storage_settings()
disk = shutil.disk_usage(PROJECT_ROOT)
st.subheader("탐색 기본값")
st.json(
    {
        "기본 병렬 워커": 4,
        "현재 GP 기본값": {
            field: getattr(defaults, field) for field in defaults.__dataclass_fields__
        },
    }
)
st.subheader("저장 공간")
st.json(
    {
        "저장 경로": str(PROJECT_ROOT),
        "저장 공간 한도(bytes)": settings["quota_bytes"],
        "사용 가능(bytes)": disk.free,
    }
)
st.info(
    "이 버전에서는 설정 화면을 읽기 전용으로 제공합니다. 저장된 사용자 설정을 변경하지 않습니다."
)
