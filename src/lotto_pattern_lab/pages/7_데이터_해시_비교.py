"""Compare current official data hashes with preserved historical metadata."""

from __future__ import annotations

import json

import streamlit as st

from lotto_pattern_lab.data_audit import compare_datasets, dataset_hashes, legacy_dataset_hash
from lotto_pattern_lab.dataset import load_csv
from lotto_pattern_lab.paths import project_path

st.set_page_config(page_title="데이터 해시 비교", layout="wide")
st.title("데이터 및 해시 비교")
official_path = project_path("data/official/draws.csv")
if not official_path.exists():
    st.error("공식 데이터 파일을 찾을 수 없습니다.")
    st.stop()
dataset = load_csv(official_path)
current_hashes = dataset_hashes(dataset)
st.subheader("현재 데이터 스냅샷")
st.json({**current_hashes, "legacy_aggregate_sha256": legacy_dataset_hash(dataset)})

experiment_files = sorted(
    project_path("experiments").glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True
)
if experiment_files:
    selected = st.selectbox("과거 실험 기록", experiment_files, format_func=lambda path: path.stem)
    try:
        metadata = json.loads(selected.read_text(encoding="utf-8")).get("dataset", {})
        comparison = compare_datasets(None, dataset, previous_metadata=metadata)
        st.subheader("과거 기록과 현재 스냅샷")
        st.json(comparison)
        st.info(
            "기존 실험 JSON에는 통합 SHA-256과 회차 범위만 저장되어 있습니다. "
            "과거 행별 CSV 스냅샷이 보존되지 않은 실험은 저장된 해시만으로 "
            "번호·날짜·보너스의 변경 필드를 복원할 수 없습니다. "
            "원본 기록은 이 화면에서 수정하지 않습니다."
        )
    except (OSError, ValueError) as error:
        st.error(f"실험 기록을 읽지 못했습니다: {error}")
