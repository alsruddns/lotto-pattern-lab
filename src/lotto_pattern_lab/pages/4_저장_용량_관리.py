"""Korean storage inventory and explicit, preview-first maintenance controls."""

import streamlit as st

from lotto_pattern_lab.storage_management import (
    DEFAULT_QUOTA_BYTES,
    checkpoint_cleanup_plan,
    compress_experiment,
    execute_checkpoint_cleanup,
    experiment_compression_plan,
    load_storage_settings,
    save_storage_settings,
    storage_inventory,
    storage_over_quota,
)

st.set_page_config(page_title="저장 용량 관리", layout="wide")
st.title("시스템 설정 · 저장 용량 관리")
st.caption(
    "실험 원본과 추천 기록은 자동 삭제하지 않습니다. 기존 실험 JSON의 압축과 체크포인트 정리는 "
    "대상 및 예상 확보 용량을 확인하고 각각 실행을 승인해야 합니다."
)

settings = load_storage_settings()
inventory = storage_inventory()
if storage_over_quota(inventory, settings["quota_bytes"]):
    st.warning(
        f"설정한 저장 용량 한도를 초과했습니다: {inventory['total_bytes'] / 1024**3:.2f} GiB / "
        f"{settings['quota_bytes'] / 1024**3:.2f} GiB."
    )

left, right = st.columns(2)
left.metric("관리 폴더 전체", f"{inventory['total_bytes'] / 1024**2:.2f} MiB")
right.metric("전체 파일 수", sum(folder["file_count"] for folder in inventory["folders"]))
st.dataframe(
    [
        {
            "폴더": folder["folder"],
            "상태": "있음" if folder["exists"] else "없음",
            "파일 수": folder["file_count"],
            "용량 (MiB)": round(folder["bytes"] / 1024**2, 3),
        }
        for folder in inventory["folders"]
    ],
    use_container_width=True,
    hide_index=True,
)

st.subheader("보존 및 경고 설정")
quota_gib = st.number_input(
    "용량 경고 한도 (GiB)",
    min_value=0.1,
    max_value=100_000.0,
    value=max(0.1, settings["quota_bytes"] / 1024**3),
    step=1.0,
    help="한도를 넘으면 연구 화면과 이 페이지에 용량 경고가 표시됩니다.",
)
auto_cleanup = st.checkbox(
    "자동 체크포인트 정리 활성화",
    value=settings["automatic_checkpoint_cleanup"],
    help=(
        "켜면 연구 작업이 종료될 때 검증된 오래된 완료/실패 체크포인트만 "
        "최근 5개 정책으로 정리합니다."
    ),
)
keep_count = st.number_input(
    "최근 체크포인트 보존 개수",
    min_value=5,
    max_value=100,
    value=max(5, settings["checkpoint_keep"]),
    step=1,
    disabled=not auto_cleanup,
)
if st.button("저장 설정 적용"):
    save_storage_settings(
        {
            "quota_bytes": int(quota_gib * 1024**3),
            "automatic_checkpoint_cleanup": auto_cleanup,
            "checkpoint_keep": int(keep_count),
        }
    )
    st.success("저장 용량 설정을 저장했습니다.")
    st.rerun()

st.subheader("기존 실험 상세 데이터 압축 미리보기")
experiment_plan = experiment_compression_plan()
if experiment_plan["candidates"]:
    st.dataframe(
        [
            {
                "실험 파일": item["path"],
                "기존 크기 (MiB)": round(item["original_bytes"] / 1024**2, 2),
                "요약+압축 상세 (MiB)": round(
                    (item["summary_bytes"] + item["compressed_bytes"]) / 1024**2, 2
                ),
                "예상 확보 (MiB)": round(item["expected_bytes_freed"] / 1024**2, 2),
                "완전 일치 이력": item["exact_match_count"],
            }
            for item in experiment_plan["candidates"]
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.info(
        f"선택 가능한 {len(experiment_plan['candidates'])}개 실험의 예상 확보 용량: "
        f"{experiment_plan['expected_bytes_freed'] / 1024**2:.2f} MiB. "
        "압축은 회차 상세 기록을 gzip으로 보존하고 JSON에는 요약과 해시 참조를 남깁니다. "
        "추천 기록은 이 작업 대상에 포함되지 않습니다."
    )
    approve_archive = st.checkbox(
        "목록의 기존 실험 JSON을 요약 JSON으로 바꾸고, 원본 전체를 "
        "검증된 gzip 상세 파일로 보존하는 데 동의합니다.",
        key="approve_experiment_archive",
    )
    if st.button("미리보기한 실험을 압축 보존", disabled=not approve_archive):
        outcomes = [compress_experiment(item["path"]) for item in experiment_plan["candidates"]]
        st.success(
            f"{sum(bool(item.get('archived')) for item in outcomes)}개 실험 상세 기록을 "
            "압축 보존했습니다. "
            f"실제 확보 용량 "
            f"{sum(item.get('bytes_freed', 0) for item in outcomes) / 1024**2:.2f} MiB"
        )
        st.rerun()
else:
    st.info("압축 후보가 없습니다. 완료/실패 상태의 1 MB 초과 기존 실험만 후보로 표시됩니다.")
if experiment_plan["protected"]:
    with st.expander("압축 후보에서 제외된 파일"):
        st.dataframe(experiment_plan["protected"], use_container_width=True, hide_index=True)

st.subheader("체크포인트 정리 미리보기")
checkpoint_plan = checkpoint_cleanup_plan(keep=int(keep_count))
st.caption(
    f"최근 {checkpoint_plan['keep']}개는 유지합니다. "
    "실행 중·재개 가능·검증 실패·완전 일치 이력이 있는 "
    "파일은 정리 대상에서 제외합니다."
)
if checkpoint_plan["candidates"]:
    st.dataframe(checkpoint_plan["candidates"], use_container_width=True, hide_index=True)
    st.write(f"예상 확보 용량: {checkpoint_plan['expected_bytes_freed'] / 1024:.1f} KiB")
    approve_checkpoints = st.checkbox(
        "위의 검증된 종료 체크포인트를 삭제하는 데 동의합니다.",
        key="approve_checkpoint_cleanup",
    )
    if st.button("미리보기한 체크포인트 정리", disabled=not approve_checkpoints):
        outcome = execute_checkpoint_cleanup(keep=int(keep_count))
        st.success(
            f"체크포인트 {len(outcome['removed'])}개 정리, "
            f"확보 용량 {outcome['bytes_freed'] / 1024:.1f} KiB"
        )
        st.rerun()
else:
    st.info("삭제 가능한 체크포인트가 없습니다.")
if checkpoint_plan["protected"]:
    with st.expander("보존되는 체크포인트"):
        st.dataframe(checkpoint_plan["protected"], use_container_width=True, hide_index=True)

st.caption(
    f"기본 저장 한도는 {DEFAULT_QUOTA_BYTES / 1024**3:.0f} GiB이며 자동 정리는 기본 꺼짐입니다. "
    "실험/예측 JSON, 압축 상세 파일, 패턴 라이브러리는 자동 삭제하지 않습니다."
)
