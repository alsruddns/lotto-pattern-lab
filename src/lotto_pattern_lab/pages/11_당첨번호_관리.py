"""Manage manually entered draws with source backups and verification history."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import date

import streamlit as st

from lotto_pattern_lab.dataset import dataset_integrity_report, load_csv
from lotto_pattern_lab.draw_management import (
    confirm_manual_official_comparison,
    create_draw_correction_request,
    draw_registration_block_reason,
    register_draw,
    verify_draw_against_official,
)
from lotto_pattern_lab.official_data import fetch_official_draw
from lotto_pattern_lab.paths import project_path

st.set_page_config(page_title="당첨번호 관리", layout="wide")
st.title("당첨번호 관리")
verification_labels = {
    "official_verified": "공식 대조 완료",
    "manual_unverified": "수동 입력·미검증",
    "verification_failed": "공식 대조 불일치",
    "locally_validated_unreconciled": "형식 검사 완료·공식 미대조",
    "unverified": "미검증",
}
csv_path = project_path("data/raw/lotto_draws_753_1246_with_bonus.csv")
official_path = project_path("data/official/draws.csv")

if not csv_path.is_file():
    st.error("기준 당첨번호 파일이 없습니다. 프로젝트의 데이터 경로를 확인하세요.")
    st.stop()

try:
    dataset = load_csv(csv_path)
except (OSError, ValueError) as error:
    st.error(f"기준 데이터 무결성 검사에 실패했습니다: {error}")
    st.stop()

raw_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
latest = dataset.records[-1]
official_records = {}
if official_path.is_file():
    try:
        official_records = {item.round_number: item for item in load_csv(official_path).records}
    except (OSError, ValueError):
        official_records = {}
latest_date = latest.draw_date or getattr(
    official_records.get(latest.round_number), "draw_date", None
)
report = dataset_integrity_report(dataset)
columns = st.columns(4)
columns[0].metric("최신 회차", f"{latest.round_number}회")
columns[1].metric("최신 추첨일", latest_date.isoformat() if latest_date else "미등록")
columns[2].metric("등록 회차 수", len(dataset.records))
columns[3].metric(
    "데이터 상태",
    verification_labels.get(dataset.verification_status, "확인 필요"),
)
st.caption(f"기준 파일: `data/raw/`의 보너스 포함 당첨번호 CSV · SHA-256 `{raw_hash}`")
if report["missing_rounds"]:
    st.warning(f"회차 누락 {len(report['missing_rounds'])}건: {report['missing_rounds'][:20]}")
else:
    st.success("회차 연속성 검사 통과")

st.subheader("새 당첨번호 등록")
st.caption(
    "등록 전 원본 CSV를 백업하고 기존 컬럼 순서·인코딩을 유지합니다. "
    "새 입력은 공식 대조 전까지 수동 미검증 상태입니다."
)
registration_block_reason = draw_registration_block_reason()
if registration_block_reason:
    st.warning(registration_block_reason)
with st.form("manual_draw_entry", clear_on_submit=False):
    round_number = st.number_input(
        "회차 번호", min_value=1, value=latest.round_number + 1, step=1, key="draw_entry_round"
    )
    draw_date = st.date_input("추첨 날짜", value=date.today(), max_value=date.today())
    number_columns = st.columns(6)
    main_numbers = [
        number_columns[index].number_input(
            f"당첨번호 {index + 1}", min_value=1, max_value=45, value=index + 1, step=1
        )
        for index in range(6)
    ]
    bonus_number = st.number_input("보너스 번호", min_value=1, max_value=45, value=7, step=1)
    submitted = st.form_submit_button(
        "당첨번호 저장",
        type="primary",
        disabled=registration_block_reason is not None,
    )

if submitted:
    try:
        with st.spinner("원본 백업, 회차 검증, 임시 저장 및 재검증을 진행합니다."):
            result = register_draw(
                csv_path,
                round_number=int(round_number),
                draw_date=draw_date,
                main_numbers=[int(value) for value in main_numbers],
                bonus_number=int(bonus_number),
                official_dates_path=official_path,
            )
        if result["status"] == "duplicate":
            st.info(f"{round_number}회는 동일한 데이터가 이미 등록되어 있습니다.")
        else:
            st.cache_data.clear()
            st.success(
                f"{result['round_number']}회 등록 완료 · "
                f"데이터 상태 {result['verification_status']}"
            )
            st.write(
                {
                    "백업 위치": result["backup_path"],
                    "신규 CSV SHA-256": result["new_csv_sha256"],
                    "신규 데이터 버전": result["version_id"],
                    "전체 회차 수": result["draw_count"],
                    "고정 과거 범위 해시 유지": (
                        result["fixed_range_before_sha256"]
                        == result["fixed_range_after_sha256"]
                    ),
                    "기존 실험 영향": "기존 연구 ID와 과거 평가 기록은 수정되지 않습니다.",
                }
            )
            st.info(
                "현재 프로세스의 데이터 캐시를 비웠습니다. "
                "새 연구에서 새 회차를 다시 불러옵니다."
            )
            st.rerun()
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"등록하지 않았습니다: {error}")

st.subheader("기존 당첨번호 조회")
left, right = st.columns([1, 2])
selected_round = left.selectbox(
    "회차 상세 조회", [item.round_number for item in reversed(dataset.records)]
)
selected = next(item for item in dataset.records if item.round_number == selected_round)
shown_date = selected.draw_date or getattr(official_records.get(selected_round), "draw_date", None)
left.write(
    {
        "회차": selected.round_number,
        "추첨일": shown_date.isoformat() if shown_date else None,
        "당첨번호": list(selected.draw.numbers),
        "보너스": selected.draw.bonus,
        "검증 상태": verification_labels.get(
            selected.verification_status or dataset.verification_status, "확인 필요"
        ),
    }
)
with right.expander("기존 회차 수정 요청 (원본은 변경하지 않음)"):
    requested_numbers = [
        int(
            st.number_input(
                f"수정 요청 번호 {index + 1}",
                min_value=1,
                max_value=45,
                value=selected.draw.numbers[index],
                key=f"correction_number_{index}",
            )
        )
        for index in range(6)
    ]
    requested_bonus = int(
        st.number_input(
            "수정 요청 보너스", min_value=1, max_value=45,
            value=selected.draw.bonus or 7, key="correction_bonus",
        )
    )
    correction_reason = st.text_area("수정 요청 사유")
    if st.button("수정 요청만 기록"):
        try:
            request_path = create_draw_correction_request(
                csv_path,
                round_number=selected_round,
                requested_record={
                    "main_numbers": requested_numbers,
                    "bonus_number": requested_bonus,
                },
                reason=correction_reason,
            )
            st.success(f"수정 요청을 별도 기록으로 저장했습니다: {request_path}")
        except (OSError, ValueError) as error:
            st.error(f"수정 요청 실패: {error}")

verification_file = csv_path.with_name(f"{csv_path.name}.metadata.json")
metadata = (
    json.loads(verification_file.read_text(encoding="utf-8"))
    if verification_file.exists()
    else {}
)
row_status = metadata.get("round_verification", {}).get(str(selected_round), {})
st.subheader("공식 결과와 대조")
if row_status.get("status") == "official_verified":
    st.success(f"자동 공식 대조 완료 · {row_status.get('verified_at')}")
elif row_status.get("status") == "verification_failed":
    st.error(f"공식 결과와 불일치 · {row_status.get('field_matches')}")
else:
    if row_status.get("manual_user_confirmation", {}).get("confirmed"):
        st.info("사용자가 공식 결과를 대조했다고 확인했습니다. 자동 공식 검증과는 별도 상태입니다.")
    if st.button("선택 회차를 공식 결과와 자동 대조", disabled=selected_round < 753):
        try:
            with st.spinner("공식 결과를 조회하고 메인번호·보너스·추첨일을 대조합니다."):
                official_record = fetch_official_draw(selected_round)
                if official_record is None:
                    raise ValueError("공식 결과가 아직 공개되지 않았습니다.")
                outcome = verify_draw_against_official(
                    csv_path, selected_round, official_record=official_record
                )
            if outcome["status"] == "official_verified":
                st.success("메인번호, 보너스 번호, 추첨일이 공식 결과와 모두 일치합니다.")
            else:
                st.error(f"공식 결과와 불일치: {outcome['field_matches']}")
            st.rerun()
        except (OSError, RuntimeError, ValueError) as error:
            st.error(f"공식 대조를 완료하지 못했습니다: {error}")
    if row_status.get("status") != "verification_failed":
        manual_confirmation = st.checkbox(
            "공식 결과 페이지에서 메인번호·보너스·추첨일을 직접 대조했습니다.",
            key=f"manual_official_confirmation_{selected_round}",
        )
        confirmation_note = st.text_input(
            "사용자 확인 메모", value="공식 결과 페이지에서 번호와 추첨일을 직접 대조"
        )
        if st.button("사용자 대조 확인을 별도 기록", disabled=not manual_confirmation):
            try:
                confirm_manual_official_comparison(
                    csv_path,
                    selected_round,
                    confirmation_note=confirmation_note,
                )
                st.success("사용자 확인을 기록했습니다. 자동 검증 상태는 변경하지 않았습니다.")
                st.rerun()
            except (OSError, ValueError) as error:
                st.error(f"사용자 확인을 기록하지 못했습니다: {error}")

st.subheader("전체 목록 및 다운로드")
rows = []
for item in dataset.records:
    item_date = item.draw_date or getattr(
        official_records.get(item.round_number), "draw_date", None
    )
    status = item.verification_status or dataset.verification_status
    rows.append(
        {
            "회차": item.round_number,
            "추첨일": item_date.isoformat() if item_date else None,
            "당첨번호": item.draw.numbers,
            "보너스": item.draw.bonus,
            "검증 상태": verification_labels.get(status, "확인 필요"),
        }
    )
st.dataframe(rows, hide_index=True, width="stretch")
csv_buffer = io.StringIO(newline="")
writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0]))
writer.writeheader()
writer.writerows(rows)
st.download_button(
    "현재 회차 목록 CSV 다운로드",
    csv_buffer.getvalue().encode("utf-8-sig"),
    file_name="당첨번호_관리_목록.csv",
    mime="text/csv",
)
st.download_button(
    "현재 데이터 버전 JSON 다운로드",
    (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    file_name="당첨번호_데이터_버전.json",
    mime="application/json",
)
