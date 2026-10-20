from pathlib import Path

from streamlit.testing.v1 import AppTest

from lotto_pattern_lab import research_queue


def test_streamlit_script_executes_without_import_errors() -> None:
    script = Path(__file__).resolve().parents[1] / "src/lotto_pattern_lab/app.py"
    app = AppTest.from_file(script).run(timeout=30)

    assert not app.exception, [exception.message for exception in app.exception]
    assert app.title[0].value == "AI 패턴 연구실"


def test_replay_and_draw_management_pages_render_from_app_navigation() -> None:
    page_root = Path(__file__).resolve().parents[1] / "src/lotto_pattern_lab/pages"
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "src/lotto_pattern_lab/app.py"
    ).run(timeout=30)

    for pattern in ("6_*.py", "11_*.py"):
        page = next(page_root.glob(pattern))
        app.switch_page(f"pages/{page.name}").run(timeout=60)
        assert not app.exception, [exception.message for exception in app.exception]


def test_formula_library_page_renders_similarity_and_lineage_sections() -> None:
    page = next(
        (Path(__file__).resolve().parents[1] / "src/lotto_pattern_lab/pages").glob("8_*.py")
    )
    app = AppTest.from_file(page).run(timeout=60)

    assert not app.exception, [exception.message for exception in app.exception]
    assert app.title[0].value == "수학 공식 라이브러리"
    assert "수식 유사도 및 그룹" in [item.value for item in app.subheader]
    assert "수식 계보" in [item.value for item in app.subheader]


def test_continuous_research_page_separates_start_and_resume(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(research_queue, "QUEUE_DB", tmp_path / "queue.sqlite")
    page = next(
        (Path(__file__).resolve().parents[1] / "src/lotto_pattern_lab/pages").glob("3_*.py")
    )
    app = AppTest.from_file(page).run(timeout=60)

    assert not app.exception, [exception.message for exception in app.exception]
    assert [tab.label for tab in app.tabs] == [
        "새 연구 시작",
        "기존 연구 이어서 하기",
        "연구 예약 실행",
    ]
    labels = [button.label for button in app.button]
    assert "연구 환경 사전 검사" in labels
    assert "새 연구 시작" in labels
    start_button = next(button for button in app.button if button.label == "새 연구 시작")
    assert not start_button.disabled
    assert "이어서 하기" in labels
    resume_selector = next(item for item in app.selectbox if item.label == "재개할 연구")
    if resume_selector.value == "962a0a57-92c3-4e88-a04f-5cf4b836ba29":
        resume_button = next(button for button in app.button if button.label == "이어서 하기")
        assert not resume_button.disabled
