from pathlib import Path

from streamlit.testing.v1 import AppTest

from lotto_pattern_lab import research_queue


def test_streamlit_script_executes_without_import_errors() -> None:
    script = Path(__file__).resolve().parents[1] / "src/lotto_pattern_lab/app.py"
    app = AppTest.from_file(script).run(timeout=30)

    assert not app.exception, [exception.message for exception in app.exception]
    assert app.title[0].value == "Lotto Pattern Lab 연구실"


def test_continuous_research_page_separates_start_and_resume(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(research_queue, "QUEUE_DB", tmp_path / "queue.sqlite")
    page = next(
        (Path(__file__).resolve().parents[1] / "src/lotto_pattern_lab/pages").glob("3_*.py")
    )
    app = AppTest.from_file(page).run(timeout=60)

    assert not app.exception, [exception.message for exception in app.exception]
    assert [tab.label for tab in app.tabs] == [
        "\uc0c8 \uc5f0\uad6c \uc2dc\uc791",
        "\uae30\uc874 \uc5f0\uad6c \uc774\uc5b4\uc11c \ud558\uae30",
        "\uc5f0\uad6c \uc608\uc57d \uc2e4\ud589",
    ]
    labels = [button.label for button in app.button]
    assert "연구 환경 사전 검사" in labels
    assert "새 연구 시작" in labels
    start_button = next(button for button in app.button if button.label == "새 연구 시작")
    assert start_button.disabled
    assert "이어서 하기" in labels
    resume_selector = next(item for item in app.selectbox if item.label == "재개할 연구")
    if resume_selector.value == "962a0a57-92c3-4e88-a04f-5cf4b836ba29":
        resume_button = next(button for button in app.button if button.label == "이어서 하기")
        assert not resume_button.disabled
