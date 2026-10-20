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


def test_future_validation_page_renders_scorecard() -> None:
    page = next(
        (Path(__file__).resolve().parents[1] / "src/lotto_pattern_lab/pages").glob("9_*.py")
    )
    app = AppTest.from_file(page).run(timeout=60)

    assert not app.exception, [exception.message for exception in app.exception]
    assert app.title[0].value == "미래 독립 검증"
    assert "공식별 미래 검증 성적표" in [item.value for item in app.subheader]


def test_continuous_research_page_separates_start_and_resume(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(research_queue, "QUEUE_DB", tmp_path / "queue.sqlite")
    page = next(
        (Path(__file__).resolve().parents[1] / "src/lotto_pattern_lab/pages").glob("3_*.py")
    )
    app = AppTest.from_file(page).run(timeout=60)

    assert not app.exception, [exception.message for exception in app.exception]
    tab_labels = [tab.label for tab in app.tabs]
    assert "지속적 패턴 연구" in tab_labels
    assert "연구 조건 자동 최적화" in tab_labels
    assert all(
        label in tab_labels
        for label in [
            "새 연구 시작",
            "기존 연구 이어서 하기",
            "연구 예약 실행",
        ]
    )
    labels = [button.label for button in app.button]
    assert "연구 환경 사전 검사" in labels
    assert "새 연구 시작" in labels
    start_button = next(button for button in app.button if button.label == "새 연구 시작")
    assert not start_button.disabled
    assert "이어서 하기" in labels
    mode = next(item for item in app.selectbox if item.key == "new_research_mode")
    assert mode.value == "cumulative"
    mode.set_value("independent").run(timeout=60)
    assert not app.exception, [exception.message for exception in app.exception]
    assert app.selectbox(key="new_research_mode").value == "independent"
    assert app.checkbox(key="new_elite_seeding_enabled").disabled
    resume_selector = next(item for item in app.selectbox if item.label == "재개할 연구")
    if resume_selector.value == "962a0a57-92c3-4e88-a04f-5cf4b836ba29":
        resume_button = next(button for button in app.button if button.label == "이어서 하기")
        assert not resume_button.disabled


def test_optimization_tab_shows_history_and_separates_new_and_resume() -> None:
    page = next(
        (Path(__file__).resolve().parents[1] / "src/lotto_pattern_lab/pages").glob("3_*.py")
    )
    app = AppTest.from_file(page).run(timeout=60)

    assert not app.exception, [exception.message for exception in app.exception]
    labels = [tab.label for tab in app.tabs]
    assert "연구 조건 자동 최적화" in labels
    assert "새 자동 최적화" in labels
    assert "기존 연구 이어하기" in labels
    assert "최적화 이력" in labels
    assert "새 자동 최적화 시작" in [button.label for button in app.button]
    assert "기존 자동 최적화 이어서 하기" in [button.label for button in app.button]

def test_optimization_history_tab_state_survives_rerun() -> None:
    page = next(
        (Path(__file__).resolve().parents[1] / "src/lotto_pattern_lab/pages").glob("3_*.py")
    )
    app = AppTest.from_file(page).run(timeout=60)

    assert not app.exception, [exception.message for exception in app.exception]
    assert len(app.tabs) >= 5
    app.session_state["research_workspace_tab"] = app.tabs[-4].label
    optimization_label = app.tabs[-4].label
    history_label = app.tabs[-1].label
    app.session_state["optimization_workspace_tab"] = history_label
    app.run(timeout=60)

    assert not app.exception, [exception.message for exception in app.exception]
    assert app.session_state["research_workspace_tab"] == optimization_label
    assert app.session_state["optimization_workspace_tab"] == history_label
    app.run(timeout=60)
    assert not app.exception, [exception.message for exception in app.exception]
    assert app.session_state["optimization_workspace_tab"] == history_label


def test_live_research_status_is_at_top_and_rerun_preserves_controls(tmp_path) -> None:
    page = next(
        (Path(__file__).resolve().parents[1] / "src/lotto_pattern_lab/pages").glob("3_*.py")
    )
    root = Path(__file__).resolve().parents[1]
    missing_checkpoint = root / "checkpoints" / "ui-test-missing-checkpoint.json"
    app = AppTest.from_file(page)
    app.session_state["monitor_research_checkpoint"] = str(missing_checkpoint)
    app.session_state["research_launch_status"] = "first_checkpoint_waiting"
    app.session_state["research_launch_started_at"] = "2026-10-20T12:00:00+00:00"
    app.session_state["research_target_duration_seconds"] = 3600
    app.run(timeout=60)

    assert not app.exception, [exception.message for exception in app.exception]
    assert any(item.value == "연구 진행 현황" for item in app.subheader)
    metrics = {item.label for item in app.metric}
    assert {"누적 실행 시간", "남은 연구 시간", "현재 시드"} <= metrics
    assert any(
        "첫 체크포인트 저장 대기 중" in item.value for item in app.markdown
    )
    assert "연구 진행 상태" not in {item.value for item in app.subheader}
    assert any(button.key == "refresh_research_status" for button in app.button)

    app.number_input(key="new_seed").set_value(54321).run(timeout=60)
    assert not app.exception, [exception.message for exception in app.exception]
    assert app.number_input(key="new_seed").value == 54321
    assert app.session_state["research_workspace_tab"] == "지속적 패턴 연구"
