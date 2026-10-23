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
    assert app.session_state["new_profile"] == "integrated_v1"
    app.session_state["new_profile"] = "calendar_round"
    app.run(timeout=60)
    assert not app.exception, [exception.message for exception in app.exception]
    assert app.session_state["new_profile"] == "calendar_round"
    tab_labels = [tab.label for tab in app.tabs]
    assert "지속적 패턴 연구" in tab_labels
    assert "연구 조건 자동 최적화" in tab_labels
    assert all(
        label in tab_labels
        for label in [
            "새 연구 시작",
            "기존 연구 이어서 하기",
            "\uc5f0\uad6c \uc608\uc57d \uad00\ub9ac",
        ]
    )
    assert any(item.label == "\uc2e4\ud589 \ubc29\ubc95" for item in app.selectbox)
    subheaders = {item.value for item in app.subheader}
    assert "\ud328\ud134 \ub77c\uc774\ube0c\ub7ec\ub9ac" not in subheaders
    labels = [button.label for button in app.button]
    assert any(button.key == "run_new_research_preflight" for button in app.button)
    assert "\ud658\uacbd \uc804\uccb4 \uc7ac\uac80\uc0ac" in labels
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
    app.session_state["research_workspace_tab"] = app.tabs[-4].label
    app.run(timeout=60)

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


def test_auto_optimization_tab_renders_without_running_hidden_research_view(monkeypatch) -> None:
    import runpy

    import lotto_pattern_lab.research as research
    import lotto_pattern_lab.research_optimization as optimization

    monkeypatch.setattr(optimization, "list_optimization_study_overviews", lambda *_: [])
    monkeypatch.setattr(optimization, "list_optimization_run_states", lambda *_: [])
    monkeypatch.setattr(optimization, "optuna_status", lambda: {"available": True})
    monkeypatch.setattr(research, "research_worker_active", lambda: True)
    original_run_path = runpy.run_path

    def guarded_run_path(path, *args, **kwargs):
        if str(path).endswith("continuous_research_view.py"):
            raise AssertionError("hidden continuous-research tab was executed")
        return original_run_path(path, *args, **kwargs)

    monkeypatch.setattr(runpy, "run_path", guarded_run_path)
    page = next(
        (Path(__file__).resolve().parents[1] / "src/lotto_pattern_lab/pages").glob("3_*.py")
    )
    app = AppTest.from_file(page)
    app.session_state["research_workspace_tab"] = "연구 조건 자동 최적화"
    app.run(timeout=30)

    assert not app.exception, [exception.message for exception in app.exception]
    assert "연구 조건 자동 최적화" in [tab.label for tab in app.tabs]
    assert "최적화 이력" in [tab.label for tab in app.tabs]
    app.session_state["optimization_workspace_tab"] = "최적화 이력"
    app.run(timeout=30)
    assert not app.exception, [exception.message for exception in app.exception]
    assert app.session_state["optimization_workspace_tab"] == "최적화 이력"


def test_optimization_trial_details_are_read_only_until_requested(monkeypatch, tmp_path) -> None:
    import runpy

    import lotto_pattern_lab.research as research
    import lotto_pattern_lab.research_optimization as optimization

    record = {
        "study_name": "temporary-study",
        "status": "completed",
        "database_available": True,
        "database_path": str(tmp_path / "temporary-study.sqlite"),
        "state_path": str(tmp_path / "temporary-study.state.json"),
        "completed_trials": 1,
        "failed_trials": 0,
        "pruned_trials": 0,
        "best_value": 0.5,
    }
    reads = []
    monkeypatch.setattr(optimization, "list_optimization_study_overviews", lambda *_: [record])
    monkeypatch.setattr(optimization, "list_optimization_run_states", lambda *_: [])
    monkeypatch.setattr(optimization, "optuna_status", lambda: {"available": True})
    monkeypatch.setattr(
        optimization,
        "load_optimization_study_detail",
        lambda value, **kwargs: reads.append((value["study_name"], kwargs))
        or {
            **value,
            "detail_available": True,
            "trials": [],
            "total_trials": 0,
            "best_parameters": {},
        },
    )
    monkeypatch.setattr(research, "research_worker_active", lambda: True)
    original_run_path = runpy.run_path
    monkeypatch.setattr(
        runpy,
        "run_path",
        lambda path, *args, **kwargs: (
            (_ for _ in ()).throw(AssertionError("hidden research view executed"))
            if str(path).endswith("continuous_research_view.py")
            else original_run_path(path, *args, **kwargs)
        ),
    )
    page = next(
        (Path(__file__).resolve().parents[1] / "src/lotto_pattern_lab/pages").glob("3_*.py")
    )
    app = AppTest.from_file(page)
    app.session_state["research_workspace_tab"] = "연구 조건 자동 최적화"
    app.session_state["optimization_workspace_tab"] = "최적화 이력"
    app.session_state["optimization_history_refresh_token"] = "test-detail-load"
    app.run(timeout=30)

    assert not app.exception, [exception.message for exception in app.exception]
    assert reads == []
    app.button(key="optimization_history_load_detail").click().run(timeout=30)
    assert not app.exception, [exception.message for exception in app.exception]
    assert reads == [("temporary-study", {"offset": 0, "limit": 100})]


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
    assert any("첫 체크포인트 저장 대기 중" in item.value for item in app.markdown)
    assert "연구 진행 상태" not in {item.value for item in app.subheader}
    assert any(button.key == "refresh_research_status" for button in app.button)

    app.number_input(key="new_seed").set_value(54321).run(timeout=60)
    assert not app.exception, [exception.message for exception in app.exception]
    assert app.number_input(key="new_seed").value == 54321
    assert app.session_state["research_workspace_tab"] == "지속적 패턴 연구"
