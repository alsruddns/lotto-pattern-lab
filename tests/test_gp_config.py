from dataclasses import asdict
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from lotto_pattern_lab import research_queue
from lotto_pattern_lab.gp_config import GPConfig
from lotto_pattern_lab.gp_engine import SearchConfig
from lotto_pattern_lab.intermittent import IntermittentConfig
from lotto_pattern_lab.research_duration import (
    duration_seconds_from_minutes,
    recommended_preset_minutes,
    validate_duration_minutes,
)


def test_searches_share_gp_defaults_and_validation() -> None:
    shared_fields = GPConfig.__dataclass_fields__
    standard = SearchConfig()
    intermittent = IntermittentConfig()
    defaults = GPConfig()

    assert {name: getattr(standard, name) for name in shared_fields} == asdict(defaults)
    assert {name: getattr(intermittent, name) for name in shared_fields} == asdict(defaults)
    with pytest.raises(ValueError, match="Train plus validation"):
        IntermittentConfig(train_ratio=0.8, validation_ratio=0.2)


@pytest.mark.parametrize("minutes", [5, 10, 45, 120, 480, 720])
def test_custom_duration_is_positive_whole_minutes_and_converts_exactly(minutes):
    assert validate_duration_minutes(minutes) == minutes
    assert duration_seconds_from_minutes(minutes) == minutes * 60


@pytest.mark.parametrize("minutes", [0, -1, 1441, 5.5, True])
def test_custom_duration_rejects_invalid_or_unsafe_values(minutes):
    with pytest.raises(ValueError):
        duration_seconds_from_minutes(minutes)


def test_custom_duration_uses_nearest_lower_recommended_gp_profile():
    presets = (30, 60, 120, 180, 360, 480)
    assert recommended_preset_minutes(5, presets) == 30
    assert recommended_preset_minutes(45, presets) == 30
    assert recommended_preset_minutes(120, presets) == 120
    assert recommended_preset_minutes(720, presets) == 480


def test_research_dashboard_exposes_settings_in_separate_start_tab(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(research_queue, "QUEUE_DB", tmp_path / "queue.sqlite")
    page = next(
        (Path(__file__).resolve().parents[1] / "src/lotto_pattern_lab/pages").glob("3_*.py")
    )
    app = AppTest.from_file(page).run(timeout=30)

    assert not app.exception, [exception.message for exception in app.exception]
    rendered = []

    def visit(node):
        rendered.append(node)
        children = getattr(node, "children", ())
        if isinstance(children, dict):
            children = children.values()
        for child in children:
            visit(child)

    visit(app.main)
    start_index = next(
        index
        for index, item in enumerate(rendered)
        if getattr(item, "key", None) == "start_new_research_now"
    )
    startup_index = next(
        index
        for index, item in enumerate(rendered)
        if index > start_index and type(item).__name__ in {"Info", "Success", "Error", "Warning"}
    )
    settings_index = next(
        index
        for index, item in enumerate(rendered)
        if getattr(item, "value", None) == "시작 전 적용 설정"
    )
    assert start_index < startup_index < settings_index
    assert rendered[startup_index].value
    detail_expander = next(item for item in app.expander if item.label == "세부 적용 설정 보기")
    assert detail_expander.label == "세부 적용 설정 보기"
    view_source = page.parents[1] / "continuous_research_view.py"
    view_text = view_source.read_text(encoding="utf-8")
    assert "settings_details = applied_settings_container.expander(" in view_text
    assert "expanded=False" in view_text[view_text.index("settings_details =") :]
    tab_labels = [tab.label for tab in app.tabs]
    assert "지속적 패턴 연구" in tab_labels
    assert "연구 조건 자동 최적화" in tab_labels
    assert all(
        label in tab_labels
        for label in (
            "\uc0c8 \uc5f0\uad6c \uc2dc\uc791",
            "\uae30\uc874 \uc5f0\uad6c \uc774\uc5b4\uc11c \ud558\uae30",
            "\uc5f0\uad6c \uc608\uc57d \uad00\ub9ac",
        )
    )
    keys = set(app.session_state)
    assert {"new_population_size", "new_crossover_probability", "new_elite_seed_ratio"} <= keys
    preset_selector = next(item for item in app.selectbox if item.key == "new_duration_minutes")
    assert preset_selector.value == 180
    assert len(preset_selector.options) == 6

    original_tab = app.session_state["research_workspace_tab"]
    app.selectbox(key="new_duration_mode").set_value("custom").run(timeout=30)
    assert app.number_input(key="new_custom_duration_minutes").value == 5
    assert app.session_state["research_current_duration_seconds"] == 300
    assert app.session_state["research_current_settings"]["new_duration_minutes"] == 5
    app.number_input(key="new_custom_duration_minutes").set_value(45).run(timeout=30)
    assert app.session_state["research_current_duration_seconds"] == 2700
    assert app.session_state["research_workspace_tab"] == original_tab
    app.selectbox(key="new_duration_mode").set_value("recommended").run(timeout=30)
    assert app.selectbox(key="new_duration_minutes").value == 180
    assert app.session_state["research_current_duration_seconds"] == 10800
    app.selectbox(key="new_duration_mode").set_value("custom").run(timeout=30)
    assert app.number_input(key="new_custom_duration_minutes").value == 45
    assert app.session_state["research_current_duration_seconds"] == 2700
    app.selectbox(key="new_duration_mode").set_value("recommended").run(timeout=30)
    presets = {
        30: (32, 10, 5, 3),
        60: (48, 15, 5, 5),
        120: (64, 20, 6, 5),
        180: (80, 25, 6, 5),
        360: (96, 30, 6, 10),
        480: (112, 35, 7, 10),
    }
    for minutes, expected in presets.items():
        app.selectbox(key="new_duration_minutes").set_value(minutes).run(timeout=30)
        actual = (
            app.number_input(key="new_population_size").value,
            app.number_input(key="new_generations").value,
            app.number_input(key="new_max_depth").value,
            app.number_input(key="new_control_runs").value,
        )
        assert actual == expected
    assert {"new_population_size", "new_generations", "new_max_depth", "new_control_runs"} <= {
        widget.key for widget in app.number_input
    }
    assert {"new_crossover_probability", "new_mutation_probability", "new_train_ratio"} <= {
        widget.key for widget in app.slider
    }

    general_page = next(
        (Path(__file__).resolve().parents[1] / "src/lotto_pattern_lab/pages").glob("1_*.py")
    )
    general_app = AppTest.from_file(general_page).run(timeout=180)
    assert not general_app.exception, [exception.message for exception in general_app.exception]
    general_keys = set(general_app.session_state)
    assert "general_crossover_probability" in general_keys
    assert "new_crossover_probability" not in general_keys
