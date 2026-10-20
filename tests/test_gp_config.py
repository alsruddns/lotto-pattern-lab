from dataclasses import asdict
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from lotto_pattern_lab import research_queue
from lotto_pattern_lab.gp_config import GPConfig
from lotto_pattern_lab.gp_engine import SearchConfig
from lotto_pattern_lab.intermittent import IntermittentConfig


def test_searches_share_gp_defaults_and_validation() -> None:
    shared_fields = GPConfig.__dataclass_fields__
    standard = SearchConfig()
    intermittent = IntermittentConfig()
    defaults = GPConfig()

    assert {name: getattr(standard, name) for name in shared_fields} == asdict(defaults)
    assert {name: getattr(intermittent, name) for name in shared_fields} == asdict(defaults)
    with pytest.raises(ValueError, match="Train plus validation"):
        IntermittentConfig(train_ratio=0.8, validation_ratio=0.2)


def test_research_dashboard_exposes_settings_in_separate_start_tab(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(research_queue, "QUEUE_DB", tmp_path / "queue.sqlite")
    page = next(
        (Path(__file__).resolve().parents[1] / "src/lotto_pattern_lab/pages").glob("3_*.py")
    )
    app = AppTest.from_file(page).run(timeout=30)

    assert not app.exception, [exception.message for exception in app.exception]
    assert [tab.label for tab in app.tabs] == [
        "\uc0c8 \uc5f0\uad6c \uc2dc\uc791",
        "\uae30\uc874 \uc5f0\uad6c \uc774\uc5b4\uc11c \ud558\uae30",
        "\uc5f0\uad6c \uc608\uc57d \uc2e4\ud589",
    ]
    keys = set(app.session_state)
    assert {"new_population_size", "new_crossover_probability", "new_elite_seed_ratio"} <= keys
    preset_selector = next(item for item in app.selectbox if item.key == "new_duration_minutes")
    assert preset_selector.value == 180
    assert len(preset_selector.options) == 6
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
    general_app = AppTest.from_file(general_page).run(timeout=30)
    assert not general_app.exception, [exception.message for exception in general_app.exception]
    general_keys = set(general_app.session_state)
    assert "general_crossover_probability" in general_keys
    assert "new_crossover_probability" not in general_keys
