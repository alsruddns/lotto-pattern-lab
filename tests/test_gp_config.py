from dataclasses import asdict
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

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


def test_research_dashboard_exposes_settings_with_isolated_widget_keys() -> None:
    page = next(
        (Path(__file__).resolve().parents[1] / "src/lotto_pattern_lab/pages").glob("3_*.py")
    )
    app = AppTest.from_file(page).run(timeout=30)

    assert not app.exception, [exception.message for exception in app.exception]
    keys = set(app.session_state)
    assert {
        "research_population_size",
        "research_generations",
        "research_crossover_probability",
        "research_mutation_probability",
        "research_max_depth",
        "research_train_ratio",
        "research_validation_ratio",
        "research_max_lag",
        "research_min_activation_rate",
        "research_hidden_states",
        "research_elite_enabled",
        "research_elite_ratio",
    } <= keys

    general_app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "src/lotto_pattern_lab/app.py"
    ).run(timeout=30)
    assert not general_app.exception, [exception.message for exception in general_app.exception]
    general_keys = set(general_app.session_state)
    assert "general_crossover_probability" in general_keys
    assert "research_crossover_probability" not in general_keys
