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


def test_research_dashboard_exposes_settings_in_separate_start_tab() -> None:
    page = next(
        (Path(__file__).resolve().parents[1] / "src/lotto_pattern_lab/pages").glob("3_*.py")
    )
    app = AppTest.from_file(page).run(timeout=30)

    assert not app.exception, [exception.message for exception in app.exception]
    assert [tab.label for tab in app.tabs] == ["새 연구 시작", "기존 연구 이어서 하기"]
    keys = set(app.session_state)
    assert {"new_population_size", "new_crossover_probability", "new_elite_seed_ratio"} <= keys
    number_labels = {widget.label for widget in app.number_input}
    slider_labels = {widget.label for widget in app.slider}
    assert {
        "총 연구 시간(분)",
        "개체군",
        "세대 수",
        "최대 수식 깊이",
        "최대 과거 참조",
        "무작위 대조 반복",
        "숨겨진 상태 수",
    } <= number_labels
    assert {
        "교차 확률",
        "돌연변이 확률",
        "학습 비율",
        "검증 비율",
        "최소 활성화 비율",
        "기존 패턴 계승 비율",
    } <= slider_labels

    general_app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "src/lotto_pattern_lab/app.py"
    ).run(timeout=30)
    assert not general_app.exception, [exception.message for exception in general_app.exception]
    general_keys = set(general_app.session_state)
    assert "general_crossover_probability" in general_keys
    assert "new_crossover_probability" not in general_keys
