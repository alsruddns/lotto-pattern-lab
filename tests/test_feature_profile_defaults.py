from __future__ import annotations

from lotto_pattern_lab.feature_schema import (
    DEFAULT_FEATURE_PROFILE,
    feature_mapping,
)
from lotto_pattern_lab.features import feature_names
from lotto_pattern_lab.intermittent import IntermittentConfig
from lotto_pattern_lab.pysr_backend import PySRConfig


def test_new_gp_and_pysr_configurations_default_to_integrated_profile() -> None:
    assert DEFAULT_FEATURE_PROFILE == "integrated_v1"
    assert IntermittentConfig().feature_profile == DEFAULT_FEATURE_PROFILE
    assert PySRConfig().feature_profile == DEFAULT_FEATURE_PROFILE


def test_integrated_profile_has_100_unique_features_at_maximum_lag_10() -> None:
    mapping = feature_mapping("integrated_v1", 10)

    assert len(mapping) == 100
    assert len({item["feature_id"] for item in mapping.values()}) == 100
    legacy = feature_mapping("legacy", 10)
    assert [item["name"] for item in legacy.values()] == list(feature_names(10))


def test_explicit_historical_profile_remains_available_for_restored_settings() -> None:
    assert IntermittentConfig(feature_profile="legacy").feature_profile == "legacy"
    assert IntermittentConfig(feature_profile="calendar_round").feature_profile == "calendar_round"
    assert IntermittentConfig(feature_profile="bonus_lag_v1").feature_profile == "bonus_lag_v1"
