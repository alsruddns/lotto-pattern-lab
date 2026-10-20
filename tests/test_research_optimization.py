from __future__ import annotations

import pytest

import lotto_pattern_lab.research_optimization as optimization
from lotto_pattern_lab.intermittent import IntermittentConfig


class _Trial:
    number = 4

    def suggest_int(self, name, low, high, **kwargs):
        return high if name == "population_size" else low

    def suggest_float(self, name, low, high):
        return (low + high) / 2


def test_quick_optimizer_trials_use_bounded_configs_and_disable_nested_controls() -> None:
    base = IntermittentConfig(population_size=16, generations=2, max_lag=10, control_runs=5)

    config = optimization._trial_config(_Trial(), base, "quick")

    assert config.population_size == 32
    assert config.generations == 1
    assert 1 <= config.max_lag <= 10
    assert config.control_runs == 0
    assert config.seed == base.seed + 4


def test_optimizer_objective_uses_validation_training_and_complexity_not_test() -> None:
    selected = {
        "training_score": 0.2,
        "validation_score": 0.1,
        "independent_test_score": 1000.0,
        "complexity": 10,
    }
    experiment = {
        "one_number_model": {"selected": selected},
        "six_number_set_model": {"selected": selected},
    }

    objective, components = optimization.validation_objective(experiment)

    assert objective == pytest.approx(0.1 - 0.01 - 0.005)
    assert components == {
        "validation_mean": 0.1,
        "training_validation_gap": pytest.approx(0.1),
        "mean_complexity": 10.0,
    }


def test_optimizer_fails_clearly_when_optional_optuna_is_missing(tmp_path, monkeypatch) -> None:
    from lotto_pattern_lab.dataset import make_random_dataset

    monkeypatch.setattr(
        optimization,
        "optuna_status",
        lambda: {
            "available": False,
            "install_extra": "install command",
        },
    )

    with pytest.raises(RuntimeError, match="install command"):
        optimization.optimize_gp(
            make_random_dataset(20, seed=31),
            IntermittentConfig(population_size=8, generations=1),
            study_name="missing-optuna",
            n_trials=1,
            storage_directory=tmp_path / "storage",
        )
