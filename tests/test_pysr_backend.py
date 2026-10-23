from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from lotto_pattern_lab.dataset import make_random_dataset
from lotto_pattern_lab.evaluation_cache import (
    EvaluationCache,
    canonical_pysr_formula,
)
from lotto_pattern_lab.pysr_backend import (
    PySRConfig,
    _prepare_julia_environment,
    run_pysr_experiment,
)
from lotto_pattern_lab.storage import ExperimentStore


def test_pysr_config_validates_library_native_options() -> None:
    assert PySRConfig().feature_profile == "integrated_v1"
    with pytest.raises(ValueError, match="Unsupported PySR binary operator"):
        PySRConfig(binary_operators=("bit_xor",))
    with pytest.raises(ValueError, match="less than one"):
        PySRConfig(train_ratio=0.8, validation_ratio=0.2)


def test_optional_backend_selects_bundled_julia_project_without_overriding_user_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "python-env" / "julia_env"
    project.mkdir(parents=True)
    (project / "Project.toml").write_text("[deps]\n", encoding="utf-8")
    app_depot = tmp_path / "app-data" / "LottoPatternLab" / "julia-depot"
    app_depot.mkdir(parents=True)
    monkeypatch.setattr("lotto_pattern_lab.pysr_backend.sys.prefix", str(project.parent))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "app-data"))
    monkeypatch.delenv("JULIA_PROJECT", raising=False)
    monkeypatch.delenv("JULIA_DEPOT_PATH", raising=False)

    _prepare_julia_environment()

    assert Path(os.environ["JULIA_PROJECT"]) == project
    assert Path(os.environ["JULIA_DEPOT_PATH"]) == app_depot


def test_pysr_formula_fingerprint_is_conservative() -> None:
    assert canonical_pysr_formula("x0 + x1") == canonical_pysr_formula("x1+x0")
    assert canonical_pysr_formula("x0 - x1") != canonical_pysr_formula("x1 - x0")
    assert canonical_pysr_formula("square(x0)") == canonical_pysr_formula("square( x0 )")
    with pytest.raises(ValueError):
        canonical_pysr_formula("__import__('os').system('bad')")


def test_evaluation_cache_is_scoped_to_dataset_and_full_conditions(tmp_path: Path) -> None:
    cache = EvaluationCache(tmp_path / "cache.sqlite")
    conditions = {"dataset_sha256": "a", "target": "main_position_1", "split": "1:10"}
    assert cache.put("x0 + x1", conditions, {"mae": 1.0})
    assert not cache.put("x1 + x0", conditions, {"mae": 2.0})
    assert cache.get("x1 + x0", conditions) == {"mae": 1.0}
    assert cache.get("x0 + x1", {**conditions, "dataset_sha256": "b"}) is None


def test_symbolic_candidates_merge_identity_but_keep_research_sources(tmp_path: Path) -> None:
    cache = EvaluationCache(tmp_path / "patterns.sqlite")
    candidate = {
        "formula": "x0 + x1",
        "target": "main_position_1",
        "classification": "PROMISING",
        "evaluation": {"validation": {"mae": 1.0}},
    }
    shared = {
        "dataset_sha256": "data-v1",
        "engine_version": "pysr-engine-v1",
        "feature_profile": "legacy",
        "feature_schema_version": 1,
    }
    first = cache.register_symbolic_candidate(
        candidate,
        experiment_id="exp-1",
        research_id="research-1",
        **shared,
    )
    second = cache.register_symbolic_candidate(
        {**candidate, "formula": "x1+x0"},
        experiment_id="exp-2",
        research_id="research-2",
        **shared,
    )
    assert first == second
    with cache._connect() as connection:
        assert connection.execute("SELECT count(*) FROM symbolic_patterns").fetchone()[0] == 1
        source_count = connection.execute(
            "SELECT count(*) FROM symbolic_pattern_sources"
        ).fetchone()[0]
        assert source_count == 2


def test_pysr_runner_persists_six_independent_position_fits_without_leakage(
    tmp_path: Path, monkeypatch
) -> None:
    class Equations:
        def to_dict(self, orient="records"):
            assert orient == "records"
            return [{"equation": "x0 + 1.0", "complexity": 3, "loss": 1.0, "score": 0.2}]

    class FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.mean = 0.0
            self.equations_ = Equations()

        def fit(self, x, y, **kwargs):
            self.mean = float(np.mean(y))

        def predict(self, x, index=None):
            return np.full(len(x), self.mean)

    monkeypatch.setattr(
        "lotto_pattern_lab.pysr_backend.initialize_optional_backend",
        lambda: (FakeModel, {"version": "test", "installed": True}),
    )
    dataset = make_random_dataset(50, seed=71)
    store = ExperimentStore(tmp_path / "experiments", minimum_compress_bytes=10**9)
    result = run_pysr_experiment(
        dataset,
        PySRConfig(
            target="main_positions",
            feature_profile="legacy",
            niterations=1,
            population_size=8,
            populations=1,
        ),
        store,
        cache_path=str(tmp_path / "cache.sqlite"),
    )
    assert len(result["metrics"]) == 6
    assert result["six_number_set_metrics"]["validation"]["target_round_count"] > 0
    assert result["dataset"]["target_data_leakage_policy"].endswith("target draw excluded")
    loaded = store.load(result["experiment_id"])
    assert loaded["engine"] == result["engine"]
    assert loaded["dataset"]["dataset_sha256"] == result["dataset"]["dataset_sha256"]
    assert loaded["config"]["binary_operators"] == list(
        result["config"]["binary_operators"]
    )
