from __future__ import annotations

import operator
import random
from copy import deepcopy
from datetime import date, timedelta
from functools import partial

import pytest
from deap import gp

from lotto_pattern_lab.dataset import DrawDataset, DrawRecord, make_random_dataset
from lotto_pattern_lab.draw import Draw
from lotto_pattern_lab.feature_schema import feature_mapping, feature_schema_version
from lotto_pattern_lab.gp_engine import _make_primitive_set, validate_primitive_set
from lotto_pattern_lab.intermittent import (
    IntermittentConfig,
    _ensure_individual_types,
    _mate_individuals,
    _mutate_individual,
    _new_individual,
    restore_elite_individual,
)
from lotto_pattern_lab.pattern_library import canonical_formula
from lotto_pattern_lab.preflight import run_research_preflight
from lotto_pattern_lab.pysr_backend import PySRConfig


def _typed_dataset(length: int = 90) -> DrawDataset:
    randomizer = random.Random(90210)
    records = []
    for offset in range(length):
        numbers = randomizer.sample(range(1, 46), 6)
        bonus = randomizer.choice([number for number in range(1, 46) if number not in numbers])
        records.append(
            DrawRecord(
                753 + offset,
                Draw.from_numbers(numbers, bonus),
                date(2017, 5, 6) + timedelta(days=7 * offset),
            )
        )
    return DrawDataset(tuple(records), "synthetic", "Synthetic typed-GP regression fixture.")


def test_missing_float_terminal_reproduces_legacy_elite_mutation_failure() -> None:
    legacy_pset = gp.PrimitiveSet("LEGACY", 1)
    legacy_pset.renameArguments(ARG0="x0")
    legacy_pset.addPrimitive(operator.add, 2, name="add")
    elite_literal = gp.Terminal(-5.896400085320264, False, float)
    individual = gp.PrimitiveTree([elite_literal])

    with pytest.raises(IndexError, match="terminal of type .*float.*none available"):
        gp.genFull(pset=legacy_pset, min_=0, max_=0, type_=float)

    with pytest.raises(
        IndexError, match="gp.generate function tried to add .*float.*none available"
    ):
        gp.mutUniform(
            individual,
            expr=partial(gp.genFull, min_=0, max_=2),
            pset=legacy_pset,
        )


def test_typed_primitive_set_has_float_terminals_and_preflights_missing_types() -> None:
    pset = _make_primitive_set(4)
    assert isinstance(pset, gp.PrimitiveSetTyped)
    assert pset.ret is float
    assert len(pset.terminals[float]) >= 5
    assert pset.primitives[float]
    validate_primitive_set(pset)

    malformed = gp.PrimitiveSetTyped("BROKEN", [int], float)
    malformed.addPrimitive(lambda value: float(value), [int], float, name="to_float")
    with pytest.raises(ValueError, match="no terminal for required type.*float"):
        validate_primitive_set(malformed)


def test_genfull_gengrow_mutation_crossover_and_operator_semantics() -> None:
    pset = _make_primitive_set(4)
    _ensure_individual_types()
    random.seed(55)
    for max_depth in range(0, 7):
        for generator in (gp.genFull, gp.genGrow):
            for _ in range(20):
                tree = gp.PrimitiveTree(generator(pset=pset, min_=0, max_=max_depth))
                assert tree.height <= max_depth
                assert all(node.ret is float for node in tree)

    individual = _new_individual(pset, outputs=1, max_depth=5)
    mutated, = _mutate_individual(individual, pset, max_depth=5, max_nodes=35)
    assert all(node.ret is float for tree in mutated for node in tree)
    other = _new_individual(pset, outputs=1, max_depth=5)
    first, second = _mate_individuals(mutated, other, max_depth=5, max_nodes=35)
    assert all(node.ret is float for tree in first + second for node in tree)

    legacy_pset = gp.PrimitiveSet("OLD", 1)
    legacy_pset.renameArguments(ARG0="x0")
    legacy_pset.addPrimitive(operator.add, 2, name="add")
    old_tree = gp.PrimitiveTree.from_string("add(x0, 2.0)", legacy_pset)
    new_tree = gp.PrimitiveTree.from_string("add(x0, 2.0)", pset)
    inputs = (3.5, 1.0, 2.0, 3.0)
    assert gp.compile(old_tree, legacy_pset)(inputs[0]) == gp.compile(new_tree, pset)(*inputs)


def test_numeric_elite_formula_restores_and_executes_in_float_grammar() -> None:
    config = IntermittentConfig(max_lag=1, feature_profile="legacy", workers=1)
    pset = _make_primitive_set(7)
    mapping = feature_mapping("legacy", 1)
    pattern = {
        "pattern_id": "historic-float-elite",
        "outputs_per_event": 1,
        "rule_index": 1,
        "feature_profile": "legacy",
        "feature_mapping": mapping,
        "feature_schema_version": feature_schema_version("legacy"),
        "primitive_set_version": "deap-safe-math-bits-digits-v1",
        "activation_condition": "-5.896400085320264",
        "formulas": ["add(x0, 1.5)"],
    }

    individual, reason = restore_elite_individual(pattern, pset, 1, config)
    assert reason is None
    assert individual is not None
    assert all(node.ret is float for tree in individual for node in tree)
    inputs = (2.0,) * 7
    assert gp.compile(individual[0], pset)(*inputs) == -5.896400085320264
    assert gp.compile(individual[1], pset)(*inputs) == 3.5
    assert canonical_formula("add(x0, 2)") == canonical_formula("add(x0, 2.0)")
    for seed in range(16):
        random.seed(seed)
        mutated, = _mutate_individual(deepcopy(individual), pset, max_depth=6, max_nodes=35)
        assert all(node.ret is float for tree in mutated for node in tree)


@pytest.mark.parametrize("profile", ["legacy", "calendar_round", "bonus_lag_v1"])
def test_gp_preflight_runs_both_models_for_supported_profiles(profile: str) -> None:
    dataset = _typed_dataset()
    config = IntermittentConfig(
        population_size=8,
        generations=1,
        max_depth=4,
        workers=1,
        max_lag=3,
        control_runs=0,
        feature_profile=profile,
    )

    report = run_research_preflight(dataset, config, ("intermittent_gp",))

    engine_report = report["engines"]["intermittent_gp"]
    assert report["status"] == "passed", engine_report
    assert engine_report["generated_individuals"] == 200
    assert engine_report["mutations"] == 200
    assert engine_report["crossovers"] == 100
    assert engine_report["experiment_json_round_trip"] is True
    assert engine_report["checkpoint_round_trip"] is True
    assert engine_report["one_number_smoke_candidates"] > 0
    assert engine_report["six_number_smoke_candidates"] > 0


def test_failed_pysr_preflight_is_attributed_to_pysr_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lotto_pattern_lab import preflight

    monkeypatch.setattr(
        preflight,
        "_run_pysr_check",
        lambda _dataset, _config: (_ for _ in ()).throw(RuntimeError("Julia smoke failed")),
    )
    report = run_research_preflight(
        make_random_dataset(60, seed=12, include_bonus=True),
        IntermittentConfig(
            population_size=8,
            generations=1,
            max_depth=3,
            workers=1,
            max_lag=3,
            control_runs=0,
        ),
        ("pysr_symbolic_regression",),
        pysr_config=PySRConfig(),
    )
    failure = report["engines"]["pysr_symbolic_regression"]
    assert report["status"] == "failed"
    assert failure["engine"] == "pysr_symbolic_regression"
    assert failure["error"] == "Julia smoke failed"
    assert "run_research_preflight" in failure["traceback"]


def test_gp_plus_pysr_preflight_keeps_engine_results_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lotto_pattern_lab import preflight

    monkeypatch.setattr(
        preflight,
        "_run_pysr_check",
        lambda _dataset, _config: {
            "status": "passed",
            "engine": "pysr_symbolic_regression",
            "julia_search": True,
        },
    )
    report = run_research_preflight(
        _typed_dataset(),
        IntermittentConfig(
            population_size=8,
            generations=1,
            max_depth=3,
            workers=1,
            max_lag=3,
            control_runs=0,
        ),
        ("intermittent_gp", "pysr_symbolic_regression"),
        pysr_config=PySRConfig(),
    )

    assert report["status"] == "passed"
    assert report["engines"]["intermittent_gp"]["status"] == "passed"
    assert report["engines"]["pysr_symbolic_regression"]["status"] == "passed"
    assert report["engines"]["intermittent_gp"]["engine"] == "intermittent_gp"
    assert (
        report["engines"]["pysr_symbolic_regression"]["engine"]
        == "pysr_symbolic_regression"
    )
