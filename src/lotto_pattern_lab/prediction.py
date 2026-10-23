"""Safely execute saved intermittent GP rules and store immutable recommendations."""

import ast
import copy
import hashlib
import json
import math
import operator
import os
import secrets
import tempfile
import uuid
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from .dataset import DrawDataset
from .feature_schema import feature_schema_version
from .features import build_bonus_features_for_target, build_features_for_target, feature_names
from .gp_engine import (
    _bit_and,
    _bit_or,
    _bit_xor,
    _concat_digits,
    _cube,
    _digit_sum,
    _digital_root,
    _factor_count,
    _fibonacci_index,
    _is_prime,
    _predict,
    _reverse_digits,
    _rotate_left,
    _rotate_right,
    _safe_div,
    _safe_mod,
    _shift_left,
    _shift_right,
    _square,
)
from .paths import project_path
from .pattern_library import pattern_fingerprint
from .prize_evaluation import classify_prize_tier
from .research_features import (
    ResearchFeatureConfig,
    build_integrated_features_for_target,
    build_research_features_for_target,
    integrated_feature_names,
    research_feature_names,
)

MODEL_VERSION = "lotto-pattern-lab/0.1.0"
RULE_COUNT = 2
_OPERATIONS: dict[str, tuple[Callable[..., float], int]] = {
    "add": (operator.add, 2),
    "sub": (operator.sub, 2),
    "mul": (operator.mul, 2),
    "safe_div": (_safe_div, 2),
    "safe_mod": (_safe_mod, 2),
    "abs": (abs, 1),
    "min2": (min, 2),
    "max2": (max, 2),
    "square": (_square, 1),
    "cube": (_cube, 1),
    "bit_xor": (_bit_xor, 2),
    "bit_and": (_bit_and, 2),
    "bit_or": (_bit_or, 2),
    "shift_left": (_shift_left, 2),
    "shift_right": (_shift_right, 2),
    "rotate_left": (_rotate_left, 2),
    "rotate_right": (_rotate_right, 2),
    "concat_digits": (_concat_digits, 2),
    "digit_sum": (_digit_sum, 1),
    "reverse_digits": (_reverse_digits, 1),
    "digital_root": (_digital_root, 1),
    "is_prime": (_is_prime, 1),
    "factor_count": (_factor_count, 1),
    "fibonacci_index": (_fibonacci_index, 1),
}


class FormulaError(ValueError):
    """Raised when a saved formula is outside the GP grammar."""


def _display_number(value: float) -> float | str:
    return float(value) if math.isfinite(value) else str(value)


def _evaluate_node(
    node: ast.AST,
    values: tuple[float, ...],
    names: tuple[str, ...],
    trace: list[dict[str, Any]],
) -> float:
    if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
        value = float(node.value)
        if not math.isfinite(value):
            raise FormulaError("Formula constants must be finite.")
        trace.append({"expression": ast.unparse(node), "value": value})
        return value
    if isinstance(node, ast.Name):
        if not node.id.startswith("x") or not node.id[1:].isdigit():
            raise FormulaError(f"Unsupported formula variable: {node.id}")
        index = int(node.id[1:])
        if index >= len(values) or index >= len(names):
            raise FormulaError(f"Formula variable {node.id} is outside the feature mapping.")
        value = values[index]
        trace.append(
            {
                "expression": node.id,
                "feature": names[index],
                "value": value,
            }
        )
        return value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        child = _evaluate_node(node.operand, values, names, trace)
        result = child if isinstance(node.op, ast.UAdd) else -child
        trace.append(
            {
                "expression": ast.unparse(node),
                "operation": "unary_sign",
                "value": _display_number(result),
            }
        )
        return result
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise FormulaError(
            "Only named GP operations, feature variables, and numeric constants are allowed."
        )
    operation = _OPERATIONS.get(node.func.id)
    if operation is None:
        raise FormulaError(f"Unsupported GP operation: {node.func.id}")
    function, arity = operation
    if len(node.args) != arity or node.keywords:
        raise FormulaError(f"Operation {node.func.id} expects {arity} positional arguments.")
    arguments = [_evaluate_node(argument, values, names, trace) for argument in node.args]
    try:
        result = float(function(*arguments))
    except (ArithmeticError, OverflowError, ValueError) as error:
        raise FormulaError(f"Operation {node.func.id} failed: {error}") from error
    trace.append(
        {
            "expression": ast.unparse(node),
            "operation": node.func.id,
            "arguments": [_display_number(value) for value in arguments],
            "value": _display_number(result),
        }
    )
    return result


def evaluate_formula(
    formula: str, values: tuple[float, ...], names: tuple[str, ...]
) -> dict[str, Any]:
    """Evaluate a DEAP string without eval/exec and return a node-by-node trace."""
    try:
        expression = ast.parse(formula, mode="eval").body
    except SyntaxError as error:
        raise FormulaError(f"Invalid formula syntax: {error.msg}") from error
    trace: list[dict[str, Any]] = []
    value = _evaluate_node(expression, values, names, trace)
    return {
        "formula": formula,
        "value": _display_number(value),
        "variable_values": {
            item["expression"]: {"feature": item["feature"], "value": item["value"]}
            for item in trace
            if "feature" in item
        },
        "steps": trace,
        "number": _predict(lambda *_: value, values),
    }


def evaluate_pysr_formula(formula: str, values: tuple[float, ...], names: tuple[str, ...]) -> float:
    """Evaluate the small, versioned PySR arithmetic grammar without eval/exec."""
    try:
        root = ast.parse(formula, mode="eval").body
    except SyntaxError as error:
        raise FormulaError(f"Invalid PySR formula syntax: {error.msg}") from error

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and type(node.value) in {float, int}:
            result = float(node.value)
        elif isinstance(node, ast.Name) and node.id.startswith("x") and node.id[1:].isdigit():
            index = int(node.id[1:])
            if index >= len(values) or index >= len(names):
                raise FormulaError(f"PySR variable {node.id} is outside the feature mapping.")
            result = float(values[index])
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            child = visit(node.operand)
            result = child if isinstance(node.op, ast.UAdd) else -child
        elif isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                result = left + right
            elif isinstance(node.op, ast.Sub):
                result = left - right
            elif isinstance(node.op, ast.Mult):
                result = left * right
            else:
                result = left / right if right else 0.0
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"square", "abs"}
            and len(node.args) == 1
            and not node.keywords
        ):
            child = visit(node.args[0])
            result = child * child if node.func.id == "square" else abs(child)
        else:
            raise FormulaError("PySR formula uses an unsupported saved operation.")
        if not math.isfinite(result):
            raise FormulaError("PySR formula produced a non-finite value.")
        return result

    return visit(root)


def pysr_replay_readiness(experiment: dict[str, Any]) -> dict[str, str]:
    """Explain whether a saved PySR result has enough metadata for honest replay."""
    if experiment.get("engine") != "pysr_symbolic_regression":
        return {
            "status": "verification_required",
            "reason": "선택한 실험은 PySR 결과가 아닙니다.",
        }
    if experiment.get("engine_version") != "pysr-engine-v1":
        return {
            "status": "verification_required",
            "reason": "PySR 실행 규약 버전이 없거나 지원되지 않습니다.",
        }
    config_payload = experiment.get("config")
    if not isinstance(config_payload, dict) or not experiment.get("dataset_sha256"):
        return {
            "status": "verification_required",
            "reason": "설정 스냅샷 또는 데이터셋 지문이 없습니다.",
        }
    from .feature_schema import feature_mapping
    from .pysr_backend import PySRConfig

    try:
        config = PySRConfig(
            **{
                key: value
                for key, value in config_payload.items()
                if key in PySRConfig.__dataclass_fields__
            }
        )
        if experiment.get("feature_mapping") != feature_mapping(
            config.feature_profile, config.max_lag
        ):
            raise ValueError("저장된 특징 스키마가 현재 스키마와 다릅니다.")
    except (TypeError, ValueError) as error:
        return {"status": "verification_required", "reason": str(error)}
    target_names = (
        [f"main_position_{index}" for index in range(1, 7)]
        if config.target == "main_positions"
        else [config.target]
    )
    candidates = experiment.get("candidates", [])
    if not candidates or any(
        not any(
            row.get("target") == target and row.get("formula") and row.get("fingerprint")
            for row in candidates
        )
        for target in target_names
    ):
        return {
            "status": "verification_required",
            "reason": "모든 예측 대상의 저장 수식과 fingerprint가 준비되지 않았습니다.",
        }
    return {
        "status": "verified",
        "reason": "저장 수식, 엔진 규약, 특징 스키마를 재실행할 수 있습니다.",
    }


def create_pysr_recommendation(
    experiment: dict[str, Any],
    dataset: DrawDataset,
    target_round: int,
    *,
    candidate_rank: int = 1,
    data_file_sha256: str | None = None,
    random_seed: int | None = None,
    official_latest_round: int | None = None,
    target_date: date | None = None,
) -> dict[str, Any]:
    """Execute a stored PySR candidate using its recorded target and transform contract."""
    if (
        experiment.get("engine") != "pysr_symbolic_regression"
        or experiment.get("status") != "completed"
    ):
        raise ValueError("PySR recommendation requires a completed PySR experiment.")
    from .feature_schema import feature_mapping
    from .pysr_backend import PySRConfig, _examples, _legal_lotto_number

    config = PySRConfig(
        **{
            key: value
            for key, value in experiment.get("config", {}).items()
            if key in PySRConfig.__dataclass_fields__
        }
    )
    examples = _examples(dataset, config)
    previous = [item for item in examples if item.round_number < target_round]
    if not previous or max(item.round_number for item in previous) != target_round - 1:
        raise ValueError("PySR inputs must contain every draw through the round before target.")
    features = tuple(float(value) for value in previous[-1].features)
    mapping = feature_mapping(config.feature_profile, config.max_lag)
    if len(features) != len(mapping) or experiment.get("feature_mapping") != mapping:
        raise ValueError("Stored PySR feature schema does not match the current input schema.")
    expected_schema = experiment.get("config", {}).get("feature_profile")
    if expected_schema != config.feature_profile or not experiment.get("dataset_sha256"):
        raise ValueError("PySR experiment lacks required schema or dataset provenance.")
    target_names = (
        [f"main_position_{index}" for index in range(1, 7)]
        if config.target == "main_positions"
        else [config.target]
    )
    by_target: dict[str, list[dict[str, Any]]] = {name: [] for name in target_names}
    for candidate in experiment.get("candidates", []):
        target_name = candidate.get("target")
        if target_name in by_target:
            by_target[target_name].append(candidate)
    formulas: list[str] = []
    candidate_fingerprints: list[str] = []
    raw_values: list[float] = []
    for target_name in target_names:
        candidates = by_target[target_name]
        if candidate_rank < 1 or candidate_rank > len(candidates):
            raise ValueError(
                f"PySR candidate rank {candidate_rank} is unavailable for {target_name}."
            )
        row = candidates[candidate_rank - 1]
        if not row.get("fingerprint") or row.get("classification") in {"INVALID", "REDUNDANT"}:
            raise ValueError("Selected PySR candidate lacks a verified evaluation fingerprint.")
        formula = str(row.get("formula", ""))
        formulas.append(formula)
        candidate_fingerprints.append(str(row["fingerprint"]))
        raw_values.append(evaluate_pysr_formula(formula, features, tuple(mapping)))
    numbers: list[int] = []
    used: set[int] = set()
    for value in raw_values:
        number = _legal_lotto_number(value, used)
        numbers.append(number)
        used.add(number)
    expected_count = 6 if config.target == "main_positions" else 1
    if (
        len(numbers) != expected_count
        or len(set(numbers)) != expected_count
        or any(not 1 <= n <= 45 for n in numbers)
    ):
        raise ValueError("PySR 실행에서 올바른 6개 번호 조합을 만들지 못했습니다.")
    latest = max(record.round_number for record in dataset.records)
    status = (
        "historical_reproduction"
        if latest >= target_round
        or (official_latest_round is not None and official_latest_round >= target_round)
        else "pre_draw_prediction"
        if official_latest_round is not None
        else "timing_unverified"
    )
    formulas_payload = json.dumps(
        {
            "formulas": formulas,
            "targets": target_names,
            "rank": candidate_rank,
            "engine_version": experiment.get("engine_version"),
            "feature_schema": mapping,
        },
        sort_keys=True,
    )
    import random

    random_seed = secrets.randbelow(2**31) if random_seed is None else random_seed
    random_generator = random.Random(random_seed)
    baseline = (
        [random_generator.randint(1, 45)]
        if expected_count == 1
        else sorted(random_generator.sample(range(1, 46), expected_count))
    )
    calculations = [
        {"target": name, "formula": formula, "raw_value": value, "final_number": number}
        for name, formula, value, number in zip(
            target_names, formulas, raw_values, numbers, strict=True
        )
    ]
    return {
        "recommendation_id": str(uuid.uuid4()),
        "target_round": target_round,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment_id": experiment["experiment_id"],
        "candidate_rank": candidate_rank,
        "engine": experiment["engine"],
        "engine_version": experiment.get("engine_version"),
        "model_type": "six_number_set" if expected_count == 6 else "one_number",
        "recommended_numbers": sorted(numbers),
        "used_formulas": formulas,
        "all_selected_candidate_formulas": formulas,
        "formula_calculations": calculations,
        "gate_result": {"branch": "always_active", "rule_index": 1},
        "gate_calculation": {"explanation": "PySR 후보식은 활성화 게이트 없이 실행됩니다."},
        "model_validation": {
            "training_score": None,
            "validation_score": None,
            "test_score": None,
            "candidate_validation": [
                by_target[name][candidate_rank - 1].get("evaluation", {}).get("validation")
                for name in target_names
            ],
        },
        "formula_fingerprint": hashlib.sha256(formulas_payload.encode()).hexdigest(),
        "candidate_fingerprints": candidate_fingerprints,
        "feature_schema": {
            "profile": config.feature_profile,
            "version": feature_schema_version(config.feature_profile),
            "mapping": mapping,
        },
        "feature_values": {f"x{i}": value for i, value in enumerate(features)},
        "data_provenance": {
            "source_kind": dataset.source_kind,
            "source_file_sha256": data_file_sha256,
            "used_draws_sha256": _canonical_data_hash(dataset, target_round - 1),
            "data_used_through_round": target_round - 1,
        },
        "official_latest_round_at_creation": official_latest_round,
        "recommendation_status": status,
        "replay_status": "verified_saved_formula_reexecution",
        "random_baseline": {
            "numbers": baseline,
            "seed": random_seed,
            "source": "independent uniform 1..45 reference; not used in the PySR recommendation",
        },
    }


def evaluate_gate(value: float) -> dict[str, Any]:
    """Match the existing intermittent engine's modulo-three gate semantics."""
    if not math.isfinite(value):
        return {"gate_value": _display_number(value), "rule_index": 0, "branch": "inactive"}
    rule_index = int(abs(value)) % (RULE_COUNT + 1)
    return {
        "gate_value": value,
        "rule_index": rule_index,
        "branch": f"rule_{rule_index}" if rule_index else "inactive",
    }


def _canonical_data_hash(dataset: DrawDataset, through_round: int) -> str:
    payload = [
        {
            "round": record.round_number,
            "numbers": list(record.draw.numbers),
            "bonus": record.draw.bonus,
            "date": record.draw_date.isoformat() if record.draw_date else None,
        }
        for record in dataset.records
        if record.round_number <= through_round
    ]
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _selected_model(
    experiment: dict[str, Any], model_type: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_key = {"one_number": "one_number_model", "six_number_set": "six_number_set_model"}.get(
        model_type
    )
    if model_key is None:
        raise ValueError("model_type must be one_number or six_number_set.")
    try:
        model = experiment[model_key]
        selected = model["selected"]
        rule_count = int(selected["rule_count"])
        output_count = int(selected["outputs_per_event"])
        formulas = selected["formulas"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Experiment JSON is missing selected GP model fields.") from error
    expected_outputs = 1 if model_type == "one_number" else 6
    if rule_count != RULE_COUNT or output_count != expected_outputs:
        raise ValueError("Selected rule count or output count is unsupported by this executor.")
    if len(formulas) != RULE_COUNT * expected_outputs:
        raise ValueError("Saved formula count does not match the model's rule/output structure.")
    return model, selected


def create_recommendation(
    experiment: dict[str, Any],
    dataset: DrawDataset,
    target_round: int,
    model_type: str,
    *,
    data_file_sha256: str | None = None,
    random_seed: int | None = None,
    official_latest_round: int | None = None,
    target_date: date | None = None,
) -> dict[str, Any]:
    """Execute one saved model using only contiguous draws before the target round."""
    if (
        experiment.get("experiment_type") != "intermittent"
        or experiment.get("status") != "completed"
    ):
        raise ValueError("Recommendation requires a completed intermittent experiment.")
    config = experiment.get("config", {})
    max_lag = int(config.get("max_lag", 10))
    feature_profile = str(config.get("feature_profile", "legacy"))
    if feature_profile == "integrated_v1":
        features = build_integrated_features_for_target(
            dataset, target_round, target_date, max_lag
        )
        mapping = integrated_feature_names(max_lag)
    elif feature_profile == "calendar_round":
        feature_config = ResearchFeatureConfig(include_calendar=True, include_round_digits=True)
        features = build_research_features_for_target(
            dataset, target_round, target_date, max_lag, feature_config
        )
        mapping = feature_names(max_lag) + research_feature_names(feature_config)
    elif feature_profile == "bonus_lag_v1":
        features = build_bonus_features_for_target(dataset, target_round, max_lag)
        mapping = feature_names(max_lag) + tuple(f"lag{lag}_bonus" for lag in range(1, max_lag + 1))
    elif feature_profile == "legacy":
        features = build_features_for_target(dataset, target_round, max_lag)
        mapping = feature_names(max_lag)
    else:
        raise ValueError(f"Unsupported saved feature profile: {feature_profile}")
    if len(features) != len(mapping):
        raise ValueError("Generated feature vector does not match the saved engine feature layout.")
    prior_records = [record for record in dataset.records if record.round_number < target_round]
    latest_input_round = max(record.round_number for record in prior_records)
    if latest_input_round != target_round - 1:
        raise ValueError(
            "Prediction inputs must include every draw through the round before target."
        )
    model, selected = _selected_model(experiment, model_type)

    gate_formula = str(selected["activation_condition"])
    gate_result = evaluate_formula(gate_formula, features, mapping)
    gate_value = float(gate_result["value"])
    gate = evaluate_gate(gate_value)
    formulas = [str(formula) for formula in selected["formulas"]]
    expected_outputs = int(selected["outputs_per_event"])
    calculations: list[dict[str, Any]] = []
    final_numbers: list[int] = []
    if gate["rule_index"]:
        branch_start = (int(gate["rule_index"]) - 1) * expected_outputs
        for formula in formulas[branch_start : branch_start + expected_outputs]:
            calculation = evaluate_formula(formula, features, mapping)
            number = int(calculation["number"])
            adjusted_number = number
            adjustments: list[int] = []
            while adjusted_number in final_numbers:
                adjustments.append(adjusted_number)
                adjusted_number = adjusted_number % 45 + 1
            calculation["normalized_number"] = number
            calculation["final_number"] = adjusted_number
            calculation["duplicate_adjustments"] = adjustments
            final_numbers.append(adjusted_number)
            calculations.append(calculation)
    expected_number_count = 1 if model_type == "one_number" else 6
    if gate["rule_index"] and (
        len(final_numbers) != expected_number_count
        or len(set(final_numbers)) != expected_number_count
        or any(number < 1 or number > 45 for number in final_numbers)
    ):
        raise ValueError("저장된 수식이 중복 없는 유효한 6개 번호 조합을 만들지 못했습니다.")
    pattern_model = "one_number_model" if model_type == "one_number" else "six_number_set_model"
    candidate_rank = int(experiment.get("selected_candidate_rank", 1))
    selected_pattern = next(
        (
            item
            for item in experiment.get("pattern_library_candidates", ())
            if item.get("model") == pattern_model
            and int(item.get("candidate_rank", 0)) == candidate_rank
            and int(item.get("rule_index", 0)) == int(gate["rule_index"] or 0)
        ),
        {},
    )

    if random_seed is None:
        random_seed = secrets.randbelow(2**31)
    import random

    random_generator = random.Random(random_seed)
    random_count = 1 if model_type == "one_number" else 6
    random_numbers = (
        [random_generator.randint(1, 45)]
        if random_count == 1
        else sorted(random_generator.sample(range(1, 46), random_count))
    )
    source_latest_round = max(record.round_number for record in dataset.records)
    if source_latest_round >= target_round or (
        official_latest_round is not None and official_latest_round >= target_round
    ):
        recommendation_status = "historical_reproduction"
    elif official_latest_round is not None:
        recommendation_status = "pre_draw_prediction"
    else:
        recommendation_status = "timing_unverified"

    try:
        training_last_round = int(model["train_rounds"][-1])
    except (KeyError, IndexError, TypeError, ValueError):
        training_last_round = int(experiment["splits"]["train_rounds"][1])
    recommendation_id = str(uuid.uuid4())
    fingerprint_payload = pattern_fingerprint(gate_formula, formulas)
    formula_fingerprint = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()
    schema_version = feature_schema_version(feature_profile)
    return {
        "recommendation_id": recommendation_id,
        "target_round": target_round,
        "created_at": datetime.now(UTC).isoformat(),
        "learning_last_round": training_last_round,
        "input_through_round": latest_input_round,
        "experiment_id": experiment["experiment_id"],
        "candidate_rank": candidate_rank,
        "experiment_config": config,
        "target_date": target_date.isoformat() if target_date else None,
        "model_type": model_type,
        "model_version": MODEL_VERSION,
        "program_version": MODEL_VERSION,
        "engine": experiment.get("engine", experiment.get("engine_version", "unknown")),
        "engine_version": experiment.get("engine_version"),
        "formula_fingerprint": formula_fingerprint,
        "feature_schema": {
            "profile": feature_profile,
            "version": schema_version,
            "mapping": {f"x{index}": name for index, name in enumerate(mapping)},
        },
        "activation_condition": gate_formula,
        "gate_result": gate,
        "gate_calculation": gate_result,
        "selected_rule": gate["rule_index"] or None,
        "pattern_context": {
            "pattern_id": selected_pattern.get("pattern_id"),
            "origin_research_id": selected_pattern.get("origin_research_id"),
            "inheritance_count": selected_pattern.get("inheritance_count", 0),
            "validation_status": selected_pattern.get("current_status", "not_registered"),
            "lineage": selected.get("lineage", {}),
            "active_for_target": bool(gate["rule_index"]),
            "prospective_round_result": "pending",
            "inheritance_count_is_not_confidence": True,
        },
        "used_formulas": [calculation["formula"] for calculation in calculations],
        "all_selected_candidate_formulas": formulas,
        "model_validation": {
            key: selected.get(key)
            for key in (
                "training_hit_rate",
                "validation_hit_rate",
                "test_hit_rate",
                "training_score",
                "validation_score",
                "test_score",
            )
        },
        "feature_mapping": {f"x{index}": name for index, name in enumerate(mapping)},
        "feature_values": {f"x{index}": value for index, value in enumerate(features)},
        "formula_calculations": calculations,
        "recommended_numbers": final_numbers,
        "random_baseline": {
            "numbers": random_numbers,
            "seed": random_seed,
            "source": "independent uniform 1..45 reference; not used in the AI recommendation",
        },
        "data_provenance": {
            "source_kind": dataset.source_kind,
            "verification_status": dataset.verification_status,
            "source_file_sha256": data_file_sha256,
            "used_draws_sha256": _canonical_data_hash(dataset, latest_input_round),
            "source_first_round": dataset.records[0].round_number,
            "source_last_round": source_latest_round,
            "data_used_through_round": latest_input_round,
        },
        "official_latest_round_at_creation": official_latest_round,
        "recommendation_status": recommendation_status,
        "result_status": "pending",
    }


def save_recommendation(
    recommendation: dict[str, Any], directory: str | Path = "predictions"
) -> Path:
    """Write a new recommendation JSON without replacing any existing prediction."""
    destination_dir = project_path(directory)
    destination_dir.mkdir(parents=True, exist_ok=True)
    recommendation_id = str(recommendation["recommendation_id"])
    destination = destination_dir / f"{recommendation_id}.json"
    if destination.exists():
        raise FileExistsError(
            f"Prediction {recommendation_id} already exists; records are immutable."
        )
    handle, temporary_name = tempfile.mkstemp(
        prefix=f"{recommendation_id}_", suffix=".tmp", dir=destination_dir
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as output:
            json.dump(recommendation, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
        os.link(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def load_recommendation(path: str | Path) -> dict[str, Any]:
    with project_path(path).open(encoding="utf-8") as source:
        return json.load(source)


def compare_recommendation(
    recommendation: dict[str, Any],
    actual_numbers: tuple[int, ...] | list[int],
    actual_bonus: int | None = None,
) -> dict[str, Any]:
    """Compare saved numbers to a draw without mutating the saved recommendation."""
    actual = set(actual_numbers)
    predicted = set(recommendation["recommended_numbers"])
    random_numbers = set(recommendation["random_baseline"]["numbers"])
    hit_count = len(predicted & actual)
    if recommendation["model_type"] != "six_number_set":
        prize_tier = None
        prize_tier_status = "해당 없음"
    else:
        classification = classify_prize_tier(
            recommendation["recommended_numbers"], tuple(actual_numbers), actual_bonus
        )
        prize_tier = classification["prize_tier"]
        prize_tier_status = classification["prize_tier_status"]
    bonus_in_prediction = (
        None if actual_bonus is None else actual_bonus in recommendation["recommended_numbers"]
    )
    return {
        "recommendation_id": recommendation["recommendation_id"],
        "target_round": recommendation["target_round"],
        "compared_at": datetime.now(UTC).isoformat(),
        "actual_numbers": sorted(actual),
        "actual_bonus": actual_bonus,
        "ai_hits": sorted(predicted & actual),
        "ai_hit_count": hit_count,
        "bonus_in_prediction": bonus_in_prediction,
        "prize_tier": prize_tier,
        "prize_tier_status": prize_tier_status,
        "random_hits": sorted(random_numbers & actual),
        "random_hit_count": len(random_numbers & actual),
        "gate_branch": recommendation["gate_result"]["branch"],
        "ai_prediction_count": len(predicted),
    }


def summarize_prediction_history(
    history: list[dict[str, Any]], dataset: DrawDataset
) -> dict[str, dict[str, Any]]:
    """Summarize saved prospective recommendations whose actual draws are now available."""
    actual_by_round = {record.round_number: record.draw for record in dataset.records}
    summaries: dict[str, dict[str, Any]] = {}
    for recommendation in history:
        if recommendation.get("recommendation_status") != "pre_draw_prediction":
            continue
        actual_draw = actual_by_round.get(int(recommendation["target_round"]))
        if actual_draw is None:
            continue
        actual = set(actual_draw.numbers)
        model_type = str(recommendation["model_type"])
        row = summaries.setdefault(
            model_type,
            {
                "recommendations": 0,
                "ai_hits": 0,
                "ai_numbers": 0,
                "random_hits": 0,
                "random_numbers": 0,
                "hit_distribution": {str(count): 0 for count in range(7)},
                "three_or_more": 0,
                "four_or_more": 0,
                "five_or_more": 0,
                "six_exact": 0,
                "second_prize_bonus": 0,
                "second_prize_bonus_evaluated": 0,
                "second_prize_bonus_unknown": 0,
                "highest_hit_count": 0,
                "recommendation_combinations": 0,
                "by_gate_branch": {},
            },
        )
        predicted = set(recommendation["recommended_numbers"])
        random_numbers = set(recommendation["random_baseline"]["numbers"])
        hits = predicted & actual
        random_hits = random_numbers & actual
        row["recommendations"] += 1
        row["ai_hits"] += len(hits)
        row["ai_numbers"] += len(predicted)
        row["random_hits"] += len(random_hits)
        row["random_numbers"] += len(random_numbers)
        hit_count = len(hits)
        row["hit_distribution"][str(hit_count)] += 1
        row["three_or_more"] += int(hit_count >= 3)
        row["four_or_more"] += int(hit_count >= 4)
        row["five_or_more"] += int(hit_count >= 5)
        row["six_exact"] += int(hit_count == 6)
        row["highest_hit_count"] = max(row["highest_hit_count"], hit_count)
        row["recommendation_combinations"] += 1
        if model_type == "six_number_set":
            classification = classify_prize_tier(
                recommendation["recommended_numbers"],
                actual_draw.numbers,
                actual_draw.bonus,
            )
            if actual_draw.bonus is None:
                row["second_prize_bonus_unknown"] += 1
            else:
                row["second_prize_bonus_evaluated"] += 1
                row["second_prize_bonus"] += int(classification["prize_tier"] == 2)
        branch = str(recommendation["gate_result"]["branch"])
        branch_summary = row["by_gate_branch"].setdefault(
            branch, {"recommendations": 0, "ai_hits": 0, "actual_draws": 0}
        )
        branch_summary["recommendations"] += 1
        branch_summary["ai_hits"] += len(hits)
        branch_summary["actual_draws"] += 1
    for model_type, row in summaries.items():
        row["ai_hit_rate_per_number"] = row["ai_hits"] / max(row["ai_numbers"], 1)
        row["random_hit_rate_per_number"] = row["random_hits"] / max(row["random_numbers"], 1)
        if model_type != "six_number_set":
            row["second_prize_bonus"] = None
            row["second_prize_status"] = "해당 없음"
        elif row["second_prize_bonus_evaluated"] == 0:
            row["second_prize_bonus"] = None
            row["second_prize_status"] = "미수집"
        elif row["second_prize_bonus_unknown"]:
            row["second_prize_status"] = "부분 수집"
        else:
            row["second_prize_status"] = "판정 완료"
    return summaries


def audit_candidate_history(
    experiment: dict[str, Any],
    dataset: DrawDataset,
    model_type: str,
    candidate_rank: int,
) -> dict[str, Any]:
    """Replay one saved candidate once per unique draw using only earlier draws as inputs."""
    model_key = {
        "one_number": "one_number_model",
        "six_number_set": "six_number_set_model",
    }.get(model_type)
    if model_key is None:
        raise ValueError("Unsupported recommendation model type.")
    replay_experiment = copy.deepcopy(experiment)
    model = replay_experiment.get(model_key, {})
    candidates = model.get("candidates") or [model.get("selected", {})]
    if candidate_rank < 1 or candidate_rank > len(candidates):
        raise ValueError("Candidate rank is outside the saved experiment.")
    model["selected"] = candidates[candidate_rank - 1]
    replay_experiment["selected_candidate_rank"] = candidate_rank
    max_lag = int(replay_experiment.get("config", {}).get("max_lag", 10))
    events: list[dict[str, Any]] = []
    distribution = {str(count): 0 for count in range(7)}
    active_count = 0
    for record in dataset.records:
        target_round = record.round_number
        if target_round - dataset.records[0].round_number < max_lag:
            continue
        recommendation = create_recommendation(
            replay_experiment,
            dataset,
            target_round,
            model_type,
            random_seed=target_round,
            official_latest_round=target_round - 1,
            target_date=record.draw_date,
        )
        predicted = set(recommendation["recommended_numbers"])
        actual = set(record.draw.numbers)
        hits = sorted(predicted & actual)
        hit_count = len(hits)
        distribution[str(hit_count)] += 1
        active_count += int(bool(predicted))
        events.append(
            {
                "round": target_round,
                "actual_numbers": sorted(actual),
                "actual_bonus": record.draw.bonus,
                "predicted_numbers": sorted(predicted),
                "matched_numbers": hits,
                "hit_count": hit_count,
                "gate_branch": recommendation["gate_result"]["branch"],
                "prediction_type": "retrospective_unique_round",
                "formula_fingerprint": recommendation["formula_fingerprint"],
            }
        )
    return {
        "evaluation_rounds": len(events),
        "active_rounds": active_count,
        "hit_distribution": distribution,
        "three_or_more": sum(event["hit_count"] >= 3 for event in events),
        "four_or_more": sum(event["hit_count"] >= 4 for event in events),
        "five_or_more": sum(event["hit_count"] >= 5 for event in events),
        "six_exact": sum(event["hit_count"] == 6 for event in events),
        "events": events,
    }


def save_comparison(
    comparison: dict[str, Any], directory: str | Path = "predictions/comparisons"
) -> Path:
    destination_dir = project_path(directory)
    destination_dir.mkdir(parents=True, exist_ok=True)
    comparison_id = str(uuid.uuid4())
    destination = destination_dir / f"{comparison['recommendation_id']}_{comparison_id}.json"
    with destination.open("x", encoding="utf-8") as output:
        json.dump(comparison, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    return destination


def load_prediction_history(directory: str | Path = "predictions") -> list[dict[str, Any]]:
    prediction_dir = project_path(directory)
    if not prediction_dir.exists():
        return []
    history = []
    for path in sorted(
        prediction_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True
    ):
        history.append(load_recommendation(path))
    return history
