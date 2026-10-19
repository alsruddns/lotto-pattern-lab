"""Safely execute saved intermittent GP rules and store immutable recommendations."""

import ast
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
from .features import build_features_for_target, feature_names
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
from .research_features import (
    ResearchFeatureConfig,
    build_research_features_for_target,
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
        {"round": record.round_number, "numbers": list(record.draw.numbers)}
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
    if feature_profile == "calendar_round":
        feature_config = ResearchFeatureConfig(include_calendar=True, include_round_digits=True)
        features = build_research_features_for_target(
            dataset, target_round, target_date, max_lag, feature_config
        )
        mapping = feature_names(max_lag) + research_feature_names(feature_config)
    elif feature_profile == "legacy":
        features = build_features_for_target(dataset, target_round, max_lag)
        mapping = feature_names(max_lag)
    else:
        raise ValueError(f"Unsupported saved feature profile: {feature_profile}")
    if len(features) != len(mapping):
        raise ValueError("Generated feature vector does not match the saved engine feature layout.")
    prior_records = [record for record in dataset.records if record.round_number < target_round]
    latest_input_round = max(record.round_number for record in prior_records)
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
    return {
        "recommendation_id": recommendation_id,
        "target_round": target_round,
        "created_at": datetime.now(UTC).isoformat(),
        "learning_last_round": training_last_round,
        "input_through_round": latest_input_round,
        "experiment_id": experiment["experiment_id"],
        "experiment_config": config,
        "target_date": target_date.isoformat() if target_date else None,
        "model_type": model_type,
        "model_version": MODEL_VERSION,
        "activation_condition": gate_formula,
        "gate_result": gate,
        "gate_calculation": gate_result,
        "selected_rule": gate["rule_index"] or None,
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
    destination_dir = Path(directory)
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
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)


def compare_recommendation(
    recommendation: dict[str, Any], actual_numbers: tuple[int, ...] | list[int]
) -> dict[str, Any]:
    """Compare saved numbers to a draw without mutating the saved recommendation."""
    actual = set(actual_numbers)
    predicted = set(recommendation["recommended_numbers"])
    random_numbers = set(recommendation["random_baseline"]["numbers"])
    return {
        "recommendation_id": recommendation["recommendation_id"],
        "target_round": recommendation["target_round"],
        "compared_at": datetime.now(UTC).isoformat(),
        "actual_numbers": sorted(actual),
        "ai_hits": sorted(predicted & actual),
        "ai_hit_count": len(predicted & actual),
        "random_hits": sorted(random_numbers & actual),
        "random_hit_count": len(random_numbers & actual),
        "gate_branch": recommendation["gate_result"]["branch"],
        "ai_prediction_count": len(predicted),
    }


def summarize_prediction_history(
    history: list[dict[str, Any]], dataset: DrawDataset
) -> dict[str, dict[str, Any]]:
    """Summarize saved prospective recommendations whose actual draws are now available."""
    actual_by_round = {record.round_number: set(record.draw.numbers) for record in dataset.records}
    summaries: dict[str, dict[str, Any]] = {}
    for recommendation in history:
        if recommendation.get("recommendation_status") != "pre_draw_prediction":
            continue
        actual = actual_by_round.get(int(recommendation["target_round"]))
        if actual is None:
            continue
        model_type = str(recommendation["model_type"])
        row = summaries.setdefault(
            model_type,
            {
                "recommendations": 0,
                "ai_hits": 0,
                "ai_numbers": 0,
                "random_hits": 0,
                "random_numbers": 0,
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
        branch = str(recommendation["gate_result"]["branch"])
        branch_summary = row["by_gate_branch"].setdefault(
            branch, {"recommendations": 0, "ai_hits": 0, "actual_draws": 0}
        )
        branch_summary["recommendations"] += 1
        branch_summary["ai_hits"] += len(hits)
        branch_summary["actual_draws"] += 1
    for row in summaries.values():
        row["ai_hit_rate_per_number"] = row["ai_hits"] / max(row["ai_numbers"], 1)
        row["random_hit_rate_per_number"] = row["random_hits"] / max(row["random_numbers"], 1)
    return summaries


def save_comparison(
    comparison: dict[str, Any], directory: str | Path = "predictions/comparisons"
) -> Path:
    destination_dir = Path(directory)
    destination_dir.mkdir(parents=True, exist_ok=True)
    comparison_id = str(uuid.uuid4())
    destination = destination_dir / f"{comparison['recommendation_id']}_{comparison_id}.json"
    with destination.open("x", encoding="utf-8") as output:
        json.dump(comparison, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    return destination


def load_prediction_history(directory: str | Path = "predictions") -> list[dict[str, Any]]:
    prediction_dir = Path(directory)
    if not prediction_dir.exists():
        return []
    history = []
    for path in sorted(
        prediction_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True
    ):
        history.append(load_recommendation(path))
    return history
