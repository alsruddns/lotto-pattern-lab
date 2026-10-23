"""Conservative GP expression cleanup with dataset-wide semantic verification."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from deap import gp

from .atomic_io import atomic_write_bytes, interprocess_lock
from .dataset import DrawDataset
from .feature_schema import PRIMITIVE_SET_VERSION, feature_mapping
from .features import build_bonus_examples, build_examples
from .gp_engine import _make_primitive_set
from .intermittent import _gate_rule, _predict_numbers, _validate_formula_ast
from .paths import project_path
from .pattern_library import canonical_formula
from .prediction import _OPERATIONS
from .research_features import (
    ResearchFeatureConfig,
    build_integrated_examples,
    build_research_examples,
)


def formula_metrics(formula: str) -> dict[str, int | str]:
    """Return canonical form, stable fingerprint, operator count, and tree depth."""
    canonical = canonical_formula(formula)
    tree = ast.parse(formula, mode="eval").body
    calls = sum(isinstance(node, ast.Call) for node in ast.walk(tree))

    def depth(node: ast.AST) -> int:
        children = list(ast.iter_child_nodes(node))
        return 1 + max((depth(child) for child in children), default=0)

    return {
        "normalized_formula": canonical,
        "normalized_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "operator_count": calls,
        "depth": depth(tree),
    }


def _constant_value(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
        value = float(node.value)
        return value if math.isfinite(value) else None
    return None


def _simplify_node(node: ast.AST) -> ast.AST:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        children = [_simplify_node(child) for child in node.args]
        name = node.func.id
        values = [_constant_value(child) for child in children]
        if all(value is not None for value in values) and name in _OPERATIONS:
            function, arity = _OPERATIONS[name]
            if len(children) == arity:
                try:
                    result = float(function(*values))
                except (ArithmeticError, OverflowError, ValueError):
                    result = math.nan
                if math.isfinite(result):
                    return ast.copy_location(ast.Constant(value=result), node)

        if len(children) == 2:
            left, right = children
            left_value, right_value = values
            if name in {"add", "sub"} and right_value == 0.0:
                return left
            if name == "add" and left_value == 0.0:
                return right
            if name == "mul" and (right_value == 1.0 or left_value == 1.0):
                return left if right_value == 1.0 else right
            if name in {"sub", "bit_xor"} and ast.dump(left) == ast.dump(right):
                return ast.copy_location(ast.Constant(value=0.0), node)
        return ast.copy_location(ast.Call(func=node.func, args=children, keywords=[]), node)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        child = _simplify_node(node.operand)
        if isinstance(node.op, ast.UAdd):
            return child
        value = _constant_value(child)
        if value is not None:
            return ast.copy_location(ast.Constant(value=-value), node)
        if isinstance(child, ast.UnaryOp) and isinstance(child.op, ast.USub):
            return child.operand
        return ast.copy_location(ast.UnaryOp(op=node.op, operand=child), node)
    return node


def simplify_formula(formula: str) -> dict[str, Any]:
    """Generate a candidate simplification; this function alone never approves it."""
    original = formula.strip()
    try:
        parsed = ast.parse(original, mode="eval")
    except SyntaxError as error:
        raise ValueError(f"수식 문법을 해석할 수 없습니다: {error.msg}") from error
    simplified_tree = ast.fix_missing_locations(_simplify_node(parsed.body))
    simplified = ast.unparse(simplified_tree)
    return {
        "original_formula": original,
        "simplified_formula": simplified,
        "original_metrics": formula_metrics(original),
        "simplified_metrics": formula_metrics(simplified),
        "equivalence_status": "not_verified",
    }


def _examples(dataset: DrawDataset, feature_profile: str, max_lag: int):
    if feature_profile == "integrated_v1":
        return build_integrated_examples(dataset, max_lag)
    if feature_profile == "calendar_round":
        return build_research_examples(
            dataset,
            max_lag,
            ResearchFeatureConfig(include_calendar=True, include_round_digits=True),
        )
    if feature_profile == "bonus_lag_v1":
        main = build_examples(dataset, max_lag)
        bonus = build_bonus_examples(dataset, max_lag)
        return tuple(
            type(left)(left.round_number, right.features, left.winning_numbers)
            for left, right in zip(main, bonus, strict=True)
        )
    if feature_profile == "legacy":
        return build_examples(dataset, max_lag)
    raise ValueError(f"지원하지 않는 특징 프로파일: {feature_profile}")


def _candidate_functions(
    candidate: dict[str, Any], experiment: dict[str, Any], model_key: str
) -> tuple[Callable[..., float], list[Callable[..., float]]]:
    schema = experiment.get("feature_schema") or {}
    config = experiment.get("config") or {}
    profile = str(schema.get("feature_profile", config.get("feature_profile", "legacy")))
    max_lag = int(config.get("max_lag", 10))
    mapping = feature_mapping(profile, max_lag)
    original_mapping = schema.get("feature_mapping") or {}
    if original_mapping and original_mapping != mapping:
        raise ValueError("저장된 특징 스키마가 현재 특징 매핑과 일치하지 않습니다.")
    primitive_version = str(schema.get("primitive_set_version", PRIMITIVE_SET_VERSION))
    if primitive_version != PRIMITIVE_SET_VERSION:
        raise ValueError("지원하지 않는 원본 연산자 버전입니다.")
    if model_key != "six_number_set_model":
        raise ValueError("현재 전체 동등성 검증은 6개 집합 GP 모델만 지원합니다.")
    if candidate.get("rule_count") != 2 or candidate.get("outputs_per_event") != 6:
        raise ValueError("두 분기·분기당 6개 출력 수식만 지원합니다.")
    pset = _make_primitive_set(len(mapping))
    formulas = [
        str(candidate.get("activation_condition", "")),
        *map(str, candidate.get("formulas", [])),
    ]
    if len(formulas) != 13:
        raise ValueError("활성화 수식 1개와 출력 수식 12개가 필요합니다.")
    compiled = []
    for formula in formulas:
        _validate_formula_ast(formula, pset)
        compiled.append(gp.compile(gp.PrimitiveTree.from_string(formula, pset), pset))
    return compiled[0], compiled[1:]


def verify_candidate_simplification(
    original_candidate: dict[str, Any],
    simplified_candidate: dict[str, Any],
    experiment: dict[str, Any],
    dataset: DrawDataset,
    *,
    model_key: str = "six_number_set_model",
) -> dict[str, Any]:
    """Compare gates, active branches, raw values and final six numbers on every row."""
    original_gate, original_outputs = _candidate_functions(
        original_candidate, experiment, model_key
    )
    simplified_gate, simplified_outputs = _candidate_functions(
        simplified_candidate, experiment, model_key
    )
    config = experiment.get("config") or {}
    schema = experiment.get("feature_schema") or {}
    profile = str(schema.get("feature_profile", config.get("feature_profile", "legacy")))
    examples = _examples(dataset, profile, int(config.get("max_lag", 10)))
    compared = 0
    differences: list[dict[str, Any]] = []
    for example in examples:
        original_rule = _gate_rule(original_gate, example.features)
        simplified_rule = _gate_rule(simplified_gate, example.features)
        if original_rule != simplified_rule:
            differences.append(
                {
                    "round": example.round_number,
                    "field": "activation_rule",
                    "original": original_rule,
                    "simplified": simplified_rule,
                }
            )
            continue
        original_values: list[float] = []
        simplified_values: list[float] = []
        if original_rule:
            offset = (original_rule - 1) * 6
            for function in original_outputs[offset : offset + 6]:
                original_values.append(float(function(*example.features)))
            for function in simplified_outputs[offset : offset + 6]:
                simplified_values.append(float(function(*example.features)))
        if original_values != simplified_values:
            differences.append(
                {
                    "round": example.round_number,
                    "field": "active_branch_values",
                    "original": original_values,
                    "simplified": simplified_values,
                }
            )
            continue
        original_numbers = (
            _predict_numbers(tuple(original_outputs[offset : offset + 6]), example.features)
            if original_rule
            else ()
        )
        simplified_numbers = (
            _predict_numbers(tuple(simplified_outputs[offset : offset + 6]), example.features)
            if simplified_rule
            else ()
        )
        if original_numbers != simplified_numbers:
            differences.append(
                {
                    "round": example.round_number,
                    "field": "final_numbers",
                    "original": original_numbers,
                    "simplified": simplified_numbers,
                }
            )
        compared += 1
    return {
        "equivalent": not differences,
        "status": "verified_equivalent" if not differences else "rejected_different_output",
        "evaluation_rounds": compared,
        "difference_count": len(differences),
        "differences": differences[:100],
    }


def simplify_candidate(candidate: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Simplify gate and output expressions while retaining all candidate metadata."""
    clone = json.loads(json.dumps(candidate, ensure_ascii=False))
    formula_results = [
        simplify_formula(str(formula))
        for formula in [candidate.get("activation_condition", ""), *candidate.get("formulas", [])]
    ]
    clone["activation_condition"] = formula_results[0]["simplified_formula"]
    clone["formulas"] = [item["simplified_formula"] for item in formula_results[1:]]
    clone["simplification_original_fingerprint"] = hashlib.sha256(
        json.dumps(
            [candidate.get("activation_condition"), candidate.get("formulas")],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    metrics = {
        "original_operator_count": sum(
            int(item["original_metrics"]["operator_count"]) for item in formula_results
        ),
        "simplified_operator_count": sum(
            int(item["simplified_metrics"]["operator_count"]) for item in formula_results
        ),
        "formula_results": formula_results,
        "status": "not_verified",
    }
    return clone, metrics


def save_verified_simplification(
    *,
    experiment_id: str,
    research_id: str | None,
    candidate_rank: int,
    original_candidate: dict[str, Any],
    simplified_candidate: dict[str, Any],
    metrics: dict[str, Any],
    verification: dict[str, Any],
    dataset_sha256: str,
    directory: str = "data/formula_simplifications",
) -> dict[str, Any]:
    """Store only proven equivalent variants as new immutable records."""
    if not verification.get("equivalent"):
        raise ValueError("동등성이 확인되지 않은 단순화 수식은 저장할 수 없습니다.")
    record = {
        "simplification_id": str(uuid.uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
        "origin_experiment_id": experiment_id,
        "origin_research_id": research_id,
        "candidate_rank": candidate_rank,
        "dataset_sha256": dataset_sha256,
        "equivalence_verification": verification,
        "metrics": metrics,
        "original_candidate": original_candidate,
        "simplified_candidate": simplified_candidate,
    }
    record["record_sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    target = project_path(directory) / f"{record['simplification_id']}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with interprocess_lock(target, scope="formula-simplification"):
        if target.exists():
            raise FileExistsError(target)
        content = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        atomic_write_bytes(target, content + b"\n")
    record["path"] = str(target)
    return record
