"""Conservative structural and observed-behavior comparisons for saved formulas."""

from __future__ import annotations

import ast
import hashlib
from collections import Counter
from collections.abc import Iterable
from typing import Any

from .pattern_library import pattern_fingerprint


def _node_tokens(node: ast.AST, depth: int = 0) -> Counter[str]:
    """Return depth-tagged AST labels; this is a structural proxy, not equivalence proof."""
    tokens: Counter[str] = Counter()
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        label = f"call:{node.func.id}"
        children = list(node.args)
    elif isinstance(node, ast.Name):
        label, children = f"var:{node.id}", []
    elif isinstance(node, ast.Constant) and type(node.value) in {int, float}:
        label, children = f"const:{node.value!r}", []
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        label = "unary:+" if isinstance(node.op, ast.UAdd) else "unary:-"
        children = [node.operand]
    elif isinstance(node, ast.BinOp):
        label = f"binary:{type(node.op).__name__}"
        children = [node.left, node.right]
    else:
        raise ValueError("저장된 수식이 지원되는 AST 구조가 아닙니다.")
    tokens[f"{depth}:{label}"] += 1
    for child in children:
        tokens.update(_node_tokens(child, depth + 1))
    return tokens


def _formula_tokens(pattern: dict[str, Any]) -> Counter[str]:
    result: Counter[str] = Counter()
    expressions = [pattern.get("activation_condition", ""), *pattern.get("formulas", [])]
    for index, expression in enumerate(expressions):
        if not expression:
            continue
        tree = ast.parse(str(expression), mode="eval").body
        for token, count in _node_tokens(tree).items():
            result[f"expression:{index}:{token}"] += count
    return result


def _jaccard(left: set[Any], right: set[Any]) -> float | None:
    union = left | right
    return len(left & right) / len(union) if union else None


def _weighted_jaccard(left: Counter[str], right: Counter[str]) -> float:
    union = left.keys() | right.keys()
    denominator = sum(max(left[key], right[key]) for key in union)
    return sum(min(left[key], right[key]) for key in union) / denominator if denominator else 1.0


def compare_patterns(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Compare canonical structure and saved event evidence without claiming proof."""
    left_exact = pattern_fingerprint(
        str(left.get("activation_condition", "")), tuple(map(str, left.get("formulas", ())))
    )
    right_exact = pattern_fingerprint(
        str(right.get("activation_condition", "")), tuple(map(str, right.get("formulas", ())))
    )
    structure = _weighted_jaccard(_formula_tokens(left), _formula_tokens(right))
    active_similarity = _jaccard(
        set(map(int, left.get("active_rounds", ()))),
        set(map(int, right.get("active_rounds", ()))),
    )
    left_outputs = {
        int(item["round_number"]): tuple(map(int, item.get("predicted_numbers", ())))
        for item in left.get("observed_outputs", ())
        if item.get("round_number") is not None and item.get("predicted_numbers")
    }
    right_outputs = {
        int(item["round_number"]): tuple(map(int, item.get("predicted_numbers", ())))
        for item in right.get("observed_outputs", ())
        if item.get("round_number") is not None and item.get("predicted_numbers")
    }
    shared_rounds = left_outputs.keys() & right_outputs.keys()
    output_similarity = (
        sum(
            left_outputs[round_number] == right_outputs[round_number]
            for round_number in shared_rounds
        )
        / len(shared_rounds)
        if shared_rounds
        else None
    )
    return {
        "exact_duplicate": left_exact == right_exact,
        "structural_similarity": structure,
        "activation_round_similarity": active_similarity,
        "output_similarity": output_similarity,
        "shared_output_rounds": len(shared_rounds),
    }


def cluster_patterns(
    patterns: Iterable[dict[str, Any]], *, structural_threshold: float = 0.8
) -> dict[str, Any]:
    """Group exact and structurally similar formulas; observed output evidence is separate."""
    if not 0.0 <= structural_threshold <= 1.0:
        raise ValueError("구조 유사도 기준은 0과 1 사이여야 합니다.")
    items = list(patterns)
    tokens = [_formula_tokens(item) for item in items]
    fingerprints = [
        pattern_fingerprint(
            str(item.get("activation_condition", "")), tuple(map(str, item.get("formulas", ())))
        )
        for item in items
    ]
    parents = list(range(len(items)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    pairwise = []
    for left_index, left in enumerate(items):
        for right_index in range(left_index + 1, len(items)):
            exact = fingerprints[left_index] == fingerprints[right_index]
            score = _weighted_jaccard(tokens[left_index], tokens[right_index])
            pairwise.append(
                {
                    "left_id": str(left.get("pattern_id", left_index)),
                    "right_id": str(items[right_index].get("pattern_id", right_index)),
                    "exact_duplicate": exact,
                    "structural_similarity": score,
                    "activation_round_similarity": _jaccard(
                        set(map(int, left.get("active_rounds", ()))),
                        set(map(int, items[right_index].get("active_rounds", ()))),
                    ),
                }
            )
            if exact or score >= structural_threshold:
                first, second = find(left_index), find(right_index)
                parents[second] = first

    groups: dict[int, list[int]] = {}
    for index in range(len(items)):
        groups.setdefault(find(index), []).append(index)
    result_groups = []
    assignments = {}
    for members in groups.values():
        def rank_key(index: int) -> tuple[int, float, str]:
            performance = items[index].get("performance") or {}
            test_score = performance.get("independent_test_score")
            return (
                int(items[index].get("complexity") or 10**9),
                -float(test_score) if test_score is not None else 0.0,
                str(items[index].get("pattern_id", index)),
            )

        ranked = sorted(
            members,
            key=rank_key,
        )
        member_ids = [str(items[index].get("pattern_id", index)) for index in members]
        content = "|".join(sorted(member_ids))
        group_id = hashlib.sha256(content.encode()).hexdigest()[:16]
        duplicate = len(members) > 1 and len({fingerprints[index] for index in members}) == 1
        result_groups.append(
            {
                "group_id": group_id,
                "member_ids": member_ids,
                "member_count": len(members),
                "representative_id": str(items[ranked[0]].get("pattern_id", ranked[0])),
                "group_type": "exact_duplicate" if duplicate else "structurally_similar",
            }
        )
        for index in members:
            assignments[str(items[index].get("pattern_id", index))] = group_id
    return {
        "patterns": len(items),
        "groups": result_groups,
        "pairwise": pairwise,
        "assignments": assignments,
        "structural_threshold": structural_threshold,
        "output_similarity_scope": "같은 회차의 저장된 예측 출력이 모두 제공된 경우에만 계산",
    }
