from __future__ import annotations

from lotto_pattern_lab.formula_similarity import cluster_patterns, compare_patterns


def _pattern(pattern_id: str, gate: str, formula: str, rounds: list[int]) -> dict:
    return {
        "pattern_id": pattern_id,
        "activation_condition": gate,
        "formulas": [formula],
        "active_rounds": rounds,
        "complexity": 2,
        "performance": {"independent_test_score": 0.1},
    }


def test_formula_similarity_separates_exact_structure_and_observed_output() -> None:
    left = _pattern("a", "x0", "add(x1, x2)", [10, 11])
    right = _pattern("b", "x0", "add(x2, x1)", [10, 11])
    right["observed_outputs"] = [{"round_number": 10, "predicted_numbers": [1, 2, 3]}]
    left["observed_outputs"] = [{"round_number": 10, "predicted_numbers": [1, 2, 3]}]

    result = compare_patterns(left, right)

    assert result["exact_duplicate"] is True
    assert result["structural_similarity"] == 1.0
    assert result["activation_round_similarity"] == 1.0
    assert result["output_similarity"] == 1.0
    assert result["shared_output_rounds"] == 1


def test_formula_clustering_uses_threshold_and_selects_simpler_representative() -> None:
    patterns = [
        _pattern("complex", "x0", "add(add(x1, x2), x3)", [10]),
        _pattern("simple", "x0", "add(add(x1, x2), x4)", [10]),
        _pattern("different", "x8", "mul(x8, x9)", [99]),
    ]
    patterns[0]["complexity"] = 9
    patterns[1]["complexity"] = 5

    result = cluster_patterns(patterns, structural_threshold=0.7)

    grouped = next(group for group in result["groups"] if group["member_count"] == 2)
    assert grouped["representative_id"] == "simple"
    assert set(grouped["member_ids"]) == {"complex", "simple"}
    assert len(result["groups"]) == 2


def test_formula_clustering_does_not_claim_unobserved_semantic_equivalence() -> None:
    comparison = compare_patterns(
        _pattern("a", "x0", "add(x1, 0)", [1, 2]),
        _pattern("b", "x0", "x1", [1, 2]),
    )

    assert comparison["exact_duplicate"] is False
    assert comparison["output_similarity"] is None
    assert comparison["activation_round_similarity"] == 1.0
