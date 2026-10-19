"""Stable identifiers and compatibility checks for GP feature and primitive schemas."""

import re
from collections.abc import Mapping
from typing import Any

from .features import feature_names
from .research_features import research_feature_names

FEATURE_SCHEMA_VERSION = 1
PRIMITIVE_SET_VERSION = "deap-safe-math-bits-digits-v1"
CODE_VERSION = "0.1.0"
_VARIABLE = re.compile(r"\bx(\d+)\b")


def feature_mapping(profile: str, max_lag: int) -> dict[str, dict[str, str]]:
    """Map current GP argument names to stable semantic IDs and readable names."""
    names = list(feature_names(max_lag))
    if profile == "calendar_round":
        names.extend(research_feature_names())
    elif profile != "legacy":
        raise ValueError(f"Unknown feature profile: {profile}")
    return {
        f"x{index}": {
            "feature_id": f"lotto.{name}",
            "name": name,
        }
        for index, name in enumerate(names)
    }


def _normalize_source_mapping(mapping: Mapping[str, Any], source_profile: str) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for variable, value in mapping.items():
        if isinstance(value, Mapping):
            feature_id = value.get("feature_id")
            name = value.get("name")
        else:
            name = str(value)
            feature_id = None
        if not feature_id and name:
            feature_id = f"lotto.{name}"
        if feature_id and re.fullmatch(r"x\d+", variable):
            normalized[variable] = str(feature_id)
    if not normalized and source_profile in {"legacy", "calendar_round"}:
        fallback = feature_mapping(source_profile, 10)
        normalized = {key: item["feature_id"] for key, item in fallback.items()}
    return normalized


def remap_formula_features(
    formula: str,
    source_mapping: Mapping[str, Any],
    target_mapping: Mapping[str, Mapping[str, str]],
    source_profile: str,
) -> tuple[str | None, str | None]:
    """Remap only known xN terminals; reject unknown or unavailable semantic inputs."""
    source_ids = _normalize_source_mapping(source_mapping, source_profile)
    target_indices = {value["feature_id"]: variable for variable, value in target_mapping.items()}
    missing: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        variable = match.group(0)
        feature_id = source_ids.get(variable)
        target = target_indices.get(feature_id or "")
        if target is None:
            missing.add(feature_id or f"unknown:{variable}")
            return variable
        return target

    remapped = _VARIABLE.sub(replace, formula)
    if missing:
        return None, "feature inputs unavailable in target profile: " + ", ".join(sorted(missing))
    return remapped, None


def formula_feature_ids(formula: str, mapping: Mapping[str, Any], profile: str) -> set[str]:
    source_ids = _normalize_source_mapping(mapping, profile)
    referenced = set(_VARIABLE.findall(formula))
    return {source_ids[f"x{index}"] for index in referenced if f"x{index}" in source_ids}
