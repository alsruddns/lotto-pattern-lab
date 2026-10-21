"""Validation and conversion helpers for research time budgets."""

from __future__ import annotations

from collections.abc import Iterable

MIN_RESEARCH_DURATION_MINUTES = 1
MAX_RESEARCH_DURATION_MINUTES = 1440


def validate_duration_minutes(value: object) -> int:
    """Accept positive whole minutes up to the one-day safety limit."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Research duration must be a whole number of minutes.")
    if not MIN_RESEARCH_DURATION_MINUTES <= value <= MAX_RESEARCH_DURATION_MINUTES:
        raise ValueError(
            "Research duration must be between "
            f"{MIN_RESEARCH_DURATION_MINUTES} and {MAX_RESEARCH_DURATION_MINUTES} minutes."
        )
    return value


def duration_seconds_from_minutes(value: object) -> int:
    """Convert validated integer minutes to an exact number of seconds."""
    return validate_duration_minutes(value) * 60


def recommended_preset_minutes(value: object, presets: Iterable[int]) -> int:
    """Choose the closest lower recommended profile for custom-time GP defaults."""
    minutes = validate_duration_minutes(value)
    available = sorted(set(presets))
    if not available:
        raise ValueError("At least one recommended duration preset is required.")
    return max((preset for preset in available if preset <= minutes), default=available[0])
