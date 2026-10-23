"""Optional calendar and round features for research without changing legacy GP vectors."""

from dataclasses import dataclass
from datetime import date

from .dataset import DrawDataset
from .features import (
    Example,
    append_lagged_bonus_features,
    build_bonus_features_for_target,
    build_examples,
    feature_names,
)


@dataclass(frozen=True, slots=True)
class ResearchFeatureConfig:
    """Feature switches kept separate from the existing saved-model feature contract."""

    include_calendar: bool = True
    include_round_digits: bool = True


CALENDAR_FEATURE_NAMES = (
    "draw_year",
    "draw_month",
    "draw_day",
    "draw_weekday_monday0",
    "draw_day_of_year",
    "draw_iso_week",
    "draw_yyyymmdd",
    "draw_digit_sum",
    "draw_reversed_digits",
    "draw_mod_31",
    "draw_mod_45",
    "draw_mod_97",
    "date_round_concat",
    "date_round_xor",
    "date_round_mod_997",
)
ROUND_FEATURE_NAMES = (
    "round_digit_sum",
    "round_reversed_digits",
    "round_mod_3",
    "round_mod_7",
    "round_mod_45",
)


def _digit_sum(value: int) -> int:
    return sum(int(character) for character in str(abs(value)))


def _reverse_digits(value: int) -> int:
    return int(str(abs(value))[::-1])


def calendar_feature_values(draw_date: date, round_number: int) -> tuple[float, ...]:
    """Return exogenous date fields; callers must supply an official or scheduled date."""
    compact = draw_date.strftime("%Y%m%d")
    value = int(compact)
    return (
        float(draw_date.year),
        float(draw_date.month),
        float(draw_date.day),
        float(draw_date.weekday()),
        float(draw_date.timetuple().tm_yday),
        float(draw_date.isocalendar().week),
        float(value),
        float(_digit_sum(value)),
        float(_reverse_digits(value)),
        float(value % 31),
        float(value % 45),
        float(value % 97),
        float(value * (10 ** len(str(round_number))) + round_number),
        float(value ^ round_number),
        float((value + round_number) % 997),
    )


def round_feature_values(round_number: int) -> tuple[float, ...]:
    return (
        float(_digit_sum(round_number)),
        float(_reverse_digits(round_number)),
        float(round_number % 3),
        float(round_number % 7),
        float(round_number % 45),
    )


def research_feature_names(
    config: ResearchFeatureConfig = ResearchFeatureConfig(),
) -> tuple[str, ...]:
    names: list[str] = []
    if config.include_calendar:
        names.extend(CALENDAR_FEATURE_NAMES)
    if config.include_round_digits:
        names.extend(ROUND_FEATURE_NAMES)
    return tuple(names)


def integrated_feature_names(max_lag: int = 10) -> tuple[str, ...]:
    """Return the integrated schema with stable first-occurrence deduplication."""
    names = (
        *feature_names(max_lag),
        *research_feature_names(),
        *(f"lag{lag}_bonus" for lag in range(1, max_lag + 1)),
    )
    return tuple(dict.fromkeys(names))


def _unique_values(values: tuple[float, ...], max_lag: int) -> tuple[float, ...]:
    raw_names = (
        *feature_names(max_lag),
        *research_feature_names(),
        *(f"lag{lag}_bonus" for lag in range(1, max_lag + 1)),
    )
    selected_indices: list[int] = []
    seen: set[str] = set()
    for index, name in enumerate(raw_names):
        if name not in seen:
            seen.add(name)
            selected_indices.append(index)
    return tuple(values[index] for index in selected_indices)


def build_research_examples(
    dataset: DrawDataset,
    max_lag: int = 10,
    config: ResearchFeatureConfig = ResearchFeatureConfig(),
) -> tuple[Example, ...]:
    """Extend the stable legacy features with verified date and round transforms.

    Calendar features are refused when a draw date is missing; dates are never fabricated.
    """
    if config.include_calendar:
        if dataset.source_kind not in {"official_verified", "synthetic"}:
            raise ValueError("Calendar features require an official verified date source.")
        if any(record.draw_date is None for record in dataset.records):
            raise ValueError(
                "Calendar features require a date for every record; missing dates are not inferred."
            )
    legacy = build_examples(dataset, max_lag)
    date_by_round = {record.round_number: record.draw_date for record in dataset.records}
    examples: list[Example] = []
    for example in legacy:
        additions: list[float] = []
        if config.include_calendar:
            draw_date = date_by_round[example.round_number]
            assert draw_date is not None
            additions.extend(calendar_feature_values(draw_date, example.round_number))
        if config.include_round_digits:
            additions.extend(round_feature_values(example.round_number))
        examples.append(
            Example(
                example.round_number,
                example.features + tuple(additions),
                example.winning_numbers,
            )
        )
    return tuple(examples)


def build_research_features_for_target(
    dataset: DrawDataset,
    target_round: int,
    target_date: date | None,
    max_lag: int = 10,
    config: ResearchFeatureConfig = ResearchFeatureConfig(),
) -> tuple[float, ...]:
    """Construct feature-only target inputs; output numbers are never read."""
    from .features import build_features_for_target

    legacy = build_features_for_target(dataset, target_round, max_lag)
    additions: list[float] = []
    if config.include_calendar:
        if target_date is None:
            raise ValueError("Calendar prediction requires an explicitly supplied target date.")
        additions.extend(calendar_feature_values(target_date, target_round))
    if config.include_round_digits:
        additions.extend(round_feature_values(target_round))
    return legacy + tuple(additions)


def build_integrated_examples(dataset: DrawDataset, max_lag: int = 10) -> tuple[Example, ...]:
    """Build base, date/round, and lagged bonus inputs with the main draw as label."""
    examples = build_research_examples(dataset, max_lag)
    merged = append_lagged_bonus_features(examples, dataset, max_lag)
    return tuple(
        Example(item.round_number, _unique_values(item.features, max_lag), item.winning_numbers)
        for item in merged
    )


def build_integrated_bonus_examples(
    dataset: DrawDataset, max_lag: int = 10
) -> tuple[Example, ...]:
    """Build integrated inputs with each target draw's bonus used only as the label."""
    examples = build_integrated_examples(dataset, max_lag)
    bonus_by_round = {record.round_number: record.draw.bonus for record in dataset.records}
    if any(value is None for value in bonus_by_round.values()):
        raise ValueError("Bonus-number research requires a bonus value for every draw.")
    return tuple(
        Example(
            item.round_number,
            item.features,
            frozenset({int(bonus_by_round[item.round_number])}),
        )
        for item in examples
    )


def build_integrated_features_for_target(
    dataset: DrawDataset, target_round: int, target_date: date | None, max_lag: int = 10
) -> tuple[float, ...]:
    """Build all integrated inputs from target metadata and draws strictly before target."""
    calendar_features = build_research_features_for_target(
        dataset,
        target_round,
        target_date,
        max_lag,
        ResearchFeatureConfig(include_calendar=True, include_round_digits=True),
    )
    bonus_features = build_bonus_features_for_target(dataset, target_round, max_lag)
    base_count = len(feature_names(max_lag))
    bonus_count = max_lag
    return _unique_values(
        calendar_features + bonus_features[base_count : base_count + bonus_count], max_lag
    )
