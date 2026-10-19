"""Order-invariant feature construction using only prior rounds."""

from dataclasses import dataclass

from .dataset import DrawDataset, DrawRecord

MAX_LAG = 10


@dataclass(frozen=True, slots=True)
class Example:
    round_number: int
    features: tuple[float, ...]
    winning_numbers: frozenset[int]


def _set_features(record: DrawRecord) -> tuple[float, ...]:
    values = record.draw.numbers
    total = sum(values)
    sum_squares = sum(number * number for number in values)
    pairwise_sum = (total * total - sum_squares) // 2
    product_mod = 1
    for number in values:
        product_mod = product_mod * number % 997
    return (
        float(total),
        float(sum_squares),
        float(pairwise_sum),
        float(product_mod),
        float(min(values)),
        float(max(values)),
    )


def feature_names(max_lag: int = MAX_LAG) -> tuple[str, ...]:
    names = ["round_scaled"]
    for lag in range(1, max_lag + 1):
        names.extend(
            f"lag{lag}_{name}" for name in ("sum", "squares", "pairs", "product_mod", "min", "max")
        )
    names.extend(f"sum_delta_{lag}_{lag + 1}" for lag in range(1, max_lag))
    return tuple(names)


def build_examples(dataset: DrawDataset, max_lag: int = MAX_LAG) -> tuple[Example, ...]:
    if not 1 <= max_lag <= MAX_LAG:
        raise ValueError(f"max_lag must be between 1 and {MAX_LAG}.")
    records = dataset.records
    examples: list[Example] = []
    set_features = [_set_features(record) for record in records]
    for target_index in range(max_lag, len(records)):
        prior = [set_features[target_index - lag] for lag in range(1, max_lag + 1)]
        features = [records[target_index].round_number / 1000.0]
        for values in prior:
            features.extend(values)
        features.extend(prior[lag - 1][0] - prior[lag][0] for lag in range(1, max_lag))
        examples.append(
            Example(
                records[target_index].round_number,
                tuple(float(value) for value in features),
                frozenset(records[target_index].draw.numbers),
            )
        )
    return tuple(examples)


def build_features_for_target(
    dataset: DrawDataset, target_round: int, max_lag: int = MAX_LAG
) -> tuple[float, ...]:
    """Build a target's inputs from exactly the preceding contiguous draws only."""
    if not 1 <= max_lag <= MAX_LAG:
        raise ValueError(f"max_lag must be between 1 and {MAX_LAG}.")
    prior_by_round = {
        record.round_number: record
        for record in dataset.records
        if record.round_number < target_round
    }
    required_rounds = range(target_round - max_lag, target_round)
    if target_round <= max_lag or any(
        round_number not in prior_by_round for round_number in required_rounds
    ):
        raise ValueError(
            f"Target {target_round} needs contiguous history for rounds "
            f"{target_round - max_lag}..{target_round - 1}."
        )
    prior = [_set_features(prior_by_round[target_round - lag]) for lag in range(1, max_lag + 1)]
    values = [target_round / 1000.0]
    for summaries in prior:
        values.extend(summaries)
    values.extend(prior[lag - 1][0] - prior[lag][0] for lag in range(1, max_lag))
    return tuple(float(value) for value in values)
