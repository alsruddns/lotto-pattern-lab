"""Chronological train, validation, and untouched test partitions."""

from dataclasses import dataclass

from .features import Example


@dataclass(frozen=True, slots=True)
class DataSplits:
    train: tuple[Example, ...]
    validation: tuple[Example, ...]
    test: tuple[Example, ...]


def chronological_split(
    examples: tuple[Example, ...], train_ratio: float = 0.6, validation_ratio: float = 0.2
) -> DataSplits:
    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1:
        raise ValueError("Split ratios must be between 0 and 1.")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("Train plus validation ratios must be less than 1.")
    train_end = int(len(examples) * train_ratio)
    validation_end = train_end + int(len(examples) * validation_ratio)
    if min(train_end, validation_end - train_end, len(examples) - validation_end) < 1:
        raise ValueError("Not enough examples for three non-empty chronological splits.")
    return DataSplits(
        examples[:train_end], examples[train_end:validation_end], examples[validation_end:]
    )
