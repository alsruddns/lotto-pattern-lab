"""Shared genetic programming configuration and validation rules."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GPConfig:
    """Settings common to the standard GP and intermittent-rule GP searches."""

    population_size: int = 80
    generations: int = 25
    crossover_probability: float = 0.65
    mutation_probability: float = 0.25
    max_depth: int = 6
    seed: int = 12345
    workers: int = 4
    train_ratio: float = 0.6
    validation_ratio: float = 0.2
    max_lag: int = 10

    def __post_init__(self) -> None:
        if self.population_size < 4 or self.generations < 1:
            raise ValueError("Population must be >= 4 and generations must be positive.")
        if not 0 <= self.crossover_probability <= 1:
            raise ValueError("Crossover probability must be between 0 and 1.")
        if not 0 <= self.mutation_probability <= 1:
            raise ValueError("Mutation probability must be between 0 and 1.")
        if not 2 <= self.max_depth <= 10:
            raise ValueError("Maximum depth must be between 2 and 10.")
        if not 1 <= self.workers <= 4:
            raise ValueError("Worker count must be between 1 and 4.")
        if not 1 <= self.max_lag <= 10:
            raise ValueError("Maximum past references must be between 1 and 10 draws.")
        if self.train_ratio <= 0 or self.validation_ratio <= 0:
            raise ValueError("Train and validation ratios must be positive.")
        if self.train_ratio + self.validation_ratio >= 1:
            raise ValueError("Train plus validation ratios must be less than 1.")
