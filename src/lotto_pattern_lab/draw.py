"""Validated Lotto 6/45 draw model."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Draw:
    """One draw with six main numbers and an optional bonus number."""

    numbers: tuple[int, ...]
    bonus: int | None = None

    def __post_init__(self) -> None:
        if len(self.numbers) != 6:
            raise ValueError("A draw must contain exactly six main numbers.")
        if any(not isinstance(number, int) or isinstance(number, bool) for number in self.numbers):
            raise ValueError("Numbers must be integers.")
        if any(number < 1 or number > 45 for number in self.numbers):
            raise ValueError("Numbers must be between 1 and 45.")
        if len(set(self.numbers)) != 6:
            raise ValueError("Main numbers must be unique.")
        if self.bonus is not None:
            if not isinstance(self.bonus, int) or isinstance(self.bonus, bool):
                raise ValueError("The bonus number must be an integer.")
            if not 1 <= self.bonus <= 45:
                raise ValueError("The bonus number must be between 1 and 45.")
            if self.bonus in self.numbers:
                raise ValueError("The bonus number must differ from the main numbers.")

    @classmethod
    def from_numbers(cls, numbers: list[int] | tuple[int, ...], bonus: int | None = None) -> "Draw":
        """Build a draw and store its main numbers in ascending order."""
        return cls(tuple(sorted(numbers)), bonus)
