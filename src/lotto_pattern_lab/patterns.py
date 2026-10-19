"""Simple descriptive summaries for a Lotto 6/45 draw."""

from dataclasses import dataclass

from .draw import Draw


@dataclass(frozen=True, slots=True)
class DrawSummary:
    """Descriptive number features for one draw."""

    total: int
    odd_count: int
    even_count: int
    decade_counts: tuple[int, int, int, int, int]


def summarize_draw(draw: Draw) -> DrawSummary:
    """Count basic number patterns without making predictive claims."""
    decade_counts = [0, 0, 0, 0, 0]
    for number in draw.numbers:
        decade_counts[min((number - 1) // 10, 4)] += 1

    odd_count = sum(number % 2 for number in draw.numbers)
    return DrawSummary(
        total=sum(draw.numbers),
        odd_count=odd_count,
        even_count=6 - odd_count,
        decade_counts=tuple(decade_counts),
    )
