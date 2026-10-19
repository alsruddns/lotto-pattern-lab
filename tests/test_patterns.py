import pytest

from lotto_pattern_lab import Draw, summarize_draw


def test_from_numbers_sorts_and_summarizes_draw() -> None:
    draw = Draw.from_numbers([44, 7, 35, 13, 22, 1], bonus=9)

    assert draw.numbers == (1, 7, 13, 22, 35, 44)
    assert summarize_draw(draw).total == 122
    assert summarize_draw(draw).odd_count == 4
    assert summarize_draw(draw).even_count == 2
    assert summarize_draw(draw).decade_counts == (2, 1, 1, 1, 1)


@pytest.mark.parametrize(
    "numbers",
    [
        (1, 2, 3, 4, 5),
        (1, 2, 3, 4, 5, 46),
        (1, 2, 3, 4, 5, 5),
        (1, 2, 3, 4, 5, True),
    ],
)
def test_invalid_draw_is_rejected(numbers: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        Draw(numbers)


def test_bonus_must_be_distinct_from_main_numbers() -> None:
    with pytest.raises(ValueError, match="differ"):
        Draw((1, 2, 3, 4, 5, 6), bonus=6)
