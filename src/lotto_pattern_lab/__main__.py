"""Command-line entry point for a single draw summary."""

import argparse

from .draw import Draw
from .patterns import summarize_draw


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize one Lotto 6/45 draw.")
    parser.add_argument("numbers", nargs=6, type=int, metavar="N", help="six main numbers")
    parser.add_argument("--bonus", type=int, help="optional bonus number")
    args = parser.parse_args()

    draw = Draw.from_numbers(args.numbers, args.bonus)
    summary = summarize_draw(draw)
    print(f"Numbers: {' '.join(map(str, draw.numbers))}")
    if draw.bonus is not None:
        print(f"Bonus: {draw.bonus}")
    print(f"Sum: {summary.total}")
    print(f"Odd/even: {summary.odd_count}/{summary.even_count}")
    print(f"Ranges (1-10, 11-20, 21-30, 31-40, 41-45): {summary.decade_counts}")


if __name__ == "__main__":
    main()
