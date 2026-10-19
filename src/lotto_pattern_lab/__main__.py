"""Command-line entry points for draw summaries and GP experiments."""

import argparse
import json
import sys

from .dataset import load_csv, make_random_dataset
from .draw import Draw
from .experiment import run_experiment
from .gp_engine import SearchConfig
from .patterns import summarize_draw
from .storage import ExperimentStore


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "search":
        _run_search_cli(sys.argv[2:])
        return
    _run_draw_cli(sys.argv[1:])


def _run_draw_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Summarize one Lotto 6/45 draw.")
    parser.add_argument("numbers", nargs=6, type=int, metavar="N", help="six main numbers")
    parser.add_argument("--bonus", type=int, help="optional bonus number")
    args = parser.parse_args(argv)

    draw = Draw.from_numbers(args.numbers, args.bonus)
    summary = summarize_draw(draw)
    print(f"Numbers: {' '.join(map(str, draw.numbers))}")
    if draw.bonus is not None:
        print(f"Bonus: {draw.bonus}")
    print(f"Sum: {summary.total}")
    print(f"Odd/even: {summary.odd_count}/{summary.even_count}")
    print(f"Ranges (1-10, 11-20, 21-30, 31-40, 41-45): {summary.decade_counts}")


def _run_search_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Run a DEAP search and chronological backtest.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", help="UTF-8 CSV with round,n1,n2,n3,n4,n5,n6 columns")
    source.add_argument(
        "--synthetic", type=int, metavar="DRAWS", help="generate labeled synthetic control data"
    )
    parser.add_argument(
        "--source-url", help="record the claimed source URL for a user supplied CSV"
    )
    parser.add_argument("--population", type=int, default=80)
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--crossover", type=float, default=0.65)
    parser.add_argument("--mutation", type=float, default=0.25)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--controls", type=int, default=3)
    parser.add_argument("--store", default="experiments")
    args = parser.parse_args(argv)

    dataset = (
        load_csv(args.csv, args.source_url)
        if args.csv
        else make_random_dataset(args.synthetic, args.seed)
    )
    config = SearchConfig(
        population_size=args.population,
        generations=args.generations,
        crossover_probability=args.crossover,
        mutation_probability=args.mutation,
        max_depth=args.max_depth,
        seed=args.seed,
        workers=args.workers,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        control_runs=args.controls,
    )
    store = ExperimentStore(args.store)

    def report(event: dict[str, object]) -> None:
        print(
            f"[{event['run']}] 세대 {event['generation']}/{event['generations']} "
            f"학습 {float(event['train_accuracy']):.3f}, "
            f"검증 {float(event['validation_accuracy']):.3f} | {event['formula']}",
            flush=True,
        )

    result = run_experiment(dataset, config, store, report)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"실험 저장: {store.directory / str(result['experiment_id'])}.json")


if __name__ == "__main__":
    main()
