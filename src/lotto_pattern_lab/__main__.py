"""Command-line entry points for draw summaries and GP experiments."""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from .dataset import load_csv, make_random_dataset, save_official_csv
from .draw import Draw
from .experiment import run_experiment
from .gp_engine import SearchConfig
from .intermittent import IntermittentConfig
from .intermittent_experiment import run_intermittent_experiment
from .official_data import FIRST_SUPPORTED_ROUND, fetch_official_dataset
from .pattern_library import PatternLibrary
from .patterns import summarize_draw
from .research import SUPPORTED_ENGINES, request_stop, run_research_schedule
from .storage import ExperimentStore


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "search":
        _run_search_cli(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "sync":
        _run_sync_cli(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "intermittent":
        _run_intermittent_cli(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "research":
        _run_research_cli(sys.argv[2:])
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


def _run_sync_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Fetch and validate official draw results.")
    parser.add_argument("--start-round", type=int, default=FIRST_SUPPORTED_ROUND)
    parser.add_argument("--end-round", type=int)
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument("--output", default="data/official/draws.csv")
    parser.add_argument("--delay", type=float, default=0.08)
    args = parser.parse_args(argv)

    def report(completed: int, total: int) -> None:
        print(f"공식 회차 수집 {completed}/{total}", flush=True)

    dataset = fetch_official_dataset(
        start_round=args.start_round,
        end_round=args.end_round,
        start_date=args.start_date,
        end_date=args.end_date,
        delay_seconds=args.delay,
        progress=report,
    )
    csv_path, manifest_path = save_official_csv(dataset, Path(args.output))
    print(
        f"검증 상태={dataset.verification_status}, "
        f"회차={dataset.records[0].round_number}~{dataset.records[-1].round_number}, "
        f"CSV={csv_path}, manifest={manifest_path}"
    )


def _run_intermittent_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Search intermittent rules and hidden-state hypotheses."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", help="CSV draw history; reads provenance manifest when present")
    source.add_argument("--synthetic", type=int, metavar="DRAWS")
    parser.add_argument("--source-url")
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--crossover", type=float, default=0.65)
    parser.add_argument("--mutation", type=float, default=0.25)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--controls", type=int, default=20)
    parser.add_argument("--min-activation", type=float, default=0.1)
    parser.add_argument("--hidden-states", type=int, default=3)
    parser.add_argument("--feature-profile", choices=("legacy", "calendar_round"), default="legacy")
    parser.add_argument("--store", default="experiments")
    args = parser.parse_args(argv)

    dataset = (
        load_csv(args.csv, args.source_url)
        if args.csv
        else make_random_dataset(args.synthetic, args.seed)
    )
    config = IntermittentConfig(
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
        min_activation_rate=args.min_activation,
        hidden_states=args.hidden_states,
        feature_profile=args.feature_profile,
    )

    def report(event: dict[str, object]) -> None:
        print(
            f"[{event['run']}] 세대 {event['generation']}/{event['generations']} "
            f"검증 점수={float(event['validation_score']):.4f} "
            f"활성률={float(event['validation_coverage']):.3f} "
            f"조건 IF {event['gate']} THEN {event['formula']}",
            flush=True,
        )

    result = run_intermittent_experiment(
        dataset, config, ExperimentStore(args.store), report, PatternLibrary()
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _run_research_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Run checkpointed, repeated local pattern research experiments."
    )
    parser.add_argument("--csv", default="data/official/draws.csv")
    parser.add_argument("--duration-minutes", type=float, default=30)
    parser.add_argument("--checkpoint", default="checkpoints/research.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stop", action="store_true", help="request a stop at the next trial boundary"
    )
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--engine", action="append", choices=SUPPORTED_ENGINES)
    parser.add_argument("--feature-profile", choices=("legacy", "calendar_round"), default="legacy")
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--controls", type=int, default=3)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args(argv)
    if args.stop:
        print(f"Stop requested: {request_stop(args.checkpoint)}")
        return
    dataset = load_csv(args.csv)
    config = IntermittentConfig(
        population_size=args.population,
        generations=args.generations,
        workers=args.workers,
        max_depth=args.max_depth,
        control_runs=args.controls,
        seed=args.seed,
        feature_profile=args.feature_profile,
    )

    def report(event: dict[str, object]) -> None:
        detail = event.get("event", {})
        evaluated = config.population_size * (int(detail.get("generation", 0)) + 1)
        print(
            f"[{event.get('engine')}] run={event.get('run_index')} "
            f"generation={detail.get('generation')}/{detail.get('generations')} "
            f"candidates-evaluated≈{evaluated}",
            flush=True,
        )

    result = run_research_schedule(
        dataset,
        config,
        duration_seconds=args.duration_minutes * 60,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
        max_runs=args.max_runs,
        engines=tuple(args.engine or SUPPORTED_ENGINES),
        progress=report,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
