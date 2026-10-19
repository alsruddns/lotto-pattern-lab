"""DEAP genetic programming search for one-number set membership rules."""

import math
import multiprocessing
import operator
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial

from deap import base, creator, gp, tools

from .backtest import DataSplits
from .features import Example

_EVALUATION_EXAMPLES: tuple[Example, ...] = ()
_MAX_DEPTH = 6
_MAX_NODES = 35


def _bounded(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(-1_000_000.0, min(1_000_000.0, value))


def _safe_div(left: float, right: float) -> float:
    return _bounded(left / right) if abs(right) > 1e-9 else _bounded(left)


def _safe_mod(left: float, right: float) -> float:
    return _bounded(left % right) if abs(right) > 1e-9 else 0.0


def _square(value: float) -> float:
    return _bounded(max(-1000.0, min(1000.0, value)) ** 2)


def _cube(value: float) -> float:
    return _bounded(max(-100.0, min(100.0, value)) ** 3)


def _ephemeral_constant() -> float:
    return random.uniform(-10.0, 10.0)


def _make_primitive_set(feature_count: int) -> gp.PrimitiveSet:
    pset = gp.PrimitiveSet("MAIN", feature_count)
    pset.renameArguments(**{f"ARG{i}": f"x{i}" for i in range(feature_count)})
    pset.addPrimitive(operator.add, 2, name="add")
    pset.addPrimitive(operator.sub, 2, name="sub")
    pset.addPrimitive(operator.mul, 2, name="mul")
    pset.addPrimitive(_safe_div, 2, name="safe_div")
    pset.addPrimitive(_safe_mod, 2, name="safe_mod")
    pset.addPrimitive(abs, 1, name="abs")
    pset.addPrimitive(min, 2, name="min2")
    pset.addPrimitive(max, 2, name="max2")
    pset.addPrimitive(_square, 1, name="square")
    pset.addPrimitive(_cube, 1, name="cube")
    pset.addEphemeralConstant("constant", _ephemeral_constant)
    return pset


def _compile_expression(
    expression: gp.PrimitiveTree, pset: gp.PrimitiveSet
) -> Callable[..., float]:
    return gp.compile(expression, pset)


def _predict(function: Callable[..., float], features: tuple[float, ...]) -> int:
    try:
        output = float(function(*features))
    except (ArithmeticError, OverflowError, ValueError):
        output = 0.0
    if not math.isfinite(output):
        output = 0.0
    return int(abs(output)) % 45 + 1


def _score(function: Callable[..., float], examples: tuple[Example, ...]) -> float:
    if not examples:
        return 0.0
    return sum(
        _predict(function, example.features) in example.winning_numbers for example in examples
    ) / len(examples)


def _worker_init(examples: tuple[Example, ...], pset: gp.PrimitiveSet, max_depth: int) -> None:
    global _EVALUATION_EXAMPLES, _WORKER_PSET, _MAX_DEPTH
    _EVALUATION_EXAMPLES = examples
    _WORKER_PSET = pset
    _MAX_DEPTH = max_depth


def _evaluate_individual(individual: gp.PrimitiveTree) -> tuple[float]:
    function = _compile_expression(individual, _WORKER_PSET)
    score = _score(function, _EVALUATION_EXAMPLES)
    complexity_penalty = 0.0005 * len(individual) + 0.002 * individual.height
    return (score - complexity_penalty,)


def _metrics(
    individual: gp.PrimitiveTree, examples: tuple[Example, ...], pset: gp.PrimitiveSet
) -> float:
    return _score(_compile_expression(individual, pset), examples)


@dataclass(frozen=True, slots=True)
class SearchConfig:
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
    control_runs: int = 3

    def __post_init__(self) -> None:
        if self.population_size < 4 or self.generations < 1:
            raise ValueError("Population size must be >= 4 and generations must be positive.")
        if not 0 <= self.crossover_probability <= 1 or not 0 <= self.mutation_probability <= 1:
            raise ValueError("Crossover and mutation probabilities must be between 0 and 1.")
        if not 2 <= self.max_depth <= 10:
            raise ValueError("Maximum depth must be between 2 and 10.")
        if not 1 <= self.workers <= 4:
            raise ValueError("Worker count must be between 1 and 4.")
        if not 1 <= self.max_lag <= 10:
            raise ValueError("Maximum lag must be between 1 and 10 draws.")
        if self.control_runs < 0:
            raise ValueError("Control run count cannot be negative.")


@dataclass(frozen=True, slots=True)
class FormulaCandidate:
    formula: str
    training_accuracy: float
    validation_accuracy: float
    test_accuracy: float | None
    training_fitness: float
    complexity: int
    height: int


@dataclass(frozen=True, slots=True)
class SearchResult:
    formula: str
    training_accuracy: float
    validation_accuracy: float
    test_accuracy: float
    training_fitness: float
    complexity: int
    height: int
    train_rounds: tuple[int, ...]
    validation_rounds: tuple[int, ...]
    test_rounds: tuple[int, ...]
    candidates: tuple[FormulaCandidate, ...]
    random_baseline: float = 6 / 45


def run_search(
    splits: DataSplits,
    config: SearchConfig,
    progress: Callable[[dict[str, int | float | str]], None] | None = None,
    run_label: str = "actual",
) -> SearchResult:
    """Run GP; validation selects the formula while test is evaluated only after selection."""
    global _EVALUATION_EXAMPLES, _WORKER_PSET, _MAX_DEPTH
    random.seed(config.seed)
    feature_count = len(splits.train[0].features)
    pset = _make_primitive_set(feature_count)
    if not hasattr(creator, "LottoFitnessMax"):
        creator.create("LottoFitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "LottoIndividual"):
        creator.create("LottoIndividual", gp.PrimitiveTree, fitness=creator.LottoFitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=min(3, config.max_depth))
    toolbox.register("individual", tools.initIterate, creator.LottoIndividual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", _compile_expression, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register(
        "mutate",
        gp.mutUniform,
        expr=partial(gp.genFull, min_=0, max_=2),
        pset=pset,
    )
    toolbox.decorate(
        "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=config.max_depth)
    )
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=_MAX_NODES))
    toolbox.decorate(
        "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=config.max_depth)
    )
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=_MAX_NODES))

    _WORKER_PSET = pset
    _EVALUATION_EXAMPLES = splits.train
    _MAX_DEPTH = config.max_depth
    pool = None
    if config.workers > 1:
        pool = multiprocessing.get_context("spawn").Pool(
            config.workers,
            initializer=_worker_init,
            initargs=(splits.train, pset, config.max_depth),
        )
        toolbox.register("map", pool.map)
    else:
        toolbox.register("map", map)
    toolbox.register("evaluate", _evaluate_individual)

    population = toolbox.population(n=config.population_size)
    hall_of_fame = tools.HallOfFame(1)
    validation_archive: dict[str, tuple[float, gp.PrimitiveTree, float, float, float]] = {}
    try:
        for generation in range(config.generations + 1):
            invalid = [individual for individual in population if not individual.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid)
            for individual, fitness in zip(invalid, fitnesses, strict=True):
                individual.fitness.values = fitness
            hall_of_fame.update(population)
            generation_best = max(population, key=lambda individual: individual.fitness.values[0])
            validation_score = _metrics(generation_best, splits.validation, pset)
            for individual in population:
                candidate_validation = _metrics(individual, splits.validation, pset)
                objective = (
                    candidate_validation - 0.0005 * len(individual) - 0.002 * individual.height
                )
                formula = str(individual)
                existing = validation_archive.get(formula)
                if existing is None or objective > existing[0]:
                    validation_archive[formula] = (
                        objective,
                        creator.LottoIndividual(individual),
                        candidate_validation,
                        _metrics(individual, splits.train, pset),
                        individual.fitness.values[0],
                    )
            if progress:
                progress(
                    {
                        "run": run_label,
                        "generation": generation,
                        "generations": config.generations,
                        "train_accuracy": _metrics(generation_best, splits.train, pset),
                        "validation_accuracy": validation_score,
                        "formula": str(generation_best),
                    }
                )
            if generation == config.generations:
                break
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))
            for first, second in zip(offspring[::2], offspring[1::2], strict=False):
                if random.random() < config.crossover_probability:
                    toolbox.mate(first, second)
                    del first.fitness.values, second.fitness.values
            for mutant in offspring:
                if random.random() < config.mutation_probability:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            population = offspring
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    ranked = sorted(validation_archive.values(), key=lambda item: item[0], reverse=True)[:10]
    assert ranked
    selected = ranked[0][1]
    candidates = tuple(
        FormulaCandidate(
            formula=str(individual),
            training_accuracy=training_accuracy,
            validation_accuracy=validation_accuracy,
            test_accuracy=_metrics(individual, splits.test, pset) if index == 0 else None,
            training_fitness=training_fitness,
            complexity=len(individual),
            height=individual.height,
        )
        for index, (
            _,
            individual,
            validation_accuracy,
            training_accuracy,
            training_fitness,
        ) in enumerate(ranked)
    )
    return SearchResult(
        formula=str(selected),
        training_accuracy=_metrics(selected, splits.train, pset),
        validation_accuracy=ranked[0][2],
        test_accuracy=candidates[0].test_accuracy or 0.0,
        training_fitness=ranked[0][4],
        complexity=len(selected),
        height=selected.height,
        train_rounds=tuple(example.round_number for example in splits.train),
        validation_rounds=tuple(example.round_number for example in splits.validation),
        test_rounds=tuple(example.round_number for example in splits.test),
        candidates=candidates,
    )


def config_to_dict(config: SearchConfig) -> dict[str, int | float]:
    return asdict(config)
