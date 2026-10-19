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
from .gp_config import GPConfig

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


def _safe_integer(value: float) -> int:
    """Convert to a bounded signed 32-bit integer for structure operations."""
    if not math.isfinite(value):
        return 0
    return max(-(2**31), min(2**31 - 1, int(value)))


def _bit_xor(left: float, right: float) -> float:
    return float((_safe_integer(left) & 0xFFFFFFFF) ^ (_safe_integer(right) & 0xFFFFFFFF))


def _bit_and(left: float, right: float) -> float:
    return float((_safe_integer(left) & 0xFFFFFFFF) & (_safe_integer(right) & 0xFFFFFFFF))


def _bit_or(left: float, right: float) -> float:
    return float((_safe_integer(left) & 0xFFFFFFFF) | (_safe_integer(right) & 0xFFFFFFFF))


def _shift_left(value: float, places: float) -> float:
    shift = max(0, min(31, _safe_integer(places)))
    return float(((_safe_integer(value) & 0xFFFFFFFF) << shift) & 0xFFFFFFFF)


def _shift_right(value: float, places: float) -> float:
    shift = max(0, min(31, _safe_integer(places)))
    return float((_safe_integer(value) & 0xFFFFFFFF) >> shift)


def _rotate_left(value: float, places: float) -> float:
    shift = _safe_integer(places) % 32
    integer = _safe_integer(value) & 0xFFFFFFFF
    if shift == 0:
        return float(integer)
    return float(((integer << shift) | (integer >> (32 - shift))) & 0xFFFFFFFF)


def _rotate_right(value: float, places: float) -> float:
    shift = _safe_integer(places) % 32
    integer = _safe_integer(value) & 0xFFFFFFFF
    if shift == 0:
        return float(integer)
    return float(((integer >> shift) | (integer << (32 - shift))) & 0xFFFFFFFF)


def _concat_digits(left: float, right: float) -> float:
    first = abs(_safe_integer(left)) % 1_000_000
    second = abs(_safe_integer(right))
    width = len(str(second))
    return float((first * (10**width) + second) % (2**32))


def _digit_sum(value: float) -> float:
    return float(sum(int(character) for character in str(abs(_safe_integer(value)))))


def _reverse_digits(value: float) -> float:
    return float(int(str(abs(_safe_integer(value)))[::-1]))


def _digital_root(value: float) -> float:
    integer = abs(_safe_integer(value))
    return float(0 if integer == 0 else 1 + (integer - 1) % 9)


def _is_prime(value: float) -> float:
    """Deterministic Miller-Rabin primality test for the supported 32-bit range."""
    number = abs(_safe_integer(value))
    if number < 2:
        return 0.0
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if number == prime:
            return 1.0
        if number % prime == 0:
            return 0.0
    odd_part = number - 1
    power = 0
    while odd_part % 2 == 0:
        odd_part //= 2
        power += 1
    for witness in (2, 3, 5, 7, 11):
        residue = pow(witness, odd_part, number)
        if residue in (1, number - 1):
            continue
        if not any(pow(residue, 2**step, number) == number - 1 for step in range(power - 1)):
            return 0.0
    return 1.0


def _factor_count(value: float) -> float:
    """Count divisors up to one million; larger inputs are explicitly clipped."""
    number = min(abs(_safe_integer(value)), 1_000_000)
    if number == 0:
        return 0.0
    count = 0
    divisor = 1
    while divisor * divisor <= number:
        if number % divisor == 0:
            count += 1 if divisor * divisor == number else 2
        divisor += 1
    return float(count)


def _fibonacci_index(value: float) -> float:
    """Return the exact Fibonacci index, or zero when the bounded integer is not Fibonacci."""
    number = abs(_safe_integer(value))
    previous, current, index = 0, 1, 0
    while previous < number:
        previous, current = current, previous + current
        index += 1
    return float(index if previous == number else 0)


def _ephemeral_constant() -> float:
    return random.uniform(-10.0, 10.0)


def _make_primitive_set(feature_count: int) -> gp.PrimitiveSetTyped:
    """Build a float-only grammar, including typed float terminals for elite literals."""
    if feature_count < 1:
        raise ValueError("A GP primitive set requires at least one input feature.")
    pset = gp.PrimitiveSetTyped("MAIN", [float] * feature_count, float)
    pset.renameArguments(**{f"ARG{i}": f"x{i}" for i in range(feature_count)})
    for function, arity, name in (
        (operator.add, 2, "add"),
        (operator.sub, 2, "sub"),
        (operator.mul, 2, "mul"),
        (_safe_div, 2, "safe_div"),
        (_safe_mod, 2, "safe_mod"),
        (abs, 1, "abs"),
        (min, 2, "min2"),
        (max, 2, "max2"),
        (_square, 1, "square"),
        (_cube, 1, "cube"),
        (_bit_xor, 2, "bit_xor"),
        (_bit_and, 2, "bit_and"),
        (_bit_or, 2, "bit_or"),
        (_shift_left, 2, "shift_left"),
        (_shift_right, 2, "shift_right"),
        (_rotate_left, 2, "rotate_left"),
        (_rotate_right, 2, "rotate_right"),
        (_concat_digits, 2, "concat_digits"),
        (_digit_sum, 1, "digit_sum"),
        (_reverse_digits, 1, "reverse_digits"),
        (_digital_root, 1, "digital_root"),
        (_is_prime, 1, "is_prime"),
        (_factor_count, 1, "factor_count"),
        (_fibonacci_index, 1, "fibonacci_index"),
    ):
        pset.addPrimitive(function, [float] * arity, float, name=name)
    pset.addEphemeralConstant("constant", _ephemeral_constant, float)
    validate_primitive_set(pset)
    return pset


def validate_primitive_set(pset: gp.PrimitiveSetTyped) -> None:
    """Fail before search if any supported return type cannot reach a terminal."""
    required_types = {pset.ret, *(argument for argument in pset.ins)}
    required_types.update(
        primitive.ret
        for values in pset.primitives.values()
        for primitive in values
    )
    required_types.update(
        argument
        for values in pset.primitives.values()
        for primitive in values
        for argument in primitive.args
    )
    missing = [
        value_type.__name__
        for value_type in required_types
        if not pset.terminals.get(value_type)
    ]
    if missing:
        raise ValueError(
            "GP primitive set has no terminal for required type(s): " + ", ".join(sorted(missing))
        )


def _compile_expression(
    expression: gp.PrimitiveTree, pset: gp.PrimitiveSetTyped
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


def _worker_init(
    examples: tuple[Example, ...], pset: gp.PrimitiveSetTyped, max_depth: int
) -> None:
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
    individual: gp.PrimitiveTree,
    examples: tuple[Example, ...],
    pset: gp.PrimitiveSetTyped,
) -> float:
    return _score(_compile_expression(individual, pset), examples)


@dataclass(frozen=True, slots=True)
class SearchConfig(GPConfig):
    control_runs: int = 3

    def __post_init__(self) -> None:
        GPConfig.__post_init__(self)
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
