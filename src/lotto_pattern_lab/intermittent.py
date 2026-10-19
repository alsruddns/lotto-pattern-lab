"""DEAP search for gated, intermittent one-number and unordered six-number rules."""

import math
import multiprocessing
import random
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial

from deap import base, creator, gp, tools

from .backtest import DataSplits
from .features import Example
from .gp_engine import _make_primitive_set, _predict

_WORKER_EXAMPLES: tuple[Example, ...] = ()
_WORKER_PSET: gp.PrimitiveSet | None = None
_WORKER_OUTPUTS = 1
_RULE_COUNT = 2
_WORKER_MIN_COVERAGE = 0.1


@dataclass(frozen=True, slots=True)
class IntermittentConfig:
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
    control_runs: int = 20
    min_activation_rate: float = 0.1
    top_candidates: int = 10
    hidden_states: int = 3

    def __post_init__(self) -> None:
        if self.population_size < 8 or self.generations < 1:
            raise ValueError("Population must be >= 8 and generations must be positive.")
        if not 0 <= self.crossover_probability <= 1 or not 0 <= self.mutation_probability <= 1:
            raise ValueError("Crossover and mutation probabilities must be between 0 and 1.")
        if not 2 <= self.max_depth <= 10 or not 1 <= self.workers <= 4:
            raise ValueError("Maximum depth must be 2..10 and workers must be 1..4.")
        if not 1 <= self.max_lag <= 10:
            raise ValueError("Maximum lag must be 1..10.")
        if not 0.05 <= self.min_activation_rate <= 1:
            raise ValueError("Minimum activation rate must be between 0.05 and 1.")
        if self.control_runs < 0 or self.top_candidates < 1:
            raise ValueError(
                "Control runs cannot be negative and at least one candidate is required."
            )
        if not 2 <= self.hidden_states <= 5:
            raise ValueError("Hypothetical hidden state count must be between 2 and 5.")
        if self.train_ratio <= 0 or self.validation_ratio <= 0:
            raise ValueError("Train and validation ratios must be positive.")
        if self.train_ratio + self.validation_ratio >= 1:
            raise ValueError("Train plus validation ratios must be less than 1.")


@dataclass(frozen=True, slots=True)
class IntermittentCandidate:
    activation_condition: str
    formulas: tuple[str, ...]
    rule_count: int
    outputs_per_event: int
    training_activation_rate: float
    validation_activation_rate: float
    test_activation_rate: float | None
    training_hit_rate: float
    validation_hit_rate: float
    test_hit_rate: float | None
    training_hit_count: int
    validation_hit_count: int
    test_hit_count: int | None
    training_score: float
    validation_score: float
    test_score: float | None
    complexity: int
    height: int
    activation_events: tuple[dict[str, object], ...] = ()
    active_gaps: tuple[int, ...] = ()
    rule_active_rounds: tuple[tuple[int, ...], ...] = ()
    rule_active_gaps: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class IntermittentSearchResult:
    outputs_per_event: int
    selected: IntermittentCandidate
    candidates: tuple[IntermittentCandidate, ...]
    train_rounds: tuple[int, ...]
    validation_rounds: tuple[int, ...]
    test_rounds: tuple[int, ...]


def _new_tree(pset: gp.PrimitiveSet, max_depth: int) -> gp.PrimitiveTree:
    nodes = gp.genHalfAndHalf(pset=pset, min_=1, max_=min(3, max_depth))
    return gp.PrimitiveTree(nodes)


def _new_individual(pset: gp.PrimitiveSet, outputs: int, max_depth: int):
    tree_count = outputs * _RULE_COUNT + 1
    return creator.IntermittentIndividual([_new_tree(pset, max_depth) for _ in range(tree_count)])


def _gate_rule(function: Callable[..., float], features: tuple[float, ...]) -> int:
    try:
        value = float(function(*features))
    except (ArithmeticError, OverflowError, ValueError):
        return 0
    if not math.isfinite(value):
        return 0
    return int(abs(value)) % (_RULE_COUNT + 1)


def _predict_numbers(
    functions: tuple[Callable[..., float], ...], features: tuple[float, ...]
) -> tuple[int, ...]:
    predictions: list[int] = []
    for function in functions:
        number = _predict(function, features)
        while number in predictions:
            number = number % 45 + 1
        predictions.append(number)
    return tuple(predictions)


def _measure(
    individual,
    examples: tuple[Example, ...],
    pset: gp.PrimitiveSet,
    outputs: int,
    min_activation_rate: float,
) -> dict[str, object]:
    gate = gp.compile(individual[0], pset)
    predictors = tuple(gp.compile(tree, pset) for tree in individual[1:])
    active_rounds: list[int] = []
    rule_rounds: list[list[int]] = [[] for _ in range(_RULE_COUNT)]
    events: list[dict[str, object]] = []
    hits = 0
    exact_matches = 0
    for example in examples:
        rule_index = _gate_rule(gate, example.features)
        active = rule_index > 0
        start = (rule_index - 1) * outputs
        rule_predictors = predictors[start : start + outputs] if active else ()
        predictions = _predict_numbers(rule_predictors, example.features) if active else ()
        hit_count = len(set(predictions) & example.winning_numbers)
        if active:
            active_rounds.append(example.round_number)
            rule_rounds[rule_index - 1].append(example.round_number)
            hits += hit_count
            exact_matches += int(set(predictions) == example.winning_numbers)
        events.append(
            {
                "round_number": example.round_number,
                "active": active,
                "rule_index": rule_index if active else None,
                "predicted_numbers": list(predictions),
                "hit_count": hit_count if active else None,
            }
        )
    active_count = len(active_rounds)
    total = len(examples)
    activation_rate = active_count / total if total else 0.0
    hit_rate = hits / (active_count * outputs) if active_count else 0.0
    random_hit_rate = 6 / 45
    score = (hits - active_count * outputs * random_hit_rate) / max(total * outputs, 1)
    if activation_rate < min_activation_rate:
        score = -1.0 - (min_activation_rate - activation_rate)
    gaps = tuple(
        right - left for left, right in zip(active_rounds, active_rounds[1:], strict=False)
    )
    return {
        "activation_rate": activation_rate,
        "hit_rate": hit_rate,
        "hit_count": hits,
        "exact_matches": exact_matches,
        "score": score,
        "active_rounds": active_rounds,
        "events": events,
        "gaps": gaps,
        "rule_rounds": rule_rounds,
        "rule_gaps": [
            [right - left for left, right in zip(rounds, rounds[1:], strict=False)]
            for rounds in rule_rounds
        ],
    }


def _fitness(individual) -> tuple[float]:
    measured = _measure(
        individual, _WORKER_EXAMPLES, _WORKER_PSET, _WORKER_OUTPUTS, _WORKER_MIN_COVERAGE
    )
    nodes = sum(len(tree) for tree in individual)
    height = max(tree.height for tree in individual)
    active_count = max(int(measured["activation_rate"] * len(_WORKER_EXAMPLES)), 1)
    # A shrinking penalty discourages rules supported by only a few active examples.
    sample_penalty = 0.01 / math.sqrt(active_count)
    complexity_penalty = 0.0005 * nodes + 0.002 * height
    return (float(measured["score"]) - sample_penalty - complexity_penalty,)


def _worker_init(
    examples: tuple[Example, ...],
    pset: gp.PrimitiveSet,
    outputs: int,
    min_activation_rate: float,
) -> None:
    global _WORKER_EXAMPLES, _WORKER_PSET, _WORKER_OUTPUTS, _WORKER_MIN_COVERAGE
    _WORKER_EXAMPLES = examples
    _WORKER_PSET = pset
    _WORKER_OUTPUTS = outputs
    _WORKER_MIN_COVERAGE = min_activation_rate


def _mate_individuals(first, second, max_depth: int, max_nodes: int):
    for index, (left, right) in enumerate(zip(first, second, strict=True)):
        original_left, original_right = deepcopy(left), deepcopy(right)
        gp.cxOnePoint(left, right)
        if left.height > max_depth or len(left) > max_nodes:
            first[index] = original_left
        if right.height > max_depth or len(right) > max_nodes:
            second[index] = original_right
    return first, second


def _mutate_individual(individual, pset: gp.PrimitiveSet, max_depth: int, max_nodes: int):
    index = random.randrange(len(individual))
    original = deepcopy(individual[index])
    (mutated,) = gp.mutUniform(
        individual[index],
        expr=partial(gp.genFull, min_=0, max_=2),
        pset=pset,
    )
    if mutated.height <= max_depth and len(mutated) <= max_nodes:
        individual[index] = mutated
    else:
        individual[index] = original
    return (individual,)


def _candidate_key(individual) -> str:
    return "IF[" + str(individual[0]) + "] THEN[" + ";".join(map(str, individual[1:])) + "]"


def _make_candidate(
    individual,
    splits: DataSplits,
    all_examples: tuple[Example, ...],
    pset,
    outputs,
    min_rate,
    include_test: bool,
):
    train = _measure(individual, splits.train, pset, outputs, min_rate)
    validation = _measure(individual, splits.validation, pset, outputs, min_rate)
    test = _measure(individual, splits.test, pset, outputs, min_rate) if include_test else None
    report_examples = all_examples if include_test else splits.train + splits.validation
    all_metrics = _measure(individual, report_examples, pset, outputs, min_rate)
    return IntermittentCandidate(
        activation_condition=str(individual[0]),
        formulas=tuple(str(tree) for tree in individual[1:]),
        rule_count=_RULE_COUNT,
        outputs_per_event=outputs,
        training_activation_rate=float(train["activation_rate"]),
        validation_activation_rate=float(validation["activation_rate"]),
        test_activation_rate=float(test["activation_rate"]) if test is not None else None,
        training_hit_rate=float(train["hit_rate"]),
        validation_hit_rate=float(validation["hit_rate"]),
        test_hit_rate=float(test["hit_rate"]) if test is not None else None,
        training_hit_count=int(train["hit_count"]),
        validation_hit_count=int(validation["hit_count"]),
        test_hit_count=int(test["hit_count"]) if test is not None else None,
        training_score=float(train["score"]),
        validation_score=float(validation["score"]),
        test_score=float(test["score"]) if test is not None else None,
        complexity=sum(len(tree) for tree in individual),
        height=max(tree.height for tree in individual),
        activation_events=tuple(all_metrics["events"]),
        active_gaps=tuple(int(gap) for gap in all_metrics["gaps"]),
        rule_active_rounds=tuple(
            tuple(int(round_number) for round_number in rounds)
            for rounds in all_metrics["rule_rounds"]
        ),
        rule_active_gaps=tuple(
            tuple(int(gap) for gap in gaps) for gaps in all_metrics["rule_gaps"]
        ),
    )


def run_intermittent_search(
    splits: DataSplits,
    config: IntermittentConfig,
    outputs_per_event: int = 1,
    progress: Callable[[dict[str, int | float | str]], None] | None = None,
    run_label: str = "actual",
) -> IntermittentSearchResult:
    """Search GP activation and outcome formulas; validation ranks all candidates."""
    if outputs_per_event not in {1, 6}:
        raise ValueError("Intermittent experiments support one-number or six-number-set output.")
    global _WORKER_EXAMPLES, _WORKER_PSET, _WORKER_OUTPUTS, _WORKER_MIN_COVERAGE
    random.seed(config.seed)
    pset = _make_primitive_set(len(splits.train[0].features))
    if not hasattr(creator, "IntermittentFitnessMax"):
        creator.create("IntermittentFitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "IntermittentIndividual"):
        creator.create("IntermittentIndividual", list, fitness=creator.IntermittentFitnessMax)
    toolbox = base.Toolbox()
    toolbox.register(
        "individual",
        _new_individual,
        pset=pset,
        outputs=outputs_per_event,
        max_depth=config.max_depth,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", _mate_individuals, max_depth=config.max_depth, max_nodes=35)
    toolbox.register(
        "mutate", _mutate_individual, pset=pset, max_depth=config.max_depth, max_nodes=35
    )
    _WORKER_EXAMPLES = splits.train
    _WORKER_PSET = pset
    _WORKER_OUTPUTS = outputs_per_event
    _WORKER_MIN_COVERAGE = config.min_activation_rate
    pool = None
    if config.workers > 1:
        pool = multiprocessing.get_context("spawn").Pool(
            config.workers,
            initializer=_worker_init,
            initargs=(splits.train, pset, outputs_per_event, config.min_activation_rate),
        )
        toolbox.register("map", pool.map)
    else:
        toolbox.register("map", map)
    toolbox.register("evaluate", _fitness)

    population = toolbox.population(n=config.population_size)
    archive: dict[str, tuple[float, object]] = {}
    try:
        for generation in range(config.generations + 1):
            invalid = [individual for individual in population if not individual.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid)
            for individual, fitness in zip(invalid, fitnesses, strict=True):
                individual.fitness.values = fitness
            for individual in population:
                validation = _measure(
                    individual,
                    splits.validation,
                    pset,
                    outputs_per_event,
                    config.min_activation_rate,
                )
                validation_score = float(validation["score"])
                validation_count = int(validation["activation_rate"] * len(splits.validation))
                penalty = 0.0005 * sum(len(tree) for tree in individual)
                penalty += 0.002 * max(tree.height for tree in individual)
                if validation_count < math.ceil(
                    config.min_activation_rate * len(splits.validation)
                ):
                    penalty += 1.0
                objective = validation_score - penalty
                key = _candidate_key(individual)
                if key not in archive or objective > archive[key][0]:
                    archive[key] = (objective, creator.IntermittentIndividual(deepcopy(individual)))
            generation_best = max(population, key=lambda individual: individual.fitness.values[0])
            if progress:
                current = _measure(
                    generation_best,
                    splits.validation,
                    pset,
                    outputs_per_event,
                    config.min_activation_rate,
                )
                progress(
                    {
                        "run": run_label,
                        "generation": generation,
                        "generations": config.generations,
                        "train_fitness": generation_best.fitness.values[0],
                        "validation_score": current["score"],
                        "validation_coverage": current["activation_rate"],
                        "gate": str(generation_best[0]),
                        "formula": ";".join(map(str, generation_best[1:])),
                    }
                )
            if generation == config.generations:
                break
            offspring = list(map(toolbox.clone, toolbox.select(population, len(population))))
            for first, second in zip(offspring[::2], offspring[1::2], strict=False):
                if random.random() < config.crossover_probability:
                    toolbox.mate(first, second)
                    del first.fitness.values, second.fitness.values
            for individual in offspring:
                if random.random() < config.mutation_probability:
                    toolbox.mutate(individual)
                    del individual.fitness.values
            population = offspring
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    ranked = sorted(archive.values(), key=lambda item: item[0], reverse=True)[
        : config.top_candidates
    ]
    if not ranked:
        raise RuntimeError("GP did not produce a candidate rule.")
    all_examples = splits.train + splits.validation + splits.test
    candidates = tuple(
        _make_candidate(
            individual,
            splits,
            all_examples,
            pset,
            outputs_per_event,
            config.min_activation_rate,
            include_test,
        )
        for index, (_, individual) in enumerate(ranked)
        for include_test in (index == 0,)
    )
    return IntermittentSearchResult(
        outputs_per_event,
        candidates[0],
        candidates,
        tuple(example.round_number for example in splits.train),
        tuple(example.round_number for example in splits.validation),
        tuple(example.round_number for example in splits.test),
    )


def config_to_dict(config: IntermittentConfig) -> dict[str, int | float]:
    return asdict(config)
