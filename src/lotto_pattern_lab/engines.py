"""Common research-engine contract and adapters for implemented discovery backends."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .dataset import DrawDataset
from .intermittent import IntermittentConfig
from .intermittent_experiment import run_intermittent_experiment
from .pysr_backend import PySRConfig, run_pysr_experiment
from .storage import ExperimentStore

GP_ENGINE_ID = "intermittent_gp"
PYSR_ENGINE_ID = "pysr_symbolic_regression"


@dataclass(frozen=True, slots=True)
class ResearchRunContext:
    dataset: DrawDataset
    config: IntermittentConfig
    store: ExperimentStore
    progress: Callable[[dict[str, Any]], None] | None = None
    elite_seeds: tuple[dict[str, object], ...] = ()
    pysr_config: PySRConfig | None = None
    evaluation_cache_path: Path | None = None


class ResearchEngine(Protocol):
    """Engines return a persisted experiment record under the shared run contract."""

    engine_id: str
    implemented: bool

    def run(self, context: ResearchRunContext) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class IntermittentGPEngine:
    """Adapter around the existing DEAP gated-rule and random-control experiment."""

    engine_id: str = GP_ENGINE_ID
    implemented: bool = True

    def run(self, context: ResearchRunContext) -> dict[str, Any]:
        return run_intermittent_experiment(
            context.dataset,
            context.config,
            context.store,
            context.progress,
            elite_seeds=context.elite_seeds,
        )


@dataclass(frozen=True, slots=True)
class PySRSymbolicRegressionEngine:
    """Independent adapter that calls the actual optional PySR implementation."""

    engine_id: str = PYSR_ENGINE_ID
    implemented: bool = True

    def run(self, context: ResearchRunContext) -> dict[str, Any]:
        if context.pysr_config is None:
            raise ValueError("PySR settings must be supplied when selecting the PySR engine.")
        return run_pysr_experiment(
            context.dataset,
            context.pysr_config,
            context.store,
            context.progress,
            str(context.evaluation_cache_path) if context.evaluation_cache_path else None,
        )


ENGINE_REGISTRY: dict[str, ResearchEngine] = {
    "intermittent_gp": IntermittentGPEngine(),
    "pysr_symbolic_regression": PySRSymbolicRegressionEngine(),
}


def get_engine(engine_id: str) -> ResearchEngine:
    try:
        engine = ENGINE_REGISTRY[engine_id]
    except KeyError as error:
        raise ValueError(f"Research engine {engine_id!r} is not implemented.") from error
    if not engine.implemented:
        raise ValueError(f"Research engine {engine_id!r} is a reserved, unimplemented backend.")
    return engine
