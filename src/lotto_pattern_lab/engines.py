"""Common research-engine contract and adapters for implemented discovery backends."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .dataset import DrawDataset
from .intermittent import IntermittentConfig
from .intermittent_experiment import run_intermittent_experiment
from .storage import ExperimentStore


@dataclass(frozen=True, slots=True)
class ResearchRunContext:
    dataset: DrawDataset
    config: IntermittentConfig
    store: ExperimentStore
    progress: Callable[[dict[str, Any]], None] | None = None


class ResearchEngine(Protocol):
    """Engines return a persisted experiment record under the shared run contract."""

    engine_id: str
    implemented: bool

    def run(self, context: ResearchRunContext) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class IntermittentGPEngine:
    """Adapter around the existing DEAP gated-rule and random-control experiment."""

    engine_id: str = "intermittent_gp"
    implemented: bool = True

    def run(self, context: ResearchRunContext) -> dict[str, Any]:
        return run_intermittent_experiment(
            context.dataset, context.config, context.store, context.progress
        )


ENGINE_REGISTRY: dict[str, ResearchEngine] = {"intermittent_gp": IntermittentGPEngine()}


def get_engine(engine_id: str) -> ResearchEngine:
    try:
        engine = ENGINE_REGISTRY[engine_id]
    except KeyError as error:
        raise ValueError(f"Research engine {engine_id!r} is not implemented.") from error
    if not engine.implemented:
        raise ValueError(f"Research engine {engine_id!r} is a reserved, unimplemented backend.")
    return engine
