"""Каркас модуля оптимизации параметров."""

from .executor import OptimizationExecutor
from .parameter_space import ParameterDefinition, ParameterSpace
from .reporting import OptimizationReporter
from .results_store import InMemoryResultsStore, OptimizationResult, ResultsStoreProtocol
from .strategies.base import SearchStrategy
from .strategies.grid_search import GridSearchStrategy
from .strategies.random_search import RandomSearchStrategy

__all__ = [
    "OptimizationExecutor",
    "OptimizationReporter",
    "InMemoryResultsStore",
    "OptimizationResult",
    "ParameterDefinition",
    "ParameterSpace",
    "ResultsStoreProtocol",
    "SearchStrategy",
    "GridSearchStrategy",
    "RandomSearchStrategy",
]

