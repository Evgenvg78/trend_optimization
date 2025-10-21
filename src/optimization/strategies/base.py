"""Base classes and interfaces for optimization search strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Protocol


class ParameterSampler(Protocol):
    """Protocol describing the minimal interface required from a parameter source."""

    def validate(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalise a parameter mapping."""

    def grid(self) -> Dict[str, tuple[Any, ...]]:
        """Return a mapping of parameter names to candidate values."""


@dataclass
class StrategyState:
    """Mutable state shared between strategy iterations."""

    iterations: int = 0
    max_iterations: int | None = None


class SearchStrategy(ABC):
    """Abstract base class for parameter search strategies."""

    def __init__(
        self,
        sampler: ParameterSampler,
        *,
        max_iterations: int | None = None,
        random_seed: int | None = None,
    ) -> None:
        self.sampler = sampler
        self.random_seed = random_seed
        self._initial_max_iterations = max_iterations
        self.state = StrategyState(iterations=0, max_iterations=max_iterations)

    @abstractmethod
    def generate(self) -> Iterator[dict[str, Any]]:
        """Produce an iterator over parameter combinations."""

    def reset(self) -> None:
        """Reset internal state to its initial values."""
        self.state = StrategyState(iterations=0, max_iterations=self._initial_max_iterations)

    def __iter__(self) -> Iterable[dict[str, Any]]:
        """Allow strategies to be used directly in for-loops."""
        self.reset()
        return self.generate()
