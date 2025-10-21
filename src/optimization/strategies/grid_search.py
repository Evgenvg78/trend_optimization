"""Deterministic grid-search strategy."""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, Iterator, Mapping, Sequence

from .base import ParameterSampler, SearchStrategy


class GridSearchStrategy(SearchStrategy):
    """Systematically traverse the full cartesian product of the parameter space."""

    def __init__(
        self,
        sampler: ParameterSampler,
        grid: Mapping[str, Sequence[Any]] | None = None,
        *,
        random_seed: int | None = None,
    ) -> None:
        super().__init__(sampler, max_iterations=None, random_seed=random_seed)
        self._grid_override = {name: tuple(values) for name, values in (grid or {}).items()}

    def generate(self) -> Iterator[dict[str, Any]]:
        grid_definition = self._resolve_grid()
        if not grid_definition:
            raise ValueError("GridSearchStrategy requires at least one parameter to explore")

        keys = list(grid_definition.keys())
        for combination in product(*(grid_definition[key] for key in keys)):
            candidate = dict(zip(keys, combination))
            self.state.iterations += 1
            yield self.sampler.validate(candidate)

    def _resolve_grid(self) -> Dict[str, tuple[Any, ...]]:
        if self._grid_override:
            return {name: tuple(values) for name, values in self._grid_override.items()}
        return self.sampler.grid()
