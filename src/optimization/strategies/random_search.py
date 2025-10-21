"""Random search strategy with reproducible sampling via seed."""

from __future__ import annotations

import random
from typing import Any, Dict, Iterator, Mapping, Sequence

from .base import ParameterSampler, SearchStrategy


class RandomSearchStrategy(SearchStrategy):
    """Sample parameter combinations uniformly at random."""

    def __init__(
        self,
        sampler: ParameterSampler,
        bounds: Mapping[str, Sequence[Any]] | None = None,
        *,
        max_iterations: int | None = None,
        random_seed: int | None = None,
    ) -> None:
        super().__init__(sampler, max_iterations=max_iterations, random_seed=random_seed)
        self._bounds_override = {name: tuple(values) for name, values in (bounds or {}).items()}
        self._rng = random.Random(random_seed)

    def reset(self) -> None:
        super().reset()
        self._rng = random.Random(self.random_seed)

    def generate(self) -> Iterator[dict[str, Any]]:
        domain = self._resolve_domain()
        if not domain:
            raise ValueError("RandomSearchStrategy requires non-empty parameter bounds")

        keys = list(domain.keys())
        limit = self.state.max_iterations
        iterations = 0

        while limit is None or iterations < limit:
            candidate: dict[str, Any] = {}
            for name in keys:
                values = domain[name]
                if not values:
                    raise ValueError(f"Candidate set for parameter {name} is empty")
                candidate[name] = self._rng.choice(values)
            iterations += 1
            self.state.iterations = iterations
            yield self.sampler.validate(candidate)

    def _resolve_domain(self) -> Dict[str, tuple[Any, ...]]:
        if self._bounds_override:
            return {name: tuple(values) for name, values in self._bounds_override.items()}
        return self.sampler.grid()
