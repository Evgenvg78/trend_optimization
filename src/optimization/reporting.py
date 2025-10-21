"""Reporting utilities for optimisation runs."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, Mapping, Sequence

from .results_store import OptimizationResult, ResultsStoreProtocol


@dataclass
class OptimizationReporter:
    """Produce tabular and aggregated views of optimisation results."""

    store: ResultsStoreProtocol

    def to_table(self, metrics: Iterable[str] | None = None) -> Any:
        """
        Return a pandas DataFrame when pandas is available, otherwise a plain dict.

        Args:
            metrics: Optional collection of metric names to include. Defaults to key metric + drawdown + sharpe.
        """
        metrics = tuple(metrics) if metrics is not None else ("total_return", "max_drawdown", "sharpe")
        rows: list[dict[str, Any]] = []
        for result in self.store:
            row = {}
            row.update({f"param_{key}": value for key, value in result.parameters.items()})
            for metric in metrics:
                row[f"metric_{metric}"] = result.metrics.get(metric)
            row["error"] = result.error
            rows.append(row)

        try:  # pragma: no cover - optional dependency
            import pandas as pd

            return pd.DataFrame(rows)
        except Exception:
            return {"rows": rows, "metrics": list(metrics)}

    def summary(
        self,
        *,
        metrics: Sequence[str] | None = None,
        top_n: int = 5,
        filters: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Return aggregated statistics: top-N, averages, min/max per metric."""
        metrics = tuple(metrics) if metrics is not None else (self.store.key_metric if hasattr(self.store, "key_metric") else "total_return",)

        filtered = _filter_results(list(self.store), filters)
        if not filtered:
            return {"top_n": [], "averages": {}, "distributions": {}}

        top_entries = self.store.top_n(top_n, filters=filters)
        top_payload = [self._result_payload(result, metrics) for result in top_entries]

        averages: dict[str, float] = {}
        distributions: dict[str, Dict[str, float]] = {}

        for metric in metrics:
            metric_values = [result.metrics.get(metric) for result in filtered if result.metrics.get(metric) is not None]
            if not metric_values:
                continue
            averages[metric] = mean(metric_values)
            distributions[metric] = {
                "min": min(metric_values),
                "max": max(metric_values),
                "mean": averages[metric],
            }

        return {
            "top_n": top_payload,
            "averages": averages,
            "distributions": distributions,
        }

    def best_parameters(self) -> dict[str, Any] | None:
        """Return parameters of the highest-ranked configuration."""
        top_results = self.store.top_n(1)
        if not top_results:
            return None
        return dict(top_results[0].parameters)

    @staticmethod
    def _result_payload(result: OptimizationResult, metrics: Sequence[str]) -> Dict[str, Any]:
        return {
            "parameters": result.parameters,
            "metrics": {metric: result.metrics.get(metric) for metric in metrics},
            "error": result.error,
            "timestamp": result.timestamp,
        }


def _filter_results(
    results: Sequence[OptimizationResult],
    filters: Mapping[str, Any] | None,
) -> list[OptimizationResult]:
    if not filters:
        return list(results)

    filtered: list[OptimizationResult] = []
    for result in results:
        include = True
        parameters_filter = filters.get("parameters") if isinstance(filters, Mapping) else None
        if parameters_filter:
            for key, value in parameters_filter.items():
                if result.parameters.get(key) != value:
                    include = False
                    break
        if not include:
            continue

        metrics_filter = filters.get("metrics") if isinstance(filters, Mapping) else None
        if metrics_filter:
            for key, threshold in metrics_filter.items():
                metric_value = result.metrics.get(key)
                if metric_value is None or metric_value < threshold:
                    include = False
                    break
        if include:
            filtered.append(result)
    return filtered
