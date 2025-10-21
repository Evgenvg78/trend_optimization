"""Storage utilities for optimization results and artifact export."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableSequence, Sequence

import json


@dataclass
class OptimizationResult:
    """Outcome of a single parameter evaluation."""

    parameters: dict[str, Any]
    metrics: dict[str, Any]
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self, *, metrics: Sequence[str] | None = None) -> dict[str, Any]:
        """Convert the result into a JSON serialisable dictionary."""
        base = {
            "parameters": self.parameters,
            "metrics": _filter_metrics(self.metrics, metrics),
            "artifacts": self.artifacts,
            "metadata": self.metadata,
            "error": self.error,
            "timestamp": self.timestamp,
        }
        return base

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OptimizationResult":
        """Instantiate a result object from a serialised representation."""
        return cls(
            parameters=dict(payload.get("parameters", {})),
            metrics=dict(payload.get("metrics", {})),
            artifacts=dict(payload.get("artifacts", {})),
            metadata=dict(payload.get("metadata", {})),
            error=payload.get("error"),
            timestamp=payload.get("timestamp", datetime.now(UTC).isoformat()),
        )


class ResultsStoreProtocol(Iterable[OptimizationResult]):
    """Protocol-like base class for type hints."""

    def append(self, result: OptimizationResult) -> None:  # pragma: no cover - interface only
        raise NotImplementedError

    def top_n(
        self,
        n: int = 10,
        *,
        key_metric: str | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[OptimizationResult]:  # pragma: no cover - interface only
        raise NotImplementedError

    def export_csv(
        self,
        destination: str | Path,
        *,
        metrics: Sequence[str] | None = None,
    ) -> Path:  # pragma: no cover - interface only
        raise NotImplementedError

    def export_json(
        self,
        destination: str | Path,
        *,
        metrics: Sequence[str] | None = None,
    ) -> Path:  # pragma: no cover - interface only
        raise NotImplementedError


class InMemoryResultsStore(ResultsStoreProtocol):
    """Thread-safe in-memory store for optimisation results."""

    def __init__(self, *, key_metric: str = "total_return") -> None:
        self.key_metric = key_metric
        self._items: MutableSequence[OptimizationResult] = []
        self._lock = Lock()

    def __iter__(self) -> Iterator[OptimizationResult]:
        with self._lock:
            return iter(list(self._items))

    def append(self, result: OptimizationResult) -> None:
        with self._lock:
            self._items.append(result)

    # Backwards compatible alias
    def add(self, result: OptimizationResult) -> None:
        self.append(result)

    def top(self, n: int = 10) -> Sequence[OptimizationResult]:
        return self.top_n(n)

    def top_n(
        self,
        n: int = 10,
        *,
        key_metric: str | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[OptimizationResult]:
        metric_name = key_metric or self.key_metric
        filtered = self._apply_filters(filters)
        return sorted(
            filtered,
            key=lambda item: item.metrics.get(metric_name, float("-inf")),
            reverse=True,
        )[:n]

    def filter(
        self,
        *,
        parameter_equals: Mapping[str, Any] | None = None,
        metric_mins: Mapping[str, float] | None = None,
    ) -> Sequence[OptimizationResult]:
        criteria: dict[str, Any] = {}
        if parameter_equals:
            criteria["parameters"] = parameter_equals
        if metric_mins:
            criteria["metrics"] = metric_mins
        return self._apply_filters(criteria)

    def export_csv(
        self,
        destination: str | Path,
        *,
        metrics: Sequence[str] | None = None,
    ) -> Path:
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        rows = self._rows(metrics=metrics)
        if not rows:
            destination_path.write_text("", encoding="utf-8")
            return destination_path

        headers = sorted(rows[0].keys())
        lines = [",".join(headers)]
        for row in rows:
            line = ",".join(_to_csv_value(row.get(header)) for header in headers)
            lines.append(line)

        destination_path.write_text("\n".join(lines), encoding="utf-8")
        return destination_path

    def export_json(
        self,
        destination: str | Path,
        *,
        metrics: Sequence[str] | None = None,
    ) -> Path:
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        payload = [item.to_dict(metrics=metrics) for item in self._snapshot()]
        destination_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        return destination_path

    def _snapshot(self) -> List[OptimizationResult]:
        with self._lock:
            return list(self._items)

    def _apply_filters(self, filters: Mapping[str, Any] | None) -> List[OptimizationResult]:
        if not filters:
            return self._snapshot()

        results = []
        for item in self._snapshot():
            if not _matches(item, filters):
                continue
            results.append(item)
        return results

    def _rows(self, *, metrics: Sequence[str] | None = None) -> List[Dict[str, Any]]:
        rows: list[Dict[str, Any]] = []
        for item in self._snapshot():
            row: Dict[str, Any] = {}
            for key, value in item.parameters.items():
                row[f"param_{key}"] = value
            metric_values = _filter_metrics(item.metrics, metrics)
            for key, value in metric_values.items():
                row[f"metric_{key}"] = value
            row["error"] = item.error
            row["timestamp"] = item.timestamp
            rows.append(row)
        return rows


def _filter_metrics(metrics: Mapping[str, Any], selected: Sequence[str] | None) -> Dict[str, Any]:
    if not selected:
        return dict(metrics)
    return {name: metrics.get(name) for name in selected}


def _matches(result: OptimizationResult, filters: Mapping[str, Any]) -> bool:
    for key, condition in filters.items():
        if key == "parameters":
            for param_key, param_value in condition.items():
                if result.parameters.get(param_key) != param_value:
                    return False
        elif key == "metrics":
            for metric_key, threshold in condition.items():
                value = result.metrics.get(metric_key)
                if value is None or value < threshold:
                    return False
        else:
            # Fallback to direct comparison against result attributes / metadata.
            value = getattr(result, key, None)
            if isinstance(condition, Mapping):
                for inner_key, inner_value in condition.items():
                    if result.metadata.get(inner_key) != inner_value:
                        return False
            elif value != condition:
                return False
    return True


def _to_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    try:
        return asdict(value)
    except TypeError:
        pass
    return str(value)
