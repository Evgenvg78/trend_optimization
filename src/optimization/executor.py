"""Orchestrator responsible for running optimisation searches end-to-end."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Protocol

import json
import time

from trend_indicator_module.signals.base_signal import BaseSignal

from .parameter_space import ParameterSpace
from .reporting import OptimizationReporter
from .results_store import OptimizationResult, ResultsStoreProtocol
from .strategies.base import SearchStrategy

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from trend_indicator_module.backtest.engine import BacktestEngine


@dataclass
class SignalConfig:
    """Bundle describing how to generate signals for a single evaluation."""

    signal: BaseSignal
    signal_params: dict[str, Any]
    indicator_params: dict[str, Any] = field(default_factory=dict)
    indicator: "pd.Series | None" = None
    backtest_params: dict[str, Any] = field(default_factory=dict)


class SignalFactory(Protocol):
    """Factory responsible for preparing signal instances and parameter splits."""

    def create(self, parameters: Mapping[str, Any]) -> SignalConfig | BaseSignal | tuple[BaseSignal, Mapping[str, Any]]:
        """Return a signal configuration ready to generate trading signals."""


Callback = Callable[[OptimizationResult], None]


@dataclass
class OptimizationExecutor:
    """Drive optimisation: generate candidates, evaluate them, store and report results."""

    backtest_engine: "BacktestEngine"
    parameter_space: ParameterSpace
    strategy: SearchStrategy
    signal_factory: SignalFactory
    results_store: ResultsStoreProtocol
    indicator_builder: Callable[[ "pd.DataFrame", Mapping[str, Any]], "pd.Series"] | None = None
    output_root: Path = Path("output/optimization")
    callbacks: Iterable[Callback] = field(default_factory=tuple)
    per_candidate_timeout: float | None = None
    max_run_time_seconds: float | None = None
    checkpoint_enabled: bool = True
    checkpoint_interval: int = 1
    resume: bool = False
    checkpoint_dir: Path | None = None
    max_failures: int | None = None
    default_max_workers: int | None = None

    def __post_init__(self) -> None:
        self._log_lock = Lock()
        self._log_path: Path | None = None
        self._checkpoint_lock = Lock()
        self._checkpoint_path: Path | None = None
        self._processed_signatures: set[str] = set()
        self._completed = 0
        self._stop_requested = False
        self._run_start: float | None = None
        self._total_failures = 0

    def run(
        self,
        market_data: "pd.DataFrame",
        indicator: "pd.Series | None" = None,
        *,
        run_id: str | None = None,
        output_root: str | Path | None = None,
        max_workers: int | None = None,
    ) -> Path:
        """Execute optimisation against provided market data."""
        if self.resume and not run_id:
            raise ValueError("run_id must be provided when resume is enabled")

        run_name = run_id or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        root = Path(output_root) if output_root else self.output_root
        run_dir = root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_root = Path(self.checkpoint_dir) if self.checkpoint_dir else (root / "checkpoints")
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        self._checkpoint_path = checkpoint_root / f"{run_name}.json"

        self._log_path = run_dir / "run.log"
        worker_count = max_workers if max_workers is not None else (self.default_max_workers or 1)
        worker_count = max(1, worker_count)

        self._run_start = time.perf_counter()
        self._stop_requested = False
        self._total_failures = 0
        if not self.resume:
            with self._checkpoint_lock:
                self._processed_signatures.clear()
            self._completed = 0
        else:
            self._load_checkpoint()

        resume_label = "resume" if self.resume else "fresh"
        self._log(f"Starting optimisation run '{run_name}' ({resume_label}) with max_workers={worker_count}")

        total_candidates = self._estimate_total_candidates()

        if worker_count > 1:
            self._run_parallel(market_data, indicator, total_candidates, worker_count)
        else:
            self._run_sequential(market_data, indicator, total_candidates)

        duration = time.perf_counter() - (self._run_start or time.perf_counter())
        status = "completed" if not self._stop_requested else "stopped"
        self._log(f"{status.capitalize()} run '{run_name}' in {duration:.2f} seconds")

        self._export_results(run_dir)
        self._write_checkpoint(force=True)
        return run_dir

    # Execution helpers -------------------------------------------------

    def _run_sequential(
        self,
        market_data: "pd.DataFrame",
        indicator: "pd.Series | None",
        total_candidates: int | None,
    ) -> None:
        for index, parameters in enumerate(self.strategy, start=1):
            if self._stop_requested or not self._check_runtime_budget():
                self._log("Runtime limit reached during sequential execution; aborting remaining candidates")
                break
            self._process_candidate(index, total_candidates, parameters, market_data, indicator)
            if self._stop_requested:
                break

    def _run_parallel(
        self,
        market_data: "pd.DataFrame",
        indicator: "pd.Series | None",
        total_candidates: int | None,
        max_workers: int,
    ) -> None:
        iterator = enumerate(self.strategy, start=1)

        def submit_next(pool: ThreadPoolExecutor, futures: set) -> bool:
            if self._stop_requested or not self._check_runtime_budget():
                return False
            try:
                index, parameters = next(iterator)
            except StopIteration:
                return False
            future = pool.submit(
                self._process_candidate,
                index,
                total_candidates,
                parameters,
                market_data,
                indicator,
            )
            futures.add(future)
            return True

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: set = set()
            for _ in range(max_workers):
                if not submit_next(executor, futures):
                    break

            while futures:
                for future in as_completed(list(futures)):
                    futures.remove(future)
                    future.result()
                    if self._stop_requested:
                        break
                    submit_next(executor, futures)
                    break
                if self._stop_requested:
                    for pending in futures:
                        pending.cancel()
                    break

    def _process_candidate(
        self,
        index: int,
        total_candidates: int | None,
        parameters: Mapping[str, Any],
        market_data: "pd.DataFrame",
        indicator: "pd.Series | None",
    ) -> None:
        iteration_label = f"{index}/{total_candidates}" if total_candidates else f"{index}"
        signature: str | None = None
        validated: dict[str, Any] | None = None
        try:
            validated = self.parameter_space.validate(dict(parameters))
            signature = self._signature(validated)
            if self._is_processed(signature):
                self._log(f"[{iteration_label}] skip (already processed)")
                return
            if not self._check_runtime_budget():
                self._log(f"[{iteration_label}] skipped due to runtime limit")
                return
            result = self._run_with_timeout(validated, market_data, indicator)
            result.metadata.setdefault("iteration", index)
            self._record_success(result, signature, iteration_label)
        except TimeoutError:
            payload = dict(validated or parameters)
            signature = signature or self._signature(payload)
            timeout_label = f"{self.per_candidate_timeout}s" if self.per_candidate_timeout else "configured timeout"
            self._record_failure(signature, payload, index, iteration_label, f"timeout after {timeout_label}")
        except Exception as exc:  # pragma: no cover - exercised via integration test
            payload = dict(validated or parameters)
            signature = signature or self._signature(payload)
            self._record_failure(signature, payload, index, iteration_label, str(exc))

    def _evaluate_single(
        self,
        parameters: Mapping[str, Any],
        market_data: "pd.DataFrame",
        indicator: "pd.Series | None",
    ) -> OptimizationResult:
        config = self._build_signal_config(parameters)
        indicator_series = self._resolve_indicator(config, market_data, indicator)

        raw_signals = config.signal.generate(market_data, indicator_series, **config.signal_params)
        signals = _ensure_series(raw_signals, market_data.index)
        backtest_kwargs = dict(config.backtest_params)
        backtest_result = self.backtest_engine.run(market_data, signals, **backtest_kwargs)

        metrics = {
            key: value
            for key, value in backtest_result.items()
            if key not in {"equity", "trades"}
        }
        artifacts = self._extract_artifacts(backtest_result)
        artifacts["signals"] = _serialise_series(signals)

        return OptimizationResult(
            parameters=dict(parameters),
            metrics=metrics,
            artifacts=artifacts,
        )

    # Utilities ---------------------------------------------------------

    def _build_signal_config(self, parameters: Mapping[str, Any]) -> SignalConfig:
        factory = self.signal_factory
        if hasattr(factory, "create") and callable(getattr(factory, "create")):
            descriptor = factory.create(parameters)
        elif callable(factory):
            descriptor = factory(parameters)
        else:
            raise TypeError("signal_factory must be callable or expose a create(parameters) method")
        if isinstance(descriptor, SignalConfig):
            return descriptor
        if isinstance(descriptor, BaseSignal):
            return SignalConfig(signal=descriptor, signal_params=dict(parameters))
        if isinstance(descriptor, tuple) and len(descriptor) == 2:
            signal, signal_params = descriptor
            if not isinstance(signal, BaseSignal):
                raise TypeError("First element of tuple returned by factory must be a BaseSignal instance")
            return SignalConfig(signal=signal, signal_params=dict(signal_params))
        raise TypeError("SignalFactory.create must return SignalConfig, BaseSignal, or (BaseSignal, params) tuple")

    def _resolve_indicator(
        self,
        config: SignalConfig,
        market_data: "pd.DataFrame",
        indicator: "pd.Series | None",
    ):
        if config.indicator is not None:
            return config.indicator
        if indicator is not None:
            return indicator
        if self.indicator_builder is not None:
            return self.indicator_builder(market_data, config.indicator_params)
        raise ValueError("Indicator series is required but was not provided")

    def _extract_artifacts(self, result: Mapping[str, Any]) -> dict[str, Any]:
        artifacts: dict[str, Any] = {}
        equity = result.get("equity")
        if equity is not None:
            artifacts["equity"] = _serialise_series(equity)
        trades = result.get("trades")
        if trades is not None:
            artifacts["trades"] = [_serialise_trade(trade) for trade in trades]
        return artifacts

    def _estimate_total_candidates(self) -> int | None:
        try:
            grid = self.parameter_space.grid()
        except ValueError:
            return None

        if not grid:
            return None

        total = 1
        for values in grid.values():
            total *= len(values)
        max_iterations = getattr(self.strategy.state, "max_iterations", None)
        if max_iterations is not None:
            return min(total, max_iterations)
        return total

    def _export_results(self, run_dir: Path) -> None:
        table_path = run_dir / "results.csv"
        json_path = run_dir / "results.json"
        summary_path = run_dir / "summary.json"

        self.results_store.export_csv(table_path)
        self.results_store.export_json(json_path)

        reporter = OptimizationReporter(self.results_store)
        summary = reporter.summary()
        summary_path.write_text(json.dumps(summary, indent=2, default=_json_fallback), encoding="utf-8")

        table = reporter.to_table()
        if _is_dataframe(table):
            table.to_csv(run_dir / "results_table.csv", index=False)
        else:
            (run_dir / "results_table.json").write_text(
                json.dumps(table, indent=2, default=_json_fallback),
                encoding="utf-8",
            )

    def _record_success(self, result: OptimizationResult, signature: str, iteration_label: str) -> None:
        self.results_store.append(result)
        for callback in self.callbacks:
            callback(result)
        self._log(f"[{iteration_label}] success")
        self._register_processed(signature)

    def _record_failure(
        self,
        signature: str,
        parameters: Mapping[str, Any],
        index: int,
        iteration_label: str,
        error: str,
    ) -> None:
        self._log(f"[{iteration_label}] error: {error}")
        error_result = OptimizationResult(
            parameters=dict(parameters),
            metrics={},
            artifacts={},
            metadata={"iteration": index},
            error=str(error),
        )
        self.results_store.append(error_result)
        self._total_failures += 1
        if self.max_failures is not None and self._total_failures >= self.max_failures:
            self._stop_requested = True
        self._register_processed(signature)

    def _register_processed(self, signature: str) -> None:
        with self._checkpoint_lock:
            self._processed_signatures.add(signature)
            self._completed += 1
            self._write_checkpoint_locked(force=False)

    def _run_with_timeout(
        self,
        parameters: Mapping[str, Any],
        market_data: "pd.DataFrame",
        indicator: "pd.Series | None",
    ) -> OptimizationResult:
        if not self.per_candidate_timeout:
            return self._evaluate_single(parameters, market_data, indicator)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._evaluate_single, parameters, market_data, indicator)
            return future.result(timeout=self.per_candidate_timeout)

    def _check_runtime_budget(self) -> bool:
        if self.max_run_time_seconds is None or self._run_start is None:
            return True
        elapsed = time.perf_counter() - self._run_start
        if elapsed <= self.max_run_time_seconds:
            return True
        self._stop_requested = True
        return False

    def _is_processed(self, signature: str) -> bool:
        with self._checkpoint_lock:
            return signature in self._processed_signatures

    def _signature(self, parameters: Mapping[str, Any]) -> str:
        return json.dumps(dict(parameters), sort_keys=True, default=_json_fallback)

    def _write_checkpoint(self, *, force: bool = False) -> None:
        if not self.checkpoint_enabled or not self._checkpoint_path:
            return
        with self._checkpoint_lock:
            self._write_checkpoint_locked(force=force)

    def _write_checkpoint_locked(self, *, force: bool) -> None:
        if not self.checkpoint_enabled or not self._checkpoint_path:
            return
        interval = max(1, self.checkpoint_interval)
        if not force and self._completed % interval != 0:
            return
        payload = {
            "version": 1,
            "updated_at": datetime.now(UTC).isoformat(),
            "processed": sorted(self._processed_signatures),
            "results": [item.to_dict() for item in self.results_store],
            "meta": {
                "completed": self._completed,
                "total_failures": self._total_failures,
                "strategy": {
                    "iterations": getattr(self.strategy.state, "iterations", None),
                    "max_iterations": getattr(self.strategy.state, "max_iterations", None),
                },
            },
        }
        tmp_path = self._checkpoint_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, default=_json_fallback), encoding="utf-8")
        tmp_path.replace(self._checkpoint_path)

    def _load_checkpoint(self) -> None:
        if not self.resume or not self._checkpoint_path or not self._checkpoint_path.exists():
            return
        try:
            payload = json.loads(self._checkpoint_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive path
            self._log(f"Failed to load checkpoint '{self._checkpoint_path}': {exc}")
            return
        processed = payload.get("processed", [])
        results = payload.get("results", [])
        meta = payload.get("meta", {})
        with self._checkpoint_lock:
            self._processed_signatures = set(processed)
            self._completed = int(meta.get("completed", len(processed)))
        self._total_failures = 0
        for item in results:
            result = OptimizationResult.from_dict(item)
            self.results_store.append(result)
            if result.error:
                self._total_failures += 1
        self._log(
            f"Loaded checkpoint '{self._checkpoint_path.name}' with "
            f"{len(processed)} processed candidates ({self._total_failures} failures)"
        )

    def _log(self, message: str) -> None:
        if not self._log_path:
            return
        timestamp = datetime.now(UTC).isoformat()
        line = f"{timestamp} {message}\n"
        with self._log_lock:
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(line)


# Serialization helpers -------------------------------------------------------

def _serialise_series(series) -> dict[str, Any]:
    return {
        "index": [str(item) for item in series.index],
        "values": [float(value) for value in series.tolist()],
    }


def _serialise_trade(trade) -> dict[str, Any]:
    payload = {
        "entry_date": getattr(trade.entry_date, "isoformat", lambda: str(trade.entry_date))(),
        "exit_date": getattr(trade.exit_date, "isoformat", lambda: str(trade.exit_date))(),
        "entry_price": float(trade.entry_price),
        "exit_price": float(trade.exit_price),
        "direction": int(trade.direction),
        "n_contracts": int(trade.n_contracts),
        "commission": float(trade.commission),
    }
    if hasattr(trade, "net_pnl"):
        payload["net_pnl"] = float(trade.net_pnl)
    return payload


def _is_dataframe(value: Any) -> bool:
    if TYPE_CHECKING:
        return False
    try:  # pragma: no cover - optional dependency
        import pandas as pd

        return isinstance(value, pd.DataFrame)
    except Exception:
        return False


def _json_fallback(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    try:
        return list(value)
    except TypeError:
        pass
    return str(value)


def _ensure_series(signals, index) -> Any:
    if hasattr(signals, "index"):
        return signals
    try:  # pragma: no cover - optional dependency
        import pandas as pd

        return pd.Series(signals, index=index)
    except Exception as exc:
        raise TypeError("Signal generator must return a pandas Series or array-like object") from exc
