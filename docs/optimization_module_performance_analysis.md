# Optimization Module Performance Analysis

## Overview
- The optimisation stack centres on `OptimizationExecutor` (`trend_indicator_module/optimization/executor.py`) coordinating candidate generation, signal construction, backtesting, and persistence.
- Parameter definitions live in `ParameterSpace` (`trend_indicator_module/optimization/parameter_space.py`), while search behaviour is implemented by strategies such as grid and random search (`trend_indicator_module/optimization/strategies`).
- Results are buffered in-memory via `InMemoryResultsStore` before being exported and checkpointed (`trend_indicator_module/optimization/results_store.py`).

## Observed Bottlenecks

1. **Full artifact capture per evaluation**  
   `trend_indicator_module/optimization/executor.py:250` and `trend_indicator_module/optimization/executor.py:291` serialise the complete signal series, equity curve, and trade list for *every* candidate. For long histories this inflates each `OptimizationResult` by thousands of points, multiplying memory usage and JSON export/checkpoint time.  
   *Recommendation:* Make artifact collection optional or scope it to the top-N configurations. Persist bulky artifacts asynchronously on demand instead of embedding them into every in-memory result.

2. **Quadratic checkpoint overhead**  
   Each checkpoint rewrites the entire result set (`trend_indicator_module/optimization/executor.py:414`), including artifacts above, whenever `_completed % checkpoint_interval == 0`. With the default interval of 1 (`trend_indicator_module/config/optimization.yaml:57`) this becomes an O(n^2) operation as runs grow.  
   *Recommendation:* Raise the default interval and redesign checkpoint payloads to store only incremental metadata (e.g., processed signatures + compressed summaries) or stream to a lightweight store such as SQLite/Parquet.

3. **Per-candidate timeout thread pools**  
   `_run_with_timeout` spins up a fresh `ThreadPoolExecutor(max_workers=1)` for every evaluation when timeouts are enabled (`trend_indicator_module/optimization/executor.py:378`). Creating and tearing down executors thousands of times is expensive and adds scheduling latency.  
   *Recommendation:* Reuse worker pools or rely on cooperative timeouts (`concurrent.futures.wait`, async alarms) instead of per-candidate pools.

4. **Thread-bound parallelism on a stateful backtester**  
   Parallel mode uses `ThreadPoolExecutor` (`trend_indicator_module/optimization/executor.py:160`), but `BacktestEngine.run` mutates shared state (`trend_indicator_module/backtest/engine.py:35`, `trend_indicator_module/backtest/engine.py:68`). CPU-heavy loops and the GIL limit throughput, while shared state makes threads unsafe.  
   *Recommendation:* Move to process-based workers (multiprocessing/Ray) that own independent backtest engines, or instantiate one engine per worker behind a worker factory.

5. **Repeated indicator recomputation without caching**  
   `_resolve_indicator` recalculates indicators for every candidate (`trend_indicator_module/optimization/executor.py:273`), even when parameter combinations reuse identical indicator inputs.  
   *Recommendation:* Cache indicator outputs keyed by parameter signature to avoid redundant computation, or precompute invariant indicators during setup.

6. **Signature hashing via JSON on the hot path**  
   Candidate deduplication serialises parameters with `json.dumps` (`trend_indicator_module/optimization/executor.py:362`). For large grids this becomes a noticeable overhead.  
   *Recommendation:* Replace JSON serialisation with faster structural hashing (e.g., tuples + `hashlib`), or pre-hash candidates inside the strategy layer.

## Recommended Optimizer Mechanism Changes

1. Introduce configurable artifact policies (all / top-N / disabled) and push heavy payloads to background writers so checkpoints stay lightweight.
2. Redesign checkpointing to be append-only or metadata-only (e.g., processed signature set + latest top results) and bump the runtime default interval to a sane value (50-100) for big sweeps.
3. Replace per-candidate timeout pools with a shared watchdog (single worker executor with `future.result(timeout=...)` or OS timers) to eliminate thousands of executor allocations.
4. Provide a process-aware executor abstraction that gives each worker its own `BacktestEngine` instance plus cached market data, unlocking safe multi-core scaling.
5. Layer in memoisation for indicator/feature builders and optionally for signal factories when parameters leave sub-components untouched.

## Broader System-Level Improvements

1. Support distributed execution backends (e.g., `ProcessPoolExecutor`, Dask, Ray) with chunked task dispatch so large grids can span multiple cores or machines.
2. Stream results to durable storage (SQLite, Parquet, Arrow IPC) and expose incremental readers, enabling real-time dashboards without keeping entire result sets in RAM.
3. Add profiling hooks and telemetry (timings per stage, cache hit ratios) so slow components are observable during long optimisation runs.
4. Provide scenario-level orchestration: batch candidates by shared indicator parameters, warm cache once, then evaluate dependent signal/backtest permutations.
5. Offer CLI/notebook utilities to resume from lightweight checkpoints, prune dominated candidates early, and integrate Bayesian/gradient-free strategies for targeted exploration.

## Next Steps
- Prototype artifact policy toggles and checkpoint slimming, then benchmark on a realistic optimisation workload.
- Validate a multiprocess executor prototype with per-worker engine factories and shared-read market data to quantify parallel speedups.
- Measure indicator caching efficacy on common strategies; document cache invalidation semantics.
- Update user-facing docs and configuration schema to surface the new runtime knobs once validated.

