"""Backtest engine that turns signals into trades and equity curves."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .commission import CommissionCalculator
from .metrics import MetricsCalculator
from .trade import Position, Trade


@dataclass
class BacktestEngine:
    """Run a simple long/short backtest on OHLCV data."""

    commission_calculator: CommissionCalculator = field(default_factory=CommissionCalculator)
    initial_capital: float = 100_000.0
    default_n_contracts: int = 1
    metrics_calculator: MetricsCalculator = field(default_factory=MetricsCalculator)

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.default_n_contracts <= 0:
            raise ValueError("default_n_contracts must be positive")
        # Internal state
        self.trades: list[Trade] = []
        self.equity_curve: Optional[pd.Series] = None
        self._last_signals: Optional[pd.Series] = None

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series | np.ndarray | list,
        entry_price_type: str = "close",
        n_contracts: Optional[int] = None,
    ) -> Dict[str, object]:
        """
        Execute the backtest for the supplied OHLCV data and signals.

        Args:
            df: OHLCV price data. Must contain columns OPEN, HIGH, LOW, CLOSE.
            signals: Trading signal per bar (values in {-1, 0, 1}).
            entry_price_type: Price source for entries/exits ("close", "open", "next_open").
            n_contracts: Contracts per trade. Defaults to engine.default_n_contracts.

        Returns:
            Dictionary with performance statistics, trades list, and equity series.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        required_cols = {"OPEN", "HIGH", "LOW", "CLOSE"}
        if not required_cols.issubset(df.columns):
            missing = ", ".join(sorted(required_cols - set(df.columns)))
            raise ValueError(f"df is missing required columns: {missing}")
        if df.empty:
            raise ValueError("df must contain at least one row")

        signals_series = self._prepare_signals(signals, df.index)
        contracts = n_contracts or self.default_n_contracts
        if contracts <= 0:
            raise ValueError("n_contracts must be positive")

        self.trades = []
        self._last_signals = signals_series

        cash = float(self.initial_capital)
        position: Optional[Position] = None
        equity_values: list[float] = []

        closes = df["CLOSE"].to_numpy(dtype=float)

        for idx, timestamp in enumerate(df.index):
            desired_direction = signals_series.iat[idx]

            # Exit conditions: go flat or reverse position.
            if position is not None and (desired_direction == 0 or desired_direction != position.direction):
                exit_price = self._get_price(df, idx, entry_price_type)
                exit_commission = self.commission_calculator.calculate_total_commission(
                    exit_price, position.n_contracts
                )
                trade = position.close(timestamp, float(exit_price), exit_commission)
                self.trades.append(trade)
                cash += trade.net_pnl
                position = None

            # Entry condition: enter when flat and signal requests exposure.
            if desired_direction != 0 and position is None:
                entry_price = self._get_price(df, idx, entry_price_type)
                entry_commission = self.commission_calculator.calculate_total_commission(entry_price, contracts)
                position = Position(
                    entry_date=timestamp,
                    entry_price=float(entry_price),
                    direction=int(desired_direction),
                    n_contracts=contracts,
                    commission=float(entry_commission),
                )

            # Update mark-to-market equity at close.
            if position is not None:
                equity = cash + position.unrealized_net_pnl(closes[idx])
            else:
                equity = cash
            equity_values.append(float(equity))

        # Force close any open position at the last available close.
        if position is not None:
            final_price = closes[-1]
            final_timestamp = df.index[-1]
            exit_commission = self.commission_calculator.calculate_total_commission(
                final_price, position.n_contracts
            )
            trade = position.close(final_timestamp, float(final_price), exit_commission)
            self.trades.append(trade)
            cash += trade.net_pnl
            position = None
            equity_values[-1] = float(cash)

        self.equity_curve = pd.Series(equity_values, index=df.index, name="equity")
        returns = self.metrics_calculator.returns(self.equity_curve)

        return {
            "total_return": self.metrics_calculator.total_return(self.equity_curve),
            "max_drawdown": self.metrics_calculator.max_drawdown(self.equity_curve),
            "recovery_factor": self.metrics_calculator.recovery_factor(self.equity_curve),
            "sharpe": self.metrics_calculator.sharpe_ratio(returns),
            "sortino": self.metrics_calculator.sortino_ratio(returns),
            "area_ab": self.metrics_calculator.area_ab(self.equity_curve),
            "n_flips": self.metrics_calculator.count_flips(signals_series),
            "n_trades": len(self.trades),
            "win_rate": self.metrics_calculator.win_rate(self.trades),
            "profit_factor": self.metrics_calculator.profit_factor(self.trades),
            # Wave-based metrics between signal flips
            "waves_count": self.metrics_calculator.waves_count(df["CLOSE"], signals_series),
            "mean_wave_mfe": self.metrics_calculator.mean_wave_mfe(df["CLOSE"], signals_series),
            "mean_wave_mfe_ratio": self.metrics_calculator.mean_wave_mfe_ratio(df["CLOSE"], signals_series),
            "equity": self.equity_curve,
            "trades": self.trades,
        }

    def _get_price(self, df: pd.DataFrame, idx: int, price_type: str) -> float:
        """Return price for trade execution respecting the requested fill type."""
        if price_type == "close":
            return float(df["CLOSE"].iat[idx])
        if price_type == "open":
            return float(df["OPEN"].iat[idx])
        if price_type == "next_open":
            if idx + 1 < len(df):
                return float(df["OPEN"].iat[idx + 1])
            return float(df["CLOSE"].iat[idx])
        raise ValueError(f"Unknown entry_price_type: {price_type}")

    @staticmethod
    def _prepare_signals(signals, index: pd.Index) -> pd.Series:
        """Clean and align signal data with the price index."""
        if isinstance(signals, pd.Series):
            series = signals.copy()
            if not series.index.equals(index):
                series = series.reindex(index, method="ffill").fillna(0)
        else:
            series = pd.Series(signals, index=index)

        series = series.fillna(0).astype(float)
        normalised_series = pd.Series(np.sign(series), index=index)

        invalid = ~normalised_series.isin([-1.0, 0.0, 1.0])
        if invalid.any():
            raise ValueError("signals must contain only -1, 0, or 1 values (after normalisation)")

        return normalised_series.astype(int)
