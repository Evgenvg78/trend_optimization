"""Performance metrics for the backtest module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass
class MetricsCalculator:
    """Utility class that calculates common performance metrics for equity curves and trades."""

    periods_per_year: int = 252

    def total_return(self, equity: pd.Series) -> float:
        """Return the total compounded return of the equity curve."""
        cleaned = self._validate_equity(equity)
        if cleaned is None or len(cleaned) < 2:
            return 0.0
        first, last = cleaned.iloc[0], cleaned.iloc[-1]
        if first == 0:
            return 0.0
        return last / first - 1.0

    def max_drawdown(self, equity: pd.Series) -> float:
        """Return the maximum drawdown (as a positive fraction)."""
        cleaned = self._validate_equity(equity)
        if cleaned is None or len(cleaned) == 0:
            return 0.0
        running_max = cleaned.cummax()
        drawdowns = 1.0 - cleaned / running_max.replace(0, np.nan)
        return float(drawdowns.max(skipna=True) or 0.0)

    def recovery_factor(self, equity: pd.Series) -> float:
        """Return the recovery factor defined as total return divided by max drawdown."""
        total = self.total_return(equity)
        max_dd = self.max_drawdown(equity)
        if max_dd == 0:
            return np.nan if total != 0 else 0.0
        return total / max_dd

    def returns(self, equity: pd.Series) -> pd.Series:
        """Return simple percentage returns of the equity curve."""
        cleaned = self._validate_equity(equity)
        if cleaned is None or len(cleaned) < 2:
            return pd.Series(dtype=float)
        return cleaned.pct_change().dropna()

    def sharpe_ratio(self, returns: pd.Series | Sequence[float] | np.ndarray) -> float:
        """Return the annualised Sharpe ratio."""
        ret_series = self._to_series(returns)
        if ret_series.empty:
            return 0.0
        std = ret_series.std(ddof=0)
        if std == 0:
            return 0.0
        return np.sqrt(self.periods_per_year) * ret_series.mean() / std

    def sortino_ratio(self, returns: pd.Series | Sequence[float] | np.ndarray) -> float:
        """Return the annualised Sortino ratio (downside deviation only)."""
        ret_series = self._to_series(returns)
        if ret_series.empty:
            return 0.0
        downside = ret_series[ret_series < 0]
        downside_std = downside.std(ddof=0)
        if downside_std == 0:
            return 0.0
        return np.sqrt(self.periods_per_year) * ret_series.mean() / downside_std

    @staticmethod
    def count_flips(signals: pd.Series | Sequence[float | int]) -> int:
        """Count the number of times the signal flips from long to short or vice versa."""
        sig = MetricsCalculator._to_series(signals).dropna()
        if sig.empty:
            return 0
        # Normalise to {-1, 0, 1}
        normalised = sig.apply(lambda x: np.sign(x) if x != 0 else 0)
        changes = normalised.diff().abs()
        return int((changes == 2).sum())

    @staticmethod
    def win_rate(trades: Iterable) -> float:
        """Return the fraction of winning trades."""
        trades_list = list(trades)
        if not trades_list:
            return 0.0
        wins = sum(1 for trade in trades_list if getattr(trade, "net_pnl", 0.0) > 0)
        return wins / len(trades_list)

    @staticmethod
    def profit_factor(trades: Iterable) -> float:
        """Return the profit factor of the trade list."""
        gross_profit = 0.0
        gross_loss = 0.0
        counted = False
        for trade in trades:
            pnl = getattr(trade, "net_pnl", 0.0)
            counted = True
            if pnl > 0:
                gross_profit += pnl
            elif pnl < 0:
                gross_loss += abs(pnl)
        if not counted:
            return 0.0
        if gross_loss == 0.0:
            return np.inf if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @staticmethod
    def _validate_equity(equity: pd.Series | Sequence[float] | np.ndarray) -> pd.Series | None:
        """Return a cleaned equity series or None if no data is available."""
        if equity is None:
            return None
        series = MetricsCalculator._to_series(equity).dropna()
        return series if not series.empty else None

    @staticmethod
    def _to_series(data: pd.Series | Sequence[float] | np.ndarray) -> pd.Series:
        """Convert arbitrary iterable input into a pandas Series."""
        if isinstance(data, pd.Series):
            return data.astype(float)
        if isinstance(data, np.ndarray):
            return pd.Series(data.astype(float))
        return pd.Series(list(data), dtype=float)

    @staticmethod
    def _segments_between_flips(price: pd.Series, signals: pd.Series) -> pd.DataFrame:
        """
        Split the series into segments of non-zero positions between flips.
        Returns DataFrame with columns:
            ['entry_idx','exit_idx','side','entry_price','exit_price',
             'extreme_price','mfe_abs','flip_move','mfe_ratio']
        Notes:
          - 'exit_idx' is the index of the bar on which the flip occurred.
          - Segments without a closing flip (unfinished) are dropped.
        """
        if price.empty or signals.empty:
            return pd.DataFrame(
                columns=[
                    "entry_idx",
                    "exit_idx",
                    "side",
                    "entry_price",
                    "exit_price",
                    "extreme_price",
                    "mfe_abs",
                    "flip_move",
                    "mfe_ratio",
                ]
            )

        price = price.dropna()
        signals = signals.reindex(price.index).ffill().fillna(0).astype(int)

        rows: list[dict[str, float | int | pd.Timestamp]] = []
        curr_side = 0
        run_start_idx = None

        for idx, side in signals.items():
            if side != curr_side:
                if (
                    curr_side != 0
                    and side != 0
                    and run_start_idx is not None
                ):
                    entry_idx = run_start_idx
                    exit_idx = idx
                    entry_price = float(price.loc[entry_idx])
                    exit_price = float(price.loc[exit_idx])

                    seg = price.loc[entry_idx:exit_idx]

                    if curr_side > 0:
                        extreme_price = float(seg.max())
                        mfe_abs = abs(extreme_price - entry_price)
                        flip_move = abs(exit_price - entry_price)
                    else:
                        extreme_price = float(seg.min())
                        mfe_abs = abs(entry_price - extreme_price)
                        flip_move = abs(exit_price - entry_price)

                    mfe_ratio = (mfe_abs / flip_move) if flip_move > 0 else np.nan

                    rows.append(
                        {
                            "entry_idx": entry_idx,
                            "exit_idx": exit_idx,
                            "side": int(curr_side),
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "extreme_price": extreme_price,
                            "mfe_abs": float(mfe_abs),
                            "flip_move": float(flip_move),
                            "mfe_ratio": float(mfe_ratio)
                            if np.isfinite(mfe_ratio)
                            else np.nan,
                        }
                    )

                run_start_idx = idx if side != 0 else None
                curr_side = side

        return pd.DataFrame(rows)

    @staticmethod
    def waves_table(price: pd.Series, signals: pd.Series) -> pd.DataFrame:
        """
        Return the table of wave segments between flips with columns:
        ['entry_idx','exit_idx','side','entry_price','exit_price',
         'extreme_price','mfe_abs','flip_move','mfe_ratio']
        """
        return MetricsCalculator._segments_between_flips(price, signals)

    @staticmethod
    def mean_wave_mfe(price: pd.Series, signals: pd.Series) -> float:
        """Return the mean absolute wave size (mean MFE across flip segments)."""
        table = MetricsCalculator._segments_between_flips(price, signals)
        return float(table["mfe_abs"].mean()) if not table.empty else np.nan

    @staticmethod
    def mean_wave_mfe_ratio(price: pd.Series, signals: pd.Series) -> float:
        """Return the mean ratio of peak-to-flip (mfe_abs / flip_move)."""
        table = MetricsCalculator._segments_between_flips(price, signals)
        col = table["mfe_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        return float(col.mean()) if not col.empty else np.nan

    @staticmethod
    def waves_count(price: pd.Series, signals: pd.Series) -> int:
        """Return the number of wave segments (between flips)."""
        table = MetricsCalculator._segments_between_flips(price, signals)
        return int(len(table))
