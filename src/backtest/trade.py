"""Trade and position entities for the backtest module."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Trade:
    """Closed trade details."""

    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    direction: int  # 1 for long, -1 for short
    n_contracts: int = 1
    commission: float = 0.0

    def __post_init__(self) -> None:
        if self.direction not in (-1, 1):
            raise ValueError("direction must be either 1 (long) or -1 (short)")
        if self.n_contracts <= 0:
            raise ValueError("n_contracts must be positive")

    @property
    def pnl(self) -> float:
        """Profit or loss without commissions."""
        return (self.exit_price - self.entry_price) * self.direction * self.n_contracts

    @property
    def net_pnl(self) -> float:
        """Profit or loss with commissions deducted."""
        return self.pnl - self.commission

    @property
    def duration(self) -> pd.Timedelta:
        """Time the trade was open."""
        return self.exit_date - self.entry_date

    @property
    def pnl_pct(self) -> float:
        """Return on capital employed expressed as a fraction."""
        notional = self.entry_price * self.n_contracts
        if notional == 0:
            return 0.0
        return self.pnl / notional


@dataclass
class Position:
    """An open position that can be converted into a Trade once closed."""

    entry_date: pd.Timestamp
    entry_price: float
    direction: int  # 1 for long, -1 for short
    n_contracts: int = 1
    commission: float = 0.0

    def __post_init__(self) -> None:
        if self.direction not in (-1, 1):
            raise ValueError("direction must be either 1 (long) or -1 (short)")
        if self.n_contracts <= 0:
            raise ValueError("n_contracts must be positive")

    @property
    def is_long(self) -> bool:
        return self.direction == 1

    @property
    def is_short(self) -> bool:
        return self.direction == -1

    def unrealized_pnl(self, current_price: float) -> float:
        """Return unrealized P/L without commissions."""
        return (current_price - self.entry_price) * self.direction * self.n_contracts

    def unrealized_net_pnl(self, current_price: float) -> float:
        """Return unrealized P/L including commissions paid so far."""
        return self.unrealized_pnl(current_price) - self.commission

    def add_commission(self, value: float) -> None:
        """Accumulate additional commission costs for the position."""
        self.commission += value

    def close(
        self,
        exit_date: pd.Timestamp,
        exit_price: float,
        exit_commission: float = 0.0,
    ) -> Trade:
        """Convert the position into a closed trade."""
        total_commission = self.commission + exit_commission
        return Trade(
            entry_date=self.entry_date,
            entry_price=self.entry_price,
            exit_date=exit_date,
            exit_price=exit_price,
            direction=self.direction,
            n_contracts=self.n_contracts,
            commission=total_commission,
        )
