"""Commission calculation utilities for the backtest module."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CommissionCalculator:
    """Calculate broker and exchange commissions for futures trades."""

    broker_fee: float = 1.0
    exchange_fee_rate: float = 0.00001

    def calculate_broker_commission(self, n_contracts: int = 1) -> float:
        """Return the fixed broker commission component."""
        return self.broker_fee * n_contracts

    def calculate_exchange_commission(self, price: float, n_contracts: int = 1) -> float:
        """Return the exchange commission component that depends on trade price."""
        return price * n_contracts * self.exchange_fee_rate

    def calculate_total_commission(self, price: float, n_contracts: int = 1) -> float:
        """Return total commission (broker + exchange)."""
        return self.calculate_broker_commission(n_contracts) + self.calculate_exchange_commission(
            price, n_contracts
        )

