"""
Technical indicators module.

This module provides a collection of technical indicators for financial analysis.
All indicators are automatically registered in the IndicatorRegistry upon import.

Usage:
    from trend_indicator_module.indicators import IndicatorRegistry
    
    # Get list of available indicators
    print(IndicatorRegistry.list_indicators())
    
    # Get specific indicator
    vm = IndicatorRegistry.get("VolatilityMedian")
    sma = IndicatorRegistry.get("SMA")
    
    # Calculate indicator values
    vm_values = vm.calculate(df, KATR=2.0, PerATR=14)
    sma_values = sma.calculate(df, period=50, price_type="close")
"""

# Import base classes
from .base_indicator import BaseIndicator
from .indicator_registry import IndicatorRegistry

# Import indicators (this triggers automatic registration via @register decorator)
from .volatility_median import VolatilityMedian
from .sma import SMA

# Define public API
__all__ = [
    "BaseIndicator",
    "IndicatorRegistry",
    "VolatilityMedian",
    "SMA",
]
