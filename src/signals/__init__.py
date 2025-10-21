"""
Trading signals generation module.
"""

from .base_signal import BaseSignal
from .signal_registry import SignalRegistry
from .slope_signal import SlopeSignal
from .deviation_signal import DeviationSignal

__all__ = [
    "BaseSignal",
    "SignalRegistry",
    "SlopeSignal",
    "DeviationSignal"
]
