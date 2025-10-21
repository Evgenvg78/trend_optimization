"""
Slope-based signal generator.
"""

import numpy as np
import pandas as pd
from .base_signal import BaseSignal
from .signal_registry import SignalRegistry


@SignalRegistry.register
class SlopeSignal(BaseSignal):
    """
    Сигнал по наклону индикатора (Вариант A)
    
    Генерирует сигналы на основе изменения (наклона) значений индикатора:
    - Long: если изменение индикатора > threshold
    - Short: если изменение индикатора < -threshold
    - Flat: если изменение в пределах [-threshold, threshold]
    
    Позиция сохраняется до появления нового сигнала (forward-fill).
    """
    
    def __init__(self):
        super().__init__("SlopeSignal")
    
    def generate(self, df: pd.DataFrame, 
                 indicator: pd.Series,
                 threshold: float = 0.01,
                 absolute: bool = False) -> pd.Series:
        """
        Генерация сигналов по наклону индикатора
        
        Args:
            df: DataFrame с OHLCV данными
            indicator: Series со значениями индикатора
            threshold: пороговое значение изменения для генерации сигнала
            absolute: True - абсолютные единицы (diff), False - относительные (pct_change)
        
        Returns:
            pd.Series с сигналами: 1 (long), -1 (short), 0 (flat)
        
        Example:
            >>> signal_gen = SignalRegistry.get("SlopeSignal")
            >>> signals = signal_gen.generate(df, indicator, threshold=0.02, absolute=False)
        """
        # Валидация параметров
        if threshold <= 0:
            raise ValueError("threshold должен быть > 0")
        
        # Расчет изменения индикатора
        if absolute:
            slope = indicator.diff()
        else:
            slope = indicator.pct_change()
        
        # Генерация сигналов на основе наклона
        raw_signals = np.where(slope > threshold, 1,
                      np.where(slope < -threshold, -1, np.nan))
        
        # Преобразование в Series с forward-fill для сохранения позиции
        signals = pd.Series(raw_signals, index=df.index).ffill().fillna(0)
        
        return signals

