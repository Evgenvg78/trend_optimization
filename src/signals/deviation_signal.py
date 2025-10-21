"""
Deviation-based signal generator.
"""

import numpy as np
import pandas as pd
from .base_signal import BaseSignal
from .signal_registry import SignalRegistry


@SignalRegistry.register
class DeviationSignal(BaseSignal):
    """
    Сигнал по отклонению цены от индикатора (Вариант B)
    
    Генерирует сигналы на основе пробоя ценой уровня индикатора с учетом смещения:
    - Long: если цена > индикатор + offset
    - Short: если цена < индикатор - offset
    
    По умолчанию использует High для определения пробоя вверх и Low для пробоя вниз,
    что обеспечивает более надежное определение направления движения.
    
    Позиция сохраняется до появления нового сигнала (forward-fill).
    """
    
    def __init__(self):
        super().__init__("DeviationSignal")
    
    def generate(self, df: pd.DataFrame, 
                 indicator: pd.Series,
                 offset: float = 0.0,
                 use_hl: bool = True) -> pd.Series:
        """
        Генерация сигналов по отклонению цены от индикатора
        
        Args:
            df: DataFrame с OHLCV данными
            indicator: Series со значениями индикатора
            offset: смещение от линии индикатора (в абсолютных единицах)
            use_hl: True - использовать High/Low для определения пробоев,
                    False - использовать только Close
        
        Returns:
            pd.Series с сигналами: 1 (long), -1 (short), 0 (flat)
        
        Example:
            >>> signal_gen = SignalRegistry.get("DeviationSignal")
            >>> signals = signal_gen.generate(df, indicator, offset=10.0, use_hl=True)
        """
        # Валидация параметров
        if offset < 0:
            raise ValueError("offset должен быть >= 0")
        
        # Генерация сигналов на основе отклонения цены от индикатора
        if use_hl:
            # Используем High для long и Low для short
            raw_signals = np.where(df["HIGH"] > indicator + offset, 1,
                          np.where(df["LOW"] < indicator - offset, -1, np.nan))
        else:
            # Используем только Close
            raw_signals = np.where(df["CLOSE"] > indicator + offset, 1,
                          np.where(df["CLOSE"] < indicator - offset, -1, np.nan))
        
        # Преобразование в Series с forward-fill для сохранения позиции
        signals = pd.Series(raw_signals, index=df.index).ffill().fillna(0)
        
        return signals

