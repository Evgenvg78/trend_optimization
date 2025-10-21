"""
Base signal class for trend indicator module.
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseSignal(ABC):
    """Базовый класс для всех генераторов сигналов"""
    
    def __init__(self, name: str):
        """
        Инициализация генератора сигналов
        
        Args:
            name: уникальное имя генератора сигналов
        """
        self.name = name
    
    @abstractmethod
    def generate(self, df: pd.DataFrame, indicator: pd.Series, **params) -> pd.Series:
        """
        Генерация торговых сигналов
        
        Args:
            df: DataFrame с OHLCV данными (колонки: OPEN, HIGH, LOW, CLOSE, VOL)
            indicator: Series со значениями индикатора
            **params: параметры генерации сигналов (зависят от конкретного генератора)
        
        Returns:
            pd.Series с сигналами (индекс совпадает с df.index):
                1 - long (длинная позиция)
                -1 - short (короткая позиция)
                0 - flat (нет позиции)
        
        Raises:
            ValueError: если параметры невалидны
        """
        pass
    
    def __repr__(self):
        """Строковое представление генератора сигналов"""
        return f"{self.__class__.__name__}(name='{self.name}')"

