"""
Base indicator class for trend indicator module.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any


class BaseIndicator(ABC):
    """Базовый класс для всех индикаторов"""
    
    def __init__(self, name: str):
        """
        Инициализация индикатора
        
        Args:
            name: уникальное имя индикатора
        """
        self.name = name
    
    @abstractmethod
    def calculate(self, df: pd.DataFrame, **params) -> pd.Series:
        """
        Рассчитать значения индикатора
        
        Args:
            df: DataFrame с OHLCV данными (колонки: OPEN, HIGH, LOW, CLOSE, VOL)
            **params: параметры индикатора (зависят от конкретного индикатора)
        
        Returns:
            pd.Series с значениями индикатора (индекс совпадает с df.index)
        
        Raises:
            ValueError: если параметры невалидны
        """
        pass
    
    @abstractmethod
    def get_param_grid(self) -> Dict[str, Any]:
        """
        Получить сетку параметров для оптимизации
        
        Returns:
            dict с параметрами и их возможными значениями.
            Ключи - имена параметров, значения - списки возможных значений.
            
        Example:
            {
                "period": [10, 20, 50, 100],
                "multiplier": [1.5, 2.0, 2.5]
            }
        """
        pass
    
    def validate_params(self, **params) -> bool:
        """
        Валидация параметров
        
        Args:
            **params: параметры для проверки
        
        Returns:
            True если параметры валидны
        
        Raises:
            ValueError: если параметры невалидны
        """
        return True
    
    def __repr__(self):
        """Строковое представление индикатора"""
        return f"{self.__class__.__name__}(name='{self.name}')"


