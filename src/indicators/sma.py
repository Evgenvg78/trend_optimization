"""
Simple Moving Average (SMA) indicator implementation.
"""

import pandas as pd
from typing import Dict, Any
from .base_indicator import BaseIndicator
from .indicator_registry import IndicatorRegistry


@IndicatorRegistry.register
class SMA(BaseIndicator):
    """
    Simple Moving Average (Простая скользящая средняя)
    
    Классический трендовый индикатор, показывающий среднюю цену
    за заданный период времени.
    
    Параметры:
        period (int): период усреднения (по умолчанию 20)
        price_type (str): тип цены для расчёта (по умолчанию "close")
            - "close": цена закрытия
            - "hl2": (High + Low) / 2
            - "hlc3": (High + Low + Close) / 3
            - "ohlc4": (Open + High + Low + Close) / 4
    """
    
    def __init__(self):
        super().__init__("SMA")
    
    def calculate(self, df: pd.DataFrame,
                  period: int = 20,
                  price_type: str = "close") -> pd.Series:
        """
        Расчет Simple Moving Average
        
        Args:
            df: DataFrame с OHLCV данными
            period: период усреднения
            price_type: тип цены ("close", "hl2", "hlc3", "ohlc4")
        
        Returns:
            pd.Series с значениями SMA
        
        Raises:
            ValueError: если параметры невалидны
        """
        # Валидация параметров
        self.validate_params(period=period, price_type=price_type)
        
        # Выбор типа цены
        price = self._get_price(df, price_type)
        
        # Расчёт простой скользящей средней
        sma = price.rolling(window=period, min_periods=period).mean()
        
        return sma
    
    def _get_price(self, df: pd.DataFrame, price_type: str) -> pd.Series:
        """
        Получить ценовой ряд для расчёта
        
        Args:
            df: DataFrame с OHLCV данными
            price_type: тип цены
        
        Returns:
            pd.Series с ценами
        """
        if price_type == "close":
            return df["CLOSE"]
        elif price_type == "hl2":
            return (df["HIGH"] + df["LOW"]) / 2
        elif price_type == "hlc3":
            return (df["HIGH"] + df["LOW"] + df["CLOSE"]) / 3
        elif price_type == "ohlc4":
            return (df["OPEN"] + df["HIGH"] + df["LOW"] + df["CLOSE"]) / 4
        else:
            raise ValueError(f"Неизвестный тип цены: {price_type}")
    
    def get_param_grid(self) -> Dict[str, Any]:
        """
        Сетка параметров для оптимизации
        
        Returns:
            dict с параметрами и их возможными значениями
        """
        return {
            "period": [10, 20, 50, 100, 200],
            "price_type": ["close", "hl2", "hlc3", "ohlc4"]
        }
    
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
        if "period" in params and params["period"] < 1:
            raise ValueError("period должен быть >= 1")
        
        if "price_type" in params:
            valid_types = ["close", "hl2", "hlc3", "ohlc4"]
            if params["price_type"] not in valid_types:
                raise ValueError(
                    f"price_type должен быть одним из: {', '.join(valid_types)}"
                )
        
        return True


