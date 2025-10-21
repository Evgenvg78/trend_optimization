"""
Volatility Median indicator implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from .base_indicator import BaseIndicator
from .indicator_registry import IndicatorRegistry


@IndicatorRegistry.register
class VolatilityMedian(BaseIndicator):
    """
    Volatility Median индикатор
    
    Трендовый индикатор на основе ATR с адаптивной волатильной полосой.
    Индикатор следует за ценой с учётом волатильности рынка.
    
    Параметры:
        KATR (float): коэффициент ATR для расчёта смещения (по умолчанию 2.0)
        PerATR (int): период для расчёта ATR (по умолчанию 14)
        SMA (int): период сглаживания результата (по умолчанию 1)
        MinRA (float): минимальное расстояние смещения (по умолчанию 0.5)
        FlATR (int): тип расчёта ATR (0 - mean, 1 - max)
        FlHL (int): тип базовой цены (0 - (H+L)/2, 1 - (H+L+C)/3)
    """
    
    def __init__(self):
        super().__init__("VolatilityMedian")
    
    def calculate(self, df: pd.DataFrame,
                  KATR: float = 2.0,
                  PerATR: int = 14,
                  SMA: int = 1,
                  MinRA: float = 0.5,
                  FlATR: int = 0,
                  FlHL: int = 0) -> pd.Series:
        """
        Расчет Volatility Median
        
        Args:
            df: DataFrame с OHLCV данными
            KATR: коэффициент ATR
            PerATR: период ATR
            SMA: период сглаживания
            MinRA: минимальное расстояние
            FlATR: флаг типа ATR (0 - mean, 1 - max)
            FlHL: флаг базовой цены (0 - HL/2, 1 - HLC/3)
        
        Returns:
            pd.Series с значениями индикатора
        
        Raises:
            ValueError: если параметры невалидны
        """
        # Валидация параметров
        self.validate_params(KATR=KATR, PerATR=PerATR, SMA=SMA, 
                           MinRA=MinRA, FlATR=FlATR, FlHL=FlHL)
        
        # Извлечение данных
        h = df["HIGH"].values
        l = df["LOW"].values
        c = df["CLOSE"].values
        n = len(df)
        
        # Расчёт True Range
        tr = np.empty(n)
        tr[:] = np.nan
        tr[1:] = np.maximum.reduce([
            h[1:] - l[1:],
            np.abs(h[1:] - c[:-1]),
            np.abs(l[1:] - c[:-1])
        ])
        
        # Расчёт ATR (среднее или максимум)
        if FlATR == 0:
            atr = pd.Series(tr).rolling(PerATR, min_periods=PerATR).mean().to_numpy()
        else:
            atr = pd.Series(tr).rolling(PerATR, min_periods=PerATR).max().to_numpy()
        
        # Выбор базовой цены
        if FlHL == 0:
            price = (h + l) / 2
        else:
            price = (h + l + c) / 3
        
        # Расчёт смещения (максимум из KATR*ATR и MinRA)
        offset = np.maximum(KATR * atr, MinRA)
        
        # Расчёт Volatility Range
        vr = np.empty(n)
        vr[:] = np.nan
        
        for i in range(n):
            if np.isnan(offset[i]):
                continue
            
            # Инициализация первого значения
            if i == 0 or np.isnan(vr[i-1]):
                vr[i] = price[i] - offset[i]
            else:
                prev = vr[i-1]
                # Если цена выше предыдущего значения индикатора
                if price[i] > prev:
                    vr[i] = max(prev, price[i] - offset[i])
                # Если цена ниже или равна
                else:
                    vr[i] = min(prev, price[i] + offset[i])
        
        # Сглаживание результата через SMA
        result = pd.Series(vr, index=df.index).rolling(SMA, min_periods=SMA).mean()
        
        return result
    
    def get_param_grid(self) -> Dict[str, Any]:
        """
        Сетка параметров для оптимизации
        
        Returns:
            dict с параметрами и их возможными значениями
        """
        return {
            "KATR": [2, 3, 5, 8, 11, 15, 18, 25],
            "PerATR": [5, 10, 14, 20, 30],
            "SMA": [1, 3, 5, 10, 20],
            "MinRA": [0.5, 1, 3, 5, 7, 10],
            "FlATR": [0, 1],
            "FlHL": [0, 1]
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
        if "KATR" in params and params["KATR"] <= 0:
            raise ValueError("KATR должен быть > 0")
        
        if "PerATR" in params and params["PerATR"] < 1:
            raise ValueError("PerATR должен быть >= 1")
        
        if "SMA" in params and params["SMA"] < 1:
            raise ValueError("SMA должен быть >= 1")
        
        if "MinRA" in params and params["MinRA"] < 0:
            raise ValueError("MinRA должен быть >= 0")
        
        if "FlATR" in params and params["FlATR"] not in [0, 1]:
            raise ValueError("FlATR должен быть 0 или 1")
        
        if "FlHL" in params and params["FlHL"] not in [0, 1]:
            raise ValueError("FlHL должен быть 0 или 1")
        
        return True


