"""
Indicator registry for managing and accessing indicators.
"""

from typing import Dict, List
from .base_indicator import BaseIndicator


class IndicatorRegistry:
    """Реестр индикаторов для централизованного управления"""
    
    _indicators: Dict[str, BaseIndicator] = {}
    
    @classmethod
    def register(cls, indicator_class):
        """
        Декоратор для регистрации индикатора
        
        Автоматически создаёт экземпляр класса и регистрирует его
        в реестре под именем indicator.name
        
        Usage:
            @IndicatorRegistry.register
            class MyIndicator(BaseIndicator):
                def __init__(self):
                    super().__init__("MyIndicator")
                ...
        
        Args:
            indicator_class: класс индикатора (наследник BaseIndicator)
        
        Returns:
            indicator_class: тот же класс (для цепочки декораторов)
        """
        instance = indicator_class()
        cls._indicators[instance.name] = instance
        return indicator_class
    
    @classmethod
    def get(cls, name: str) -> BaseIndicator:
        """
        Получить индикатор по имени
        
        Args:
            name: имя индикатора
        
        Returns:
            BaseIndicator: экземпляр индикатора
        
        Raises:
            ValueError: если индикатор не зарегистрирован
        """
        if name not in cls._indicators:
            available = ", ".join(cls._indicators.keys()) or "нет доступных индикаторов"
            raise ValueError(
                f"Индикатор '{name}' не зарегистрирован. "
                f"Доступные индикаторы: {available}"
            )
        return cls._indicators[name]
    
    @classmethod
    def list_indicators(cls) -> List[str]:
        """
        Список всех зарегистрированных индикаторов
        
        Returns:
            List[str]: список имён индикаторов
        """
        return list(cls._indicators.keys())
    
    @classmethod
    def get_all(cls) -> Dict[str, BaseIndicator]:
        """
        Получить все индикаторы
        
        Returns:
            Dict[str, BaseIndicator]: словарь {имя: индикатор}
        """
        return cls._indicators.copy()


