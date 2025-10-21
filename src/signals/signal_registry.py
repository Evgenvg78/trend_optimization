"""
Signal registry for managing and accessing signal generators.
"""

from typing import Dict, List
from .base_signal import BaseSignal


class SignalRegistry:
    """Реестр генераторов сигналов для централизованного управления"""
    
    _signals: Dict[str, BaseSignal] = {}
    
    @classmethod
    def register(cls, signal_class):
        """
        Декоратор для регистрации генератора сигналов
        
        Автоматически создаёт экземпляр класса и регистрирует его
        в реестре под именем signal.name
        
        Usage:
            @SignalRegistry.register
            class MySignal(BaseSignal):
                def __init__(self):
                    super().__init__("MySignal")
                ...
        
        Args:
            signal_class: класс генератора сигналов (наследник BaseSignal)
        
        Returns:
            signal_class: тот же класс (для цепочки декораторов)
        """
        instance = signal_class()
        cls._signals[instance.name] = instance
        return signal_class
    
    @classmethod
    def get(cls, name: str) -> BaseSignal:
        """
        Получить генератор сигналов по имени
        
        Args:
            name: имя генератора сигналов
        
        Returns:
            BaseSignal: экземпляр генератора сигналов
        
        Raises:
            ValueError: если генератор сигналов не зарегистрирован
        """
        if name not in cls._signals:
            available = ", ".join(cls._signals.keys()) or "нет доступных генераторов сигналов"
            raise ValueError(
                f"Генератор сигналов '{name}' не зарегистрирован. "
                f"Доступные генераторы: {available}"
            )
        return cls._signals[name]
    
    @classmethod
    def list_signals(cls) -> List[str]:
        """
        Список всех зарегистрированных генераторов сигналов
        
        Returns:
            List[str]: список имён генераторов сигналов
        """
        return list(cls._signals.keys())
    
    @classmethod
    def get_all(cls) -> Dict[str, BaseSignal]:
        """
        Получить все генераторы сигналов
        
        Returns:
            Dict[str, BaseSignal]: словарь {имя: генератор сигналов}
        """
        return cls._signals.copy()

