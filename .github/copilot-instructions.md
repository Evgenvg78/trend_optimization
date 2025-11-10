## Цель
Короткие, практичные инструкции для AI-агента, который будет помогать править и дополнять этот репозиторий.

Ожидаемый фокус — код в папке `src/` (optimization, backtest, data, signals, indicators), а также ноутбуки и скрипты в корне и `notebooks/`.

## Ключевая архитектура (big picture)
- Компоненты:
  - `src/optimization/` — генерация параметров (ParameterSpace), стратегии поиска, исполнение (`executor.py`) и хранение/экспорт результатов (`results_store.py`, `reporting.py`).
  - `src/backtest/` — движок бэктеста (`engine.py`), расчёт метрик (`metrics.py`), представление сделок/позиций (`trade.py`).
  - `src/data/` — загрузчики данных (`loader.py`, `csv_loader.py`), кэш-менеджер; ожидается DataFrame с колонками OPEN/HIGH/LOW/CLOSE/VOL.
  - `src/signals/` и `src/indicators/` — плагины индикаторов и генераторов сигналов; регистрация выполняется через декораторы `SignalRegistry.register` и `IndicatorRegistry.register`.

## Входные контракты и важные конвенции
- Таблицы рынка: DataFrame с индексом datetime и колонками EXACTLY `OPEN`, `HIGH`, `LOW`, `CLOSE`, `VOL` (заглавные имена). Многие функции проверяют это строго (`BacktestEngine.run`).
- Сигналы: Series значений {-1, 0, 1} после нормализации. Используйте `BaseSignal.generate(df, indicator, **params)` и возвращайте pd.Series, совпадающий по индексу с df.
- SignalFactory: может быть callable или объект с методом `create(parameters)` и должен возвращать либо `SignalConfig`, либо `BaseSignal`, либо `(BaseSignal, params)` — см. `src/optimization/executor.py` (метод `_build_signal_config`).
- Indicator builder: функция signature `(market_data: pd.DataFrame, params: Mapping) -> pd.Series` (см. `OptimizationExecutor.indicator_builder`).
- Параметры поиска: используйте `ParameterDefinition` / `ParameterSpace` (см. `src/optimization/parameter_space.py`). Проверки строгие — валидаторы и шаги/диапазоны требуют согласованности.

## Как запускается оптимизация (рабочий процесс)
- Установите зависимости: `pip install -r requirements.txt` (в корне).
- Создайте BacktestEngine и OptimizationExecutor и вызовите `executor.run(market_data_df, indicator_series, run_id="...", output_root=...)`.
- Output: результаты сохраняются в `output/optimization/<run_id>/` (файлы: `results.csv`, `results.json`, `summary.json`, `results_table.csv`). Контрольные точки пишутся в `output/optimization/checkpoints/<run_id>.json`.
- Важные параметры исполнения: `per_candidate_timeout`, `max_run_time_seconds`, `max_failures`, `resume` — они влияют на параллельность и поведение при ошибках.

## Паттерны и примеры (конкретно для этого проекта)
- Реестр сигналов/индикаторов: используйте декоратор регистрации: `@SignalRegistry.register` (файл `src/signals/signal_registry.py`). Это автоматически создаёт экземпляр и регистрирует по `instance.name`.
- Экспорт результатов: `ResultsStore` реализует `export_csv`/`export_json`; `OptimizationReporter` собирает summary и таблицу (см. `src/optimization/reporting.py`).
- Формат параметров: `ParameterDefinition(values=[...])` для дискретных, либо `lower_bound`/`upper_bound`/`step` для непрерывных (см. `parameter_space.generate_candidates`).

## Частые ошибки и как их исправлять
- Неправильные имена колонок в DataFrame -> ValueError в `BacktestEngine.run`. Всегда проверьте колонки и типы.
- Сигналы не нормализованы -> `signals` после `_prepare_signals` может выбросить ошибку (требуется -1/0/1).
- Неправильный тип, возвращаемый SignalFactory.create -> TypeError в `_build_signal_config`. Возвращайте строго `SignalConfig` / `BaseSignal` / `(BaseSignal, params)`.

## Быстрые указания по изменению кода
- При добавлении нового сигнала: создайте класс наследник `BaseSignal`, реализуйте `generate`, зарегистрируйте через `@SignalRegistry.register`.
- При добавлении индикатора: наследуйте `BaseIndicator`, зарегистрируйте через `@IndicatorRegistry.register`.
- Для изменения экспорта/форматов — правьте `src/optimization/results_store.py` и `src/optimization/reporting.py`.

## Что НЕ документировано тут
- Любые секреты/внешние API ключи — отсутствуют в репозитории. Интеграция со сторонними сервисами не обнаружена (офлайн CSV-данные).

## Где смотреть для деталей
- Основные файлы: `src/optimization/executor.py`, `src/optimization/parameter_space.py`, `src/backtest/engine.py`, `src/data/csv_loader.py`, `src/signals/*`, `src/indicators/*`.

Если нужно, могу сократить или перевести документ на английский, добавить пример minimal runner (import + executor.run) или вставить команды для запуска ноутбуков. Пожалуйста, укажите, какие разделы стоит расширить.
