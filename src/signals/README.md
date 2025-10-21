# Signals Module

## Overview

The signals module provides a framework for generating trading signals based on indicator values and price data. It follows the same architecture pattern as the indicators module for consistency and ease of use.

## Architecture

### BaseSignal
Abstract base class that all signal generators must inherit from. Defines the core interface:
- `generate(df, indicator, **params)` - generates trading signals (1=long, -1=short, 0=flat)

### SignalRegistry
Centralized registry for managing signal generators using the decorator pattern:
- `@SignalRegistry.register` - decorator for auto-registration
- `SignalRegistry.get(name)` - retrieve a signal generator by name
- `SignalRegistry.list_signals()` - list all registered signals

## Available Signals

### 1. SlopeSignal (Variant A)

Generates signals based on the slope/direction of the indicator.

**Logic:**
- **Long**: when indicator change > threshold
- **Short**: when indicator change < -threshold
- **Flat**: when change is within [-threshold, threshold]

**Parameters:**
- `threshold` (float, default=0.01): minimum change to trigger signal
- `absolute` (bool, default=False): 
  - True: use absolute units (diff)
  - False: use relative units (pct_change)

**Example:**
```python
from trend_indicator_module.signals import SignalRegistry
from trend_indicator_module.indicators import IndicatorRegistry
from trend_indicator_module.data import CSVLoader

# Load data
loader = CSVLoader("path/to/data")
df = loader.load("TICKER", timeframe=5)

# Calculate indicator
indicator_gen = IndicatorRegistry.get("SMA")
indicator = indicator_gen.calculate(df, period=20)

# Generate signals
signal_gen = SignalRegistry.get("SlopeSignal")
signals = signal_gen.generate(df, indicator, threshold=0.002, absolute=False)

# signals is a pd.Series with values: 1 (long), -1 (short), 0 (flat)
```

### 2. DeviationSignal (Variant B)

Generates signals based on price deviation from the indicator line.

**Logic:**
- **Long**: when price > indicator + offset
- **Short**: when price < indicator - offset

**Parameters:**
- `offset` (float, default=0.0): distance from indicator line (absolute units)
- `use_hl` (bool, default=True):
  - True: use High for long signals, Low for short signals
  - False: use Close for both

**Example:**
```python
from trend_indicator_module.signals import SignalRegistry
from trend_indicator_module.indicators import IndicatorRegistry
from trend_indicator_module.data import CSVLoader

# Load data
loader = CSVLoader("path/to/data")
df = loader.load("TICKER", timeframe=5)

# Calculate indicator
indicator_gen = IndicatorRegistry.get("VolatilityMedian")
indicator = indicator_gen.calculate(df, KATR=2.0, PerATR=14, SMA=5)

# Generate signals
signal_gen = SignalRegistry.get("DeviationSignal")
signals = signal_gen.generate(df, indicator, offset=10.0, use_hl=True)
```

## Signal Behavior

### Position Persistence
All signals use forward-fill to maintain positions until a new signal is generated. This means:
- Once a long signal (1) is generated, it persists until a short signal (-1) is generated
- Positions don't automatically close - they remain active until explicitly changed

### NaN Handling
- Initial NaN values (e.g., from indicator warm-up period) are filled with 0 (flat/no position)
- Signals are always clean: no NaN values in output

### Signal Values
- `1`: Long position (buy/hold long)
- `-1`: Short position (sell/hold short)
- `0`: Flat/no position

## Creating Custom Signals

To create a custom signal generator:

```python
from trend_indicator_module.signals import BaseSignal, SignalRegistry
import pandas as pd

@SignalRegistry.register
class MyCustomSignal(BaseSignal):
    """Description of your signal"""
    
    def __init__(self):
        super().__init__("MyCustomSignal")
    
    def generate(self, df: pd.DataFrame, 
                 indicator: pd.Series,
                 **params) -> pd.Series:
        """
        Generate signals based on your logic
        
        Args:
            df: DataFrame with OHLCV data
            indicator: Series with indicator values
            **params: your custom parameters
        
        Returns:
            pd.Series with signals (1, -1, 0)
        """
        # Your signal logic here
        # Must return pd.Series with same index as df
        # Values must be 1, -1, or 0
        pass
```

## Testing

Comprehensive unit tests are available in `tests/test_signals.py`:
- Test coverage: 30 tests
- Coverage areas: base classes, registry, signal logic, edge cases, integration

Run tests:
```bash
pytest trend_indicator_module/tests/test_signals.py -v
```

## Integration with Backtest Module

The signals module is designed to integrate seamlessly with the backtest module (Week 4):

```python
# Example workflow (backtest module to be implemented)
from trend_indicator_module.data import CSVLoader
from trend_indicator_module.indicators import IndicatorRegistry
from trend_indicator_module.signals import SignalRegistry
# from trend_indicator_module.backtest import BacktestEngine  # Week 4

# Load data
loader = CSVLoader("path/to/data")
df = loader.load("TICKER", timeframe=5)

# Calculate indicator
indicator = IndicatorRegistry.get("VolatilityMedian").calculate(df, KATR=2.0)

# Generate signals
signals = SignalRegistry.get("DeviationSignal").generate(df, indicator, offset=5.0)

# Run backtest (Week 4)
# engine = BacktestEngine()
# results = engine.run(df, signals)
```

## Design Principles

1. **Consistency**: Same architecture as indicators module
2. **Vectorization**: Uses numpy/pandas vectorized operations for performance
3. **Type Safety**: Full type hints throughout
4. **Documentation**: Comprehensive docstrings
5. **Testability**: Easy to test and validate
6. **Extensibility**: Easy to add new signal types

## Status

✅ **Week 3 Complete**
- BaseSignal abstract class implemented
- SignalRegistry implemented
- SlopeSignal (Variant A) implemented
- DeviationSignal (Variant B) implemented
- Comprehensive unit tests (30 tests, all passing)
- Full integration with existing modules

**Next Steps (Week 4):**
- Integrate with backtest module
- Test signals with real trading scenarios
- Optimize signal parameters

