# Migration Guide: Reducing Coupling with Dependency Injection

## Overview
This guide shows how to refactor existing subsystem code to use the new dependency injection patterns and reduce coupling between subsystems.

## Current Coupling Issues Identified

### Cross-Subsystem Imports (12 total)
1. **Analyze → Acquire**: 6 modules import `acquire.DailyData`
2. **Analyze → Analyze**: 2 modules import `analyze.PULSyrGain_pct` 
3. **Monitor → Acquire**: 2 modules import `acquire.DailyData`
4. **Monitor → Analyze**: 1 module imports `analyze.PULSyrGain_pct`
5. **Select_assets → Acquire/Analyze**: 2 modules import from both

## Migration Steps

### Step 1: Replace Direct Acquire Imports

#### Before (Current Pattern)
```python
# In Analyze modules
from acquire.DailyData import daily_data

def analyze_symbol(symbol):
    data = daily_data(symbol, period='1y')
    # analysis logic
    return result
```

#### After (Refactored Pattern)
```python
# In Analyze modules
from common.providers import get_default_data_provider

def analyze_symbol(symbol, data_provider=None):
    if data_provider is None:
        data_provider = get_default_data_provider()
    
    market_data = data_provider.get_market_data(symbol, period='1y')
    df = market_data.to_dataframe()
    # analysis logic
    return result
```

### Step 2: Update Portfolio Position Usage

#### Before (Current Pattern)
```python
# In Monitor modules
from monitor.read_portfolio_positions import read_portfolio_positions

def portfolio_analysis():
    df, file_path, timestamp = read_portfolio_positions('data/accountActivity/')
    # analysis logic
    return result
```

#### After (Refactored Pattern)
```python
# In Monitor modules
from common.providers import get_default_portfolio_provider

def portfolio_analysis():
    portfolio_provider = get_default_portfolio_provider()
    metrics = portfolio_provider.get_portfolio_metrics()
    positions = metrics.positions
    # analysis logic
    return result
```

### Step 3: Refactor Specific Modules

#### Files to Update

**Analyze Subsystem:**
1. `Analyze/PULSyrGain_pct.py`
2. `Analyze/plot_MyTickers.py`
3. `Analyze/plot_annualized_returns_grid.py`
4. `Analyze/correlation_top.py`
5. `Analyze/market_cycle.py`
6. `Analyze/top_etfs_1d_return.py`

**Monitor Subsystem:**
1. `Monitor/portfolio_performance.py`
2. `Monitor/portfolio_value_6mo.py`

**Select_assets Subsystem:**
1. `Select_assets/screen.py`

#### Example: Refactoring PULSyrGain_pct.py

```python
# Before
from acquire.DailyData import daily_data

def PULSyrGain_pct(symbol='PULS', period='10y'):
    data = daily_data(symbol, period=period)
    # PULS-specific analysis
    return yearly_gains

# After
from common.providers import get_default_data_provider

def PULSyrGain_pct(symbol='PULS', period='10y', data_provider=None):
    if data_provider is None:
        data_provider = get_default_data_provider()
    
    market_data = data_provider.get_market_data(symbol, period=period)
    df = market_data.to_dataframe()
    # PULS-specific analysis
    return yearly_gains
```

### Step 4: Update Test Files

#### Before (Current Test Pattern)
```python
# In test files
from acquire.DailyData import daily_data

def test_analysis():
    data = daily_data('SPY', period='1mo')
    # test logic
```

#### After (Refactored Test Pattern)
```python
# In test files
from common.providers import MockDataProvider
from common.refactoring_examples import create_test_data

def test_analysis():
    mock_provider = MockDataProvider(create_test_data())
    data = mock_provider.get_market_data('SPY', period='1mo')
    # test logic
```

## Step-by-Step Migration Process

### Phase 1: Setup (Day 1)
1. ✅ Create `common/` directory structure
2. ✅ Implement `data_contracts.py`
3. ✅ Implement `providers.py`
4. ✅ Move `portfolio_utils.py` to common
5. ✅ Create `refactoring_examples.py`

### Phase 2: Core Dependencies (Days 2-3)
1. Update `Analyze/PULSyrGain_pct.py` (most imported module)
2. Update `Monitor/portfolio_performance.py`
3. Update `Analyze/plot_MyTickers.py`
4. Update `Analyze/plot_annualized_returns_grid.py`

### Phase 3: Remaining Modules (Days 4-5)
1. Update remaining Analyze modules
2. Update Monitor modules
3. Update Select_assets modules
4. Update all test files to use mock providers

### Phase 4: Validation (Day 6)
1. Run full test suite
2. Verify all functionality works
3. Update documentation
4. Add deprecation warnings to old import paths

## Testing Strategy

### Before Migration
```bash
# Run current tests to establish baseline
pytest tests/ -v
```

### During Migration
```bash
# Test specific modules as they're refactored
pytest tests/test_PULSyrGain_pct.py -v
pytest tests/test_portfolio_performance.py -v
```

### After Migration
```bash
# Run full test suite with new patterns
pytest tests/ -v --cov=common
```

## Benefits of Migration

### Reduced Coupling
- **Before**: 12 cross-subsystem imports
- **After**: 0 cross-subsystem imports (all through interfaces)

### Improved Testability
- **Before**: Hard to mock external dependencies
- **After**: Easy dependency injection for testing

### Better Maintainability
- **Before**: Changes in Acquire break Analyze/Monitor
- **After**: Interface changes are backward compatible

### Enhanced Flexibility
- **Before**: Fixed data sources
- **After**: Swappable data providers (cached, live, mock)

## Rollback Plan

If migration causes issues:
1. Keep original files as backup (`*_original.py`)
2. Use feature flags to switch between old/new implementations
3. Gradual rollout with monitoring
4. Quick revert by restoring original imports

## Validation Checklist

- [ ] All tests pass with new patterns
- [ ] No performance regression
- [ ] Documentation updated
- [ ] Deprecation warnings added
- [ ] Migration guide completed
- [ ] Team training conducted

## Common Issues and Solutions

### Issue: Import Errors
**Solution**: Update `__init__.py` files to properly export new interfaces

### Issue: Test Failures
**Solution**: Create comprehensive mock data generators for testing

### Issue: Performance Regression
**Solution**: Benchmark critical paths before/after migration

### Issue: Team Adoption
**Solution**: Provide training sessions and clear documentation

## Timeline

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 1 | Phase 1-2 | Core structure, key modules refactored |
| 2 | Phase 3 | All modules refactored, tests updated |
| 3 | Phase 4 | Validation, documentation, training |

## Success Metrics

- **Coupling**: Reduce cross-subsystem imports from 12 to 0
- **Test Coverage**: Maintain or improve current coverage (~75%)
- **Performance**: No more than 5% performance impact
- **Adoption**: 100% of new code uses dependency injection patterns

This migration will transform the trading system into a more modular, testable, and maintainable architecture while preserving all existing functionality.
