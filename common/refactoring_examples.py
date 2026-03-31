"""Example refactored subsystem functions demonstrating dependency injection via common interfaces."""

from typing import List, Dict, Any

import numpy as np
import pandas as pd

from config import RISK_FREE_RATE
from .data_contracts import MarketData, PortfolioPosition, TradingSignal
from .financial_utils import calculate_max_drawdown
from .providers import DataProvider, PortfolioProvider, get_default_data_provider, get_default_portfolio_provider


# Example 1: Refactored Analyze subsystem function
def calculate_correlation_matrix_refactored(symbols: List[str],
                                          data_provider: DataProvider = None,
                                          period: str = "1y") -> pd.DataFrame:
    """Calculate correlation matrix using DataProvider dependency injection."""
    if data_provider is None:
        data_provider = get_default_data_provider()
    
    # Get data for all symbols through the provider interface
    market_data_dict = data_provider.get_multiple_symbols(symbols, period)
    
    # Extract close prices for correlation
    close_prices = {}
    for symbol, market_data in market_data_dict.items():
        close_prices[symbol] = market_data.close
    
    # Create DataFrame and calculate correlation
    price_df = pd.DataFrame(close_prices)
    correlation_matrix = price_df.pct_change().corr()
    
    return correlation_matrix


# Example 2: Refactored portfolio performance analysis
def analyze_portfolio_performance_refactored(portfolio_provider: PortfolioProvider = None,
                                           data_provider: DataProvider = None) -> dict:
    """Analyze portfolio performance using PortfolioProvider and DataProvider interfaces."""
    if portfolio_provider is None:
        portfolio_provider = get_default_portfolio_provider()
    if data_provider is None:
        data_provider = get_default_data_provider()
    
    # Get portfolio data through provider
    metrics = portfolio_provider.get_portfolio_metrics()
    positions = metrics.positions
    
    # Get historical data for performance analysis
    symbols = [pos.symbol for pos in positions]
    historical_data = data_provider.get_multiple_symbols(symbols, period="1y")
    
    # Calculate performance metrics
    performance_results = {}
    for position in positions:
        if position.symbol in historical_data:
            market_data = historical_data[position.symbol]
            # Calculate various performance metrics
            returns = market_data.close.pct_change().dropna()
            performance_results[position.symbol] = {
                'volatility': returns.std() * np.sqrt(252),
                'current_return': position.unrealized_pnl() / position.cost_basis if position.cost_basis > 0 else 0,
                'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            }
    
    return {
        'portfolio_metrics': metrics,
        'individual_performance': performance_results,
        'total_positions': len(positions),
        'analysis_date': pd.Timestamp.now()
    }


# Example 3: Refactored market cycle analysis
def analyze_market_cycle_refactored(symbols: List[str] = None,
                                  data_provider: DataProvider = None,
                                  period: str = "24mo") -> dict:
    """Compute market breadth indicators for a set of symbols using DataProvider injection."""
    if symbols is None:
        symbols = ['SPY', 'QQQ', 'IWM', 'VTI', 'VOO']
    
    if data_provider is None:
        data_provider = get_default_data_provider()
    
    # Get market data through provider
    market_data_dict = data_provider.get_multiple_symbols(symbols, period)
    
    # Calculate market breadth indicators
    breadth_metrics = {}
    for symbol, market_data in market_data_dict.items():
        prices = market_data.close
        returns = prices.pct_change().dropna()
        
        # Calculate moving averages
        sma_50 = prices.rolling(window=50).mean()
        sma_200 = prices.rolling(window=200).mean()
        
        # Market breadth indicators
        above_sma_50 = (prices > sma_50).mean()
        above_sma_200 = (prices > sma_200).mean()
        
        breadth_metrics[symbol] = {
            'above_sma_50_pct': above_sma_50 * 100,
            'above_sma_200_pct': above_sma_200 * 100,
            'current_trend': 'bull' if prices.iloc[-1] > sma_200.iloc[-1] else 'bear',
            'volatility': returns.std() * np.sqrt(252)
        }
    
    return {
        'symbols_analyzed': symbols,
        'period': period,
        'breadth_metrics': breadth_metrics,
        'analysis_date': pd.Timestamp.now()
    }


# Example 4: Refactored strategy backtesting
def backtest_strategy_refactored(strategy_func: callable,
                               symbols: List[str],
                               data_provider: DataProvider = None,
                               period: str = "2y") -> dict:
    """Backtest a strategy function across symbols using DataProvider dependency injection.
    Args:
        strategy_func: Function that takes MarketData and returns TradingSignal
        symbols: Symbols to backtest on
        data_provider: Data provider instance
        period: Backtest period
    """
    if data_provider is None:
        data_provider = get_default_data_provider()
    
    results = {}
    
    for symbol in symbols:
        try:
            # Get historical data
            market_data = data_provider.get_market_data(symbol, period)
            
            # Generate signals using strategy
            signals = []
            for i in range(50, len(market_data.timestamp)):  # Skip first 50 for indicators
                # Create slice of data for strategy
                slice_data = MarketData(
                    symbol=symbol,
                    timestamp=market_data.timestamp[:i+1],
                    open=market_data.open[:i+1],
                    high=market_data.high[:i+1],
                    low=market_data.low[:i+1],
                    close=market_data.close[:i+1],
                    adjusted_close=market_data.adjusted_close[:i+1],
                    volume=market_data.volume[:i+1]
                )
                
                signal = strategy_func(slice_data)
                if signal:
                    signals.append(signal)
            
            # Calculate performance
            if signals:
                returns = calculate_strategy_returns(signals, market_data)
                # Convert returns to cumulative prices for drawdown calculation
                cumulative_prices = (1 + returns).cumprod()
                results[symbol] = {
                    'total_signals': len(signals),
                    'returns': returns,
                    'sharpe_ratio': calculate_sharpe_ratio(returns),
                    'max_drawdown': calculate_max_drawdown(cumulative_prices)
                }
            else:
                results[symbol] = {'error': 'No signals generated'}
                
        except Exception as e:
            results[symbol] = {'error': str(e)}
    
    return results


# Helper functions for backtesting
def calculate_strategy_returns(signals: List[TradingSignal], market_data: MarketData) -> pd.Series:
    """Calculate returns from trading signals."""
    # Simple implementation - would be more sophisticated in practice
    returns = pd.Series(0.0, index=market_data.timestamp)
    
    for signal in signals:
        if signal.signal_type == 'BUY' and signal.is_actionable():
            # Mark entry point
            entry_time = signal.timestamp
            if entry_time in returns.index:
                # Calculate return from entry to next signal or end
                exit_time = market_data.timestamp[-1]
                if entry_time in market_data.timestamp:
                    entry_price = market_data.close[market_data.timestamp.get_loc(entry_time)]
                    exit_price = market_data.close[-1]
                    period_return = (exit_price / entry_price - 1)
                    returns.loc[entry_time:] = period_return
    
    return returns


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> float:
    """Calculate Sharpe ratio."""
    if returns.std() == 0:
        return 0.0
    excess_returns = returns.mean() * 252 - risk_free_rate
    return excess_returns / (returns.std() * np.sqrt(252))


# Example usage with mock providers for testing
class MockDataProvider(DataProvider):
    """Mock data provider for testing."""

    def __init__(self, mock_data: dict = None) -> None:
        """Initialize provider."""
        self.mock_data = mock_data or {}

    def get_market_data(self, symbol: str, period: str = None, start: pd.Timestamp = None) -> MarketData:
        """Fetch market data for a symbol."""
        if symbol not in self.mock_data:
            raise ValueError(f"No mock data for {symbol}")
        return self.mock_data[symbol]

    def get_multiple_symbols(self, symbols: List[str], period: str = None) -> dict:
        """Fetch market data for multiple symbols."""
        return {symbol: self.get_market_data(symbol, period) for symbol in symbols}

    def is_data_available(self, symbol: str, period: str = None) -> bool:
        """Check if data is available for a symbol."""
        return symbol in self.mock_data


def create_test_data() -> dict:
    """Create test market data for examples."""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=252)
    
    spy_data = MarketData(
        symbol='SPY',
        timestamp=dates,
        open=pd.Series(100 + np.random.randn(252).cumsum()),
        high=pd.Series(102 + np.random.randn(252).cumsum()),
        low=pd.Series(98 + np.random.randn(252).cumsum()),
        close=pd.Series(100 + np.random.randn(252).cumsum()),
        adjusted_close=pd.Series(100 + np.random.randn(252).cumsum()),
        volume=pd.Series(np.random.randint(1000000, 10000000, 252))
    )
    
    return {'SPY': spy_data}


# Example of how to use the refactored code
if __name__ == "__main__":
    # Using default providers (production)
    correlation_matrix = calculate_correlation_matrix_refactored(['SPY', 'QQQ', 'IWM'])
    print("Correlation Matrix:")
    print(correlation_matrix)
    
    # Using mock providers (testing)
    mock_provider = MockDataProvider(create_test_data())
    test_correlation = calculate_correlation_matrix_refactored(['SPY'], data_provider=mock_provider)
    print("\nTest Correlation:")
    print(test_correlation)
