"""Standardized data contracts (dataclasses/schemas) for consistency across all trading subsystems."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np


@dataclass
class MarketData:
    """Standard contract for market data across all subsystems."""
    symbol: str
    timestamp: pd.DatetimeIndex
    open: pd.Series
    high: pd.Series
    low: pd.Series
    close: pd.Series
    volume: pd.Series
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to standard DataFrame format."""
        df = pd.DataFrame({
            'Open': self.open.values,
            'High': self.high.values,
            'Low': self.low.values,
            'Close': self.close.values,
            'Volume': self.volume.values
        }, index=self.timestamp)
        df.index.name = 'Date'
        return df
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, symbol: str) -> 'MarketData':
        """Create from standard DataFrame format."""
        return cls(
            symbol=symbol,
            timestamp=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            volume=df['Volume']
        )


@dataclass
class PortfolioPosition:
    """Standard contract for portfolio positions."""
    symbol: str
    quantity: float
    cost_basis: float
    current_price: float
    current_value: float
    last_updated: datetime
    
    def unrealized_pnl(self) -> float:
        """Calculate unrealized profit/loss."""
        if self.cost_basis == 0 or self.quantity == 0:
            return 0.0
        return (self.current_price - self.cost_basis) * self.quantity
    
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized profit/loss percentage."""
        if self.cost_basis == 0 or self.quantity == 0:
            return 0.0
        return (self.current_price - self.cost_basis) / self.cost_basis


@dataclass
class PortfolioMetrics:
    """Standard contract for portfolio performance metrics."""
    total_value: float
    total_cost: float
    total_pnl: float
    total_pnl_pct: float
    positions: List[PortfolioPosition]
    cash_balance: float
    last_updated: datetime
    
    def get_position_weights(self) -> Dict[str, float]:
        """Calculate position weights as percentage of total portfolio."""
        if self.total_value == 0:
            return {}
        return {
            pos.symbol: pos.current_value / self.total_value 
            for pos in self.positions
        }
    
    def get_top_positions(self, n: int = 10) -> List[PortfolioPosition]:
        """Get top N positions by current value."""
        return sorted(self.positions, key=lambda x: x.current_value, reverse=True)[:n]


@dataclass
class TradingSignal:
    """Standard contract for trading signals."""
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    strength: float   # Signal strength 0-1
    confidence: float # Confidence level 0-1
    timestamp: datetime
    metadata: Dict[str, Any]
    
    def is_actionable(self) -> bool:
        """Check if signal is strong enough to act on."""
        return (self.signal_type in ['BUY', 'SELL'] and 
                self.strength > 0.5 and 
                self.confidence > 0.6)


@dataclass
class BacktestResult:
    """Standard contract for backtest results."""
    strategy_name: str
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    equity_curve: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]
    
    def get_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio."""
        return self.metrics.get('sharpe_ratio', 0.0)
    
    def get_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        return self.metrics.get('max_drawdown', 0.0)
    
    def get_total_return(self) -> float:
        """Calculate total return."""
        return self.metrics.get('total_return', 0.0)


# Standard DataFrame column names for consistency
class DataColumns:
    """Standard column names used across the system."""
    
    # Market data columns
    DATE = 'Date'
    OPEN = 'Open'
    HIGH = 'High'
    LOW = 'Low'
    CLOSE = 'Close'
    VOLUME = 'Volume'
    
    # Portfolio columns
    SYMBOL = 'Symbol'
    QUANTITY = 'Quantity'
    COST_BASIS = 'Cost Basis'
    CURRENT_PRICE = 'Current Price'
    CURRENT_VALUE = 'Current Value'
    LAST_UPDATED = 'Last Updated'
    
    # Analysis columns
    RETURNS = 'Returns'
    LOG_RETURNS = 'Log Returns'
    SMA_20 = 'SMA20'
    SMA_50 = 'SMA50'
    SMA_200 = 'SMA200'
    RSI = 'RSI'
    
    @classmethod
    def get_market_data_columns(cls) -> List[str]:
        """Get standard market data column list."""
        return [cls.OPEN, cls.HIGH, cls.LOW, cls.CLOSE, cls.VOLUME]
    
    @classmethod
    def validate_market_data(cls, df: pd.DataFrame) -> bool:
        """Validate DataFrame has required market data columns."""
        required_cols = cls.get_market_data_columns()
        return all(col in df.columns for col in required_cols)
