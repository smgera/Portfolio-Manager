"""Abstract data provider interfaces and concrete implementations for dependency injection."""

from abc import ABC, abstractmethod
import logging
from typing import Optional, Dict, Any, List

import pandas as pd

from config import ACCOUNT_ACTIVITY_DIR
from .data_contracts import MarketData, PortfolioPosition, PortfolioMetrics


class DataProvider(ABC):
    """Abstract interface for market data providers."""
    
    @abstractmethod
    def get_market_data(self, symbol: str, period: Optional[str] = None,
                       start: Optional[pd.Timestamp] = None) -> MarketData:
        """Get market data for a symbol.
        Args:
            symbol: Stock ticker symbol
            period: Relative period (e.g., '1mo', '1y')
            start: Explicit start date
        Returns:
            MarketData object with standardized format
        """
        pass

    @abstractmethod
    def get_multiple_symbols(self, symbols: List[str], period: Optional[str] = None) -> Dict[str, MarketData]:
        """Get market data for multiple symbols.
        Args:
            symbols: List of stock ticker symbols
            period: Relative period for all symbols
        Returns:
            Dictionary mapping symbols to MarketData objects
        """
        pass

    @abstractmethod
    def is_data_available(self, symbol: str, period: Optional[str] = None) -> bool:
        """Check if data is available for a symbol and period.
        Args:
            symbol: Stock ticker symbol
            period: Relative period
        Returns:
            True if data is available, False otherwise
        """
        pass


class PortfolioProvider(ABC):
    """Abstract interface for portfolio data providers."""
    
    @abstractmethod
    def get_current_positions(self) -> List[PortfolioPosition]:
        """Get current portfolio positions.
        Returns:
            List of current portfolio positions
        """
        pass

    @abstractmethod
    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Get portfolio performance metrics.
        Returns:
            PortfolioMetrics object with performance data
        """
        pass


class CachedDataProvider(DataProvider):
    """Implementation that uses the existing Acquire subsystem with caching."""

    def __init__(self) -> None:
        """Initialize CachedDataProvider with empty cache."""
        self.logger = logging.getLogger(__name__)

    def get_market_data(self, symbol: str, period: Optional[str] = None,
                       start: Optional[pd.Timestamp] = None) -> MarketData:
        """Get market data using the existing DailyData caching system."""
        try:
            # Import here to avoid circular dependencies
            from Acquire.DailyData import daily_data
            
            self.logger.debug(f"Getting market data for {symbol}, period={period}")
            
            # Use existing daily_data function
            df = daily_data(symbol, period=period, start=start)
            
            if df is None or df.empty:
                raise ValueError(f"No data available for symbol {symbol}")
            
            # Ensure required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            return MarketData.from_dataframe(df, symbol)
            
        except Exception as e:
            self.logger.error(f"Failed to get market data for {symbol}: {e}")
            raise
    
    def get_multiple_symbols(self, symbols: List[str], period: Optional[str] = None) -> Dict[str, MarketData]:
        """Get market data for multiple symbols."""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_market_data(symbol, period)
            except Exception as e:
                self.logger.warning(f"Failed to get data for {symbol}: {e}")
                continue
        return results
    
    def is_data_available(self, symbol: str, period: Optional[str] = None) -> bool:
        """Check if data is available."""
        try:
            from Acquire.DailyData import daily_data
            df = daily_data(symbol, period=period)
            return df is not None and not df.empty
        except Exception:
            return False


class FileBasedPortfolioProvider(PortfolioProvider):
    """Implementation that reads portfolio data from CSV files."""

    def __init__(self, data_directory: str = None) -> None:
        """Initialize FileBasedPortfolioProvider."""
        if data_directory is None:
            self.data_directory = str(ACCOUNT_ACTIVITY_DIR)
        else:
            self.data_directory = data_directory
        self.logger = logging.getLogger(__name__)
    
    def get_current_positions(self) -> List[PortfolioPosition]:
        """Get current positions from CSV files."""
        try:
            from Monitor.read_portfolio_positions import read_portfolio_positions
            
            df, newest_file, timestamp = read_portfolio_positions(self.data_directory)
            
            if df.empty:
                return []
            
            positions = []
            for _, row in df.iterrows():
                position = PortfolioPosition(
                    symbol=row['Symbol'],
                    quantity=row.get('Quantity', 0),
                    cost_basis=row.get('Cost Basis', row['Last Price']),
                    current_price=row['Last Price'],
                    current_value=row['Current Value'],
                    last_updated=timestamp
                )
                positions.append(position)
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Failed to read portfolio positions: {e}")
            return []
    
    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Get portfolio metrics."""
        try:
            positions = self.get_current_positions()
            
            total_value = sum(pos.current_value for pos in positions)
            total_cost = sum(pos.cost_basis * pos.quantity for pos in positions)
            total_pnl = sum(pos.unrealized_pnl() for pos in positions)
            total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
            
            return PortfolioMetrics(
                total_value=total_value,
                total_cost=total_cost,
                total_pnl=total_pnl,
                total_pnl_pct=total_pnl_pct,
                positions=positions,
                cash_balance=0.0,  # Would need to be calculated from account data
                last_updated=pd.Timestamp.now()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate portfolio metrics: {e}")
            # Return empty metrics
            return PortfolioMetrics(
                total_value=0.0,
                total_cost=0.0,
                total_pnl=0.0,
                total_pnl_pct=0.0,
                positions=[],
                cash_balance=0.0,
                last_updated=pd.Timestamp.now()
            )


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
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_market_data(symbol, period)
            except ValueError:
                # Skip symbols that don't have mock data
                continue
        return results

    def is_data_available(self, symbol: str, period: str = None) -> bool:
        """Check if data is available for a symbol."""
        return symbol in self.mock_data


class ProviderFactory:
    """Factory for creating data providers with proper configuration."""
    
    @staticmethod
    def create_data_provider(provider_type: str = "cached") -> DataProvider:
        """Create a data provider instance.
        Args:
            provider_type: Type of provider to create ('cached', 'live', etc.)
        Returns:
            DataProvider instance
        """
        if provider_type == "cached":
            return CachedDataProvider()
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
    
    @staticmethod
    def create_portfolio_provider(provider_type: str = "file",
                                 data_directory: str = None) -> PortfolioProvider:
        """Create a portfolio provider instance.
        Args:
            provider_type: Type of provider to create
            data_directory: Directory for portfolio data files (defaults to config ACCOUNT_ACTIVITY_DIR)
        Returns:
            PortfolioProvider instance
        """
        if provider_type == "file":
            return FileBasedPortfolioProvider(data_directory)
        else:
            raise ValueError(f"Unknown portfolio provider type: {provider_type}")


# Module-level registry for default provider instances (use dict to avoid global keyword)
_PROVIDER_REGISTRY: Dict[str, Any] = {
    'data': None,
    'portfolio': None,
}


def get_default_data_provider() -> DataProvider:
    """Get the default data provider instance."""
    if _PROVIDER_REGISTRY['data'] is None:
        _PROVIDER_REGISTRY['data'] = ProviderFactory.create_data_provider()
    return _PROVIDER_REGISTRY['data']


def get_default_portfolio_provider() -> PortfolioProvider:
    """Get the default portfolio provider instance."""
    if _PROVIDER_REGISTRY['portfolio'] is None:
        _PROVIDER_REGISTRY['portfolio'] = ProviderFactory.create_portfolio_provider()
    return _PROVIDER_REGISTRY['portfolio']


def set_default_data_provider(provider: DataProvider) -> None:
    """Set the default data provider (useful for testing)."""
    _PROVIDER_REGISTRY['data'] = provider


def set_default_portfolio_provider(provider: PortfolioProvider) -> None:
    """Set the default portfolio provider (useful for testing)."""
    _PROVIDER_REGISTRY['portfolio'] = provider
