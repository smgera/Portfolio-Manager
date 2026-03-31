"""
Common utilities and interfaces for the trading system.

This module provides shared functionality that cuts across subsystems,
including data contracts, provider interfaces, and utility functions.
"""

from .data_contracts import MarketData, PortfolioPosition, PortfolioMetrics
from .providers import DataProvider, CachedDataProvider, MockDataProvider
from .portfolio_utils import clean_portfolio_data
from .financial_utils import (
    calculate_returns,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_information_ratio
)
from .symbols_from_description import extract_symbol_from_description
from .logging_setup import setup_root_logger, setup_named_logger

__all__ = [
    'MarketData',
    'PortfolioPosition', 
    'PortfolioMetrics',
    'DataProvider',
    'CachedDataProvider',
    'MockDataProvider',
    'clean_portfolio_data',
    'calculate_returns',
    'calculate_volatility',
    'calculate_sharpe_ratio',
    'calculate_max_drawdown',
    'calculate_sortino_ratio',
    'calculate_calmar_ratio',
    'calculate_information_ratio',
    'extract_symbol_from_description',
    'setup_root_logger',
    'setup_named_logger'
]
