"""Common financial calculation utilities reused across trading subsystems."""

from typing import Optional, Union
import warnings

import numpy as np
import pandas as pd

from config import TRADING_DAYS_PER_YEAR, RISK_FREE_RATE

warnings.filterwarnings('ignore', category=FutureWarning)


# ── Annualization ──────────────────────────────────────────────────────────────

def annualize_return(period_return: float, period_days: int) -> float:
    """Annualize a period return using compound growth.
    Args:
        period_return: Raw period return (as decimal, e.g., 0.05 for 5%)
        period_days: Number of trading days in the period.
            A period of N trading days requires N+1 price points:
            1 baseline price + N daily returns.
            1d = 2 points, 2d = 3 points, 5d = 6 points, etc.
    Raises:
        ValueError: If period_days is zero or negative
    """
    if period_days <= 0:
        raise ValueError(f"period_days must be positive, got {period_days}")
    return (1 + period_return) ** (TRADING_DAYS_PER_YEAR / period_days) - 1


def annualize_volatility(daily_pct_return_std: float) -> float:
    """Annualize a daily standard deviation by scaling by sqrt(TRADING_DAYS_PER_YEAR).
    Args:
        daily_pct_return_std: Daily std of percent returns (units preserved — decimal or percentage)
    """
    return daily_pct_return_std * TRADING_DAYS_PER_YEAR ** 0.5


# ── Sharpe / GainSharpe ────────────────────────────────────────────────────────

def calculate_sharpe_ratio(
    annualized_return: float,
    annualized_volatility: float,
) -> float:
    """Calculate Sharpe ratio using the configured RISK_FREE_RATE.
    Args:
        annualized_return: Annualized return (as decimal, e.g., -0.086 for -8.6%)
        annualized_volatility: Annualized volatility (as decimal)
    Raises:
        ValueError: If volatility is zero or negative
    """
    if annualized_volatility <= 0:
        raise ValueError(f"Volatility must be positive, got {annualized_volatility}")
    return (annualized_return - RISK_FREE_RATE) / annualized_volatility


def calculate_sharpe_ratios(
    annual_returns: pd.Series,
    annual_volatility: pd.Series,
) -> pd.Series:
    """Vectorized Sharpe ratio for a Series of assets.
    Args:
        annual_returns: Annualized returns per asset (decimal)
        annual_volatility: Annualized volatility per asset (decimal)
    Raises:
        ValueError: If any volatility is zero or negative
    """
    if (annual_volatility <= 0).any():
        bad = annual_volatility[annual_volatility <= 0].index.tolist()
        raise ValueError(f"Volatility must be positive; non-positive entries: {bad}")
    return (annual_returns - RISK_FREE_RATE) / annual_volatility


def calculate_gain_sharpe(
    period_return: float,
    sharpe: float,
) -> float:
    """Calculate gainSharpe metric.
    Args:
        period_return: Raw period return (as decimal)
        sharpe: Sharpe ratio (must be mathematically consistent)
    Returns:
        gainSharpe = abs(period_return) * sharpe
    Note:
        - If period_return is negative, sharpe must be negative
        - Result will be negative for negative returns with negative Sharpe
        - Result will be positive for positive returns with positive Sharpe
    """
    if period_return < 0 and sharpe >= 0:
        raise ValueError("Mathematical inconsistency: negative return must have negative Sharpe")
    return abs(period_return) * sharpe


# ── Raw-returns-based utilities ────────────────────────────────────────────────

def calculate_sharpe_from_returns(
    returns: pd.Series,
    risk_free_rate: Optional[Union[float, pd.Series]] = None,
    risk_free_ticker: str = 'BIL',
    annualize: bool = True,
    trading_days: int = 252,
) -> float:
    """Calculate annualized Sharpe ratio from a daily returns series.
    Args:
        returns: Series of daily returns (as decimals, not percentages)
        risk_free_rate: Fixed annual rate, aligned Series, or None to fetch via risk_free_ticker
        risk_free_ticker: Ticker for risk-free rate when risk_free_rate is None (default: 'BIL')
        annualize: Whether to annualize the ratio (default: True)
        trading_days: Number of trading days per year for annualization (default: 252)
    Raises:
        ValueError: If returns is empty
    """
    if len(returns) == 0:
        raise ValueError("Returns series is empty")

    if risk_free_rate is None:
        from Acquire.DailyData import daily_data
        from .financial_utils import calculate_returns as _calc_returns
        cal_days = 2 * len(returns)
        risk_free_data = daily_data(risk_free_ticker, period=f'{cal_days}d')
        if risk_free_data is None or risk_free_data.empty:
            warnings.warn(f"Could not fetch risk-free data for {risk_free_ticker}, using 0")
            excess_returns = returns
        else:
            risk_free_returns = _calc_returns(risk_free_data['Close']).iloc[-len(returns):]
            excess_returns = returns - risk_free_returns
    elif isinstance(risk_free_rate, (int, float)):
        daily_rf_rate = risk_free_rate / trading_days
        excess_returns = returns - daily_rf_rate
    else:
        excess_returns = returns - risk_free_rate

    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()

    if std_excess == 0:
        return 0.0 if mean_excess <= 0 else np.inf

    sharpe = mean_excess / std_excess

    if annualize:
        sharpe *= np.sqrt(trading_days)

    return sharpe


def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate daily returns from price series."""
    return prices.pct_change().dropna()


def calculate_volatility(
    returns: pd.Series,
    annualize: bool = True,
    trading_days: int = 252,
) -> float:
    """Calculate volatility (standard deviation) of returns."""
    vol = returns.std()
    if annualize:
        vol *= np.sqrt(trading_days)
    return vol


def calculate_max_drawdown(prices: pd.Series) -> float:
    """Calculate maximum drawdown as a percentage."""
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak
    return drawdown.min()


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: Optional[float] = None,
    annualize: bool = True,
    trading_days: int = 252,
) -> float:
    """Calculate Sortino ratio using downside deviation instead of standard deviation.
    Args:
        returns: Series of daily returns (as decimals)
        risk_free_rate: Annual risk-free rate (default: 0)
        annualize: Whether to annualize the ratio
        trading_days: Number of trading days per year
    """
    if risk_free_rate is None:
        risk_free_rate = 0.0

    daily_rf_rate = risk_free_rate / trading_days
    excess_returns = returns - daily_rf_rate

    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return np.inf if excess_returns.mean() > 0 else 0.0

    downside_deviation = np.sqrt((downside_returns ** 2).mean())

    if downside_deviation == 0:
        return 0.0 if excess_returns.mean() <= 0 else np.inf

    sortino = excess_returns.mean() / downside_deviation

    if annualize:
        sortino *= np.sqrt(trading_days)

    return sortino


def calculate_calmar_ratio(
    returns: pd.Series,
    prices: pd.Series,
    annualize: bool = True,
) -> float:
    """Calculate Calmar ratio (annual return / maximum drawdown).
    Args:
        returns: Series of daily returns
        prices: Series of prices for drawdown calculation
        annualize: Whether to annualize returns
    """
    if annualize:
        annual_return = (1 + returns.mean()) ** 252 - 1
    else:
        annual_return = returns.mean()

    max_dd = calculate_max_drawdown(prices)

    if max_dd == 0:
        return np.inf if annual_return > 0 else 0.0

    return annual_return / abs(max_dd)


def calculate_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    annualize: bool = True,
    trading_days: int = 252,
) -> float:
    """Calculate Information Ratio (active return / tracking error).
    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns
        annualize: Whether to annualize
        trading_days: Number of trading days per year
    """
    returns, benchmark_returns = returns.align(benchmark_returns, join='inner')
    active_returns = returns - benchmark_returns

    if active_returns.std() == 0:
        return 0.0

    info_ratio = active_returns.mean() / active_returns.std()

    if annualize:
        info_ratio *= np.sqrt(trading_days)

    return info_ratio


def calculate_hurst_exponent(
    prices: pd.Series,
    min_lags: int = 2,
    max_lags: Optional[int] = None,
    min_data_points: int = 20,
) -> float:
    """Calculate the Hurst exponent for a price series using R/S analysis.
    Args:
        prices: Series of prices
        min_lags: Minimum number of lags to use
        max_lags: Maximum number of lags to use (default: len(prices)//2 or 100)
        min_data_points: Minimum number of data points required
    Returns:
        Hurst exponent value or np.nan if calculation fails
    """
    if len(prices) < min_data_points:
        return np.nan

    if max_lags is None:
        max_lags = min(len(prices) // 2, 100)

    max_lags = max(max_lags, min_lags + 1)

    returns = np.log(prices).diff().dropna()
    lags = range(min_lags, max_lags)
    rs_values = []

    for lag in lags:
        n_chunks = len(returns) // lag
        if n_chunks < 2:
            continue

        chunks = returns[:n_chunks * lag].values.reshape(n_chunks, lag)
        mean_adj = chunks - chunks.mean(axis=1, keepdims=True)
        cum_dev = np.cumsum(mean_adj, axis=1)

        R = cum_dev.max(axis=1) - cum_dev.min(axis=1)
        S = chunks.std(axis=1, ddof=1)

        valid = S > 0
        if not valid.any():
            continue

        RS = (R / S)[valid].mean()
        rs_values.append(RS)

    if len(rs_values) < 2:
        return np.nan

    log_lags = np.log(list(lags)[:len(rs_values)])
    log_rs = np.log(rs_values)
    slope = np.polyfit(log_lags, log_rs, 1)[0]

    return float(slope)


def calculate_adf_statistic(
    prices: pd.Series,
    min_data_points: int = 10,
) -> float:
    """Calculate the Augmented Dickey-Fuller test statistic for a price series.
    Args:
        prices: Series of prices
        min_data_points: Minimum number of data points required
    Returns:
        ADF test statistic; more negative = stronger evidence of stationarity;
        np.nan if statsmodels unavailable or insufficient data
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return np.nan

    if len(prices) < min_data_points:
        return np.nan

    try:
        log_prices = np.log(prices).dropna()
        if len(log_prices) < min_data_points:
            return np.nan
        result = adfuller(log_prices, autolag='AIC')
        return float(result[0])
    except Exception:
        return np.nan
