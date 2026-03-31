"""Price data fetching and processing utilities.
Downloads historical data from yfinance, adds financial metrics, and stores as feather files.
4 Functions:
- fetch_ticker_data_yf: Download data from yfinance API
- get_fresh_ticker_data: get/refresh/create feather files with metrics
- prep_data: Process multiple tickers in batch
- add_metrics: Add financial metrics
Data Flow:
1. Check if data feather file exists and is fresh (current date)
2. If fresh: use existing data
3. If stale/missing: fetch latest N years, add metrics, overwrite feather file
Side Effects:
- Network requests to yfinance API
- Creates/updates feather files in HISTORY_1D_FOLDER
- Prints progress information
- Development logging when enabled
"""

from datetime import timedelta, datetime
import os
from pathlib import Path
import sys
import time
from typing import List

import pandas as pd
import yfinance as yf

from acquire_config import HISTORY_1D_FOLDER, Sharpe_WINDOW, MAX_RETRIES, RETRY_DELAY, HISTORY_LENGTH
from portfolio_manager.utils.dev_logging import get_dev_logger, log_data_fetch
from portfolio_manager.utils.file_io import load_feather, save_feather, safe_to_modify_file
from portfolio_manager.utils.latest_close import latest_mkt_close
from portfolio_manager.core.calculations import calculate_returns_metrics

def add_metrics(df: pd.DataFrame, window: int = Sharpe_WINDOW) -> pd.DataFrame:  # only called by get_fresh_ticker_data below
    """Add calculated metrics to price data.
    Args:
        df: DataFrame with price data
        window: Window for Sharpe calculations
    Returns:
        DataFrame with additional metrics columns
    """
    
    # Add window for reference
    df['_window'] = window
    
    # Calculate returns and volatility
    df = calculate_returns_metrics(df, window)
    
    return df

def fetch_ticker_data_yf(ticker: str, start_date: str) -> pd.DataFrame: # only called by get_fresh_ticker_data below
    """Fetch historical data for a ticker from yfinance.
    Args:
        ticker: Ticker symbol
        start_date: Start date for data fetch
    Returns:
        DataFrame with historical data
    Side Effects:
    - Network request to yfinance
    - Prints retry information on failure
    """
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try: # this try is an exception to the error rule of not using try/except, because it is a network or API call
            data = yf.download(ticker, start=start_date, auto_adjust=False, actions=True, progress=False)
            return data
        except Exception as e:
            retry_count += 1
            if retry_count < MAX_RETRIES:
                print(f"  Retry {retry_count}/{MAX_RETRIES} after {RETRY_DELAY}s...", end=' ')
                time.sleep(RETRY_DELAY)
            else:
                raise(Exception(f"Failed to fetch {ticker} history from yfinance after {MAX_RETRIES} retries"))

def is_feather_fresh(ticker_symbol: str) -> bool:
    """Check if feather file is fresh without loading full data.
    Args:
        ticker_symbol: Stock ticker symbol
    Returns:
        True if data is fresh, False otherwise
    Side Effects:
    - Reads only Date column from feather file
    """
    feather_file_path = f"{HISTORY_1D_FOLDER}/{ticker_symbol}.feather"
    
    if not os.path.exists(feather_file_path):
        return False
    
    # Quick freshness check using file timestamp first
    file_timestamp = os.path.getmtime(feather_file_path)
    local_file_datetime = datetime.fromtimestamp(file_timestamp).astimezone()
    local_latest_mkt_close = latest_mkt_close()
    
    if local_file_datetime <= local_latest_mkt_close:
        return False
    
    # File is new enough, check actual latest date
    lastrowdate = pd.read_feather(feather_file_path, columns=['Date']).iloc[-1]['Date']
    
    return lastrowdate.date() >= latest_mkt_close().date()

def get_fresh_ticker_data(ticker_symbol: str) -> pd.DataFrame:
    """Refresh or create data file for a ticker symbol.
    Args:
        ticker_symbol: Stock ticker symbol
    Returns:
        DataFrame with refreshed historical data and metrics
    Side Effects:
    - Reads existing feather file if present
    - Downloads new data if needed
    - Saves/updates feather file
    - Prints progress information
    """
    dev_logger = get_dev_logger()
    
    # Check if data is fresh using optimized function
    if is_feather_fresh(ticker_symbol):
        print(f"{ticker_symbol} Data fresh and valid")
        return load_feather(ticker_symbol)
    
    # Fetch fresh data
    historical_start_date = (datetime.now() - timedelta(days=365 * HISTORY_LENGTH)).strftime('%Y-%m-%d')
    print(f"Fetching from yfinance for {ticker_symbol} from {historical_start_date}...", end=' ')
    downloaded_data = fetch_ticker_data_yf(ticker_symbol, historical_start_date)
    log_data_fetch(ticker_symbol, "download", len(downloaded_data), start_date=historical_start_date)
    
    if downloaded_data.empty:
        print(f"No data found")
        return pd.DataFrame()
    
    # Process and save with memory-efficient operations
    if isinstance(downloaded_data.columns, pd.MultiIndex):
        downloaded_data.columns = downloaded_data.columns.get_level_values(0)
    
    # Use inplace operation to avoid creating copy
    downloaded_data.rename(columns={'Splits': 'Stock Splits'}, inplace=True)
    print(f"Adding metrics...", end=' ')
    processed_data = add_metrics(downloaded_data, window=Sharpe_WINDOW)
    
    feather_file_path = f"{HISTORY_1D_FOLDER}/{ticker_symbol}.feather"
    if safe_to_modify_file(feather_file_path):
        save_feather(processed_data, ticker_symbol)
        print(f"+ Created: {len(processed_data)} rows")
    else:
        print(f"Skipping save for {ticker_symbol} - file has date in name or is read-only")
    
    # Memory cleanup
    del downloaded_data
    
    return processed_data

def prepare_market_data(ticker_symbols: List[str]) -> List[str]: # calls refresh_ticker_data for each ticker
    """Prepare market price history data and metrics for a list of ticker symbols.
    Args:
        ticker_symbols: List of asset symbols to process
    Returns:
        List of successfully processed ticker symbols (failed ones removed)
    Side Effects:
        - Creates/checks price data feather files in HISTORY_1D_FOLDER
        - Prints progress information
        - Makes network requests to yfinance API
    """
    successful_tickers = []
    for ticker_index, ticker_symbol in enumerate(ticker_symbols, 1):
        print(f"[{ticker_index}/{len(ticker_symbols)}] ", end="")
        result = get_fresh_ticker_data(ticker_symbol) # from fresh cache or yfinance
        if not result.empty:
            successful_tickers.append(ticker_symbol)
        else:
            print(f"FAILED: {ticker_symbol} - No data returned")
    return successful_tickers

def show_feather_columns(ticker_symbol: str) -> None:
    """Print the column names from the feather file for the given ticker symbol."""
    data = load_feather(ticker_symbol)
    print(data.columns)

if __name__ == "__main__":
    prepare_market_data(["AAPL", "GLDM", "GOOGL"]) # calls refresh_ticker_data for each ticker
    get_fresh_ticker_data("MU") # ensures feather freshness, fetches if needed
    print(load_feather("aapl")) # does not fetch or check freshness
    
    from portfolio_manager.utils.print_feather import print_feather_data
    print_feather_data("GLDM") # does not fetch or check freshness
    show_feather_columns("GLDM")
