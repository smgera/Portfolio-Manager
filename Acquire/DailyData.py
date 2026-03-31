"""Fetch (download or cache) and filter daily historical stock data.
Uses yfinance and feather files.
TIMEZONE HANDLING
-----------------
All data in this module is stored and processed as timezone-naive DatetimeIndex.
Timezone stripping is enforced at two entry boundaries:
- ``_default_download_data``: strips tz from yfinance results (which may return
  tz-aware America/New_York or UTC timestamps depending on version/ticker).
- ``read_cache``: strips tz after converting the feather Date column to DatetimeIndex
  (feather does not preserve timezone info, but this makes the contract explicit).
This means all internal code — cache staleness checks, incremental merges, and
filtering — can assume tz-naive DatetimeIndex throughout and never needs to
normalize timezones reactively.
Market close detection uses ``latest_mkt_close()`` (returns tz-aware local time);
the staleness check converts to ``.date()`` on both sides to compare calendar
dates only, which is correct regardless of the caller's timezone.
"""
import logging
import os
from pathlib import Path
import sys
import time
from typing import Optional, Dict, Any, Callable

import matplotlib
# matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt GUI issues
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# import mplfinance as mpf
import pandas as pd
import yfinance as yf

from config import (
    DAILY_DATA_CACHE_DIR,
    MIN_CACHE_FILE_SIZE,
    CACHE_COLUMNS,
    REQUIRED_COLUMNS,
    USE_DATETIMEINDEX,
    LOG_LEVEL,
    LOG_FORMAT,
    MONEY_MARKET_FUNDS
)

from Acquire.latest_close import latest_mkt_close

# Use named logger instead of root logger for better control
logger = logging.getLogger(__name__)
# Suppress yfinance's own error logs - DailyData handles errors via its own logger
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
class DailyDataService:
    """Service for fetching daily stock data with dependency injection.
    Side Effects:
        - Reads and writes feather cache files to cache_dir
        - Makes network requests via yfinance downloader
    """
    def __init__(self,
                 cache_dir: Path = DAILY_DATA_CACHE_DIR,
                 downloader: Optional[Callable] = None,
                 market_close_func: Optional[Callable] = None,
                 clock_func: Optional[Callable] = None) -> None:
        """Initialize service with injectable dependencies.
        Args:
            cache_dir: Directory for cache files
            downloader: Function to download data (signature: ticker, **kwargs -> DataFrame)
            market_close_func: Function to get latest market close
            clock_func: Function to get current time
        """
        self.cache_dir = cache_dir
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.downloader = downloader or self._default_download_data
        self.market_close_func = market_close_func or latest_mkt_close
        self.clock_func = clock_func or pd.Timestamp.now
    def _default_download_data(self, ticker: str, **kwargs: Any) -> Optional[pd.DataFrame]:
        """Default yfinance downloader with exponential retry."""
        kwargs['auto_adjust'] = True
        max_retries = 3
        base_backoff = 1.0
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading data for {ticker} with params: {kwargs} (attempt {attempt + 1}/{max_retries})")
                data: pd.DataFrame = yf.Ticker(ticker).history(**kwargs)
                if data.empty:
                    logger.warning(f"No data returned for {ticker}")
                    return None
                # Strip timezone to enforce tz-naive boundary (yfinance may return tz-aware index)
                if data.index.tz is not None:
                    data.index = data.index.tz_localize(None)
                return data.round(3) # adjusted (auto_adjust=True)
            except Exception as e:
                if attempt < max_retries - 1:
                    backoff = base_backoff * (2 ** attempt)
                    logger.warning(f"Error downloading data for {ticker} (attempt {attempt + 1}): {e}. Retrying in {backoff}s...")
                    time.sleep(backoff)
                else:
                    logger.error(f"Error downloading data for {ticker} after {max_retries} attempts: {e}")
                    return None
    def get_cache_file_path(self, ticker: str) -> Path:
        """Get the cache file path for a given ticker."""
        return self.cache_dir / f'{ticker}.feather'
    def read_cache(self, cache_file: Path) -> Optional[pd.DataFrame]:
        """Read cached data from feather file and return as DataFrame with DatetimeIndex.
        Side Effects:
            - Deletes corrupted cache files from disk
            - Logs errors and warnings via logger
        """
        try:
            logger.debug(f"Reading cache from: {cache_file}")
            df: pd.DataFrame = pd.read_feather(cache_file, columns=CACHE_COLUMNS)
            # Convert to DatetimeIndex for consistency
            if USE_DATETIMEINDEX and 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            # Enforce tz-naive boundary (feather doesn't preserve tz, but be explicit)
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df
        except Exception as e:
            logger.error(f"Error reading cache file {cache_file}: {e}")
            # Delete corrupted cache files
            if "too small" in str(e).lower() or cache_file.stat().st_size < MIN_CACHE_FILE_SIZE:
                try:
                    cache_file.unlink()
                    logger.warning(f"Deleted corrupted cache file: {cache_file}")
                except Exception as del_err:
                    logger.error(f"Failed to delete corrupted cache file: {del_err}")
            return None
    def write_cache(self, data: pd.DataFrame, cache_file: Path) -> None:
        """Write DataFrame to feather cache file with validation.
        Side Effects:
            - Writes feather file to disk
            - Deletes undersized cache files from disk
            - Logs debug/error messages via logger
        """
        try:
            # Reset index to convert DatetimeIndex to 'Date' column for storage
            data_to_write: pd.DataFrame = data.reset_index() if 'Date' not in data.columns else data.copy()
            data_to_write = data_to_write.rename(columns={data_to_write.index.name or 'index': 'Date'})
            # Ensure Date column exists and is first
            if 'Date' not in data_to_write.columns:
                data_to_write.reset_index(inplace=True)
                data_to_write.rename(columns={data_to_write.columns[0]: 'Date'}, inplace=True)
            columns: list[str] = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            data_to_write[columns].to_feather(str(cache_file))
            # Validate file size
            if cache_file.stat().st_size < MIN_CACHE_FILE_SIZE:
                logger.error(f"Cache file {cache_file} is too small ({cache_file.stat().st_size} bytes), deleting")
                cache_file.unlink()
                raise ValueError(f"Cache file {cache_file} is too small")
            logger.debug(f"Successfully wrote cache to: {cache_file}")
        except Exception as e:
            logger.error(f"Error writing cache file {cache_file}: {e}")
            raise
    def _get_offset(self, period: str) -> pd.DateOffset:
        """Parse period string into DateOffset.
        'Nd' periods use BusinessDay (trading days), not calendar days.
        Combined with filter_data's extra BDay subtraction, callers receive N+1 rows:
        a period of N trading days spans N+1 price points (1 baseline + N returns).
        1d = 2 points, 2d = 3 points, 5d = 6 points, etc.
        """
        if period.endswith('mo'):
            return pd.DateOffset(months=int(period[:-2]))
        elif period.endswith('y'):
            return pd.DateOffset(years=int(period[:-1]))
        elif period.endswith('d'):
            return pd.tseries.offsets.BDay(int(period[:-1]))
        raise ValueError(f"Invalid period format: {period}. Use format like '1y', '6mo', '30d'")
    def filter_data(self, data: pd.DataFrame, period: str) -> Optional[pd.DataFrame]:
        """Filter data by period relative to the latest date in data.
        Always returns N+1 rows for an N-day period:
          - A period of N trading days spans N+1 price points
          - 1d = 2 points (baseline close + 1 return), 2d = 3 points, 5d = 6 points, etc.
          - The extra baseline row is required so pct_change() yields exactly N returns
            and period_return = (price[-1] / price[-(N+1)]) - 1 uses the correct baseline.
        Args:
            data: DataFrame with DatetimeIndex
            period: Period string like '1y', '6mo', '30d'
        Returns:
            Filtered DataFrame or None on error
        Side Effects:
            - Logs debug/info/error messages via logger
        """
        try:
            offset = self._get_offset(period)
        except ValueError as e:
            logger.error(e)
            raise
        try:
            # Work with DatetimeIndex
            date_index: pd.DatetimeIndex = pd.to_datetime(data.index)
            latest: pd.Timestamp = date_index.max()
            start: pd.Timestamp = latest - offset
            # N+1 row guarantee:
            #   'Nd' periods: BDay(N) sets start = Nth trading day before latest (inclusive),
            #   giving exactly N+1 rows. 1d = 2 points, 2d = 3 points, 5d = 6 points.
            #   'Nmo'/'Ny' calendar offsets: subtract 1 extra BDay so callers always get
            #   N+1 rows — 1 baseline price + N daily returns for pct_change / period_return.
            if not period.endswith('d'):
                start = start - pd.tseries.offsets.BDay(1)
            logger.debug(f"Filtering data from {start} to {latest} (period: {period})")
            filtered: pd.DataFrame = data[date_index >= start]
            logger.info(f"Filtered to {len(filtered)} rows for period {period}")
            return filtered
        except Exception as e:
            logger.error(f"Error filtering data for period {period}: {e}")
            return None
    def download_data(self, ticker: str, **kwargs: Any) -> Optional[pd.DataFrame]:
        """Download data using injected downloader."""
        return self.downloader(ticker, **kwargs)
    def _make_download_kwargs(self, period_val: Any, beginning_period_date: Optional[pd.Timestamp], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Return kwargs for download_data, converting relative period to explicit start.
        If period_val is a relative string like '6mo', this removes 'period' from kwargs
        and adds a 'start' date so that yfinance receives an explicit start boundary
        consistent with our cache coverage logic.
        """
        dl_kwargs: Dict[str, Any] = dict(kwargs)
        # For relative, non-max periods, we want to avoid sending both 'period' and 'start'
        # to yfinance. If the caller already provided an explicit 'start', we respect it
        # and simply drop 'period'. Otherwise, we synthesize 'start' from
        # beginning_period_date.
        if isinstance(period_val, str) and period_val != 'max':
            dl_kwargs.pop('period', None)
            if beginning_period_date is not None and 'start' not in dl_kwargs:
                # yfinance accepts datetime/date for start
                dl_kwargs['start'] = beginning_period_date
        return dl_kwargs
    @staticmethod
    def _has_split_artifact(data: pd.DataFrame, threshold: float = 0.5) -> bool:
        """Return True if data contains a single-day Close move > threshold (split artifact)."""
        if data is None or data.empty or len(data) < 2:
            return False
        return bool((data['Close'].pct_change().abs() > threshold).any())

    def _download_with_repair(self, ticker: str) -> Optional[pd.DataFrame]:
        """Re-download max history with yfinance repair=True to fix split artifacts."""
        try:
            data = yf.Ticker(ticker).history(period='max', repair=True, auto_adjust=True)
            if data is None or data.empty:
                return None
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            return data.round(3)
        except Exception as e:
            logger.warning(f"yfinance repair=True failed for {ticker}: {e}")
            return None

    def _download_from_tiingo(self, ticker: str) -> Optional[pd.DataFrame]:
        """Download full adjusted history from Tiingo REST API as fallback.
        Requires TIINGO_API_KEY environment variable.
        """
        import requests  # noqa: PLC0415 — deferred: low-probability fallback, avoids mandatory dep
        from dotenv import load_dotenv  # noqa: PLC0415 — deferred: low-probability fallback
        _secrets_env = Path(__file__).parent.parent.parent / "secrets" / ".env"
        if _secrets_env.exists():
            load_dotenv(_secrets_env)
        api_key = os.environ.get('tiingo_key', '') or os.environ.get('TIINGO_API_KEY', '')
        if not api_key:
            logger.warning("TIINGO_API_KEY not set — skipping Tiingo fallback")
            return None
        try:
            url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
            params = {'startDate': '2000-01-01', 'resampleFreq': 'daily', 'token': api_key}
            resp = requests.get(url, params=params, headers={'Content-Type': 'application/json'}, timeout=30)
            resp.raise_for_status()
            records = resp.json()
            if not records:
                logger.warning(f"Tiingo returned empty data for {ticker}")
                return None
            df = pd.DataFrame(records)
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            df = df.set_index('date').sort_index()
            df = df.rename(columns={
                'adjOpen': 'Open', 'adjHigh': 'High', 'adjLow': 'Low',
                'adjClose': 'Close', 'adjVolume': 'Volume',
            })
            missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                logger.warning(f"Tiingo response missing columns {missing} for {ticker}")
                return None
            return df[REQUIRED_COLUMNS].round(3)
        except Exception as e:
            logger.warning(f"Tiingo fallback failed for {ticker}: {e}")
            return None

    def daily_data(self, ticker: str, **kwargs: Any) -> Optional[pd.DataFrame]:
        """Fetch and cache historical data for a ticker with incremental update support.
        Args:
            ticker: The ticker symbol for the stock/ETF.
            kwargs: Additional parameters for yfinance.Ticker.history method (period, interval, start, end).
        Returns:
            pd.DataFrame containing the historical data for the given ticker.
        Side Effects:
            - Reads and writes feather cache files to disk
            - Makes network requests via yfinance when cache is missing or stale
            - Logs info/warning/error messages via logger
        """
        start_time: float = time.time()
        ticker = ticker.replace('.', '-')
        # Set default period if not provided
        if not kwargs or 'period' not in kwargs:
            kwargs['period'] = 'max'
        cache_file: Path = self.get_cache_file_path(ticker)
        logger.debug(f"Processing {ticker}, cache file: {cache_file}")
        # Check if cache exists and is readable
        if cache_file.exists():
            try:
                data = self.read_cache(cache_file)
            except Exception as e:
                logger.error(f"Exception reading cache {cache_file}: {e}")
                data = None
        else:
            data = None
        if data is not None:
            # Check if cache is up to date (using DatetimeIndex)
            latest_market_close: Optional[pd.Timestamp] = self.market_close_func()
            if latest_market_close is None:
                logger.warning("Could not determine latest market close")
            else:
                # Work in timezone-naive DATE space to avoid tz-aware vs tz-naive comparison issues
                cache_latest_date = data.index[-1].date()
                latest_market_close_date = latest_market_close.date()
                # If cache is missing latest date, fetch incremental data
                if latest_market_close_date > cache_latest_date:
                    logger.info(
                        f"Cache {ticker} is stale ({cache_latest_date} < {latest_market_close_date}), freshening with incremental data"
                    )
                    # Incremental update: only fetch data newer than cache to minimize API calls
                    incremental_start = cache_latest_date + pd.Timedelta(days=1)
                    incremental_kwargs = kwargs.copy()
                    incremental_kwargs['start'] = incremental_start
                    incremental_kwargs['period'] = 'max'  # Get all data from start to now
                    # Special handling for Fidelity money market funds
                    download_ticker = ticker
                    if ticker in MONEY_MARKET_FUNDS:
                        download_ticker = 'BIL'
                    incremental_data: Optional[pd.DataFrame] = self.download_data(download_ticker, **incremental_kwargs)
                    if incremental_data is not None and not incremental_data.empty:
                        # Drop Adj Close if present - we only store OHLCV
                        incremental_data = incremental_data[[c for c in REQUIRED_COLUMNS if c in incremental_data.columns]]
                        # Normalize timezones before comparison
                        if incremental_data.index.tz != data.index.tz:
                            if incremental_data.index.tz is not None and data.index.tz is None:
                                incremental_data.index = incremental_data.index.tz_localize(None)
                            elif incremental_data.index.tz is None and data.index.tz is not None:
                                incremental_data.index = incremental_data.index.tz_localize(data.index.tz)
                        incremental_data = incremental_data[incremental_data.index > data.index[-1]]
                        if not incremental_data.empty:
                            # Combine existing cache with new data
                            combined_data = pd.concat([data, incremental_data])
                            # Sort by date to maintain order
                            combined_data = combined_data.sort_index()
                            # Remove any duplicate indices (keep the last occurrence for most recent data)
                            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                            # Write combined data back to cache
                            self.write_cache(combined_data, cache_file)
                            logger.info(f"Added {len(incremental_data)} new rows to cache {ticker}")
                            data = combined_data
                        else:
                            logger.info(f"No new data available for {ticker}")
                    else:
                        logger.warning(f"Incremental download failed for {ticker}, using existing cache")
            # Log cache info using DatetimeIndex
            start_dt: pd.Timestamp = data.index[0]
            end_dt: pd.Timestamp = data.index[-1]
            date_diff: pd.Timedelta = end_dt - start_dt
            logger.info(
                f"Cache {ticker}: {len(data)} rows, "
                f"{start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}, "
                f"duration: {date_diff.days} days")
            # Filter data
            user_start = kwargs.get('start')
            if user_start is not None:
                 # Filter by start if provided
                 start_ts = pd.Timestamp(user_start)
                 if start_ts.tz is None and data.index.tz is not None:
                     start_ts = start_ts.tz_localize(data.index.tz)
                 logger.debug(f"Filtering data from start={start_ts}")
                 data = data[data.index >= start_ts]
                 # NEW: Filter by end if period is provided (duration from start)
                 if kwargs.get('period') and kwargs.get('period') != 'max':
                     offset = self._get_offset(kwargs['period'])
                     end_ts = start_ts + offset
                     logger.debug(f"Filtering data to end={end_ts} (duration {kwargs['period']})")
                     data = data[data.index <= end_ts]
            elif kwargs.get('period') != 'max':
                data = self.filter_data(data, kwargs['period'])
                if data is None or data.empty:
                    logger.error(f"No data after filtering for {ticker} with period {kwargs['period']}")
                    return None
        # If no data yet (no cache or cache read failed), download fresh
        if data is None:
            # No cache exists: ALWAYS download 'max' period data to populate cache,
            # then filter to return requested range (matching documented strategy)
            period_val = kwargs.get('period')
            user_start = kwargs.get('start')
            # Force 'max' period for initial cache population
            cache_download_kwargs = kwargs.copy()
            cache_download_kwargs['period'] = 'max'
            # Special handling for Fidelity money market funds, substitute BIL for SPAXX, FDRXX, FZFXX
            download_ticker = ticker
            if ticker in MONEY_MARKET_FUNDS:
                download_ticker = 'BIL'
            # Remove start/end for max download to get complete history
            cache_download_kwargs.pop('start', None)
            cache_download_kwargs.pop('end', None)
            # Download max data for cache
            data: Optional[pd.DataFrame] = self.download_data(download_ticker, **cache_download_kwargs)
            if data is None or data.empty:
                logger.error(f"No data downloaded for {ticker}")
                return None
            # Ensure we only keep the required columns (consistent with cache)
            data = data[REQUIRED_COLUMNS]
            # Log download info using DatetimeIndex
            latest_dt: pd.Timestamp = data.index[-1]
            logger.info(
                f"No cache for {ticker}: downloaded {len(data)} rows (max period for cache), "
                f"{time.time() - start_time:.2f}s, latest: {latest_dt.strftime('%Y-%m-%d')}")
            try:
                self.write_cache(data, cache_file)
                logger.info(f"Populated cache for {ticker} with {len(data)} rows of max period data")
            except Exception as e:
                logger.error(f"Failed to write cache for {ticker}: {e}")
            # Now filter the max data to return requested range
            if user_start is not None:
                 start_ts = pd.Timestamp(user_start)
                 if start_ts.tz is None and data.index.tz is not None:
                     start_ts = start_ts.tz_localize(data.index.tz)
                 data = data[data.index >= start_ts]
                 # Filter by end if period is provided (duration from start)
                 if isinstance(period_val, str) and period_val != 'max':
                     offset = self._get_offset(period_val)
                     end_ts = start_ts + offset
                     logger.debug(f"Filtering max data to end={end_ts} (duration {period_val})")
                     data = data[data.index <= end_ts]
            elif isinstance(period_val, str) and period_val != 'max':
                data = self.filter_data(data, period_val)
                if data is None or data.empty:
                    logger.error(f"No data after filtering for {ticker} with period {period_val}")
                    return None
        # Detect split artifact in returned window; attempt repair → Tiingo fallback
        period_str = kwargs.get('period', 'max')
        user_start = kwargs.get('start')
        if data is not None and user_start is None and self._has_split_artifact(data):
            logger.warning(f"{'!'*60}")
            logger.warning(f"Split artifact detected for {ticker} in {period_str} window — trying repair=True ...")
            repaired_max = self._download_with_repair(ticker)
            if repaired_max is not None:
                repaired_max = repaired_max[[c for c in REQUIRED_COLUMNS if c in repaired_max.columns]]
                repaired_window = self.filter_data(repaired_max, period_str) if period_str != 'max' else repaired_max
                if repaired_window is not None and not self._has_split_artifact(repaired_window):
                    self.write_cache(repaired_max, cache_file)
                    logger.warning(f"repair=True fixed {ticker} — cache updated")
                    return repaired_window
            logger.warning(f"repair=True still bad for {ticker} — trying Tiingo ...")
            tiingo_max = self._download_from_tiingo(ticker)
            if tiingo_max is not None:
                tiingo_window = self.filter_data(tiingo_max, period_str) if period_str != 'max' else tiingo_max
                if tiingo_window is not None and not self._has_split_artifact(tiingo_window):
                    self.write_cache(tiingo_max, cache_file)
                    logger.warning(f"Tiingo fixed {ticker} — cache updated")
                    return tiingo_window
            raise ValueError(
                f"\n{'!'*60}\n"
                f"BAD DATA: split artifact in {ticker} persists after repair=True + Tiingo\n"
                f"ACTION REQUIRED: Remove {ticker} from your ticker list in tickerLists/\n"
                f"{'!'*60}"
            )
        return data
# Default service instance for backward compatibility
_default_service = DailyDataService()
# Backward-compatible module functions
def daily_data(ticker: str, **kwargs: Any) -> Optional[pd.DataFrame]:
    """Fetch daily data using default service (backward compatibility)."""
    return _default_service.daily_data(ticker, **kwargs)
def download_data(ticker: str, **kwargs: Any) -> Optional[pd.DataFrame]:
    """Download data using default service (backward compatibility)."""
    return _default_service.download_data(ticker, **kwargs)
def read_cache(cache_file: Path) -> Optional[pd.DataFrame]:
    """Read cache using default service (backward compatibility)."""
    return _default_service.read_cache(cache_file)
def write_cache(data: pd.DataFrame, cache_file: Path) -> None:
    """Write cache using default service (backward compatibility)."""
    return _default_service.write_cache(data, cache_file)
def filter_data(data: pd.DataFrame, period: str) -> Optional[pd.DataFrame]:
    """Filter data using default service (backward compatibility)."""
    return _default_service.filter_data(data, period)
def get_cache_file_path(ticker: str) -> Path:
    """Get cache file path using default service (backward compatibility)."""
    return _default_service.get_cache_file_path(ticker)
def _test_make_download_kwargs() -> None:
    """Basic tests for _make_download_kwargs period->start conversion."""
    today = pd.Timestamp('2025-11-17').date()
    period_val = '6mo'
    # Simulate a beginning_period_date computed elsewhere
    beginning_period_date = today - pd.DateOffset(months=6)
    kwargs: Dict[str, Any] = {'period': period_val, 'interval': '1d'}
    service = DailyDataService()
    dl_kwargs = service._make_download_kwargs(period_val, beginning_period_date, kwargs)
    assert 'period' not in dl_kwargs, "period should be removed for relative periods"
    assert 'start' in dl_kwargs, "start should be set for relative periods"
    assert dl_kwargs['start'] == beginning_period_date, "start date should match beginning_period_date"
    # For 'max', kwargs should pass through unchanged
    kwargs_max: Dict[str, Any] = {'period': 'max', 'interval': '1d'}
    dl_kwargs_max = service._make_download_kwargs('max', None, kwargs_max)
    assert dl_kwargs_max.get('period') == 'max', "period 'max' should be preserved"
    assert 'start' not in dl_kwargs_max, "start should not be set for period='max'"
    print("_make_download_kwargs tests passed.")
# Example usage and basic tests
if __name__ == "__main__":
    # Configure INFO level logging for script execution
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT
    )
    logger.info("Starting DailyData script execution")
    # plot 60d PULS Adjusted Close using matplotlib
    # logger.info("Fetching PULS data for plotting")
    # symbol = 'PULS'
    # period = '60d'
    # data = daily_data(symbol, period=period)    
    # ax = data.plot(y='Close')   # close is adjClose, accounts for dividends and splits
    # ax.set_title(f'{symbol} {period} Adjusted Close')
    # ax.set_ylabel('Price')
    # plt.tight_layout()
    # plt.show()
    # plot OHLC Candlestick using mplfinance
    symbol = 'PULS'
    period = '1mo'
    # beginning from 2mo ago
    start = pd.Timestamp.now() - pd.DateOffset(months=2)
    data = daily_data(symbol, period=period, start=start)
    print('requested period:', period, ' from start:', start)
    print('received qty:', len(data), ' start:', data.index[0], ' end:', data.index[-1])
    # if data is not None and not data.empty:
    #     mpf.plot(
    #         data,
    #         type='candle',
    #         title=f'{symbol} ({period} Daily Candlestick from {start.strftime("%Y-%m-%d")}',
    #         ylabel='Price',
    #         style='yahoo',
    #     )
    # plot PULs gains vs MM funds (normalized)
    symbols = ['PULS', 'BIL', 'SGOV', 'SWVXX', 'SPAXX', 'FDRXX', 'FZFXX']
    mm_period = '10d'  # Define period for money market funds plot
    # Collect all data first to find common date range
    plot_data = {}
    for symbol in symbols:
        data = daily_data(symbol, period=mm_period)
        if data is not None and not data.empty:
            # Normalize timezone - remove timezone info to ensure consistency
            if hasattr(data.index, 'tz') and data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            plot_data[symbol] = data
        else:
            logger.warning(f"No data for {symbol}")
    # Reindex all series to a common date range and fill missing values
    all_dates = pd.Index([])
    for symbol, data in plot_data.items():
        all_dates = all_dates.union(data.index)
    for symbol, data in plot_data.items():
        data = data.reindex(all_dates, method='ffill')
        # Normalize to start at 0%
        normalized = ((data.Close / data.Close.iloc[0]) - 1) * 100
        plt.plot(data.index, normalized, label=symbol)
    plt.legend()
    plt.title(f'Money Market Funds ({mm_period}) - Normalized to 0%')
    plt.ylabel('Return (%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig('money_market_funds_plot.png')
    plt.show()
