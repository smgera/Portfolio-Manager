from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from config import CALENDAR_CACHE_DIR, DEFAULT_MARKET_CALENDAR, CALENDAR_LOOKBACK_DAYS
from utils.io_utils import read_tsv

import exchange_calendars as mcal

# Set logging level to DEBUG to see the debug messages
logging.basicConfig(level=logging.DEBUG)

# ============================================================================
# TIMEZONE HANDLING IN THIS MODULE
# ============================================================================
# This module handles timezone conversions for market calendar data:
#
# INPUT HANDLING:
# - Timezone-naive datetimes are treated as SYSTEM LOCAL TIME
# - Timezone-aware datetimes are converted to UTC for comparison
#
# INTERNAL PROCESSING:
# - All market calendar data is stored and compared in UTC
# - Cache filenames use LOCAL date (timezone-naive) for consistency
#
# OUTPUT:
# - Returns timezone-aware timestamp in LOCAL timezone
# - The timestamp represents the latest market close in your local time
#
# EXAMPLES:
# - System in New York: Returns market close in EST/EDT
# - System in California: Returns market close in PST/PDT  
# - System in London: Returns market close in GMT/BST
#
# NOTE: NYSE market hours are 9:30 AM - 4:00 PM Eastern Time
# ============================================================================

def latest_mkt_close(dt: Optional[pd.Timestamp] = None, calendar: str = DEFAULT_MARKET_CALENDAR, cache_dir: Optional[Path] = None) -> Optional[pd.Timestamp]:
    """Return the datetime of the latest market close prior to or on the provided datetime.
    Args:
        dt: The reference datetime. If None, uses now. Timezone-naive is treated as local time.
        calendar: Market calendar to use (default: NYSE).
        cache_dir: Directory for cache files. If None, uses default.
    Returns:
        Timezone-aware timestamp of the latest market close in LOCAL timezone, or None if not found.
    Side Effects:
        - Reads and writes market calendar cache files to disk
        - Makes network requests to exchange_calendars API on cache miss
        - Logs debug/info/warning messages via logging
    """
    if dt is None:
        dt = pd.Timestamp.now()
    
    # Handle case where system clock might be set to future date
    # Clamp to current actual date if in the future
    actual_now = pd.Timestamp.now()
    if dt > actual_now + pd.Timedelta(days=1):  # Allow 1 day buffer for timezone differences
        logging.warning(f"Provided date {dt} is in the future, using current date {actual_now}")
        dt = actual_now

    # Use provided cache_dir or default
    calendar_cache_dir = cache_dir if cache_dir is not None else CALENDAR_CACHE_DIR
    # Ensure calendar cache directory exists
    calendar_cache_dir.mkdir(parents=True, exist_ok=True)

    end_raw: pd.Timestamp = pd.to_datetime(dt)
    # Use local date for cache filename
    cache_date: pd.Timestamp = end_raw.tz_localize(None) if end_raw.tzinfo else end_raw
    
    # Normalize end to UTC for reliable comparison with schedule closes
    if end_raw.tzinfo is None:
        # If timezone-naive, assume it's local time and convert to UTC
        # Create a timezone-aware datetime in local time, then convert to UTC
        local_dt = datetime.combine(end_raw.date(), end_raw.time()).astimezone()
        end = pd.Timestamp(local_dt).tz_convert('UTC')
    else:
        end = end_raw.tz_convert('UTC')

    # Find the market close date prior or equal to dt, using exchange_calendars, with calendar caching
    cal_cache_file: Path = calendar_cache_dir / f'{calendar.lower()}_schedule_{cache_date.strftime("%Y%m%d")}.tsv'
    logging.debug(f"Looking for calendar cache at: {cal_cache_file}")
    if cal_cache_file.exists():
        logging.debug(f"Loading calendar from cache: {cal_cache_file}")
        schedule: pd.DataFrame = read_tsv(cal_cache_file, index_col=0, parse_dates=['market_open_utc', 'market_close_utc'])
        # Rename columns back to standard names for consistency
        schedule.rename(columns={'market_open_utc': 'market_open', 'market_close_utc': 'market_close'}, inplace=True)
        # Ensure datetime columns are timezone-aware (UTC)
        for col in ['market_open', 'market_close']:
            if col in schedule.columns:
                schedule[col] = pd.to_datetime(schedule[col], utc=True)
    else:
        logging.info(f"Fetching {calendar} calendar schedule from API")
        cal = mcal.get_calendar(calendar)
        
        # Normalize cache_date to date-only for comparison with calendar bounds
        cache_date = cache_date.normalize()
        
        # Ensure cache_date doesn't exceed calendar's last session
        if cache_date > cal.last_session:
            logging.warning(f"cache_date {cache_date} exceeds calendar last session {cal.last_session}, clamping")
            cache_date = cal.last_session
        
        # Get a schedule window that covers at least CALENDAR_LOOKBACK_DAYS before dt
        start_date = (cache_date - pd.Timedelta(days=CALENDAR_LOOKBACK_DAYS)).normalize()
        end_date = cache_date.normalize()
        
        # Ensure start_date is not before calendar's first session
        if start_date < cal.first_session:
            logging.warning(f"start_date {start_date} before calendar first session {cal.first_session}, clamping")
            start_date = cal.first_session
        
        # exchange_calendars uses sessions_in_range to get trading sessions (requires date-only inputs)
        logging.debug(f"Requesting sessions from {start_date} to {end_date}")
        
        # exchange_calendars expects date strings, not datetime objects
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        logging.debug(f"Calling sessions_in_range with: start='{start_str}', end='{end_str}'")
        sessions = cal.sessions_in_range(start_str, end_str)
        
        # Build schedule DataFrame with open/close times
        schedule = pd.DataFrame({
            'market_open': [cal.session_open(session) for session in sessions],
            'market_close': [cal.session_close(session) for session in sessions]
        }, index=sessions)
        # Rename columns to indicate UTC timezone in cache
        schedule_to_cache = schedule.rename(columns={'market_open': 'market_open_utc', 'market_close': 'market_close_utc'})
        schedule_to_cache.to_csv(str(cal_cache_file), sep='\t')
        logging.debug(f"Cached calendar to: {cal_cache_file}")

    if schedule.empty:
        logging.warning(f"No market schedule found for {calendar} before {end}")
        return None
    
    # Find the last close at or before dt
    closes: pd.Series = schedule['market_close']
    
    # Ensure closes are timezone-aware in UTC
    if closes.dt.tz is None:
        closes = closes.dt.tz_localize('UTC')
    else:
        closes = closes.dt.tz_convert('UTC')
    
    # Filter closes to include only those on or before dt
    closes_filtered: pd.Series = closes[closes <= end]
    
    if closes_filtered.empty:
        logging.warning(f"No market close found on or before {end}")
        return None
    
    latest_close: pd.Timestamp = closes_filtered.iloc[-1]
    # Convert to local timezone for display
    # Get the local timezone by converting to Python datetime then back
    latest_close_py = latest_close.to_pydatetime().astimezone()
    latest_close_local: pd.Timestamp = pd.Timestamp(latest_close_py)
    logging.debug(f"Latest market close (UTC): {latest_close}, (local): {latest_close_local}")
    return latest_close_local


if __name__ == "__main__":
    # Configure logging for tests
    from config import LOG_LEVEL, LOG_FORMAT
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    
    # Basic manual tests
    now_py = datetime.now().astimezone()
    now = pd.Timestamp(now_py).round('s')
    print(f"Latest NYSE close before now {now}:", latest_mkt_close())

    # Specific mid-week date/time (timezone-naive treated as local time)
    test_dt = datetime(2025, 9, 24, 15, 0, 0).astimezone()
    test_date = pd.Timestamp(test_dt)
    print(f"Latest NYSE close before {test_date}:", latest_mkt_close(test_date))

    # Weekend date (timezone-naive treated as local time, should resolve to previous Friday close)
    weekend_dt = datetime(2025, 9, 20, 0, 0, 0).astimezone()  # Saturday
    weekend_date = pd.Timestamp(weekend_dt)
    print(f"Latest NYSE close before {weekend_date}:", latest_mkt_close(weekend_date))

