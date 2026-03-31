"""Acquire-module-specific configuration constants.
Import this instead of 'config' when you need Acquire-specific settings.
Root config constants (TICKER_LISTS_DIR, PARENT_DATA_DIR, etc.) come from
the root 'config' package which is always available via the editable install.
"""
from pathlib import Path
import logging

# Time periods
Sharpe_WINDOW = 22
Sharpe_RECALC_DAYS = 5
OPTIMIZATION_LOOKBACK_DAYS = 63

# Logging
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEV_LOGGING_ENABLED = True
DEV_LOG_DIR = "logs"
DEV_LOG_LEVEL = "DEBUG"

# Data storage (derived from absolute path, no root config import needed)
_DATA_DIR = Path(__file__).parent.parent.parent / "data"
DATA_FOLDER = str(_DATA_DIR)
HISTORY_1D_FOLDER = str(_DATA_DIR / "history_1d")
RESULTS_FOLDER = "results"

# Network retry
MAX_RETRIES = 3
RETRY_DELAY = 1

# Data fetch settings
HISTORY_LENGTH = 1  # Fetch data from N years ago when stale

# Display
TABLE_SEPARATOR_WIDTH = 70

# Portfolio optimization
GAIN_CAP = 5.0
MAX_TICKERS = 2000
MIN_VOLATILITY = 0.07
MIN_GAIN = 0.10
MIN_PRICE = 10.0
MIN_VOLUME_M = 0.5
MIN_GAINSHARPE = 2.0
MIN_ASSET_COUNT = 8
SPECIFIC_ASSET_CONSTRAINTS = {
    # 'GLDM': [0.05, 0.15],
}
OTHER_MAX_POS_SIZE = 0.1
OTHER_MIN_POS_SIZE = 0.00

TICKER_FILES = [
    str(_DATA_DIR / "tickerLists" / "myTickers.tsv"),  # keep 1st
    str(_DATA_DIR / "tickerLists" / "inverse-etf.com-2026-01.csv"),
    str(_DATA_DIR / "tickerLists" / "bigMixedAssets.tsv"),
    # str(_DATA_DIR / "tickerLists" / "bigStocks500.tsv"),
    # str(_DATA_DIR / "tickerLists" / "sector_etfs.tsv"),
]
