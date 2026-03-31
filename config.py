"""
Configuration settings for the trading analysis project.

WARNING: This is the ONLY config file for the project. 
Do NOT create a config.py file in the project root - it will cause confusion.
All configuration changes must be made here in config/__init__.py
"""
from pathlib import Path
from typing import Final
import sys
import os
import warnings

# ============================================================================
# Environment Setup
# ============================================================================
# Clear PYTHONPATH to avoid import conflicts with other projects
# This ensures clean module resolution for the trading project
# ============================================================================
if 'PYTHONPATH' in os.environ:
    print("PYTHONPATH found in environment variables, removing")
    del os.environ['PYTHONPATH']
else:
    print("PYTHONPATH not found in environment variables")

# ============================================================================
# Global Warning Suppression
# ============================================================================
# Suppress common deprecation warnings from third-party libraries
# This keeps output clean while maintaining functionality
# ============================================================================

# Suppress common yfinance pandas deprecation warnings globally
warnings.filterwarnings("ignore", category=FutureWarning, message=".*Timestamp.utcnow.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*Timestamp.utcnow.*")

# Add external data-pipeline directory to path (not part of this package)  # noqa: sys-path-external
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "data" / "data-pipeline"))  # noqa: sys-path-external
try:
    from get_history import get_history_df
except ImportError:
    # Fallback for environments where this module isn't available
    def get_history_df(*args, **kwargs):
        return None


from pathlib import Path
from typing import Final
import sys
import warnings


# Base directories
PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent
PARENT_DATA_DIR: Final[Path] = PROJECT_ROOT.parent / "data"
RESULTS_DIR: Final[Path] = PARENT_DATA_DIR / "results"
CACHE_DIR: Final[Path] = PARENT_DATA_DIR
REPORT_DIR: Final[Path] = RESULTS_DIR
TICKER_LISTS_DIR: Final[Path] = PARENT_DATA_DIR / "tickerLists"
ACCOUNT_ACTIVITY_DIR: Final[Path] = PARENT_DATA_DIR / "accountActivity"
PLOTS_DIR: Final[Path] = RESULTS_DIR
FEATHER_CACHE_DIR: Final[Path] = PARENT_DATA_DIR / "feather_ts"
AUDITS_DIR: Final[Path] = PROJECT_ROOT.parent / "audits"
METADATA_REGISTRY_PATH: Final[Path] = PARENT_DATA_DIR / "metadata_registry.tsv"
METADATA_EXPIRY_DAYS: Final[int] = 7  # Re-fetch metadata older than this

# All ticker list files whose tickers should be in the metadata registry
METADATA_TICKER_FILES: Final[list] = [
    PARENT_DATA_DIR / "tickerLists" / "myTickers.tsv",
    PARENT_DATA_DIR / "tickerLists" / "bigETFs.tsv",
    PARENT_DATA_DIR / "tickerLists" / "bigStocks500.tsv",
    PARENT_DATA_DIR / "tickerLists" / "bigMixedAssets.tsv",
]

# Aliases for backward compatibility
data_dir = CACHE_DIR
priceHistory_dir = CACHE_DIR / "history_1d"
ticker_dir = TICKER_LISTS_DIR

# Cache subdirectories
DAILY_DATA_CACHE_DIR: Final[Path] = CACHE_DIR / "history_1d"
CALENDAR_CACHE_DIR: Final[Path] = CACHE_DIR / "calendar"
OPTIONS_CACHE_DIR: Final[Path] = CACHE_DIR / "options"

# Data format settings
USE_DATETIMEINDEX: Final[bool] = True  # True = DatetimeIndex, False = 'Date' column

# Logging settings
LOG_LEVEL: Final[str] = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT: Final[str] = "%(asctime)s - %(levelname)s - %(message)s"

# Market settings
DEFAULT_MARKET_CALENDAR: Final[str] = "NYSE"  # NYSE trading calendar (9:30 AM - 4:00 PM ET, Mon-Fri)
TRADING_DAYS_PER_YEAR: Final[int] = 252
TRADING_DAYS_PER_QUARTER: Final[int] = 63
TRADING_DAYS_PER_MONTH: Final[int] = 21
TRADING_DAYS_PER_WEEK: Final[int] = 5

# ============================================================================
# TIMEZONE CONFIGURATION
# ============================================================================
# IMPORTANT: This codebase uses the SYSTEM LOCAL TIMEZONE by default.
# No explicit timezone (like EST/EDT or PST/PDT) is hardcoded.
#
# Timezone Handling Behavior:
# 1. Timezone-naive datetimes are treated as LOCAL TIME
# 2. Market calendar data is stored in UTC for consistency
# 3. All timezone conversions happen automatically in the background
#
# Examples:
# - If your system is in New York (EST/EDT): timezone-naive = Eastern Time
# - If your system is in California (PST/PDT): timezone-naive = Pacific Time
# - If your system is in London: timezone-naive = GMT/BST
# - If your system is in Tokyo: timezone-naive = JST
#
# To change timezone behavior:
# - Change your system's timezone settings
# - Or modify the code to explicitly handle timezones (advanced users only)
# ============================================================================

# Cache settings
CALENDAR_LOOKBACK_DAYS: Final[int] = 10  # Days to look back for market calendar
MIN_CACHE_FILE_SIZE: Final[int] = 10  # Minimum valid cache file size in bytes

# Data columns
REQUIRED_COLUMNS: Final[list[str]] = ["Open", "High", "Low", "Close", "Volume"]
CACHE_COLUMNS: Final[list[str]] = ["Date", "Open", "High", "Low", "Close", "Volume"]

# Analysis settings
# Symbols to ignore (Cash, Money Market, etc.)
# SGOV is considered the ETF closest to a risk-free return, U.S. Treasurys three months or less
# BIL is similar and has more history
# Money market funds that need BIL substitution for yfinance
MONEY_MARKET_FUNDS: Final[list[str]] = ['SPAXX', 'FDRXX', 'FZFXX', 'FZDXX', 'SWVXX', 'VMFXX', 'SPRXX', 'VUSXX', 'IMRXX', 'MJLXX']

# File patterns for common data files
FIDELITY_ACTIVITY_PATTERN: Final[str] = 'fidelity*activity.csv'
PORTFOLIO_POSITIONS_PATTERN: Final[str] = 'Portfolio_Positions_*.csv'
TICKER_FILE_PATTERN: Final[str] = '*ticker*.tsv'

# Portfolio freshness settings
PORTFOLIO_FRESHNESS_THRESHOLD_DAYS: Final[int] = 0  # Maximum age of positions CSV in days (0 = only today's data is fresh)

# Portfolio sizing constants
PORTFOLIO_TARGET_SIZE: Final[int] = 600000  # $600k target portfolio size
SCRIPT_TIMEOUT_SECONDS: Final[int] = 300  # Subprocess timeout (5 minutes)

# Screening/filter thresholds
MIN_SHARPE_THRESHOLD: Final[float] = 0.1  # Minimum Sharpe ratio to include in results
MIN_SHARPE_FILTER: Final[float] = -0.1  # Minimum Sharpe ratio for loose filter

# Plot resolution settings
PLOT_DPI: Final[int] = 300        # Standard plot DPI for high-resolution output
PLOT_DPI_MEDIUM: Final[int] = 150  # Medium plot DPI (market breadth, breadth charts, etc.)

# Financial constants
RISK_FREE_RATE: Final[float] = 0.045   # Annual risk-free rate (approx. 3-month T-bill yield)
SLIPPAGE_DEFAULT: Final[float] = 0.001  # Default slippage/transaction cost per trade

# Hurst exponent interpretation thresholds
HURST_MEAN_REVERTING_THRESHOLD: Final[float] = 0.45  # H <= this → mean-reverting regime
HURST_TRENDING_THRESHOLD: Final[float] = 0.55        # H >= this → trending regime

# ETF data file paths
ETF_PRECALCULATED_CSV: Final[str] = "etf_precalculated_only.csv"  # Output CSV for pre-calculated ETF metrics
BARCHART_HIGN_CSV: Final[str] = "hign_data.csv"  # Input CSV for Barchart $HIGN data

# External URL constants
URL_BARCHART_HIGN: Final[str] = "https://www.barchart.com/stocks/quotes/$HIGN/price-history/historical"
URL_FRED_NASDAQHLR: Final[str] = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=NASDAQHLR"
URL_YAHOO_STOCK_GAINERS: Final[str] = "https://finance.yahoo.com/screener/predefined/day_gainers"
URL_YAHOO_MOST_ACTIVE_ETFS: Final[str] = "https://finance.yahoo.com/screener/predefined/most_actives_etfs"

# FRED series identifiers
FRED_SERIES_NASDAQ_HL_RATIO: Final[str] = "NASDAQHLR"
FRED_SERIES_NYSE_ADVANCES: Final[str] = "NYSEADVANCE"
FRED_SERIES_NYSE_DECLINES: Final[str] = "NYSEDECLINE"

# Fidelity action strings (values from Fidelity CSV activity files)
FIDELITY_ACTION_BOUGHT: Final[str] = 'BOUGHT'
FIDELITY_ACTION_SOLD: Final[str] = 'SOLD'
FIDELITY_ACTION_CANCELLED: Final[str] = 'CANCELLED'

# Fidelity-specific money market symbols (subset of MONEY_MARKET_FUNDS)
FIDELITY_MONEY_MARKET_SYMBOLS: Final[list[str]] = ['FDRXX', 'SPAXX', 'FZFXX']

# Cash equivalent symbols for portfolio performance (Fidelity MM funds used in portfolio)
FIDELITY_CASH_SYMBOLS: Final[list[str]] = ['FDRXX', 'SPAXX', 'SWVXX', 'FZFXX']

# Compliance output status strings
STATUS_PASSED: Final[str] = 'PASSED'
STATUS_FAILED: Final[str] = 'FAILED'

# Portfolio report labels
PORTFOLIO_TOTAL_LABEL: Final[str] = 'PORTFOLIO_TOTAL'

# Environment variable names for configurable paths
ETF_LIST_PATH_ENV: Final[str] = 'ETF_LIST_PATH'

# Market breadth symbols (major ETF holdings used as market breadth proxy)
BREADTH_SYMBOLS: Final[list[str]] = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
    'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'HD', 'CVX',
    'MRK', 'ABBV', 'KO', 'PEP', 'COST', 'AVGO', 'TMO', 'MCD',
    'CSCO', 'ACN', 'LIN', 'ABT', 'DHR', 'NKE', 'TXN', 'NEE',
    'PM', 'UNP', 'ORCL', 'COP', 'BMY', 'UPS', 'RTX', 'HON',
    'QCOM', 'LOW', 'INTU', 'AMGN', 'SBUX', 'AMD', 'CAT', 'GE',
    'BA', 'IBM',
]

# Default ticker list for sorted download scripts
SORTED_DOWNLOAD_TICKERS: Final[list[str]] = [
    'SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB', 'TSLA', 'NVDA', 'PYPL', 'ADBE',
]

# Invalid symbols loading
def load_invalid_symbols(invalid_symbols_file):
    """Load invalid symbols from file."""
    file_path = ticker_dir / invalid_symbols_file
    if file_path.exists():
        with open(file_path, 'r') as f:
            symbols = set(f.read().splitlines())
            print(f"{len(symbols)} invalid symbols from {file_path}")
            return symbols
    else:
        return set()

# Load invalid symbols from separate files
invalid_yfinance_symbols = load_invalid_symbols('yfinance_invalid_symbols.txt')
invalid_alpaca_symbols = load_invalid_symbols('alpaca_invalid_symbols.txt')

def validate_config_paths():
    """Validate that all critical config directories are accessible."""
    validation_errors = []
    
    critical_paths = {
        'CACHE_DIR': CACHE_DIR,
        'DAILY_DATA_CACHE_DIR': DAILY_DATA_CACHE_DIR,
        'CALENDAR_CACHE_DIR': CALENDAR_CACHE_DIR,
        'TICKER_LISTS_DIR': TICKER_LISTS_DIR,
        'ACCOUNT_ACTIVITY_DIR': ACCOUNT_ACTIVITY_DIR,
    }
    
    for name, path in critical_paths.items():
        try:
            # Test if path can be resolved and created
            path.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                validation_errors.append(f"{name}: Cannot create directory {path}")
        except Exception as e:
            validation_errors.append(f"{name}: {e}")
    
    if validation_errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(validation_errors)
        raise RuntimeError(error_msg)
    
    return True

# Validate configuration on import
validate_config_paths()

# Ensure directories exist (redundant with validation but kept for compatibility)
CACHE_DIR.mkdir(exist_ok=True)
DAILY_DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CALENDAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
OPTIONS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
