# Portfolio Manager

A comprehensive portfolio analytics and management platform built with Python. This project combines a desktop/web UI powered by Flet with sophisticated backend modules for portfolio performance analysis, data acquisition, and risk management.

---

## Overview

The Portfolio Manager consists of three main components:

1. **Portmanv2.py** - Modern web-based UI with 24 tabs across 8 categories
2. **Monitor Module** - Portfolio performance analysis and reporting
3. **Acquire Module** - Data acquisition and market data caching

### Key Features

- **Web Interface**: Dark-themed UI accessible via browser at http://localhost:8550
- **Data Import**: Supports CSV, TSV, Excel, JSON, Parquet, Feather, HTML
- **Real-time Data**: Live price feeds via yfinance
- **Performance Analytics**: Risk metrics, Sharpe ratios, drawdown analysis
- **Portfolio Optimization**: Target position calculations
- **Automated Reporting**: Generates TSV reports and myTickers.tsv

---

## Architecture

### Directory Structure

```
Portfolio-Manager/
├── Portmanv2.py              # Main application UI
├── config/                   # Configuration settings
│   └── __init__.py          # All project constants and paths
├── Monitor/                  # Portfolio analysis module
│   ├── portfolio_performance.py    # Performance analysis & myTickers.tsv
│   ├── read_portfolio_positions.py  # Position data processing
│   └── accountActivity.py           # Activity data processing
├── Acquire/                  # Data acquisition module
│   ├── DailyData.py         # Market data fetching & caching
│   ├── latest_close.py      # Latest price data
│   └── refresh_metadata_registry.py  # Symbol metadata
├── common/                   # Shared utilities
│   ├── financial_utils.py   # Financial calculations
│   └── portfolio_utils.py   # Portfolio helper functions
├── utils/                    # I/O utilities
│   └── io_utils.py          # File reading helpers
└── ultra_portfolio_v25.db   # DuckDB database (auto-created)
```

### Data Flow

1. **Import**: Brokerage exports (CSV/Excel) → DuckDB database
2. **Process**: Position & activity data → Portfolio metrics
3. **Analyze**: Risk calculations → Performance reports
4. **Output**: myTickers.tsv → Portfolio optimization

---

## Setup

### Prerequisites

- Python 3.10+
- Conda (recommended) or pip
- Internet connection for live data

### Installation

#### Conda (Recommended)

```bash
# Create environment from spec
conda env create -f environment.yml
conda activate portman

# Run the application
python Portmanv2.py
```

#### pip

```bash
# Install dependencies
pip install flet pandas numpy yfinance duckdb quantstats pyportfolioopt matplotlib openpyxl

# Run the application
python Portmanv2.py
```

The application will open in your default browser at `http://localhost:8550`

---

## Configuration

### Directory Settings

All paths are configured in `config/__init__.py`:

- `ACCOUNT_ACTIVITY_DIR`: Location for brokerage exports (default: `../data/accountActivity`)
- `TICKER_LISTS_DIR`: Output directory for ticker lists (default: `../data/tickerLists`)
- `RESULTS_DIR`: Report output directory (default: `../data/results`)

### Key Constants

- `TRADING_DAYS_PER_YEAR`: 252
- `TRADING_DAYS_PER_MONTH`: 21
- `PORTFOLIO_FRESHNESS_THRESHOLD_DAYS`: 0 (only today's data is fresh)
- `MONEY_MARKET_FUNDS`: List of money market symbols to filter

---

## Usage

### 1. Import Data

1. Launch the web interface
2. Navigate to the Dashboard tab
3. Enter folder path containing brokerage exports
4. Click **SCAN** to list files
5. Click **IMPORT SELECTED** to load data

**Supported Brokers**: Tested with Fidelity position and activity exports

**File Patterns**:
- Positions: `Fidelity_Positions_YY-MM-DD.csv` or `Portfolio_Positions_YYYYMonDD.csv`
- Activity: `Fidelity_Activity_YYYY-QX.csv`

### 2. Generate Ticker List

The system automatically generates `myTickers.tsv` with current open positions:

```bash
# Run standalone
python Monitor/portfolio_performance.py
```

This script:
- Reads positions from last 3 months
- Excludes closed positions (quantity = 0)
- Aggregates cash into PULS
- Outputs to `../data/tickerLists/myTickers.tsv`

### 3. Portfolio Analysis

Access 24 analytical tools across 8 categories:

#### Dashboard
- Portfolio KPIs
- Daily P/L chart
- Health score

#### Progress
- Growth trajectory
- Asset breakdowns

#### Portfolio
- Holdings table
- Top movers
- Trade log

#### Analysis
- Correlation matrix
- Sector allocation
- Performance attribution
- Returns distribution
- Monthly heatmap
- Benchmark comparison (SPY/QQQ/DIA/IWM)

#### Risk
- Risk radar
- Diversification score
- Market regime detection
- Drawdown chart
- Rolling Sharpe (60d)
- Rolling volatility (30d)
- Win/loss streaks

---

## Data Management

### Database

- **Engine**: DuckDB (in-process SQL database)
- **File**: `ultra_portfolio_v25.db`
- **Tables**: `records` (transactions, positions, activities)

### Caching

Market data is cached for performance:
- **Location**: `../data/history_1d/`
- **Format**: Feather files (.feather)
- **Refresh**: Automatic when data is stale

### File Formats

**Input Formats Supported**:
- CSV/TSV
- Excel (.xlsx, .xls)
- JSON
- Parquet
- Feather
- HTML tables

**Output Formats**:
- TSV reports (tab-separated)
- DuckDB database
- Matplotlib charts (embedded in UI)

---

## Advanced Features

### Performance Metrics

Calculates comprehensive risk-adjusted metrics:
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Volatility (daily/annual)
- Beta and alpha
- Information ratio

### Portfolio Optimization

The `calculate_target_positions()` function provides:
- Optimal position sizing based on Kelly criterion
- Risk-adjusted allocation
- Target vs current position comparison
- Rebalancing suggestions

### Data Pipeline

Automated data processing:
1. **Read**: Parse brokerage exports
2. **Clean**: Standardize column names
3. **Filter**: Remove closed positions
4. **Aggregate**: Combine cash positions
5. **Output**: Generate actionable reports

---

## Troubleshooting

### Common Issues

1. **"Data is stale" warning**
   - Update position files in ACCOUNT_ACTIVITY_DIR
   - Check PORTFOLIO_FRESHNESS_THRESHOLD_DAYS setting

2. **Import errors**
   - Verify file format is supported
   - Check column names match expected format
   - Ensure files are not password protected

3. **Missing market data**
   - Check internet connection
   - Verify symbol validity
   - Clear cache if needed

### Debug Mode

Enable verbose logging:
```python
# In config/__init__.py
LOG_LEVEL = "DEBUG"
DEV_LOGGING_ENABLED = True
```

---

## Development

### Project Structure

The project follows a modular architecture:
- **UI Layer**: Flet-based web interface
- **Business Logic**: Portfolio analysis in Monitor/
- **Data Layer**: Acquisition in Acquire/
- **Utilities**: Shared code in common/ and utils/

### Adding New Features

1. UI changes: Modify Portmanv2.py
2. New metrics: Add to Monitor/portfolio_performance.py
3. Data sources: Extend Acquire/DailyData.py
4. Configuration: Update config/__init__.py

### Testing

Run individual modules:
```bash
# Test portfolio performance
python Monitor/portfolio_performance.py

# Test data acquisition
python Acquire/latest_close.py

# Test configuration
python -c "import config; print(config.ACCOUNT_ACTIVITY_DIR)"
```

---

## License

Private project - All rights reserved

---

## Support

For issues or questions:
1. Check this README
2. Review todo.md for pending features
3. Examine error logs in console output
