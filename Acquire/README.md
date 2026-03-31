# Acquire Subsystem

## Responsibility
Fetching data from external sources (APIs, web scraping, etc.).

## Inputs
- Configuration (API keys, symbols list)
- `config.py` settings

## Outputs
- Raw data files (CSV, JSON, Parquet) stored in `cache/`
- Database entries

## Key Components
- `fetchHistory.py`: Downloads historical price data.
- `scrape*.py`: Scrapes fundamental data (market cap, etc.).
- `DailyData.py`: Interface for retrieving daily price data.
