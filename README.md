# Portfolio Manager

A desktop portfolio analytics application built with [Flet](https://flet.dev/) (Python).
Imports brokerage export files, downloads live prices via `yfinance`, and renders charts and stats in a dark-themed UI.

---

## Navigation

Six sections accessible from the sidebar:

### Dashboard
Overview KPIs, summary statistics, portfolio health score, and daily P/L chart.

### Progress
Portfolio progression analysis with asset-by-asset breakdowns and growth trajectory.

### Portfolio
Holdings table · Holdings bar chart · Top movers (gainers/losers) · Trade log

### Analysis
Correlation matrix · Market graphs · Sector allocation · Performance attribution ·
Returns distribution · Monthly heatmap · Benchmark comparison (SPY/QQQ/DIA/IWM) ·
Correlation deep-dive · Risk-adjusted returns (CAPM alpha/beta)

### Risk
Risk radar · Diversification score · Market regime detector · Drawdown chart ·
Rolling Sharpe (60d) · Rolling volatility (30d) · Win/loss streaks

### About
App info and feature summary.

---

## Setup

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate portfolio-manager
python Portmanv2.py
```

### pip

```bash
pip install flet pandas numpy yfinance duckdb quantstats pyportfolioopt matplotlib openpyxl
python Portmanv2.py
```

---

## Importing Data

1. Enter a folder path (or file path) in the import bar at the top of the Dashboard.
2. Click **SCAN** to list importable files.
3. Click **IMPORT SELECTED** to load data.

**Supported formats:** CSV, TSV, Excel (.xlsx/.xls), JSON, Parquet, Feather, HTML tables

**Tested with:** Fidelity position exports and activity exports.

Column names are auto-mapped — standard brokerage column names like `Symbol`, `Quantity`,
`Current Value`, `Cost Basis Total`, `Gain/Loss Dollar` are recognized automatically.

---

## Data Storage

All data is stored locally in `ultra_portfolio_v25.db` (DuckDB), created automatically in the project directory.

---

## Files

| File | Description |
|------|-------------|
| `Portmanv2.py` | Main application |
| `Portmanv1.py` | Previous version (reference) |
| `environment.yml` | Conda environment spec |
| `ultra_portfolio_v25.db` | Local DuckDB database (auto-created) |

---

## Requirements

- Python 3.10+
- Internet connection for live price data (`yfinance`)
