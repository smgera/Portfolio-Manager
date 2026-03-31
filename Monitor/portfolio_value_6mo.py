"""Compute and plot 6-month portfolio value and daily returns.
This script uses:
- Latest Fidelity `Portfolio_Positions*` file in `accountActivity/` to infer current
  dollar weights per symbol.
- `daily_data` from DailyData.py to fetch/cached daily price history.
It builds a daily time series of portfolio value over the last 6 months
(assuming constant share quantities equal to current position sizes) and
plots:
- Portfolio value over time
- Daily return series over the same period
"""

from __future__ import annotations

import csv
import glob
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import REPORT_DIR, ACCOUNT_ACTIVITY_DIR, PLOT_DPI_MEDIUM, FIDELITY_ACTION_BOUGHT

from Acquire.DailyData import daily_data
from Monitor.read_portfolio_positions import read_portfolio_positions

ACCT_DIR = ACCOUNT_ACTIVITY_DIR

def load_activity_trades(period: str = "6mo") -> pd.DataFrame:
    """Load BOUGHT/SOLD trades from fidelity*activity.csv for a given period.
    Returns a DataFrame with columns: Date (datetime64), Symbol, QtyDelta (float).
    Starting quantity is assumed to be zero before the first trade in the period.
    Side Effects:
        - Reads activity CSV files from disk
    """
    pattern = str(ACCT_DIR / "fidelity*activity.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No activity files found matching {pattern}")

    # Convert period like '6mo' to a start date
    now = pd.Timestamp.today().normalize()
    if period.endswith("mo"):
        months = int(period[:-2])
        start_date = now - pd.DateOffset(months=months)
    elif period.endswith("y"):
        years = int(period[:-1])
        start_date = now - pd.DateOffset(years=years)
    elif period.endswith("d"):
        days = int(period[:-1])
        start_date = now - pd.DateOffset(days=days)
    else:
        raise ValueError(f"Unsupported period format: {period}")

    records: List[Dict[str, object]] = []

    for fpath in files:
        with open(fpath, newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]
        # Remove empty rows
        rows = [row for row in rows if any(cell.strip() for cell in row)]

        # Find header row with both Action and Symbol
        header_row_idx = None
        header: List[str] | None = None
        for i, row in enumerate(rows):
            row_lower = [c.strip() for c in row]
            if "Action" in row_lower and "Symbol" in row_lower:
                header_row_idx = i
                header = row
                break
        if header_row_idx is None or header is None:
            continue

        col_index = {name: idx for idx, name in enumerate(header)}
        # Required columns
        date_col = col_index.get("Run Date") or col_index.get("Run Date/Time")
        action_col = col_index.get("Action")
        symbol_col = col_index.get("Symbol")
        qty_col = col_index.get("Quantity")
        if None in (date_col, action_col, symbol_col, qty_col):
            continue

        # Process rows until footer
        for row in rows[header_row_idx + 1 :]:
            if any("The data and information" in cell for cell in row):
                break
            try:
                date_val = pd.to_datetime(row[date_col], errors="coerce")
            except Exception:
                continue
            if pd.isna(date_val) or date_val < start_date:
                continue

            action = row[action_col].upper()
            symbol = row[symbol_col].strip()
            qty_str = row[qty_col].replace(",", "").strip()
            if not symbol or not qty_str:
                continue
            try:
                qty = float(qty_str)
            except ValueError:
                continue

            if FIDELITY_ACTION_BOUGHT in action:
                qty_delta = qty
            elif "SOLD" in action:
                qty_delta = -qty
            else:
                continue  # ignore non-trade actions

            records.append({"Date": date_val.normalize(), "Symbol": symbol, "QtyDelta": qty_delta})

    if not records:
        # No trades in the lookback period; return empty frame so callers can
        # fall back to snapshot-only positions (constant holdings).
        return pd.DataFrame(columns=["Date", "Symbol", "QtyDelta"])

    trades = pd.DataFrame.from_records(records)
    trades.sort_values(["Date", "Symbol"], inplace=True)
    return trades


def build_portfolio_series_from_trades(period: str = "6mo") -> pd.DataFrame:
    """Build portfolio value and daily return series using actual trade history.
    Reconstructs daily share counts per symbol from BOUGHT/SOLD actions and
    multiplies by daily Close prices from daily_data over the same period.
    Cash is not modeled; this tracks only the marked-to-market value of holdings.
    """

    trades = load_activity_trades(period=period)

    # Normalize known tickers like BRKB -> BRK-B
    symbol_map: Dict[str, str] = {"BRKB": "BRK-B"}
    trades["Symbol"] = trades["Symbol"].replace(symbol_map)

    # Load latest positions snapshot to anchor quantities
    df_anchor, _ = read_portfolio_positions(str(ACCT_DIR))

    # Build daily date index for the period
    end_date = pd.Timestamp.today().normalize()
    if period.endswith("mo"):
        months = int(period[:-2])
        start_date = end_date - pd.DateOffset(months=months)
    elif period.endswith("y"):
        years = int(period[:-1])
        start_date = end_date - pd.DateOffset(years=years)
    elif period.endswith("d"):
        days = int(period[:-1])
        start_date = end_date - pd.DateOffset(days=days)
    else:
        raise ValueError(f"Unsupported period format: {period}")

    date_index = pd.date_range(start=start_date, end=end_date, freq="D")

    # Use union of symbols from trades and snapshot so static positions are included
    symbols = sorted(set(trades["Symbol"].unique()).union(set(df_anchor["Symbol"].unique())))

    # Reconstruct daily share positions from trades only (starting from zero)
    pos_frames: Dict[str, pd.Series] = {}
    for sym in symbols:
        df_sym = trades[trades["Symbol"] == sym].copy()
        if not df_sym.empty:
            daily_changes = df_sym.groupby("Date")["QtyDelta"].sum()
            qty = daily_changes.cumsum().reindex(date_index, method="ffill").fillna(0.0)
        else:
            # No trades in window -> quantity stays at 0 until anchor offset is applied
            qty = pd.Series(0.0, index=date_index)
        pos_frames[sym] = qty

    positions = pd.DataFrame(pos_frames, index=date_index)

    # Fetch prices and compute anchor shares from snapshot's dollar values
    price_frames: Dict[str, pd.Series] = {}
    anchor_values = dict(zip(df_anchor["Symbol"], df_anchor["Current Value"]))
    anchor_shares: Dict[str, float] = {}

    for sym in symbols:
        print(f"Fetching {period} history for {sym}...")
        data = daily_data(sym, period=period)
        if data is None or data.empty or "Close" not in data.columns:
            print(f"  Skipping {sym}: no price data")
            continue
        s = data["Close"].copy()
        # Ensure DatetimeIndex is tz-naive for consistent comparisons with anchor_date
        idx = pd.to_datetime(s.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(None)
        s.index = idx.normalize()
        s = s.sort_index()
        price_frames[sym] = s

        # If symbol is in the snapshot, estimate shares using latest available price
        if sym in anchor_values:
            p_anchor = s.iloc[-1]
            if p_anchor > 0:
                anchor_shares[sym] = anchor_values[sym] / p_anchor

    if not price_frames:
        raise RuntimeError("No price series available for any symbols.")

    prices = pd.DataFrame(price_frames)
    # Align both positions and prices on common dates and symbols
    common_dates = positions.index.intersection(prices.index)
    common_syms = sorted(set(positions.columns).intersection(prices.columns))
    positions = positions.loc[common_dates, common_syms]
    prices = prices.loc[common_dates, common_syms]

    # Debug: inspect alignment and anchor_shares
    print("\nAligned positions shape:", positions.shape)
    print("Aligned prices shape:", prices.shape)
    print("Common symbols:", common_syms)
    print("Anchor shares (non-zero):", {k: v for k, v in anchor_shares.items() if v != 0})

    # Apply anchor offset so that positions match snapshot at a chosen anchor date
    if common_dates.empty:
        raise RuntimeError("No overlapping dates between positions and prices.")

    # Choose anchor date as the last common date (most recent)
    anchor_effective = common_dates[-1]

    anchor_pos_from_trades = positions.loc[anchor_effective]
    offsets = {}
    for sym in common_syms:
        q_anchor = anchor_shares.get(sym, 0.0)
        q_trades_anchor = float(anchor_pos_from_trades.get(sym, 0.0))
        offsets[sym] = q_anchor - q_trades_anchor

    offsets_series = pd.Series(offsets)
    positions = positions.add(offsets_series, axis="columns")

    # Portfolio value is sum of shares * Close per symbol
    portfolio_value = (positions * prices).sum(axis=1)
    daily_return = portfolio_value.pct_change().fillna(0.0)

    result = pd.DataFrame({
        "PortfolioValue": portfolio_value,
        "DailyReturn": daily_return,
    })
    return result


def build_portfolio_series(df_pos: pd.DataFrame, period: str = "6mo") -> pd.DataFrame:
    """Return DataFrame with portfolio value and daily return for given period.
    Assumes:
    - df_pos has columns ['Symbol', 'Current Value'] with current dollar value per symbol.
    - Quantities are held constant over the period; only prices move.
    """

    symbols = df_pos["Symbol"].tolist()
    values = df_pos["Current Value"].to_numpy(float)
    total_value = values.sum()
    if total_value <= 0:
        raise ValueError("Total portfolio value must be positive.")

    weights = values / total_value
    print("Total current value:", f"${total_value:,.2f}")

    price_frames: Dict[str, pd.Series] = {}

    for sym in symbols:
        print(f"Fetching {period} history for {sym}...")
        data = daily_data(sym, period=period)
        if data is None or data.empty or "Close" not in data.columns:
            print(f"  Skipping {sym}: no price data")
            continue
        # Ensure DatetimeIndex and sort
        s = data["Close"].copy()
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        price_frames[sym] = s

    if not price_frames:
        raise RuntimeError("No price series available for any symbols.")

    # Align on common date index (inner join)
    prices = pd.DataFrame(price_frames)
    prices = prices.dropna(how="any")
    prices = prices[sorted(prices.columns)]  # stable order

    print(f"Aligned price matrix shape: {prices.shape}")

    # Reorder weights to match columns
    sym_to_weight = dict(zip(symbols, weights))
    w_vec = np.array([sym_to_weight[sym] for sym in prices.columns], dtype=float)

    # Normalize prices to 1 at first date
    norm_prices = prices / prices.iloc[0]

    # Portfolio value index (starting at current total_value)
    portfolio_index = (norm_prices * w_vec).sum(axis=1)
    portfolio_value = portfolio_index * total_value

    # Daily returns
    daily_return = portfolio_value.pct_change().fillna(0.0)

    result = pd.DataFrame(
        {
            "PortfolioValue": portfolio_value,
            "DailyReturn": daily_return,
        }
    )
    return result


def plot_portfolio(result: pd.DataFrame, title: str = "Portfolio Value and Daily Return (6mo)") -> None:
    """Plot portfolio value and daily return on two axes.
    Side Effects:
        - Saves plot file to disk
    """

    # Simple debug output to inspect ranges
    print("\nPortfolio series preview:")
    print(result.head())
    print("PortfolioValue min/max:", result["PortfolioValue"].min(), result["PortfolioValue"].max())
    print("DailyReturn min/max:", result["DailyReturn"].min(), result["DailyReturn"].max())

    # Compute 5-day EMA of daily returns for smoothing
    ema_5d = result["DailyReturn"].ewm(span=5, adjust=False).mean()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Portfolio value
    ax1.plot(result.index, result["PortfolioValue"], color="tab:blue", label="Portfolio Value")
    ax1.set_ylabel("Value ($)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Daily returns on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(result.index, result["DailyReturn"], color="tab:orange", alpha=0.4, label="Daily Return")
    ax2.plot(result.index, ema_5d, color="tab:red", linewidth=1.2, label="Daily Return EMA 5d")
    # Flat reference line at 0 daily return
    ax2.axhline(0.0, color="gray", linewidth=1.0, linestyle="--", label="0 Gain")
    ax2.set_ylabel("Daily Return", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.set_title(title)
    ax1.set_xlabel("Date")

    # Vertical grid: weekly ticks on the date axis (relative to latest date)
    week_locator = mdates.WeekdayLocator(interval=1)  # every week
    ax1.xaxis.set_major_locator(week_locator)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.grid(True, axis="x", linestyle=":", alpha=0.5)

    # Horizontal grid on daily return axis only
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    fig.tight_layout()

    # Save figure to reports/ and open with default image viewer
    imagefile = REPORT_DIR / "portfolio_value_6mo.png"
    fig.savefig(imagefile, dpi=PLOT_DPI_MEDIUM)
    plt.close(fig)

    try:
        os.startfile(imagefile)  # type: ignore[attr-defined]
    except Exception:
        # Fallback: print path if auto-open fails
        print(f"Plot saved to {imagefile}")


def main() -> None:
    """Build and plot 6-month portfolio value from trades."""
    # Trade-based series using actual BOUGHT/SOLD history
    result = build_portfolio_series_from_trades(period="6mo")
    plot_portfolio(result)


if __name__ == "__main__":
    main()
