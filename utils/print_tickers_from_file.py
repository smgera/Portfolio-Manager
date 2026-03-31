"""Utility to print tickers and latest adjusted close prices from a TSV file."""

import sys
from pathlib import Path

import pandas as pd

from config import TICKER_LISTS_DIR
from utils.io_utils import read_tsv

from Acquire.DailyData import daily_data

TICKER_FILE = TICKER_LISTS_DIR / 'myTickers.tsv'


def print_tickers_and_latest_adj_close(filepath: Path) -> None:
    """Read tickers from a TSV file and print each ticker symbol and latest close.
    Args:
        filepath: Path to a TSV file with a 'Symbol' column
    Side Effects:
        - Reads TSV file from disk
        - Fetches price data via daily_data() (network/cache)
    """
    df = read_tsv(filepath)
    if 'Symbol' not in df.columns:
        print(f"Column 'Symbol' not found in {filepath}")
        return

    print(f"{'Symbol':<10}  {'Latest Adj Close':>18}")
    print('-' * 32)
    for symbol in df['Symbol']:
        try:
            data = daily_data(symbol, period='1d')
            if data is not None and not data.empty and 'Close' in data.columns:
                latest_adj_close = data['Close'].iloc[-1]
                print(f"{symbol:<10}  {latest_adj_close:18.2f}")
            else:
                print(f"{symbol:<10}  {'N/A':>18}")
        except Exception:
            print(f"{symbol:<10}  {'Error':>18}")


if __name__ == "__main__":
    filepath = Path(sys.argv[1]) if len(sys.argv) > 1 else TICKER_FILE
    print_tickers_and_latest_adj_close(filepath)
