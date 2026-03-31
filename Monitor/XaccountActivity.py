"""Analyze Fidelity account activity files and list traded symbols.
This script reads one or more Fidelity account activity CSV files,
cleans and combines the transaction rows into a pandas DataFrame, then
identifies unique ticker symbols and prints each symbol with its
description fetched from Yahoo Finance via yfinance.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import glob
import os
import sys

import pandas as pd
import yfinance as yf

from config import ACCOUNT_ACTIVITY_DIR

# Normalization map for symbols that differ from Yahoo's ticker format
SYMBOL_MAP = {
    "BRKB": "BRK-B",
}


def _fetch_description(symbol: str) -> tuple[str, str]:
    """Fetch longName for a symbol from yfinance, returning (printed_symbol, description).

    Uses SYMBOL_MAP to translate from the CSV symbol (e.g. 'BRKB') to the
    Yahoo Finance ticker (e.g. 'BRK-B') and prints the normalized symbol.
    """
    try:
        lookup_symbol = SYMBOL_MAP.get(symbol, symbol)
        ticker = yf.Ticker(lookup_symbol)
        info = ticker.info
        description = info.get("longName", "")
        # Return the normalized symbol so printing uses BRK-B instead of BRKB
        return lookup_symbol, description
    except Exception as e:  # noqa: BLE001
        return symbol, f"[Error fetching description] {e}"


def main() -> None:
    """Parse and display account activity data.
    Side Effects: Reads CSV files from disk."""
    filepattern = str(ACCOUNT_ACTIVITY_DIR / 'fidelity*activity.csv')
    # filepattern = 'accountActivity/fidelity2025Q1activity.csv'
    files = glob.glob(filepattern)
    print(files)
    files.sort(reverse=True)
    table = []
    header = None
    for f in files:
        print(f"{f}")
        with open(f, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]
            # Remove empty rows
            rows = [row for row in rows if any(cell.strip() for cell in row)]

            # Find the actual header row (the first row that contains 'Symbol')
            header_row_idx = None
            for i, row in enumerate(rows):
                if any('Symbol' in cell for cell in row):
                    header_row_idx = i
                    break

            if header_row_idx is None:
                print(f"Skipping {f}: no header row containing 'Symbol' found")
                continue

            # Use detected header row, limit and pad to 18 columns
            raw_header = [col.lstrip('\ufeff') for col in rows[header_row_idx][:18]]
            file_header = raw_header + [''] * (18 - len(raw_header))

            if header is None:
                header = file_header
                table.append(header)

            # Align each row to 18 columns and remove footer
            for row in rows[header_row_idx + 1:]:
                if any('The data and information' in cell for cell in row):
                    break
                row = row[:18] + [''] * (18 - len(row)) if len(row) < 18 else row[:18]
                table.append(row)
        print(f"{f}: {len(rows)-1} rows")

    # make a dataframe from table  variable
    if not table or len(table) < 2:
        print("No data found in account activity files.")
        raise SystemExit(0)

    df = pd.DataFrame(table[1:], columns=table[0])

    """Print sorted symbols (<=5 chars) with descriptions from yfinance."""

    # Try to detect the symbol column name robustly
    symbol_col = None
    for col in df.columns:
        if "symbol" in str(col).lower():
            symbol_col = col
            break

    if symbol_col is None:
        print("No symbol-like column found. Available columns:")
        print(list(df.columns))
        raise SystemExit(1)

    # Print sorted symbols vertically, without duplicates and no index,
    # exclude those with more than 5 characters in the symbol column
    unique_symbols = sorted(
        set(
            df[symbol_col]
            .dropna()
            .astype(str)
            .str.strip()
            # Keep only non-empty symbols with length 1..5
            .loc[lambda x: (x.str.len() > 0) & (x.str.len() <= 5)]
        )
    )

    # Fetch descriptions in parallel to avoid slow sequential network calls
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_fetch_description, symbol): symbol for symbol in unique_symbols}
        for future in as_completed(futures):
            symbol, description = future.result()
            print(f"{symbol}: {description}")


if __name__ == '__main__':
    main()
