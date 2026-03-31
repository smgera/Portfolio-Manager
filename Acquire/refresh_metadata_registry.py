"""
Refresh the metadata registry for all tickers used across the trading system.
Fetches from yfinance, applies PE sanitization, and writes to metadata_registry.tsv.
Run this before myPE.py and screen.py to ensure fresh data.
Side Effects:
    - Reads ticker list TSV/CSV files
    - Writes metadata_registry.tsv via fetch_and_update_registry
"""

import sys
from pathlib import Path

# Add project root to path to ensure we import the root config package
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config import METADATA_TICKER_FILES, METADATA_EXPIRY_DAYS
from utils.io_utils import read_tsv
from common.metadata_io import fetch_and_update_registry, load_metadata_registry


def load_tickers_from_file(path: Path) -> list:
    """Load Symbol column from a TSV/CSV ticker file, ignoring missing files.
    Args:
        path: Path to ticker list file
    Returns:
        List of ticker symbols with '.' replaced by '-'
    Side Effects:
        - Prints skip message to stdout on file read failure
    """
    try:
        sep = '\t' if str(path).endswith('.tsv') else ','
        df = read_tsv(path, sep=sep)
        return [s.replace('.', '-') for s in df['Symbol'].tolist()]
    except Exception as e:
        print(f"Skipping {path}: {e}")
        return []


def main() -> None:
    """Main function to refresh metadata registry."""
    # Load all tickers from configured ticker files
    all_tickers: list = []
    for f in METADATA_TICKER_FILES:
        all_tickers.extend(load_tickers_from_file(f))

    # Remove duplicates
    seen: set = set()
    unique_tickers = [t for t in all_tickers if not (t in seen or seen.add(t))]
    
    # Load current registry to find missing tickers
    registry = load_metadata_registry()
    registry_tickers = set(registry.keys())
    
    # Find tickers that are in myTickers but not in registry
    missing_tickers = [t for t in unique_tickers if t not in registry_tickers]
    
    if missing_tickers:
        print(f"Found {len(missing_tickers)} tickers in myTickers but not in registry:")
        for ticker in missing_tickers:
            print(f"  - {ticker}")
        print(f"Fetching {len(missing_tickers)} missing tickers...")
        
        # Fetch only the missing tickers
        fetch_and_update_registry(missing_tickers)
    else:
        print("All tickers from myTickers are already in registry.")
    
    # Also refresh any stale tickers (older than expiry days)
    print(f"Checking for stale tickers (older than {METADATA_EXPIRY_DAYS} days)...")
    fetch_and_update_registry(unique_tickers)
    
    print("Metadata registry refresh complete.")


if __name__ == "__main__":
    main()
