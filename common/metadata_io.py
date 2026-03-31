"""Metadata registry I/O: load, save, and concurrent fetch/update from yfinance."""

import concurrent.futures
from io import StringIO
import math
from pathlib import Path
import time
from typing import Dict, Any, List

import pandas as pd
from tqdm import tqdm
import yfinance as yf

from config import PARENT_DATA_DIR, METADATA_EXPIRY_DAYS
from utils.io_utils import read_tsv

METADATA_REGISTRY_PATH = PARENT_DATA_DIR / "metadata_registry.tsv"


def sanitize_pe(value: object) -> float:
    """Sanitize a PE ratio value.
    - None or NaN  → NaN  (missing/not available)
    - <= 0         → 1000 (negative earnings, not meaningful)
    - positive     → kept as-is
    """
    if value is None:
        return float('nan')
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float('nan')
    if math.isnan(v):
        return float('nan')
    if v <= 0:
        return 1000.0
    return v


def load_metadata_registry() -> Dict[str, Any]:
    """Load metadata registry from TSV file.
    Returns:
        Dictionary mapping tickers to metadata dicts
    Side Effects:
        - Reads TSV file from disk
    """
    registry = {}
    reg_path = METADATA_REGISTRY_PATH

    if reg_path.exists():
        with open(reg_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        data_start = 0
        for i, line in enumerate(lines):
            if not line.startswith('#'):
                data_start = i
                break

        if data_start < len(lines):
            data_str = ''.join(lines[data_start:])
            df = read_tsv(StringIO(data_str))
            for _, row in df.iterrows():
                ticker = row['Ticker']
                registry[ticker] = row.to_dict()

    return registry


def save_metadata_registry(registry: Dict[str, Any]) -> None:
    """Save registry to TSV file with timestamp header.
    Args:
        registry: Dictionary mapping tickers to metadata dicts
    Side Effects:
        - Creates/overwrites metadata_registry.tsv
    """
    reg_path = METADATA_REGISTRY_PATH

    registry_list = [{'Ticker': k, **v} for k, v in registry.items()]
    registry_df = pd.DataFrame(registry_list)

    with open(reg_path, 'w', encoding='utf-8') as f:
        f.write(f"# Last Updated: {pd.Timestamp.now().isoformat()}\n")
        f.write(registry_df.to_csv(sep='\t', index=False, lineterminator='\n'))


def _fetch_one_ticker(ticker: str) -> Dict[str, Any]:
    """Fetch metadata for a single ticker from yfinance.
    This is single place where external metadata is fetched.
    PE sanitization via sanitize_pe() happens here.
    Side Effects:
        - Network request to yfinance API
    """
    max_retries = 3
    base_backoff = 1.0
    for attempt in range(max_retries):
        try:
            info = yf.Ticker(ticker).info

            market_cap = info.get('marketCap')
            market_cap_b = round(market_cap / 1e9, 2) if market_cap else float('nan')

            avg_volume = info.get('averageVolume')
            avg_volume_m = round(avg_volume / 1e6, 2) if avg_volume else float('nan')

            return {
                'Name': info.get('shortName', ''),
                'Sector': info.get('sector', '') or info.get('category', ''),
                'Industry': info.get('industry', ''),
                'Region': info.get('country', ''),
                'MarketCapB': market_cap_b,
                'AvgVolumeM': avg_volume_m,
                'TrailPE': sanitize_pe(info.get('trailingPE')),
                'FwdPE': sanitize_pe(info.get('forwardPE')),
                'Beta': info.get('beta', float('nan')),
                'PEGRatio': info.get('pegRatio', float('nan')),
                'ProfitMargin': info.get('profitMargins', float('nan')),
                'DebtToEquity': info.get('debtToEquity', float('nan')),
                'ROE': info.get('returnOnEquity', float('nan')),
                'MetadataUpdated': pd.Timestamp.now().isoformat(),
            }
        except Exception:
            if attempt < max_retries - 1:
                backoff = base_backoff * (2 ** attempt)
                time.sleep(backoff)
            else:
                return {}
    return {}


def fetch_and_update_registry(tickers: List[str], max_workers: int = 8) -> Dict[str, Any]:
    """Fetch metadata for tickers from yfinance, sanitize PE, and save to registry.
    This is the single authoritative entry point for external metadata fetching.
    PE sanitization (<=0 → 1000, None/NaN → NaN) is applied here before saving.
    Skips tickers whose MetadataUpdated is within METADATA_EXPIRY_DAYS.
    Args:
        tickers: List of ticker symbols to fetch/update
        max_workers: Thread pool size for concurrent fetching
    Returns:
        Updated registry dict
    Side Effects:
        - Makes network API calls via yfinance
        - Writes metadata_registry.tsv to disk
    """
    registry = load_metadata_registry()
    expiry_cutoff = pd.Timestamp.now() - pd.Timedelta(days=METADATA_EXPIRY_DAYS)

    stale = []
    for t in tickers:
        updated = registry.get(t, {}).get('MetadataUpdated')
        try:
            fresh = bool(updated and pd.Timestamp(updated) > expiry_cutoff)
        except Exception:
            fresh = False
        if not fresh:
            stale.append(t)

    if not stale:
        print(f"All {len(tickers)} tickers fresh (within {METADATA_EXPIRY_DAYS} days). Skipping fetch.")
        return registry

    print(f"Fetching metadata for {len(stale)}/{len(tickers)} stale tickers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one_ticker, t): t for t in stale}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(stale), desc="Fetching metadata"):
            ticker = futures[future]
            result = future.result()
            if result:
                registry[ticker] = result

    save_metadata_registry(registry)
    print(f"Registry saved with {len(registry)} tickers.")
    return registry


def ensure_symbols_in_registry(symbols: List[str]) -> Dict[str, Any]:
    """Ensure symbols are present in metadata registry, fetching missing ones.
    This is the single authoritative function for ensuring symbols exist in registry.
    Args:
        symbols: List of ticker symbols to ensure are in registry
    Returns:
        Updated registry dict with all symbols present
    Side Effects:
        - Makes network API calls via yfinance for missing symbols
        - Writes metadata_registry.tsv to disk if new symbols are fetched
    """
    registry = load_metadata_registry()
    
    # Find symbols not yet in registry
    missing = [s for s in symbols if s not in registry]
    if missing:
        print(f"Fetching metadata for {len(missing)} symbols not in registry: {missing}")
        registry = fetch_and_update_registry(missing)
    
    return registry


if __name__ == "__main__":
    print("* Metadata I/O utilities loaded successfully")
    loaded_registry = load_metadata_registry()
    print(f"* Loaded registry with {len(loaded_registry)} tickers")
