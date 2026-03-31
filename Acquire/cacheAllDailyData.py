import concurrent.futures
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm
import yfinance as yf

from config import DAILY_DATA_CACHE_DIR
from utils.io_utils import read_tsv

from Acquire.DailyData import daily_data
from Acquire.latest_close import latest_mkt_close


def cache_daily_data_list_parallel(tickers: List[str]) -> None:
    """Download and cache daily data for a list of tickers in parallel using up to 15 threads."""
    # Use ThreadPoolExecutor to run cache_max_daily_data in parallel with a maximum of 15 threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        # Wrap the executor.map with tqdm to show a progress bar
        list(tqdm(executor.map(lambda ticker: daily_data(ticker, period='max'), tickers), total=len(tickers), desc="",))

def download_and_cache(symbols: List[str], cache_dir: str, kwargs: Dict) -> None:
    """Download data for symbols and save per-symbol Feather files; skips already-cached symbols.
    Side Effects:
        - Makes network requests via yfinance
        - Writes .feather files to cache_dir
        - Prints progress messages to stdout
    """
    # make list of cache files that have modified timestamp later than latest market close
    latest_close = latest_mkt_close() # latest market close in local time
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.feather')]
    # Compare seconds-since-epoch (os.path.getmtime) to latest_close in epoch seconds
    cutoff_epoch = latest_close.timestamp() if latest_close is not None else float('inf')
    cache_files = [
        f for f in cache_files
        if Path(cache_dir, f).stat().st_mtime > cutoff_epoch
    ]
    skip_symbols = [f.split('.')[0] for f in cache_files]
    print(f"skipping {len(skip_symbols)} symbols")  #: {skip_symbols}")
    symbols = [s for s in symbols if s not in skip_symbols]
    print(f"downloading {len(symbols)} symbols")

    batch_size=20
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        print(f"Downloading {kwargs.get('period', 'unknown period')} batch: {batch}")
        # Use a fresh copy of kwargs per batch and ensure 'end' is not set so latest market close is used
        params = {**kwargs}
        params.pop('end', None)
        df = yf.download(batch, **params, group_by='ticker')
        if df is None or df.empty:
            print(f"\nNo data returned for batch: {batch}")
            continue
        df = df.round(5)
        # Handle both MultiIndex (multi-ticker) and single-ticker columns
        if isinstance(df.columns, pd.MultiIndex):
            tickers = list(df.columns.levels[0])
            for symbol in tickers:
                file = Path(cache_dir) / f"{symbol}.feather"
                print("Saving to " + file + " ... ", end="")
                df[symbol].dropna().reset_index().to_feather(file) # feather has fast reads and writes, see feather_benchmark
                print("")
        else:
            # Single ticker response
            # Try to infer symbol from the batch when length is 1; otherwise skip to avoid mislabeling
            if len(batch) == 1:
                symbol = batch[0]
                file = Path(cache_dir) / f"{symbol}.feather"
                print("Saving to " + file + " ... ", end="")
                df.dropna().reset_index().to_feather(file, index=False)
                print("")
            else:
                print("Unexpected non-MultiIndex columns for multi-ticker batch; skipping save to avoid mistakes.")

def cache_daily_data_listfile(tsv_file: str) -> None:
    """Read tickers from a TSV file and download/cache their daily market data."""
    # Read tickers from the tsv file, skipping header row
    df = read_tsv(tsv_file, header=None, skiprows=1)
    tickers = df[0].tolist()
    tickers = [s.replace('.', '-') for s in tickers]
    # cache_daily_data_list_parallel(tickers)
    download_and_cache(tickers, str(DAILY_DATA_CACHE_DIR), {'period': '13mo'}) # PERIOD: 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max

if __name__ == "__main__":
    starttime = pd.Timestamp.now()
    cache_daily_data_listfile('tickerLists/bigStocks.tsv')
    # cache_daily_data_listfile('tickerLists/bigStocks500.tsv')
    cache_daily_data_listfile('tickerLists/bigETFs.tsv')
    # cache_daily_data_listfile('tickerLists/500ETFsPE.tsv')
    cache_daily_data_listfile('tickerLists/sp-500-stocks.tsv')
    cache_daily_data_listfile('tickerLists/watchTickers.tsv')
    cache_daily_data_listfile('tickerLists/myTickers.tsv')
    print(f"total time: {pd.Timestamp.now() - starttime}")
