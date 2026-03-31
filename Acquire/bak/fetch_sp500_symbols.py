#!/usr/bin/env python3
"""
Fetch current S&P 500 symbols from Wikipedia and enrich with yfinance data

This script:
1. Scrapes S&P 500 symbols from Wikipedia
2. Fetches financial metrics from yfinance for each symbol
3. Saves to TSV with specified columns

Source: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies

Output: tickerLists/sp500.tsv
Columns: ['Symbol', 'Name', 'Class''Sector', 'Industry', 'MarketCap', 'TrailingPE', 'ForwardPE', 'Beta', 'AvgVolume']
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.runtime_version')

import pandas as pd
import yfinance as yf
import os
from datetime import datetime
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed


def fetch_sp500_symbols() -> pd.DataFrame:
    """Fetch S&P 500 symbols from Wikipedia."""
    
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    print(f"Fetching S&P 500 data from Wikipedia...")
    print(f"URL: {url}")
    
    try:
        # Add User-Agent header to avoid 403 Forbidden error
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Read all tables from the page
        tables = pd.read_html(url, header=0, storage_options=headers)
        
        # The first table contains the current S&P 500 components
        df = tables[0]
        
        print(f"Successfully fetched {len(df)} S&P 500 symbols")
        
        # Standardize column names (Wikipedia may change them)
        # Common columns: Symbol, Security, GICS Sector, GICS Sub-Industry, etc.
        print(f"\nColumns found: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()


def get_ticker_info(symbol: str) -> Dict[str, Any]:
    """Fetch financial metrics for a single ticker."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        return {
            'Symbol': symbol,
            'Name': info.get('longName', info.get('shortName', '')),
            'Sector': info.get('sector', ''),
            'Industry': info.get('industry', ''),
            'MarketCap': info.get('marketCap', None),
            'TrailingPE': info.get('trailingPE', None),
            'ForwardPE': info.get('forwardPE', None),
            'Beta': info.get('beta', None),
            'AvgVolume': info.get('averageVolume', None),
        }
    except Exception as e:
        print(f"  Error fetching {symbol}: {e}")
        return {
            'Symbol': symbol,
            'Name': '',
            'Sector': '',
            'Industry': '',
            'MarketCap': None,
            'TrailingPE': None,
            'ForwardPE': None,
            'Beta': None,
            'AvgVolume': None,
        }


def enrich_with_yfinance(symbols: list, max_workers: int = 10) -> pd.DataFrame:
    """Fetch financial data for all symbols using yfinance with parallel processing."""
    print(f"\nFetching financial data for {len(symbols)} symbols from yfinance...")
    print(f"Using {max_workers} parallel workers...")
    
    data = []
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_symbol = {executor.submit(get_ticker_info, symbol): symbol for symbol in symbols}
        
        # Process completed tasks
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                ticker_data = future.result()
                data.append(ticker_data)
                completed += 1
                
                if completed % 50 == 0:
                    print(f"  Progress: {completed}/{len(symbols)} ({completed/len(symbols)*100:.1f}%)")
            except Exception as e:
                print(f"  Error processing {symbol}: {e}")
                # Add empty data for failed symbol
                data.append({
                    'Symbol': symbol,
                    'Name': '',
                    'Sector': '',
                    'Industry': '',
                    'MarketCap': None,
                    'TrailingPE': None,
                    'ForwardPE': None,
                    'Beta': None,
                    'AvgVolume': None,
                })
                completed += 1
    
    df = pd.DataFrame(data)
    print(f"Successfully fetched data for {len(df)} symbols")
    
    return df


def save_to_tsv(df: pd.DataFrame, output_file: str) -> None:
    """Save dataframe to TSV with specified columns."""
    
    if df.empty:
        print("No data to save")
        return
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Ensure columns are in the specified order
    columns = ['Symbol', 'Name', 'Sector', 'Industry', 'MarketCap', 'TrailingPE', 'ForwardPE', 'Beta', 'AvgVolume']
    df = df[columns]
    
    # Sort by MarketCap descending (largest first)
    df = df.sort_values('MarketCap', ascending=False, na_position='last')
    
    # Save to TSV
    df.to_csv(output_file, sep='\t', index=False)
    print(f"\nSaved {len(df)} symbols to: {output_file}")
    
    # Show summary
    print(f"\nSummary:")
    print(f"  Total symbols: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    # Sector breakdown
    sector_counts = df['Sector'].value_counts()
    print(f"\n  Top 5 Sectors:")
    for sector, count in sector_counts.head(5).items():
        print(f"    {sector}: {count}")
    
    # Show first few rows
    print(f"\nFirst 5 rows:")
    print(df.head(5).to_string())


def main():
    output_file = 'tickerLists/sp500.tsv'
    
    # Step 1: Fetch symbols from Wikipedia
    wiki_df = fetch_sp500_symbols()
    
    if wiki_df.empty:
        print("\nFailed to fetch S&P 500 symbols from Wikipedia")
        return
    
    # Extract symbols
    if 'Symbol' in wiki_df.columns:
        symbols = wiki_df['Symbol'].str.strip().tolist()
    else:
        print("Error: 'Symbol' column not found in Wikipedia data")
        return
    
    # Step 2: Enrich with yfinance data
    enriched_df = enrich_with_yfinance(symbols)
    
    # Step 3: Save to TSV
    save_to_tsv(enriched_df, output_file)
    
    print(f"\nDone! S&P 500 data saved to {output_file}")


if __name__ == "__main__":
    main()
