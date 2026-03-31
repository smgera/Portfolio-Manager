#!/usr/bin/env python3
"""
Fetch ETF data and enrich with yfinance metrics

This script:
1. Fetches ETF symbols from companiesmarketcap.com (already sorted by market cap)
2. Fetches financial metrics from yfinance for each symbol
3. Saves top 500 by MarketCap to TSV with specified columns

Source: https://companiesmarketcap.com/etfs/largest-etfs-by-marketcap/
Output: tickerLists/500ETFs.tsv (top 500 by MarketCap)
Columns: ['Symbol', 'Name', 'Class', 'Sector', 'Industry', 'MarketCap', 'TrailingPE', 'ForwardPE', 'Beta', 'AvgVolume', 'Region', 'Fee']
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.runtime_version')

import pandas as pd
import yfinance as yf
import os
from datetime import datetime
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed


def fetch_etf_list() -> list:
    """Fetch list of ETF symbols from companiesmarketcap.com (already sorted by market cap)."""
    print("Fetching ETF list from companiesmarketcap.com...")
    
    try:
        import requests
        from bs4 import BeautifulSoup
        
        # Use companiesmarketcap.com's largest ETFs page
        url = 'https://companiesmarketcap.com/etfs/largest-etfs-by-marketcap/'
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        print(f"Fetching from: {url}")
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all ticker symbols on the page
        all_symbols = []
        
        # Look for ticker symbols in the table
        # The page typically has a table with company/ETF data
        # Try to find all rows in tables
        for row in soup.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) > 0:
                # Look for ticker symbol - usually in a specific cell or link
                for cell in cells:
                    # Check for links that might contain ticker symbols
                    links = cell.find_all('a')
                    for link in links:
                        text = link.get_text(strip=True)
                        # Ticker symbols are usually 2-5 uppercase letters
                        if text and 2 <= len(text) <= 5 and text.isupper() and text.isalpha():
                            all_symbols.append(text)
                    
                    # Also check cell text directly
                    text = cell.get_text(strip=True)
                    if text and 2 <= len(text) <= 5 and text.isupper() and text.isalpha():
                        # Avoid common non-ticker words
                        if text not in ['ETF', 'USD', 'USA', 'US', 'UK', 'EU', 'JP', 'CN']:
                            all_symbols.append(text)
        
        # Remove duplicates while preserving order (important since they're sorted by market cap)
        seen = set()
        ordered_symbols = []
        for symbol in all_symbols:
            if symbol not in seen:
                seen.add(symbol)
                ordered_symbols.append(symbol)
        
        print(f"\nFound {len(ordered_symbols)} unique ETF symbols from companiesmarketcap.com")
        print(f"First 10 symbols: {ordered_symbols[:10]}")
        
        if len(ordered_symbols) < 100:
            print(f"Warning: Only found {len(ordered_symbols)} symbols. Trying alternative method...")
            return fetch_etf_list_fallback()
        
        return ordered_symbols
        
    except Exception as e:
        print(f"Error fetching ETF list from companiesmarketcap.com: {e}")
        print("Trying fallback method...")
        return fetch_etf_list_fallback()


def fetch_etf_list_fallback() -> list:
    """Fallback method: Use a comprehensive list of known ETF ticker ranges."""
    print("Using fallback method: scanning ticker ranges...")
    
    # Generate potential ETF tickers by common patterns
    # Most ETFs are 2-4 letters
    import string
    
    potential_symbols = []
    
    # Add common ETF prefixes
    common_prefixes = ['V', 'I', 'S', 'Q', 'X', 'A', 'F', 'E', 'D', 'B', 'G', 'P', 'R', 'T', 'U', 'M', 'N', 'K', 'H', 'L', 'C', 'J', 'O', 'W', 'Y', 'Z']
    
    # Generate 3-letter combinations with common prefixes
    for prefix in common_prefixes:
        for second in string.ascii_uppercase:
            for third in string.ascii_uppercase:
                potential_symbols.append(f"{prefix}{second}{third}")
    
    print(f"Generated {len(potential_symbols)} potential ticker symbols")
    print("This will take a while to validate. Limiting to first 1000 for testing...")
    
    # Limit to reasonable number for testing
    potential_symbols = potential_symbols[:1000]
    
    # Validate which ones are actually ETFs using yfinance
    print("Validating symbols with yfinance (this may take several minutes)...")
    
    valid_etfs = []
    for i, symbol in enumerate(potential_symbols):
        if i % 100 == 0:
            print(f"  Validated {i}/{len(potential_symbols)} symbols, found {len(valid_etfs)} ETFs so far...")
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if info.get('quoteType') == 'ETF':
                valid_etfs.append(symbol)
        except:
            pass
    
    print(f"\nFound {len(valid_etfs)} valid ETFs")
    return valid_etfs


def get_ticker_info(symbol: str) -> Dict[str, Any]:
    """Fetch financial metrics for a single ticker."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Determine asset class
        quote_type = info.get('quoteType', '')
        category = info.get('category', '')
        
        # Map to Class
        if quote_type == 'ETF':
            if 'bond' in str(category).lower() or 'fixed' in str(category).lower():
                asset_class = 'Fixed Income'
            elif 'commodity' in str(category).lower() or 'gold' in str(category).lower() or 'silver' in str(category).lower():
                asset_class = 'Commodity'
            elif 'currency' in str(category).lower() or 'bitcoin' in str(category).lower() or 'crypto' in str(category).lower():
                asset_class = 'Currency'
            elif 'allocation' in str(category).lower():
                asset_class = 'Asset Allocation'
            else:
                asset_class = 'Equity'
        else:
            asset_class = 'Equity'
        
        return {
            'Symbol': symbol,
            'Name': info.get('longName', info.get('shortName', '')),
            'Class': asset_class,
            'Sector': info.get('sector', ''),
            'Industry': info.get('industry', ''),
            'MarketCap': info.get('marketCap', info.get('totalAssets', None)),
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
            'Class': '',
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
                    'Class': '',
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


def save_to_tsv(df: pd.DataFrame, output_file: str, top_n: int = 500) -> None:
    """Save dataframe to TSV with specified columns."""
    
    if df.empty:
        print("No data to save")
        return
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Ensure columns are in the specified order
    columns = ['Symbol', 'Name', 'Class', 'Sector', 'Industry', 'MarketCap', 'TrailingPE', 'ForwardPE', 'Beta', 'AvgVolume']
    df = df[columns]
    
    # Sort by MarketCap descending (largest first)
    df = df.sort_values('MarketCap', ascending=False, na_position='last')
    
    # Keep only top N ETFs by MarketCap
    original_count = len(df)
    df = df.head(top_n)
    print(f"\nFiltered from {original_count} to top {len(df)} ETFs by MarketCap")
    
    # Save to TSV
    df.to_csv(output_file, sep='\t', index=False)
    print(f"\nSaved {len(df)} symbols to: {output_file}")
    
    # Show summary
    print(f"\nSummary:")
    print(f"  Total symbols: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    # Class breakdown
    class_counts = df['Class'].value_counts()
    print(f"\n  Asset Class Breakdown:")
    for asset_class, count in class_counts.items():
        print(f"    {asset_class}: {count}")
    
    # Show first few rows
    print(f"\nFirst 5 rows:")
    print(df.head(5).to_string())


def main():
    output_file = 'tickerLists/500ETFs.tsv'
    
    # Step 1: Fetch ETF symbols
    symbols = fetch_etf_list()
    
    if not symbols:
        print("\nFailed to fetch ETF symbols")
        return
    
    print(f"\nFound {len(symbols)} ETF symbols to process")
    
    # Step 2: Enrich with yfinance data
    enriched_df = enrich_with_yfinance(symbols)
    
    # Step 3: Save to TSV (top 500 by MarketCap)
    save_to_tsv(enriched_df, output_file, top_n=500)
    
    print(f"\nDone! Top 500 largest ETFs saved to {output_file}")


if __name__ == "__main__":
    main()
