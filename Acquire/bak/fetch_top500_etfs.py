#!/usr/bin/env python3
"""
Fetch top 500 ETFs by market cap and enrich with yfinance data

This script:
1. Scrapes top 500 ETFs by market cap from companiesmarketcap.com
2. Fetches financial metrics from yfinance for each ETF
3. Saves to TSV with specified columns

Source: https://companiesmarketcap.com/etfs/largest-etfs-by-marketcap/

Output: tickerLists/500ETFs.tsv
Columns: ['Symbol', 'Name', 'Class', 'Sector', 'Industry', 'MarketCap', 'TrailingPE', 'ForwardPE', 'Beta', 'AvgVolume', 'Region', 'Fee']
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.runtime_version')

import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import os
import time
import re
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed


def scrape_etf_symbols(max_etfs: int = 500) -> List[Dict[str, str]]:
    """Scrape ETF symbols and basic info from companiesmarketcap.com."""
    
    base_url = 'https://companiesmarketcap.com/etfs/largest-etfs-by-marketcap/'
    
    print(f"Scraping ETF data from companiesmarketcap.com...")
    print(f"URL: {base_url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    etf_data = []
    page = 1
    
    while len(etf_data) < max_etfs:
        try:
            # Construct URL with page parameter
            if page == 1:
                url = base_url
            else:
                url = f"{base_url}?page={page}"
            
            print(f"  Fetching page {page}...")
            
            # Try using pandas read_html which can handle JavaScript-rendered tables
            try:
                tables = pd.read_html(url, storage_options={'User-Agent': headers['User-Agent']})
                
                if not tables:
                    print(f"  No tables found on page {page}")
                    break
                
                # The first table should contain ETF data
                df = tables[0]
                print(f"  Found table with {len(df)} rows and columns: {list(df.columns)}")
                
                # Parse the table - columns are typically: Rank, Name, Market Cap, etc.
                for idx, row in df.iterrows():
                    if len(etf_data) >= max_etfs:
                        break
                    
                    # Try to extract symbol and name from the Name column
                    # Format is typically "NameSymbol" (no space)
                    if 'Name' in df.columns:
                        name_text = str(row['Name'])
                    elif len(df.columns) > 1:
                        name_text = str(row[df.columns[1]])
                    else:
                        continue
                    
                    # Skip empty or invalid entries
                    if not name_text or name_text == 'nan':
                        continue
                    
                    # Parse symbol from the end using same logic as BeautifulSoup
                    # Try to match symbol with country code (e.g., VUSA.DE)
                    match = re.search(r'^(.+?)([A-Z]{1,5}\.[A-Z]{2,3})$', name_text)
                    if match:
                        name = match.group(1).strip()
                        symbol = match.group(2)
                    else:
                        # Try to match regular symbol (e.g., IAU, SPY)
                        match = re.search(r'^(.+?)([A-Z]{1,5})$', name_text)
                        if match:
                            name = match.group(1).strip()
                            symbol = match.group(2)
                        else:
                            # If no clear symbol, skip this entry
                            continue
                    
                    # Skip if symbol looks invalid
                    if len(symbol.split('.')[0]) < 1 or len(symbol.split('.')[0]) > 5:
                        continue
                    
                    etf_data.append({
                        'Symbol': symbol,
                        'Name': name,
                        'MarketCapText': ''
                    })
                
                print(f"  Found {len(etf_data)} ETFs so far")
                
            except Exception as e:
                print(f"  pandas.read_html failed: {e}")
                print(f"  Trying BeautifulSoup approach...")
                
                # Fallback to BeautifulSoup
                response = requests.get(url, headers=headers)
                
                if response.status_code != 200:
                    print(f"  Error: HTTP {response.status_code}")
                    break
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for all rows in the ETF table
                # Structure: <div class="company-name">Name</div> and <div class="company-code">SYMBOL</div>
                
                # Find all company name divs
                name_divs = soup.find_all('div', class_='company-name')
                
                for name_div in name_divs:
                    if len(etf_data) >= max_etfs:
                        break
                    
                    # Get the name
                    name = name_div.get_text(strip=True)
                    if not name:
                        continue
                    
                    # Find the symbol - it should be in a sibling or nearby element
                    # Try to find company-code div in the same parent
                    parent = name_div.parent
                    if parent:
                        symbol_div = parent.find('div', class_='company-code')
                        if symbol_div:
                            symbol = symbol_div.get_text(strip=True)
                        else:
                            # Try to find it as next sibling
                            symbol_div = name_div.find_next_sibling('div', class_='company-code')
                            if symbol_div:
                                symbol = symbol_div.get_text(strip=True)
                            else:
                                # Look for any text after the name in the parent
                                full_text = parent.get_text(strip=True)
                                symbol = full_text.replace(name, '').strip()
                    else:
                        continue
                    
                    # Clean up symbol (remove quotes, whitespace)
                    symbol = symbol.strip().strip('"').strip("'").strip()
                    
                    # Validate symbol
                    if not symbol or len(symbol) > 10:
                        continue
                    
                    # Skip if symbol looks invalid
                    if not re.match(r'^[A-Z]{1,5}(\.[A-Z]{2,3})?$', symbol):
                        continue
                    
                    etf_data.append({
                        'Symbol': symbol,
                        'Name': name,
                        'MarketCapText': ''
                    })
                
                print(f"  Found {len(etf_data)} ETFs so far")
            
            # Check if we should continue to next page
            if len(etf_data) >= max_etfs:
                break
            
            # Check if there's a next page
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            next_link = soup.find('a', string=lambda x: x and 'Next' in x)
            
            if not next_link:
                print(f"  No more pages available")
                break
            
            page += 1
            time.sleep(0.5)  # Be polite to the server
            
        except Exception as e:
            print(f"  Error scraping page {page}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print(f"\nSuccessfully scraped {len(etf_data)} ETF symbols")
    return etf_data


def get_etf_info(symbol: str) -> Dict[str, Any]:
    """Fetch financial metrics for a single ETF using yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Determine asset class
        quote_type = info.get('quoteType', '')
        if quote_type == 'ETF':
            asset_class = 'Equity'  # Most ETFs are equity-based
        else:
            asset_class = quote_type
        
        # Get category/sector info
        category = info.get('category', '')
        sector = info.get('sector', '')
        industry = info.get('industry', '')
        
        # Get region
        region = info.get('region', '')
        if not region:
            # Try to infer from name or other fields
            fund_family = info.get('fundFamily', '')
            if 'international' in category.lower() or 'world' in category.lower():
                region = 'Global'
            elif 'us' in category.lower() or 'america' in category.lower():
                region = 'US'
            else:
                region = ''
        
        # Get expense ratio (fee)
        expense_ratio = info.get('annualReportExpenseRatio', info.get('expenseRatio', None))
        
        return {
            'Symbol': symbol,
            'Name': info.get('longName', info.get('shortName', '')),
            'Class': asset_class,
            'Sector': sector,
            'Industry': industry,
            'MarketCap': info.get('totalAssets', info.get('marketCap', None)),
            'TrailingPE': info.get('trailingPE', None),
            'ForwardPE': info.get('forwardPE', None),
            'Beta': info.get('beta', None),
            'AvgVolume': info.get('averageVolume', info.get('averageVolume10days', None)),
            'Region': region,
            'Fee': expense_ratio,
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
            'Region': '',
            'Fee': None,
        }


def is_valid_symbol(symbol: str) -> bool:
    """Check if a symbol looks valid for yfinance."""
    # Filter out symbols with common invalid patterns
    invalid_patterns = [
        r'^Acc',           # Starts with Acc
        r'^Dist',          # Starts with Dist
        r'^\(',            # Starts with parenthesis
        r'\(',             # Contains parenthesis
        r'Accumulation',   # Contains full word
        r'Distribution',   # Contains full word
        r'Trust[A-Z]',     # Trust followed by uppercase (e.g., TrustOUNZ)
        r'^ETF[A-Z]',      # Starts with ETF followed by letter
        r'^ETNs',          # Starts with ETNs
        r'A-dis',          # Contains A-dis
        r'^[A-Z]-',        # Single letter followed by dash
    ]
    
    for pattern in invalid_patterns:
        if re.search(pattern, symbol):
            return False
    
    # Valid symbols are typically 1-5 uppercase letters, optionally with .XX suffix
    if re.match(r'^[A-Z]{1,5}(\.[A-Z]{2,3})?$', symbol):
        return True
    
    return False


def enrich_with_yfinance(etf_list: List[Dict[str, str]], max_workers: int = 10) -> pd.DataFrame:
    """Fetch financial data for all ETFs using yfinance with parallel processing."""
    print(f"\nFetching financial data for {len(etf_list)} ETFs from yfinance...")
    
    # Filter out invalid symbols
    symbols = [etf['Symbol'] for etf in etf_list]
    valid_symbols = [s for s in symbols if is_valid_symbol(s)]
    invalid_symbols = [s for s in symbols if not is_valid_symbol(s)]
    
    if invalid_symbols:
        print(f"Filtered out {len(invalid_symbols)} invalid symbols: {invalid_symbols[:10]}{'...' if len(invalid_symbols) > 10 else ''}")
    
    print(f"Processing {len(valid_symbols)} valid symbols with {max_workers} parallel workers...")
    
    data = []
    completed = 0
    
    symbols = valid_symbols
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_symbol = {executor.submit(get_etf_info, symbol): symbol for symbol in symbols}
        
        # Process completed tasks
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                etf_data = future.result()
                data.append(etf_data)
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
                    'Region': '',
                    'Fee': None,
                })
                completed += 1
    
    df = pd.DataFrame(data)
    print(f"Successfully fetched data for {len(df)} ETFs")
    
    return df


def save_to_tsv(df: pd.DataFrame, output_file: str) -> None:
    """Save dataframe to TSV with specified columns."""
    
    if df.empty:
        print("No data to save")
        return
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Ensure columns are in the specified order
    columns = ['Symbol', 'Name', 'Class', 'Sector', 'Industry', 'MarketCap', 
               'TrailingPE', 'ForwardPE', 'Beta', 'AvgVolume', 'Region', 'Fee']
    df = df[columns]
    
    # Sort by MarketCap descending (largest first)
    df = df.sort_values('MarketCap', ascending=False, na_position='last')
    
    # Save to TSV
    df.to_csv(output_file, sep='\t', index=False)
    print(f"\nSaved {len(df)} ETFs to: {output_file}")
    
    # Show summary
    print(f"\nSummary:")
    print(f"  Total ETFs: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    # Class breakdown
    if 'Class' in df.columns:
        class_counts = df['Class'].value_counts()
        print(f"\n  Asset Classes:")
        for asset_class, count in class_counts.items():
            print(f"    {asset_class}: {count}")
    
    # Region breakdown
    if 'Region' in df.columns:
        region_counts = df['Region'].value_counts()
        print(f"\n  Top 5 Regions:")
        for region, count in region_counts.head(5).items():
            print(f"    {region}: {count}")
    
    # Show first few rows
    print(f"\nFirst 5 rows:")
    print(df.head(5).to_string())


def main():
    output_file = 'tickerLists/500ETFs.tsv'
    
    # Step 1: Scrape ETF symbols from companiesmarketcap.com
    etf_list = scrape_etf_symbols(max_etfs=500)
    
    if not etf_list:
        print("\nFailed to scrape ETF symbols from companiesmarketcap.com")
        return
    
    # Step 2: Enrich with yfinance data
    enriched_df = enrich_with_yfinance(etf_list)
    
    # Step 3: Save to TSV
    save_to_tsv(enriched_df, output_file)
    
    print(f"\nDone! Top 500 ETFs data saved to {output_file}")


if __name__ == "__main__":
    main()
