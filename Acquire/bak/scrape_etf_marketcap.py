"""
Scrape and print the largest ETFs by market cap from companiesmarketcap.com
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd


def scrape_etf_page(url, headers):
    """
    Scrape a single page of ETF data
    
    Args:
        url: URL to scrape
        headers: HTTP headers for the request
        
    Returns:
        List of ETF data dictionaries
    """
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    # Parse HTML
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the table - typically in a tbody element
    table = soup.find('tbody')
    
    if not table:
        print(f"Could not find table on page: {url}")
        return []
    
    # Extract data from table rows
    etf_data = []
    
    for row in table.find_all('tr'):
        cells = row.find_all('td')
        if len(cells) >= 4:  # Ensure we have enough columns (fav, rank, name, marketcap)
            # Cell 0 is favorite icon (skip)
            # Cell 1 is rank (skip)
            
            # Cell 2 contains name and symbol in separate divs
            name_cell = cells[2]
            name_div = name_cell.find('div', class_='company-name')
            symbol_div = name_cell.find('div', class_='company-code')
            
            name = name_div.get_text(strip=True) if name_div else ''
            symbol = symbol_div.get_text(strip=True) if symbol_div else ''
            
            # Cell 3 is market cap
            marketcap = cells[3].get_text(strip=True)
            
            # Extract additional columns if they exist
            price = cells[4].get_text(strip=True) if len(cells) > 4 else ''
            # Cell 5 is change (skip)
            # Cell 6 is sparkline chart (skip)
            fee = cells[7].get_text(strip=True) if len(cells) > 7 else ''
            market = cells[8].get_text(strip=True) if len(cells) > 8 else ''
            
            etf_data.append({
                'Name': name,
                'Symbol': symbol,
                'Market Cap': marketcap,
                'Price': price,
                'Fee': fee,
                'Market': market
            })
    
    return etf_data


def scrape_etf_table(base_url="https://companiesmarketcap.com/etfs/largest-etfs-by-marketcap/", num_pages=15):
    """
    Scrape ETF data from companiesmarketcap.com across multiple pages
    
    Args:
        base_url: Base URL to scrape (default: largest ETFs page)
        num_pages: Number of pages to scrape (default: 15 for ~1500 ETFs)
        
    Returns:
        DataFrame with ETF data from all pages
    """
    # Set headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    all_etf_data = []
    
    for page in range(1, num_pages + 1):
        url = f"{base_url}?page={page}"
        print(f"Fetching page {page}/{num_pages} from {url}...")
        
        page_data = scrape_etf_page(url, headers)
        all_etf_data.extend(page_data)
        
        print(f"  Scraped {len(page_data)} ETFs from page {page}. Total so far: {len(all_etf_data)}")
    
    # Create DataFrame
    df = pd.DataFrame(all_etf_data)
    return df

def print_etf_table(df, max_rows=None):
    """
    Print ETF data in a formatted table
    
    Args:
        df: DataFrame with ETF data
        max_rows: Maximum number of rows to display (None for all)
    """
    if df is None or df.empty:
        print("No data to display")
        return
    
    # Set pandas display options for better formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    if max_rows:
        print(f"\nTop {max_rows} ETFs by Market Cap:")
        print("=" * 100)
        print(df.head(max_rows).to_string(index=False))
    else:
        print(f"\nAll {len(df)} ETFs:")
        print("=" * 100)
        print(df.to_string(index=False))
    
    print("=" * 100)


def convert_marketcap_to_billions(marketcap_str):
    """
    Convert market cap from trillions or millions to billions for consistency
    
    Args:
        marketcap_str: Market cap string (e.g., "$1.5 T", "$500 B", "$750 M")
        
    Returns:
        Market cap string in billions
    """
    if 'T' in marketcap_str:
        # Extract numeric value
        value_str = marketcap_str.replace('$', '').replace('T', '').strip()
        try:
            value = float(value_str)
            # Convert trillions to billions (multiply by 1000)
            billions = value * 1000
            return f"${billions:.2f} B"
        except ValueError:
            return marketcap_str
    elif 'M' in marketcap_str:
        # Extract numeric value
        value_str = marketcap_str.replace('$', '').replace('M', '').strip()
        try:
            value = float(value_str)
            # Convert millions to billions (divide by 1000)
            billions = value / 1000
            return f"${billions:.3f} B"
        except ValueError:
            return marketcap_str
    return marketcap_str


def save_to_tsv(df, filename):
    """
    Save ETF data to TSV file
    
    Args:
        df: DataFrame with ETF data
        filename: Output filename
    """
    if df is None or df.empty:
        print("No data to save")
        return
    
    df.to_csv(filename, sep='\t', index=False)
    print(f"\nData saved to {filename}")


if __name__ == "__main__":
    import os
    
    # Check if output file already exists
    filename = 'tickerLists/scraped_etfs_marketcap.tsv'
    if os.path.exists(filename):
        print(f"⚠️  File already exists: {filename}")
        response = input("Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            print("❌ Scraping cancelled by user")
            exit(0)
        print("✅ Will overwrite existing file\n")
    
    # Scrape the data
    etf_df = scrape_etf_table()
    
    if etf_df is not None and not etf_df.empty:
        # Convert trillions and millions to billions in Market Cap column
        print("\nConverting market cap values (T -> B, M -> B)...")
        etf_df['Market Cap'] = etf_df['Market Cap'].apply(convert_marketcap_to_billions)
        
        # Print top 50 ETFs
        print_etf_table(etf_df, max_rows=50)
        
        # Print summary stats
        print(f"\nTotal ETFs scraped: {len(etf_df)}")
        
        # Save to TSV
        filename = 'tickerLists/scraped_etfs_marketcap.tsv'
        save_to_tsv(etf_df, filename)
        
        # Read the TSV back into a dataframe
        print(f"\nReading back from {filename}...")
        df_from_tsv = pd.read_csv(filename, sep='\t')
        print(f"Loaded {len(df_from_tsv)} ETFs from TSV file")
        print(f"Columns: {list(df_from_tsv.columns)}")
    else:
        print("Failed to scrape ETF data")
