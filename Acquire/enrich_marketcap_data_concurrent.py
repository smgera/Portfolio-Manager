"""Enrich marketcap TSV files with additional financial data (Concurrent version).
Processes 2 symbols concurrently (default) to avoid rate limiting.
Adds columns: Exchange, Sector, Industry, TrailingPE, ForwardPE, Beta, AvgVolume,
              PEGRatio, ProfitMargin, DebtToEquity, ROE, InceptionDate
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
import yfinance as yf
from utils.io_utils import read_tsv


def ensure_scraped_tsv(tsv_file: Union[str, Path]) -> bool:
    """Check if TSV file exists; if not, run the appropriate scraper to create it.
    Args:
        tsv_file: Path to the TSV file
    Returns:
        Boolean indicating if file exists (or was created successfully)
    Side Effects:
        - Runs scraper subprocess if TSV file is missing
        - Prints status messages to stdout
    """
    if os.path.exists(tsv_file):
        print(f"[OK] TSV file exists: {tsv_file}")
        return True
    
    print(f"⚠️  TSV file not found: {tsv_file}")
    
    # Determine which scraper to run based on filename
    if 'etfs' in tsv_file.lower():
        scraper = 'scrape_etf_marketcap.py'
        data_type = 'ETF'
    elif 'stocks' in tsv_file.lower():
        scraper = 'scrape_stock_marketcap.py'
        data_type = 'Stock'
    else:
        print(f"[FAIL] Cannot determine scraper for: {tsv_file}")
        return False
    
    print(f"📥 Running {scraper} to create {data_type} data...")
    
    try:
        # Run the scraper script
        result = subprocess.run(['py', scraper], 
                              capture_output=True, 
                              text=True, 
                              timeout=600)  # 10 minute timeout
        
        if result.returncode == 0:
            print(f"[OK] {data_type} data scraped successfully!")
            return os.path.exists(tsv_file)
        else:
            print(f"[FAIL] Scraper failed with return code {result.returncode}")
            print(f"Error: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"[FAIL] Scraper timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"[FAIL] Error running scraper: {e}")
        return False


def get_ticker_info(symbol: str) -> Tuple[str, Dict[str, Any]]:
    """Fetch additional financial info for a ticker symbol via yfinance.
    Args:
        symbol: Ticker symbol
    Returns:
        Tuple of (symbol, dictionary with financial data)
    Side Effects:
        - Makes network request via yfinance
        - Prints error messages to stdout on failure
    """
    try:
        # Add delay to avoid rate limiting
        # sleep(0.2)  # 200ms delay between requests
        
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Check if we got valid data (empty info dict means symbol not found)
        if not info or len(info) < 1:
            # Return empty data for invalid symbols
            return (symbol, {
                'Symbol': symbol,
                'Exchange': '',
                'Sector': 'Empty info',
                'Industry': '',
                'TrailingPE': '',
                'ForwardPE': '',
                'Beta': '',
                'AvgVolume': '',
                'PEGRatio': '',
                'ProfitMargin': '',
                'DebtToEquity': '',
                'ROE': '',
                'InceptionDate': ''
            })
        
        exchange = info.get('exchange', '')
        # check if exchange is US
        # if exchange not in ['NMS', 'NYQ', 'PCX', 'NGM', 'ASE', 'BTS', 'BATS', 'ARCA', 'AMEX', 'NASDAQ', 'NYSE']:
        #     return (symbol, {
        #         'Symbol': symbol,
        #         'Exchange': exchange,
        #         'Sector': 'Non-US',
        #         'Industry': '',
        #         'TrailingPE': '',
        #         'ForwardPE': '',
        #         'Beta': '',
        #         'AvgVolume': '',
        #         'PEGRatio': '',
        #         'ProfitMargin': '',
        #         'DebtToEquity': '',
        #         'ROE': '',
        #         'InceptionDate': ''
        #     })
        
        avgVolume = info.get('averageVolume', '')
        # # check if avgVolume > 100000
        # if avgVolume < 100000:
        #     return (symbol, {
        #         'Symbol': symbol,
        #         'Exchange': exchange,
        #         'Sector': '',
        #         'Industry': '',
        #         'TrailingPE': '',
        #         'ForwardPE': '',
        #         'Beta': '',
        #         'AvgVolume': '',
        #         'PEGRatio': '',
        #         'ProfitMargin': '',
        #         'DebtToEquity': '',
        #         'ROE': '',
        #         'InceptionDate': ''
        #     })

        # ETFs use 'category' instead of 'sector' and 'industry'
        # Stocks have sector and industry, ETFs have category
        sector = info.get('sector', '')
        industry = info.get('industry', '')
        
        # For ETFs, put category in Sector only, leave Industry empty
        if not sector and not industry:
            category = info.get('category', '')
            sector = category  # e.g., "Large Blend", "Large Growth"
            industry = ''  # Leave empty for ETFs
        
        # ETFs use 'beta3Year' instead of 'beta'
        beta = info.get('beta', '')
        if not beta:
            beta = info.get('beta3Year', '')
        
        # Get inception date if available - try multiple field names
        inception_date = ''
        # Try different field names that might contain inception/start date
        date_fields = ['firstTradeDateEpochUtc', 'fundInceptionDate', 'ipoDate', 'startDate']
        for field in date_fields:
            if field in info and info[field]:
                try:
                    if isinstance(info[field], (int, float)):
                        # Unix timestamp
                        inception_date = datetime.fromtimestamp(info[field]).strftime('%Y-%m-%d')
                    elif isinstance(info[field], str):
                        # Already a string date
                        inception_date = info[field]
                    break
                except:
                    continue
        
        # Calculate PEG ratio if not available directly
        peg_ratio = info.get('pegRatio', '')
        if not peg_ratio:
            # Try to calculate: PEG = PE / Growth Rate
            # Use trailingPegRatio or calculate from earningsGrowth
            peg_ratio = info.get('trailingPegRatio', '')
            if not peg_ratio:
                trailing_pe = info.get('trailingPE', None)
                earnings_growth = info.get('earningsGrowth', None) or info.get('earningsQuarterlyGrowth', None)
                if trailing_pe and earnings_growth and earnings_growth > 0:
                    peg_ratio = round(trailing_pe / (earnings_growth * 100), 2)
        
        data = {
            'Symbol': symbol,
            'Exchange': exchange, # info.get('exchange', ''),
            'Sector': sector,
            'Industry': industry,
            'TrailingPE': info.get('trailingPE', ''),
            'ForwardPE': info.get('forwardPE', ''),
            'Beta': beta,
            'AvgVolume': avgVolume, #info.get('averageVolume', ''),
            'PEGRatio': peg_ratio,
            'ProfitMargin': info.get('profitMargins', ''),
            'DebtToEquity': info.get('debtToEquity', ''),
            'ROE': info.get('returnOnEquity', ''),
            'InceptionDate': inception_date
        }
        return (symbol, data)
    except Exception as e:
        # Log errors to understand what's failing
        error_str = str(e)
        # Log all errors, not just non-401/404
        print(f"    ⚠️  Error fetching {symbol}: {error_str[:100]}")
        # Return empty data on error, with ERROR marker in Sector field
        data = {
            'Symbol': symbol,
            'Exchange': '',
            'Sector': f"ERROR fetching {symbol}: {error_str[:100]}",
            'Industry': '',
            'TrailingPE': '',
            'ForwardPE': '',
            'Beta': '',
            'AvgVolume': '',
            'PEGRatio': '',
            'ProfitMargin': '',
            'DebtToEquity': '',
            'ROE': '',
            'InceptionDate': ''
        }
        return (symbol, data)


def enrich_tsv_with_financial_data(tsv_file: Union[str, Path], output_file: Optional[Union[str, Path]] = None, max_workers: int = 2) -> Optional[pd.DataFrame]:
    """Enrich a TSV file with additional financial data from yfinance (concurrent version).
    Args:
        tsv_file: Path to input TSV file
        output_file: Path to output TSV file (optional, will create enriched version if not provided)
        max_workers: Number of concurrent workers (default: 2, reduce if getting 401 errors)
    Returns:
        Enriched DataFrame
    Side Effects:
        - Makes concurrent network requests via yfinance for each symbol
        - Writes enriched TSV file to disk periodically and at completion
        - Prints progress messages to stdout
    """
    # Generate enriched filename if output not specified
    if output_file is None:
        # Insert 'enriched_' before the filename
        p = Path(tsv_file)
        output_file = p.parent / f"enriched_{p.name}"
    
    # Check if enriched file already exists - use it as checkpoint
    resume_mode = False
    if os.path.exists(output_file):
        print(f"📂 Found existing enriched file: {output_file}")
        response = input("Resume from this file? (y/n): ").strip().lower()
        if response == 'y':
            resume_mode = True
            print("[OK] Will resume enrichment from existing file")
        else:
            print("[FAIL] Keeping existing enriched file")
            return None
    
    # Ensure TSV file exists (run scraper if needed)
    if not ensure_scraped_tsv(tsv_file):
        print(f"[FAIL] Cannot proceed: Scraped TSV file does not exist and could not be created")
        return None
    
    # Load data - either from existing enriched file or fresh from source
    if resume_mode:
        print(f"\nReading existing enriched file: {output_file}")
        df = read_tsv(output_file)
        print(f"  Loaded {len(df)} rows with existing enrichment data")
        # filter out rows with sector containing ERROR
        df = df[df['Sector'].str.contains('ERROR', na=False) == False]
        print(f"  Processing {len(df)} rows")
    else:
        print(f"\nReading TSV from: {tsv_file}")
        df = read_tsv(tsv_file)
        print(f"  Loaded {len(df)} rows")
        print(f"  Current columns: {list(df.columns)}")

        # keep only if Symbol NOT contains . 
        df = df[~df['Symbol'].str.contains(r'\.', na=False, regex=True)]
        
        # if column 'Market' or 'Country' exists, filter out rows not US
        # if 'Market' in df.columns:
        #     df = df[df['Market'] == 'US']
        # if 'Country' in df.columns:
        #     df = df[df['Country'] == 'USA']
        print(f"{len(df)} non . sysmbol rows")
        
        # Initialize new columns
        df['Exchange'] = ''
        df['Sector'] = ''
        df['Industry'] = ''
        df['TrailingPE'] = ''
        df['ForwardPE'] = ''
        df['Beta'] = ''
        df['AvgVolume'] = ''
        df['PEGRatio'] = ''
        df['ProfitMargin'] = ''
        df['DebtToEquity'] = ''
        df['ROE'] = ''
        df['InceptionDate'] = ''
    
    # Determine which symbols need processing
    # A symbol is already processed if it has Exchange OR Sector OR AvgVolume data
    if resume_mode:
        df['_needs_processing'] = (
            (df['Exchange'].isna() | (df['Exchange'] == '')) &
            (df['Sector'].isna() | (df['Sector'] == '')) &
            (df['AvgVolume'].isna() | (df['AvgVolume'] == ''))
        )
        symbols_to_process = df[df['_needs_processing']]['Symbol'].tolist()
        already_processed = len(df) - len(symbols_to_process)
        print(f"  Already processed: {already_processed}/{len(df)} symbols")
        print(f"  Remaining to process: {len(symbols_to_process)} symbols")
    else:
        symbols_to_process = df['Symbol'].tolist()
        already_processed = 0
    
    if len(symbols_to_process) == 0:
        print(f"\n[OK] All {len(df)} symbols already processed!")
    else:
        print(f"\nFetching financial data for {len(symbols_to_process)} symbols using {max_workers} concurrent workers...")
    
    completed = already_processed
    total_symbols = len(df)
    
    # Track timing for progress updates
    start_time = time.time()
    last_batch_time = start_time
    
    # Use ThreadPoolExecutor to fetch data concurrently
    if len(symbols_to_process) > 0:
        batch_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {executor.submit(get_ticker_info, symbol): symbol for symbol in symbols_to_process}
            
            # Process completed tasks
            for future in as_completed(future_to_symbol):
                try:
                    symbol, data = future.result()
                    batch_results.append((symbol, data))
                    
                    # Update dataframe immediately
                    symbol_idx = df[df['Symbol'] == symbol].index
                    if len(symbol_idx) > 0:
                        idx = symbol_idx[0]
                        df.at[idx, 'Exchange'] = data['Exchange']
                        df.at[idx, 'Sector'] = data['Sector']
                        df.at[idx, 'Industry'] = data['Industry']
                        df.at[idx, 'TrailingPE'] = data['TrailingPE']
                        df.at[idx, 'ForwardPE'] = data['ForwardPE']
                        df.at[idx, 'Beta'] = data['Beta']
                        df.at[idx, 'AvgVolume'] = data['AvgVolume']
                        df.at[idx, 'PEGRatio'] = data['PEGRatio']
                        df.at[idx, 'ProfitMargin'] = data['ProfitMargin']
                        df.at[idx, 'DebtToEquity'] = data['DebtToEquity']
                        df.at[idx, 'ROE'] = data['ROE']
                        df.at[idx, 'InceptionDate'] = data['InceptionDate']
                    
                    completed += 1
                    
                    # Save progress to file every 50 symbols
                    if completed % 50 == 0:
                        try:
                            df.to_csv(output_file, index=False)
                            print(f"  💾 Progress saved ({completed}/{total_symbols})")
                        except Exception as e:
                            print(f"  ⚠️  Error saving progress: {e}")
                    
                    # Show progress every 100 symbols
                    if completed % 100 == 0 or completed == total_symbols:
                        current_time = time.time()
                        batch_time = current_time - last_batch_time
                        total_time = current_time - start_time
                        avg_per_symbol = total_time / (completed - already_processed) if (completed - already_processed) > 0 else 0
                        eta = avg_per_symbol * (total_symbols - completed)
                        
                        print(f"  Progress: {completed}/{total_symbols} ({completed/total_symbols*100:.1f}%) | "
                              f"Batch: {batch_time:.1f}s | Total: {total_time:.1f}s | ETA: {eta:.1f}s")
                        last_batch_time = current_time
                except Exception as e:
                    # If a future failed unexpectedly, log and continue
                    symbol = future_to_symbol.get(future, 'unknown')
                    print(f"  ⚠️  Unexpected error processing {symbol}: {str(e)[:50]}")
                    completed += 1
    
    print(f"\n[OK] Completed fetching data for all symbols")
    
    # Count successful enrichments
    print("Analyzing enrichment results...")
    enriched_count = 0
    failed_count = 0
    failed_symbols = []
    
    for idx, row in df.iterrows():
        symbol = row['Symbol']
        # Count if any data was retrieved
        if (row.get('Sector') and row['Sector'] != '') or (row.get('AvgVolume') and row['AvgVolume'] != '') or (row.get('Exchange') and row['Exchange'] != ''):
            enriched_count += 1
        else:
            failed_count += 1
            failed_symbols.append(symbol)
    
    # Remove temporary column if it exists
    if '_needs_processing' in df.columns:
        df = df.drop(columns=['_needs_processing'])
    
    print(f"  Final columns: {list(df.columns)}")
    print(f"  Successfully enriched: {enriched_count}/{len(df)} symbols")
    if failed_count > 0:
        print(f"  Failed/unavailable: {failed_count} symbols")
        if len(failed_symbols) > 0:
            print(f"  Examples of failed symbols: {', '.join(failed_symbols[:10])}")
    
    # Convert AvgVolume to numeric for sorting (coerce errors to NaN)
    df['AvgVolume'] = pd.to_numeric(df['AvgVolume'], errors='coerce')

    # Filter out rows with no enrichment data
    # original_count = len(df)
    # df_filtered = df[
    #     (df['Sector'].notna() & (df['Sector'] != '')) | 
    #     (df['Exchange'].notna() & (df['Exchange'] != '')) |
    #     (df['AvgVolume'].notna() & (df['AvgVolume'] != ''))
    # ].copy()
    # removed_count = original_count - len(df_filtered)
    
    # if removed_count > 0:
    #     print(f"\n  Removing {removed_count} rows with no enrichment data...")
    #     print(f"  Rows remaining: {len(df_filtered)}/{original_count}")
    

    
    # sort by volume
    # df_filtered = df_filtered.sort_values(by='AvgVolume', ascending=False)
    
    # Save final filtered and sorted data to TSV
    print(f"\nSaving final enriched data to: {output_file}")
    df.to_csv(output_file, index=False)
    print(f"  [OK] Saved successfully")
    
    return df


def display_sample(df: Optional[pd.DataFrame], num_rows: int = 10) -> None:
    """Display sample rows from the enriched dataframe.
    Args:
        df: DataFrame to display
        num_rows: Number of rows to display
    Side Effects:
        - Prints formatted sample data to stdout
        - Modifies pandas display options globally
    """
    if df is None or df.empty:
        print("No data to display")
        return
    
    print(f"\nSample data (first {num_rows} rows):")
    print("=" * 150)
    
    # Set display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)
    
    print(df.head(num_rows).to_string(index=False))
    print("=" * 150)


if __name__ == "__main__":
    # Enrich ETF data
    print("=" * 80)
    print("ENRICHING ETF DATA (CONCURRENT)")
    print("=" * 80)
    etf_file = 'tickerLists/largest_etfs_marketcap.tsv'
    etf_df = enrich_tsv_with_financial_data(etf_file, max_workers=3)
    
    if etf_df is not None:
        display_sample(etf_df, num_rows=10)
    
    print("\n\n")
    
    # Enrich Stock data
    print("=" * 80)
    print("ENRICHING STOCK DATA (CONCURRENT)")
    print("=" * 80)
    stock_file = 'tickerLists/largest_stocks_marketcap.tsv'
    stock_df = enrich_tsv_with_financial_data(stock_file, max_workers=3)
    
    if stock_df is not None:
        display_sample(stock_df, num_rows=10)
    
    print("\n\n")
    print("=" * 80)
    print("ENRICHMENT COMPLETE")
    print("=" * 80)
    if etf_df is not None:
        print(f"ETFs: {len(etf_df)} symbols enriched")
    if stock_df is not None:
        print(f"Stocks: {len(stock_df)} symbols enriched")
