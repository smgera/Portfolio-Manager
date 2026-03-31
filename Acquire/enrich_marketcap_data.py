"""Enrich marketcap TSV files with additional financial data.
Adds columns: Sector, Industry, TrailingPE, ForwardPE, Beta, AvgVolume
"""

from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Any, Dict, Optional, Union

import pandas as pd
import yfinance as yf

from config import TICKER_LISTS_DIR
from utils.io_utils import read_tsv


def get_ticker_info(symbol: str) -> Dict[str, Any]:
    """Fetch additional financial info for a ticker symbol via yfinance.
    Args:
        symbol: Ticker symbol
    Returns:
        Dictionary with financial data
    Side Effects:
        - Makes network request via yfinance
        - Prints error messages to stdout on failure
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
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
        
        return {
            'Exchange': info.get('exchange', ''),
            'Sector': sector,
            'Industry': industry,
            'TrailingPE': info.get('trailingPE', ''),
            'ForwardPE': info.get('forwardPE', ''),
            'Beta': beta,
            'AvgVolume': info.get('averageVolume', ''),
            'PEGRatio': peg_ratio,
            'ProfitMargin': info.get('profitMargins', ''),
            'DebtToEquity': info.get('debtToEquity', ''),
            'ROE': info.get('returnOnEquity', ''),
            'InceptionDate': inception_date
        }
    except Exception as e:
        print(f"  ⚠️  Error fetching data for {symbol}: {e}")
        return {
            'Exchange': '',
            'Sector': '',
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


def enrich_tsv_with_financial_data(tsv_file: Union[str, Path], sleep_interval: float = 0.1, output_file: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Enrich a TSV file with additional financial data from yfinance.
    Args:
        tsv_file: Path to input TSV file
        sleep_interval: Sleep interval between API calls
        output_file: Path to output file (defaults to overwriting input)
    Returns:
        Enriched DataFrame
    Side Effects:
        - Makes network requests via yfinance for each symbol
        - Writes enriched TSV file to disk
        - Prints progress messages to stdout
    """
    # Use input file as output if not specified
    if output_file is None:
        output_file = tsv_file
    
    # Ensure output_file is a Path object
    output_file = Path(output_file)
    
    print(f"Reading TSV from: {tsv_file}")
    df = read_tsv(tsv_file)
    
    print(f"  Loaded {len(df)} rows")
    print(f"  Current columns: {list(df.columns)}")
    
    # Check if we have existing enriched data with today's date
    today_str = datetime.now().strftime('%Y-%m-%d')
    existing_enriched = pd.DataFrame()
    symbols_to_skip = set()
    
    if output_file.exists():
        try:
            existing_enriched = read_tsv(output_file)
            # Check if LastEnriched column exists and has today's data
            if 'LastEnriched' in existing_enriched.columns:
                today_symbols = existing_enriched[
                    existing_enriched['LastEnriched'] == today_str
                ]['Symbol'].tolist()
                symbols_to_skip = set(today_symbols)
                print(f"  Found {len(symbols_to_skip)} symbols already enriched today")
            else:
                print(f"  Existing file found but no LastEnriched column - will process all")
        except Exception as e:
            print(f"  Could not read existing file: {e}")
    
    # Keep original df intact for final merge, create working copy
    df_original = df.copy()
    
    # Filter out symbols already enriched today for processing only
    if symbols_to_skip:
        skipped_count = len(symbols_to_skip)
        df = df[~df['Symbol'].isin(symbols_to_skip)].copy()
        print(f"  Skipping {skipped_count} symbols already enriched today")
    
    if len(df) == 0:
        print("  All symbols already enriched today!")
        return existing_enriched
    
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
    df['LastEnriched'] = ''
    
    print(f"\nFetching financial data for {len(df)} symbols...")
    
    # Iterate through each row and fetch data
    for idx, row in df.iterrows():
        symbol = row['Symbol']
        
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(df)} ({idx/len(df)*100:.1f}%)")
        
        # Get ticker info
        ticker_data = get_ticker_info(symbol)
        
        # Update dataframe
        df.at[idx, 'Exchange'] = ticker_data['Exchange']
        df.at[idx, 'Sector'] = ticker_data['Sector']
        df.at[idx, 'Industry'] = ticker_data['Industry']
        df.at[idx, 'TrailingPE'] = ticker_data['TrailingPE']
        df.at[idx, 'ForwardPE'] = ticker_data['ForwardPE']
        df.at[idx, 'Beta'] = ticker_data['Beta']
        df.at[idx, 'AvgVolume'] = ticker_data['AvgVolume']
        df.at[idx, 'PEGRatio'] = ticker_data['PEGRatio']
        df.at[idx, 'ProfitMargin'] = ticker_data['ProfitMargin']
        df.at[idx, 'DebtToEquity'] = ticker_data['DebtToEquity']
        df.at[idx, 'ROE'] = ticker_data['ROE']
        df.at[idx, 'InceptionDate'] = ticker_data['InceptionDate']
        df.at[idx, 'LastEnriched'] = today_str
        
        # Sleep to avoid rate limiting
        if sleep_interval > 0:
            sleep(sleep_interval)
    
    print(f"\n✅ Completed fetching data for all symbols")
    print(f"  Final columns: {list(df.columns)}")
    
    # Combine with existing enriched data
    if not existing_enriched.empty and 'LastEnriched' in existing_enriched.columns:
        # Get today's existing enriched data
        today_existing = existing_enriched[existing_enriched['LastEnriched'] == today_str]
        
        # Get symbols that were skipped (already enriched today)
        if symbols_to_skip:
            try:
                # Get original data for skipped symbols
                skipped_original = df_original[df_original['Symbol'].isin(symbols_to_skip)].copy()
                
                # Merge with existing enriched data for those symbols
                skipped_enriched = today_existing[today_existing['Symbol'].isin(symbols_to_skip)].copy()
                
                # Ensure we have all original columns in the final data
                if not skipped_enriched.empty:
                    # Define enriched columns that should exist
                    enriched_cols = ['Symbol', 'Exchange', 'Sector', 'Industry', 'TrailingPE', 
                                    'ForwardPE', 'Beta', 'AvgVolume', 'PEGRatio', 
                                    'ProfitMargin', 'DebtToEquity', 'ROE', 'InceptionDate', 
                                    'LastEnriched']
                    
                    # Only use columns that actually exist in the enriched data
                    available_cols = [col for col in enriched_cols if col in skipped_enriched.columns]
                    
                    # Merge original data with enriched data, prioritizing enriched columns
                    final_skipped = skipped_original.merge(
                        skipped_enriched[available_cols], 
                        on='Symbol', 
                        how='left',
                        suffixes=('', '_enriched')
                    )
                    
                    # Remove duplicate columns from merge
                    for col in final_skipped.columns:
                        if col.endswith('_enriched'):
                            original_col = col.replace('_enriched', '')
                            final_skipped[original_col] = final_skipped[col]
                            final_skipped.drop(columns=[col], inplace=True)
                    
                    # Combine with newly enriched data
                    df_combined = pd.concat([final_skipped, df], ignore_index=True)
                    print(f"  Combined with {len(final_skipped)} existing enriched symbols")
                else:
                    df_combined = df
            except Exception as e:
                print(f"  ⚠️  Error merging existing enriched data: {e}")
                print("  Proceeding with newly enriched data only...")
                df_combined = df
        else:
            df_combined = df
    else:
        df_combined = df
    
    # Filter out rows with no enrichment data
    original_count = len(df_combined)
    df_filtered = df_combined[
        (df_combined['Sector'].notna() & (df_combined['Sector'] != '')) | 
        (df_combined['Exchange'].notna() & (df_combined['Exchange'] != '')) |
        (df_combined['AvgVolume'].notna() & (df_combined['AvgVolume'] != ''))
    ].copy()
    removed_count = original_count - len(df_filtered)
    
    if removed_count > 0:
        print(f"\n  Removing {removed_count} rows with no enrichment data...")
        print(f"  Rows remaining: {len(df_filtered)}/{original_count}")
    
    # Save to TSV
    print(f"\nSaving enriched data to: {output_file}")
    df_filtered.to_csv(output_file, index=False)
    print(f"  ✅ Saved successfully")
    
    return df_filtered


def display_sample(df: pd.DataFrame, num_rows: int = 10) -> None:
    """Display sample rows from the enriched dataframe.
    Args:
        df: DataFrame to display
        num_rows: Number of rows to display
    Side Effects:
        - Prints formatted sample data to stdout
        - Modifies pandas display options globally
    """
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
    print("ENRICHING ETF DATA")
    print("=" * 80)
    etf_file = TICKER_LISTS_DIR / 'largest_etfs_marketcap.tsv'
    etf_df = enrich_tsv_with_financial_data(etf_file, sleep_interval=0.1)
    display_sample(etf_df, num_rows=10)
    
    print("\n\n")
    
    # Enrich Stock data
    print("=" * 80)
    print("ENRICHING STOCK DATA")
    print("=" * 80)
    stock_file = TICKER_LISTS_DIR / 'largest_stocks_marketcap.tsv'
    stock_df = enrich_tsv_with_financial_data(stock_file, sleep_interval=0.1)
    display_sample(stock_df, num_rows=10)
    
    print("\n\n")
    print("=" * 80)
    print("ENRICHMENT COMPLETE")
    print("=" * 80)
    print(f"ETFs: {len(etf_df)} symbols enriched")
    print(f"Stocks: {len(stock_df)} symbols enriched")
