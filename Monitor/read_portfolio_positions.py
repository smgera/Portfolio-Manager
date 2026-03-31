import csv
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import numpy as np

from config import ACCOUNT_ACTIVITY_DIR, MONEY_MARKET_FUNDS
from utils.io_utils import read_tsv
from common.symbols_from_description import extract_symbol_from_description

_OPTION_RE = re.compile(r'^[\s\-]*([A-Za-z]+)(\d{6})([CP])(\d+(?:\.\d+)?)$')

def _normalize_option_symbol(sym: object) -> str:
    """Normalize an options ticker to standard OCC-like format.
    Strips leading ' -' prefix and zero-pads strikeprice*1000 to 8 digits.
    e.g. ' -XLE260320C52.5' -> 'XLE260320C00052500'
    Non-option symbols are returned unchanged.
    """
    if pd.isna(sym):
        return sym # NaN symbols are returned unchanged
    if not isinstance(sym, str):
        raise ValueError(f"Invalid symbol: {sym}")
    m = _OPTION_RE.match(sym)
    if not m:
        return sym # non-option symbols are returned unchanged
    underlying, date, cp, strike = m.groups()
    strike1000 = int(round(float(strike) * 1000))
    return f"{underlying}{date}{cp}{strike1000:08d}"

def clean_portfolio_data(df: pd.DataFrame, data_type: str = 'positions') -> pd.DataFrame:
    """Clean and normalize portfolio data DataFrame.
    Args:
        df: Raw portfolio DataFrame
        data_type: 'positions' for current positions, 'snapshot' for historical snapshots
    Returns:
        Cleaned DataFrame with standardized symbols and numeric quantities
    """
    if df.empty:
        return df
    
    # Make a copy to avoid side effects
    df = df.copy()
    
    # Symbol normalization
    # First extract symbols from descriptions if needed
    df["Symbol"] = df["Symbol"].apply(lambda x: extract_symbol_from_description(x) if pd.notna(x) else x)
    df["Symbol"] = df["Symbol"].apply(_normalize_option_symbol)
    
    # Then apply any additional mappings
    symbol_map = {"BRKB": "BRK-B"}
    df["Symbol"] = df["Symbol"].replace(symbol_map)
    
    # Special SPY mapping (only for positions data)
    if data_type == 'positions' and 'Description' in df.columns:
        df.loc[df['Description'] == 'S&P 500 EQUITY INDEX', 'Symbol'] = 'SPY'
    
    # Remove empty symbols
    df = df[df['Symbol'].notna() & (df['Symbol'] != '')]
    
    # Clean numeric columns
    if 'Current Value' in df.columns:
        df['Current Value'] = df['Current Value'].astype(str).replace(r'[\$,]', '', regex=True)
        df['Current Value'] = pd.to_numeric(df['Current Value'], errors='coerce')
        if data_type == 'positions':
            df = df[df['Current Value'].notna()]
    
    # Print MONEY_MARKET_FUNDS symbols
    print('Money Market Funds may not be available to trade, may be needed for settlements:')
    print(df[df['Symbol'].isin(MONEY_MARKET_FUNDS)][['Symbol','Current Value']].groupby('Symbol', as_index=False).agg({'Current Value': 'sum'}),'\n')
    # filter out MONEY_MARKET_FUNDS symbols
    df = df[~df['Symbol'].isin(MONEY_MARKET_FUNDS)]

    if 'Last Price' in df.columns:
        df['Last Price'] = df['Last Price'].astype(str).replace(r'[\$,]', '', regex=True)
        df['Last Price'] = pd.to_numeric(df['Last Price'], errors='coerce')
        # Mark invalid prices for positions data
        if data_type == 'positions':
            df.loc[(df['Last Price'].isna()) | (df['Last Price'] == 1.00), 'Symbol'] = '**'

    # Calculate or clean Quantity
    if 'Quantity' not in df.columns:
        if 'Current Value' in df.columns and 'Last Price' in df.columns:
            df['Quantity'] = df['Current Value'] / df['Last Price']
        else:
            df['Quantity'] = np.nan
    else:
        df['Quantity'] = df['Quantity'].astype(str).replace(r'[\$,]', '', regex=True)
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    
    # Filter out ignored symbols again (in case ** was added)
    df = df[~df['Symbol'].isin(MONEY_MARKET_FUNDS)]
    
    return df


def read_portfolio_positions(acctDir: str = None, target_date: datetime = None, max_days_threshold: int = None) -> Tuple[pd.DataFrame, Optional[str], Optional[datetime]]:
    """Finds portfolio positions file in the specified directory, reads it, cleans the data,
    and returns a DataFrame with 'Symbol', 'Current Value', and 'Last Price'.
    Args:
        acctDir: Directory containing Portfolio_Positions*.csv files.
                If None, uses ACCOUNT_ACTIVITY_DIR from config.
        target_date: datetime object or string 'YYYY-MM-DD' to find file closest to this date.
                    If None, returns the newest file by creation time.
        max_days_threshold: Maximum allowed days difference from target_date.
                           If None, no threshold is applied.
    Returns:
        tuple: (DataFrame with positions, path to the file used, file_date)
    Side Effects:
        - Reads portfolio CSV files from disk
    """
    # Use config default if no directory specified
    if acctDir is None:
        acctDir = ACCOUNT_ACTIVITY_DIR
    
    files = [f for f in os.listdir(acctDir) if re.search(r'_Positions_\d{2}-\d{2}-\d{2}\.csv$', f)]
    if not files:
        raise FileNotFoundError(f"No files matching '*_Positions_YY-MM-DD.csv' pattern found in {acctDir}")
    
    # Select file based on target_date or newest by creation time
    file_date = None
    if target_date is not None:
        # Convert target_date to datetime if it's a string
        if isinstance(target_date, str):
            target_date = datetime.strptime(target_date, '%Y-%m-%d')
        
        # Parse dates from filenames and find closest to target_date
        file_dates = []
        for f in files:
            # Extract date from filename like 'Something_Positions_25-01-26.csv'
            date_match = re.search(r'_(\d{2})-(\d{2})-(\d{2})\.csv$', f)
            if date_match:
                yy, mm, dd = date_match.groups()
                # Convert 2-digit year to 4-digit year (assuming 2000s)
                yyyy = f"20{yy}"
                file_date = datetime.strptime(f"{yyyy}-{mm}-{dd}", '%Y-%m-%d')
                days_diff = abs((file_date - target_date).days)
                file_dates.append((f, file_date, days_diff))
            else:
                raise ValueError(f"Filename {f} does not match expected format '*_Positions_YY-MM-DD.csv'")

        
        if not file_dates:
            raise ValueError(f"No files matching '*_Positions_YY-MM-DD.csv' pattern found in {acctDir}")    
        else:
            # Sort by days difference and pick the closest
            file_dates.sort(key=lambda x: x[2])
            closest_file, closest_date, days_diff = file_dates[0]
            
            # Check max_days_threshold if provided
            if max_days_threshold is not None and days_diff > max_days_threshold:
                available_files = [(f[1].strftime('%Y-%m-%d'), f[2]) for f in file_dates[:5]]
                raise ValueError(f"No portfolio files within {max_days_threshold} days of {target_date.strftime('%Y-%m-%d')}. "
                               f"Closest is {closest_date.strftime('%Y-%m-%d')} ({days_diff} days away). "
                               f"Available: {available_files}")
            
            newest_positions_file = Path(acctDir) / closest_file
            file_date = closest_date
            
    else:
        # Original behavior: newest file by date in filename (not creation time)
        def get_date_from_filename(filename: str) -> datetime:
            """Extract date string from a portfolio filename."""
            # Extract date from filename like 'Something_Positions_25-01-26.csv'
            date_match = re.search(r'_(\d{2})-(\d{2})-(\d{2})\.csv$', filename)
            if date_match:
                yy, mm, dd = date_match.groups()
                # Convert 2-digit year to 4-digit year (assuming 2000s)
                yyyy = f"20{yy}"
                return datetime.strptime(f"{yyyy}-{mm}-{dd}", '%Y-%m-%d')
            else:
                raise ValueError(f"Filename {filename} does not match expected format '*_Positions_YY-MM-DD.csv'")
        
        newest_filename = max(files, key=get_date_from_filename)
        newest_positions_file = Path(acctDir) / newest_filename
        file_date = get_date_from_filename(newest_filename)

    # Read portfolio positions CSV with BOM handling
    with open(newest_positions_file, 'r', encoding='utf-8-sig') as f:
        # Check if data rows have trailing commas
        reader = csv.reader(f)
        header = next(reader)
        first_row = next(reader)
        has_trailing_comma = first_row and first_row[-1] == ''
    
    # Read with proper parameters
    if has_trailing_comma:
        # Skip trailing comma column
        df = read_tsv(newest_positions_file, sep=',', encoding='utf-8-sig', usecols=range(len(header)))
    else:
        df = read_tsv(newest_positions_file, sep=',', encoding='utf-8-sig')

    if 'Symbol' not in df.columns:
        raise ValueError("CSV must have a 'Symbol' column.")

    # Clean portfolio data using shared function
    df = clean_portfolio_data(df, data_type='positions')
    if df.empty:
        raise ValueError("No valid data found after cleaning.")

    cols_to_keep = ['Symbol', 'Current Value']
    if 'Last Price' in df.columns:
        cols_to_keep.append('Last Price')
    if 'Quantity' in df.columns:
        cols_to_keep.append('Quantity')

    df = df[cols_to_keep]

    # combine symbols with same name, summing Current Value and Quantity, max last price
    agg_dict = {'Current Value': 'sum'}
    if 'Last Price' in df.columns:
        agg_dict['Last Price'] = 'max'
    if 'Quantity' in df.columns:
        agg_dict['Quantity'] = 'sum'
        
    df = df.groupby('Symbol', as_index=False).agg(agg_dict)
    # df = df.sort_values(by='Current Value', ascending=False)
    
    return df, newest_positions_file, file_date

if __name__ == "__main__":
    df, path, file_date = read_portfolio_positions()

    print(df.to_string()) # to string for cleaner output
    print(f"\nTotal value: ${df['Current Value'].sum():,.2f}")
    print(f"DataFrame shape: {df.shape}\n")