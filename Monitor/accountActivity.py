"""This module reads one or more Fidelity account activity CSV files,
cleans and combines the transaction rows into a pandas DataFrame.
Can be imported to get the activity data or run as a script.
"""
import csv
from datetime import datetime, timedelta
import glob
import itertools
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import REPORT_DIR, MONEY_MARKET_FUNDS, FIDELITY_ACTIVITY_PATTERN, PORTFOLIO_POSITIONS_PATTERN, ACCOUNT_ACTIVITY_DIR, FIDELITY_MONEY_MARKET_SYMBOLS, FIDELITY_ACTION_BOUGHT, FIDELITY_ACTION_SOLD, FIDELITY_ACTION_CANCELLED
from utils.io_utils import read_tsv

from common.portfolio_utils import clean_portfolio_data
from Monitor.read_portfolio_positions import read_portfolio_positions

def all_activity_data(filepattern: str = None,
                       verbose: bool = False,
                       save_csv: bool = False,
                       use_cache: bool = True) -> pd.DataFrame:
    """Load and combine Fidelity account activity CSV files into a DataFrame.
    Args:
        filepattern: Glob pattern to match activity CSV files (defaults to FIDELITY_ACTIVITY_PATTERN)
        verbose: If True, print progress information
        save_csv: If True, save combined data to CSV file
        use_cache: If True, use cached combined data if available
    Returns:
        Combined DataFrame with activity data
    Side Effects:
        - Reads activity CSV files from disk
        - Optionally writes combined CSV to REPORT_DIR when save_csv=True
        - Prints processing status to stdout
    """
    # Use default pattern if not specified
    if filepattern is None:
        filepattern = str(ACCOUNT_ACTIVITY_DIR / FIDELITY_ACTIVITY_PATTERN)

    output_file = f"{REPORT_DIR}/allActivity_{datetime.now().strftime('%Y%m%d')}.csv"
    if use_cache and os.path.exists(output_file):
        # read output_file
        df = read_tsv(output_file, sep=',')
        if 'Run Date' in df.columns:
            df['Run Date'] = pd.to_datetime(df['Run Date'])
        return df

    files = glob.glob(filepattern)
    if not files:
        print(f"No files found matching pattern: {filepattern}")
        input("Press any key to continue...\n")
        return pd.DataFrame()

    files.sort(reverse=True)
    table = []
    
    for f in files:
        print(f"Processing {f}")
        with open(f, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            # read all rows having more than 1 column
            rows = [row for row in reader if len(row) > 1] 

            assert 'Symbol' in rows[0] # expect 'Symbol' in header

            if not table:
                rows[0][0] = rows[0][0].replace('\ufeff', '')  # remove BOM
                table.append(rows[0][0:18]) # append header to empty table
            
            for row in rows[1:]:
                table.append(row[0:18])

    # Create DataFrame from table
    if not table or len(table) < 2:
        if verbose:
            print("No data found in account activity files.")
        return pd.DataFrame()

    df = pd.DataFrame(table[1:], columns=table[0])

    # Remove unnecessary columns (handle missing columns gracefully)
    columns_to_drop = ['Description', 'Settlement Date', 'Account Number', 'Exchange Currency', 'Exchange Rate', 'Exchange Quantity']
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    if existing_columns:
        df = df.drop(columns=existing_columns)

    # Remove rows with QACA accounts
    df = df[~df['Account'].str.contains('QACA', case=False, na=False)]  
    print('removed QACA accounts rows')

    # Remove rows with non-trading actions (dividends, contributions, transfers, etc.)
    # Removed 'REINVESTMENT' to correctly capture share increases from dividend reinvestments
    excluded_actions = (
        'DIRECT DEBIT|Check Paid|NORMAL DISTR|INTEREST FULLY PAID|INSUFFICIENT FUNDS|'
        'DEBIT CARD PURCHAS|PARTIC CONTR|Dividend|Journaled|YOU LOANED|ADJ COLLATERAL|'
        'LOAN RETURNED|DIRECT DEPOSIT|CONTRIBUTION|ROLLOVER|TRANSFERRED|'
        'EXCHANGED TO|TRANSFER OF|CURRENCY EXCHANGE|FOREX|FEE|FINANCING'
    )
    df = df[~df['Action'].str.contains(excluded_actions, case=False, na=False)]
    
    # Remove all financing transactions (these have non-numeric quantities)
    df = df[df['Type'] != 'Financing']

    # simplify action names
    df.loc[df['Action'].str.contains('SOLD', case=False, na=False), 'Action'] = 'SOLD'
    df.loc[df['Action'].str.contains('BOUGHT', case=False, na=False), 'Action'] = 'BOUGHT'

    # where Currency is not USD, rotate values columns Quantity, Currency, Price
    df.loc[df['Currency'] != 'USD', ['Quantity', 'Currency', 'Price']] = df.loc[df['Currency'] != 'USD', ['Price', 'Quantity', 'Currency']].values

    # Consolidate money market funds into PULS (instead of filtering them out)
    money_market_symbols = FIDELITY_MONEY_MARKET_SYMBOLS
    df.loc[df['Symbol'].isin(money_market_symbols), 'Symbol'] = 'PULS'

    # check cancelled trades
    for index, row in df.iterrows():
        if 'CANCELLED TRADE' in str(row['Action']).upper():
            # Find corresponding previous row where action is SOLD or BOUGHT, confirm same symbol and opposite quantity  
            if index > 0:
                prev_row = df.loc[index-1]
                if prev_row['Action'] in [FIDELITY_ACTION_SOLD, FIDELITY_ACTION_BOUGHT] and prev_row['Symbol'] == row['Symbol'] and float(prev_row['Quantity']) == -float(row['Quantity']):
                    df.loc[index-1, 'Action'] = FIDELITY_ACTION_CANCELLED
                    df.loc[index, 'Action'] = FIDELITY_ACTION_CANCELLED
                else:
                    print('a cancelled trade without previous buy or sell at')
                    print([prev_row, row])
                    input("Press any key to continue...\n")

    # convert Run Date to datetime
    df['Run Date'] = pd.to_datetime(df['Run Date'])
    #sort by Run Date
    df = df.sort_values(by='Run Date')  
    
    # Convert numeric columns to float, replacing non-numeric values with 0
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)  # Amount is net after fees and commission
    df['Commission'] = pd.to_numeric(df['Commission'], errors='coerce').fillna(0)
    df['Fees'] = pd.to_numeric(df['Fees'], errors='coerce').fillna(0)
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0)

    # Ensure Quantity signs are correct based on Action
    # SOLD should decrease quantity (negative)
    mask_sold = df['Action'].str.contains('SOLD', case=False, na=False)
    df.loc[mask_sold, 'Quantity'] = -df.loc[mask_sold, 'Quantity'].abs()

    # BOUGHT and REINVESTMENT should increase quantity (positive)
    mask_positive = df['Action'].str.contains('BOUGHT|REINVESTMENT', case=False, na=False)
    df.loc[mask_positive, 'Quantity'] = df.loc[mask_positive, 'Quantity'].abs()

    # DEBUG: Check GLDM
    gldm_rows = df[df['Symbol'] == 'GLDM']
    if not gldm_rows.empty:
        print("\n--- DEBUG: GLDM Activity ---")
        print(gldm_rows[['Run Date', 'Action', 'Quantity', 'Amount']].to_string())
        print("----------------------------\n")

    # for QACA accounts with Quantity > 0, set Price = Amount / Quantity
    qaca_mask = df['Account'].str.contains('QACA', case=False, na=False) & (df['Quantity'] != 0)
    df.loc[qaca_mask, 'Price'] = df.loc[qaca_mask, 'Amount'] / df.loc[qaca_mask, 'Quantity']    
    
    # Calculate derived columns
    df['gross'] = df['Amount'] + df['Commission'] + df['Fees']
    df['eachdiff'] = abs(df['gross'] / df['Quantity']) - abs(df['Price']) # abs to work with QACA accounts
    df['totaldiff'] = df['eachdiff'] * df['Quantity']

    # Optionally save processed data to CSV
    if save_csv:
        df.to_csv(output_file, index=False)
        print(f"Saved to {output_file}")   
    
    return df


def get_unique_symbols(df: pd.DataFrame, max_length: int = 5) -> list[str]:
    """Extract unique ticker symbols from activity DataFrame.
    Args:
        df: Activity DataFrame with a Symbol column
        max_length: Maximum length of symbols to include (default 5)
    Returns:
        Sorted list of unique symbols
    """
    # Try to detect the symbol column name robustly
    symbol_col = None
    for col in df.columns:
        if "symbol" in str(col).lower():
            symbol_col = col
            break

    if symbol_col is None:
        return []

    # Get sorted unique symbols, excluding those with more than max_length characters
    unique_symbols = sorted(
        set(
            df[symbol_col]
            .dropna()
            .astype(str)
            .str.strip()
            # Keep only non-empty symbols with length 1..max_length
            .loc[lambda x: (x.str.len() > 0) & (x.str.len() <= max_length)]
        )
    )
    
    return unique_symbols


def create_symbol_series(df_activity: pd.DataFrame, df_positions: pd.DataFrame, symbols_to_skip: list = None) -> tuple:
    """Create symbol series showing cumulative quantity over time from activity data.
    Args:
        df_activity: DataFrame containing transaction activity data
        df_positions: DataFrame containing current portfolio positions with quantities
        symbols_to_skip: List of symbols to skip (default: ['PULS'])
    Returns:
        tuple: (symbol_series dict, global_start_date)
        symbol_series: {symbol: pandas Series with datetime index and cumulative quantities}
        global_start_date: Earliest date in the activity data
    """
    # Input validation
    if df_activity.empty:
        raise ValueError("df_activity cannot be empty")
    if df_positions.empty:
        raise ValueError("df_positions cannot be empty")
    
    required_activity_cols = ['Run Date', 'Quantity', 'Symbol']
    missing_activity_cols = [col for col in required_activity_cols if col not in df_activity.columns]
    if missing_activity_cols:
        raise ValueError(f"df_activity missing required columns: {missing_activity_cols}")
    
    required_position_cols = ['Symbol', 'Current Value', 'Last Price']
    missing_position_cols = [col for col in required_position_cols if col not in df_positions.columns]
    if missing_position_cols:
        raise ValueError(f"df_positions missing required columns: {missing_position_cols}")
    
    if symbols_to_skip is None:
        symbols_to_skip = ['PULS']
    
    # Work with a copy to avoid side effects on caller's DataFrame
    df_positions = df_positions.copy()
    
    # 1. Prepare df_positions to have a complete 'Quantity' column for targets
    if 'Quantity' not in df_positions.columns:
        df_positions['Quantity'] = np.nan
    
    # Fill missing/NaN quantities using Current Value / Last Price
    calc_qty = df_positions['Current Value'] / df_positions['Last Price']
    df_positions['Quantity'] = df_positions['Quantity'].fillna(calc_qty)
    
    # Create lookup Series: Symbol -> Target Quantity
    target_qtys = df_positions.set_index('Symbol')['Quantity']
    
    # 2. Process each symbol independently
    symbol_series = {}
    
    # Group activity by symbol
    grouped = df_activity.groupby('Symbol')
    
    current_time = datetime.now()
    global_start_date = pd.to_datetime(df_activity['Run Date'].min())
    
    for symbol, group in grouped:
        # Skip specified symbols
        if symbol in symbols_to_skip: 
            continue
            
        # Sort by date
        group = group.sort_values('Run Date')
        
        # Create Time Series of Cumulative Quantity
        # We use 'Run Date' as the index, 'Quantity' is the delta
        group['Run Date'] = pd.to_datetime(group['Run Date'])
        ts = group.set_index('Run Date')['Quantity'].cumsum()
        
        # Determine Target Quantity
        target_qty = target_qtys.get(symbol, None)
        
        # Calculate Shift: Target - Last_Calculated
        if not ts.empty:
            if target_qty is not None:
                # Symbol exists in current positions, shift to target
                shift = target_qty - ts.iloc[-1]
                ts = ts + shift
            else:
                # Symbol not in current positions (sold out), keep last calculated quantity
                # Don't shift - preserve the historical quantity
                pass
            
            # --- Stretch to Left (Global Start) ---
            # Calculate quantity BEFORE the first trade
            # ts.iloc[0] is the value AFTER the first trade. 
            # group.iloc[0]['Quantity'] is the first delta.
            first_trade_date = ts.index[0]
            first_delta = group.iloc[0]['Quantity']
            start_qty = ts.iloc[0] - first_delta
            
            # Create extension points: 
            # 1. At global start -> start_qty
            # 2. At first trade date (before trade) -> start_qty
            # This ensures a flat line from start to first trade, then a vertical step
            extension = pd.Series(
                [start_qty, start_qty], 
                index=[global_start_date, first_trade_date - timedelta(seconds=1)]
            )
            ts = pd.concat([extension, ts])
            
        # Append "Now" point with Target Quantity or last calculated
        if target_qty is not None:
            ts.loc[current_time] = target_qty
        else:
            # Symbol not in current positions, preserve last calculated quantity
            ts.loc[current_time] = ts.iloc[-1]
        
        # Sort index to ensure correct plotting order
        ts = ts.sort_index()
        
        # Remove duplicate indices, keeping the last one (final cumulative quantity for that timestamp)
        # This prevents "ValueError: cannot reindex on an axis with duplicate labels" when creating the combined DataFrame
        ts = ts[~ts.index.duplicated(keep='last')]
        
        # Store the series
        symbol_series[symbol] = ts
    
    return symbol_series, global_start_date

def main() -> None:
    """Main function when run as a script.
    Side Effects:
        - Writes combined activity CSV file to disk
    """
    # Load activity data with verbose output
    df_activity = all_activity_data(save_csv=True)
    
    if df_activity.empty:
        print("No activity data loaded.")
        return
    
    print(f"\nDataFrame shape: {df_activity.shape}")
    print(f"\nColumns: {list(df_activity.columns)}")

    # Get unique symbols
    symbols = get_unique_symbols(df_activity)
    
    print(f"\nFound {len(symbols)} unique symbols (<=5 chars):")
    print(f"Activity symbols: {sorted(symbols)}")
    
    # read latest portfolio positions using the shared module
    df_positions, _, _ = read_portfolio_positions(str(ACCOUNT_ACTIVITY_DIR))
    print(f"Portfolio symbols: {sorted(df_positions['Symbol'].unique())}")

    # Create symbol series from activity and position data
    symbol_series, global_start_date = create_symbol_series(df_activity, df_positions)

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 10))
    
    markers = itertools.cycle(['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_'])

    # Sort symbols for consistent legend order
    sorted_symbols = sorted(symbol_series.keys())

    # save symbol_series to csv
    # Combine all series into a single DataFrame (outer join on index)
    df_all_series = pd.DataFrame(symbol_series).sort_index()
    
    # Forward fill to propagate last known quantity to all dates
    # (Because outer join creates NaNs for symbols that didn't trade on a specific date)
    df_all_series = df_all_series.ffill().fillna(0)
    
    # Save to reports folder using config constant
    series_output_file = f"{REPORT_DIR}/symbol_series_qty_{datetime.now().strftime('%Y%m%d')}.csv"
    df_all_series.to_csv(series_output_file)
    print(f"Saved quantity history to {series_output_file}")

    for symbol in sorted_symbols:
        ts = symbol_series[symbol]
        marker = next(markers)
        ax.plot(ts.index, ts.values, label=symbol, linewidth=1.5, alpha=0.7, marker=marker)
    
    # Legend with multiple columns to fit 40 symbols
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), ncol=3, fontsize=8, 
              framealpha=0.9, title='Symbols')
    
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Format dates on x-axis
    fig.autofmt_xdate()
    
    plt.tight_layout()
    # plt.show()
        
    # ---------------------------------------------------------
    # Compare with Historical Position Snapshots
    # ---------------------------------------------------------
    print("\nComparing calculated history with Portfolio_Positions snapshots...")
    
    snapshot_files = glob.glob(str(ACCOUNT_ACTIVITY_DIR / PORTFOLIO_POSITIONS_PATTERN))
    discrepancies = []
    
    for snap_file in snapshot_files:
        # Parse date from filename (e.g., Portfolio_Positions_2025Nov28.csv)
        basename = os.path.basename(snap_file)
        # Expected format: Portfolio_Positions_YYYYMonDD.csv
        # Simple parse approach: assume date is between second underscore and .csv
        try:
            date_part = basename.split('_')[2].replace('.csv', '')
            snap_date = pd.to_datetime(date_part, format='%Y%b%d')
        except Exception as e:
            print(f"Skipping {basename}: Could not parse date ({e})")
            continue
            
        # Read snapshot
        try:
            df_snap = read_tsv(snap_file, sep=',')
        except:
            continue
            
        if 'Symbol' not in df_snap.columns:
            continue
            
        # Clean snapshot data using shared function
        df_snap = clean_portfolio_data(df_snap, data_type='snapshot')
        
        if df_snap.empty:
            continue
            
        # Group by symbol to handle duplicates in snapshot
        snap_qtys = df_snap.groupby('Symbol')['Quantity'].sum()
        
        # Compare with calculated history at snap_date
        # Use asof to find the calculated quantity on or before the snapshot date
        # df_all_series index is datetime
        # We need to check if snap_date is within range
        
        if df_all_series.empty:
             continue
             
        # Find nearest index in history <= snap_date
        # df_all_series is sorted by index
        idx_loc = df_all_series.index.asof(snap_date)
        
        if pd.isna(idx_loc):
            # Snapshot is before history starts
            continue
            
        calculated_row = df_all_series.loc[idx_loc]
        
        for symbol in snap_qtys.index:
            if symbol in MONEY_MARKET_FUNDS: continue
            
            snap_qty = snap_qtys[symbol]
            
            if symbol in calculated_row:
                calc_qty = calculated_row[symbol]
                if pd.isna(calc_qty): calc_qty = 0.0
                
                diff = calc_qty - snap_qty
                
                discrepancies.append({
                    'Snapshot File': basename,
                    'Snapshot Date': snap_date,
                    'Symbol': symbol,
                    'Snapshot Qty': snap_qty,
                    'Calculated Qty': calc_qty,
                    'Difference': diff,
                    'Status': 'Mismatch' if abs(diff) > 0.01 else 'Match'
                })
            else:
                # Symbol in snapshot but not in calculated history
                 discrepancies.append({
                        'Snapshot File': basename,
                        'Snapshot Date': snap_date,
                        'Symbol': symbol,
                        'Snapshot Qty': snap_qty,
                        'Calculated Qty': 0.0,
                        'Difference': -snap_qty,
                        'Status': 'Not in Activity Logs'
                    })

    # Save Full Comparison Report
    if discrepancies:
        df_diff = pd.DataFrame(discrepancies)
        # Sort by Symbol (A-Z), then Snapshot Date (Newest-Oldest)
        df_diff = df_diff.sort_values(['Symbol', 'Snapshot Date'], ascending=[True, False])
        
        diff_file = f"{REPORT_DIR}/qty_comparison_{datetime.now().strftime('%Y%m%d')}.csv"
        df_diff.to_csv(diff_file, index=False)
        print(f"\nSaved full comparison report to {diff_file}")
        
        mismatches = df_diff[df_diff['Status'] != 'Match']
        print(f"Total comparisons: {len(df_diff)}")
        print(f"Mismatches found: {len(mismatches)}")
    else:
        print("\nNo comparisons generated.")


if __name__ == "__main__":
    main()
