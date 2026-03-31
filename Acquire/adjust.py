from datetime import timedelta

import pandas as pd
import yfinance as yf

# Reuse the efficient vector logic from the previous turn
def calculate_adjustments(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate backward-adjusted prices for Open, High, Low, Close using dividends and splits."""
    df = data.sort_index().copy()
    # Handle cases where Dividends/Splits might be missing or NaN
    cols = ['Dividends', 'Stock Splits']
    for col in cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    # Avoid division by zero for splits
    df['Stock Splits'] = df['Stock Splits'].replace(0, 1.0)
    
    # 1. Calculate Factors (Raw Logic)
    # Dividend Factor: (Close - Div) / Close
    # Split Factor: 1 / Split Ratio
    df['div_factor'] = (df['Close'] - df['Dividends']) / df['Close']
    df['split_factor'] = 1.0 / df['Stock Splits']
    
    # 2. Cumulative Multiplier (Backwards)
    df['combined_factor'] = df['div_factor'] * df['split_factor']
    df['adj_multiplier'] = df['combined_factor'].iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0)
    
    # 3. Apply
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df[f'Adj_{col}'] = df[col] * df['adj_multiplier']
            
    return df.drop(columns=['div_factor', 'split_factor', 'combined_factor', 'adj_multiplier'])

def update_market_data(ticker: str, existing_df: pd.DataFrame = None) -> pd.DataFrame:
    """Download incremental market data for ticker and re-calculate adjusted prices.
    Side Effects:
        - Makes network request via yfinance
        - Prints progress messages to stdout
    """
    
    # --- Step 1: Determine Start Date ---
    if existing_df is not None and not existing_df.empty:
        last_date = existing_df.index[-1]
        # Start download from the next day to avoid duplicates
        start_date = last_date + timedelta(days=1)
        print(f"Existing data found. Updating {ticker} from {start_date.date()}...")
    else:
        start_date = "1980-01-01" # Default to full history if no data
        print(f"No local data. Downloading full history for {ticker}...")
        existing_df = pd.DataFrame()

    # --- Step 2: Download New Raw Data ---
    # auto_adjust=False ensures we get RAW prices + Dividends + Splits
    new_data = yf.download(ticker, start=start_date, auto_adjust=False, actions=True, progress=False)
    
    if new_data.empty:
        print("No new data found.")
        return calculate_adjustments(existing_df) if not existing_df.empty else existing_df

    # Flatten MultiIndex columns if present (common yfinance issue)
    if isinstance(new_data.columns, pd.MultiIndex):
        new_data.columns = new_data.columns.get_level_values(0)

    # Clean up column names to match standard format
    # yfinance sometimes returns 'Stock Splits' or 'Splits'
    new_data = new_data.rename(columns={'Splits': 'Stock Splits'})

    # --- Step 3: Append ---
    # Combine old raw data with new raw data
    # Ensure we only keep raw columns for the master list
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
    
    # Filter only columns that exist (to avoid errors on empty frames)
    final_cols = [c for c in required_cols if c in new_data.columns]
    
    full_raw_df = pd.concat([existing_df, new_data[final_cols]])
    
    # Remove duplicates just in case
    full_raw_df = full_raw_df[~full_raw_df.index.duplicated(keep='last')]

    # --- Step 4: Re-Adjust ---
    # This is the "Magic": We re-run the adjustment on the full raw set.
    # It is instant and guarantees mathematical correctness.
    final_df = calculate_adjustments(full_raw_df)
    
    print(f"Success. Total rows: {len(final_df)}")
    return final_df