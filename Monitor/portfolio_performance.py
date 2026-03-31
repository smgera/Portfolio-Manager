"""Portfolio performance analysis module.
This module calculates portfolio performance metrics including returns, volatility,
Sharpe ratios, and risk metrics for different time periods.
"""

from datetime import datetime, timedelta
import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional
import warnings
import tkinter as tk
from tkinter import messagebox

import numpy as np
import pandas as pd

import config
from config import (MONEY_MARKET_FUNDS, REPORT_DIR, FIDELITY_CASH_SYMBOLS, TRADING_DAYS_PER_YEAR, 
                   TRADING_DAYS_PER_MONTH, PORTFOLIO_FRESHNESS_THRESHOLD_DAYS)
from utils.io_utils import read_tsv

from Acquire.DailyData import daily_data
from common.financial_utils import (
    calculate_returns,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    annualize_volatility,
)
from Monitor.read_portfolio_positions import read_portfolio_positions, clean_portfolio_data

warnings.filterwarnings('ignore', category=FutureWarning)


def check_positions_freshness() -> bool:
    """Check if the positions CSV file is fresh within the threshold.
    Returns:
        bool: True if fresh, False if stale
    Side Effects:
        - Shows message box and exits if data is stale
    """
    try:
        # Get the newest positions file and its date
        _, _, file_date = read_portfolio_positions()
        
        if file_date is None:
            messagebox.showerror(
                "Data Error", 
                "Could not determine portfolio positions file date.\n"
                "Please ensure portfolio positions files are available."
            )
            sys.exit(1)
        
        # Calculate age in days
        now = datetime.now()
        age_days = (now - file_date).days
        
        # Check if data is stale (older than threshold)
        if age_days > PORTFOLIO_FRESHNESS_THRESHOLD_DAYS:
            messagebox.showerror(
                "Stale Data Warning",
                f"Portfolio positions data is {age_days} days old.\n"
                f"Maximum allowed age is {PORTFOLIO_FRESHNESS_THRESHOLD_DAYS} days.\n\n"
                f"File date: {file_date.strftime('%Y-%m-%d')}\n"
                f"Current date: {now.strftime('%Y-%m-%d')}\n\n"
                "Please update your portfolio positions data before running analysis."
            )
            sys.exit(1)
        
        # Check if data is from today (for threshold = 0)
        if PORTFOLIO_FRESHNESS_THRESHOLD_DAYS == 0 and age_days > 0:
            messagebox.showerror(
                "Stale Data Warning",
                f"Portfolio positions data is {age_days} days old.\n"
                "Only today's data is considered fresh.\n\n"
                f"File date: {file_date.strftime('%Y-%m-%d')}\n"
                f"Current date: {now.strftime('%Y-%m-%d')}\n\n"
                "Please update your portfolio positions data before running analysis."
            )
            sys.exit(1)
        
        print(f"Portfolio positions data is fresh ({age_days} days old)")
        return True
        
    except FileNotFoundError as e:
        messagebox.showerror(
            "File Not Found",
            f"Portfolio positions file not found:\n{str(e)}\n\n"
            "Please ensure portfolio positions files are available in the account activity directory."
        )
        sys.exit(1)
    except Exception as e:
        messagebox.showerror(
            "Error Checking Freshness",
            f"Error checking portfolio data freshness:\n{str(e)}\n\n"
            "Please check your data files and try again."
        )
        sys.exit(1)


def calculate_risk_metrics(df: pd.DataFrame, period: str = '1mo') -> None:
    """Calculate risk metrics for all symbols in the dataframe."""
    print(f"Calculating {period} risk metrics for {len(df)} symbols...")
    
    # Convert period string to trading days for annualization
    period_trading_days = {
        '1mo': TRADING_DAYS_PER_MONTH,
        '2mo': TRADING_DAYS_PER_MONTH * 2,
        '3mo': TRADING_DAYS_PER_MONTH * 3,
        '6mo': TRADING_DAYS_PER_MONTH * 6,
        '1y':  TRADING_DAYS_PER_YEAR,
    }.get(period, np.nan)
    
    # Initialize new columns
    df[f'{period}Gain%'] = np.nan
    df['yrGain%'] = np.nan
    df['dayStdev%'] = np.nan
    df['yrStdev%'] = np.nan
    df['MaxDD%'] = np.nan
    df['Sharpe'] = np.nan
    df['$Value'] = df['Current Value']
    df['1moRisk%'] = np.nan
    df['$1moRisk'] = np.nan
    
    # Get data for all symbols
    symbols = df['Symbol'].tolist()
    print(f"Got data for {len(symbols)} symbols for {period}")
    
    # Process each symbol
    for idx, symbol in enumerate(symbols):
        try:
            # # Special handling for PULS (cash equivalent)
            # if symbol == 'PULS':
            #     # Cash has minimal returns and volatility
            #     df.loc[df['Symbol'] == symbol, f'{period}Gain%'] = 0.01  # Minimal return (money market yield)
            #     df.loc[df['Symbol'] == symbol, 'yrGain%'] = 0.5  # ~0.5% annual money market yield
            #     df.loc[df['Symbol'] == symbol, 'dayStdev%'] = 0.01  # Near-zero daily volatility
            #     df.loc[df['Symbol'] == symbol, 'yrStdev%'] = 0.1  # Near-zero annual volatility
            #     df.loc[df['Symbol'] == symbol, 'Sharpe'] = 0.1  # Low but positive Sharpe for cash
            #     current_value = df.loc[df['Symbol'] == symbol, 'Current Value'].iloc[0]
            #     df.loc[df['Symbol'] == symbol, '1moRisk%'] = 0.1  # Minimal risk
            #     df.loc[df['Symbol'] == symbol, '$1moRisk'] = current_value * 0.1 / 100
            #     continue
            
            # Fetch historical data
            data = daily_data(symbol, period=period)
            
            if data is not None and not data.empty and 'Close' in data.columns:
                prices = data['Close']
                
                # Calculate metrics
                if len(prices) >= 2:
                    # Period return into Gain% column
                    period_return = (prices.iloc[-1] / prices.iloc[0] - 1) # not %
                    if np.isnan(period_return) and sys.stdin.isatty():
                        input('NaN for ' + symbol + ' Press Enter to continue')

                    df.loc[df['Symbol'] == symbol, f'{period}Gain%'] = period_return * 100 # %
                    
                    # Annualized period return
                    annual_return = (1 + period_return) ** (TRADING_DAYS_PER_YEAR / period_trading_days) - 1 # not %
                    df.loc[df['Symbol'] == symbol, 'yrGain%'] = annual_return * 100 # %
                    
                    # Volatility
                    daily_returns = calculate_returns(prices) # not %, market days
                    daily_vol = calculate_volatility(daily_returns, annualize=False) # not %
                    annual_vol = annualize_volatility(daily_vol) # not %
                    df.loc[df['Symbol'] == symbol, 'dayStdev%'] = daily_vol * 100 # %
                    df.loc[df['Symbol'] == symbol, 'yrStdev%'] = annual_vol * 100 # %

                    # Sharpe ratio
                    if len(daily_returns) > 1 and annual_vol > 0:
                        sharpe = calculate_sharpe_ratio(annual_return, annual_vol)
                        df.loc[df['Symbol'] == symbol, 'Sharpe'] = sharpe
                    
                    # Max drawdown
                    max_dd = calculate_max_drawdown(prices) * 100
                    df.loc[df['Symbol'] == symbol, 'MaxDD%'] = max_dd
                    
                    current_value = df.loc[df['Symbol'] == symbol, 'Current Value'].iloc[0]
                    
                    # Risk metrics (simplified - using 2x daily volatility, covers ~95% of normal outcomes)
                    typ_mo_days = 252/12 # 252 mkt days/yr / 12mo 
                    risk_pct = daily_vol * 2 * np.sqrt(typ_mo_days) * 100 # Approximate monthly risk %, * 2 for normal 95% confidence
                    df.loc[df['Symbol'] == symbol, '1moRisk%'] = risk_pct
                    df.loc[df['Symbol'] == symbol, '$1moRisk'] = current_value * risk_pct / 100
            else:
                print(f'No data for {symbol}, skipping')

        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue
    
    print(f"After calculating {period} metrics, columns in df: {[col for col in df.columns if period in col]}")


def save_myTickers_tsv(df: pd.DataFrame) -> pd.DataFrame:
    """Save ticker list to myTickers.tsv with cash aggregation.
    Args:
        df: Portfolio DataFrame with position data
    Returns:
        DataFrame with cash aggregated into PULS
    Side Effects:
        - Reads portfolio positions CSV files from disk
        - Writes myTickers.tsv to disk
    """
    # Save ticker list - include PULS and treat other cash equivalents as PULS
    ticker_list_path = Path(__file__).parent.parent.parent / 'data' / 'tickerLists' / 'myTickers.tsv'
    
    # Read portfolio positions including cash equivalents
    # We need to read the raw file to get cash positions that are filtered out by default
    account_dir = Path('../data/accountActivity/')
    files = list(account_dir.glob('Portfolio_Positions*.csv'))
    
    if files:
        # Parse date from filename instead of using file modification time
        def get_date_from_filename(filepath: object) -> datetime:
            """Extract date string from a filename containing a date."""
            filename = filepath.name
            # Extract date from filename like 'Portfolio_Positions_2026Jan26.csv'
            date_part = filename.split('_')[2].replace('.csv', '')
            return datetime.strptime(date_part, '%Y%b%d')
        
        newest_file = max(files, key=get_date_from_filename)
        
        # Check if df is empty or missing Symbol column
        if df.empty or 'Symbol' not in df.columns:
            print("Warning: DataFrame is empty or missing Symbol column, skipping cash aggregation")
        else:
            # Read raw CSV without filtering cash equivalents
            raw_df = read_tsv(newest_file, sep=',', encoding='utf-8-sig', 
                                skipinitialspace=True,
                                usecols=range(16))  # Only read first 16 columns
            
            # Clean data but without filtering ignored symbols (to keep cash)
            temp_ignored = MONEY_MARKET_FUNDS.copy()
            # Temporarily remove cash symbols from ignored list
            cash_symbols = FIDELITY_CASH_SYMBOLS + ['PULS']
            non_cash_ignored = [s for s in temp_ignored if s not in cash_symbols]
            
            # Create a copy of config for temporary modification
            original_ignored = config.MONEY_MARKET_FUNDS
            config.MONEY_MARKET_FUNDS = non_cash_ignored
            
            try:
                # Clean data with modified config (keeps cash symbols)
                raw_df = clean_portfolio_data(raw_df, data_type='positions')
            finally:
                # Restore original config
                config.MONEY_MARKET_FUNDS = original_ignored
            
            # Get current PULS price
            puls_data = daily_data('PULS', period='5d')
            puls_price = puls_data['Close'].iloc[-1]
            
            # Sum all cash equivalents
            cash_positions = raw_df[raw_df['Symbol'].isin(cash_symbols)]
            total_cash_value = cash_positions['Current Value'].sum()
            
            # Combine with existing df (excluding any PULS that might be there)
            df = df[df['Symbol'] != 'PULS']
            if total_cash_value > 0:
                # Create single PULS entry with all cash value
                puls_row = pd.DataFrame([{
                    'Symbol': 'PULS',
                    'Current Value': total_cash_value,
                    'Last Price': puls_price,
                    'tPE': None,
                    'fPE': None,
                    'Beta': None
                }])
                df = pd.concat([df, puls_row], ignore_index=True)

    # Save ticker list sorted by Sharpe ratio (highest first)
    # Note: Sharpe is calculated in calculate_risk_metrics, so we sort by Current Value initially
    df_sorted = df.sort_values('Current Value', ascending=False)
    df_sorted.to_csv(ticker_list_path, index=False, sep='\t')
    print('saved myTickers.tsv', flush=True)
    
    return df


def generate_portfolio_report(df: pd.DataFrame, output_dir: Optional[str] = None, periods: Optional[list] = None) -> None:
    """Generate a comprehensive portfolio performance report.
    Args:
        df: Portfolio DataFrame with position data
        output_dir: Directory to save reports (optional)
        periods: List of periods to analyze (default: ['2mo', '1mo'])
    Side Effects:
        - Writes TSV report files to disk
    """

    if output_dir is None:
        output_dir = str(REPORT_DIR)
    
    if periods is None:
        periods = ['2mo', '1mo']
    
    # Save ticker list to myTickers.tsv
    df = save_myTickers_tsv(df)
    
    # Print summary - sort by Current Value since Sharpe not yet calculated
    print("\nPrinting portfolio summary:", flush=True)
    # Sort by Current Value descending for display
    df_sorted = df.sort_values('Current Value', ascending=False)
    print(df_sorted) #[['Symbol', 'Current Value', 'Beta', 'tPE', 'fPE']].to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
    print("Portfolio summary printed", flush=True)
    
    # Generate detailed reports for each period
    for period in periods:
        print(f"\n=== Starting {period} analysis ===")
        # Calculate risk metrics for this specific period
        calculate_risk_metrics(df, period)
        
        # Debug: Check if columns were added
        print(f"After calculating {period} metrics, columns in df: {[col for col in df.columns if period in col]}")
        
        if period+'Gain%' in df.columns:
            print(f"Found {period+'Gain%'} column, proceeding with analysis...")
            # Check if we have the required columns
            required_cols = ['Symbol', period+'Gain%', 'yrGain%', 'dayStdev%', 'yrStdev%', 'MaxDD%', 'Sharpe', 
                           '$Value', '1moRisk%', '$1moRisk']
            
            # Only include columns that exist
            available_cols = [col for col in required_cols if col in df.columns]
            print(f"Available columns for display: {available_cols}")
            
            # Don't filter columns - keep all original columns plus calculated ones
            # result_df = df[available_cols].copy()
            result_df = df.copy()
            
            # Sort by Sharpe ratio (highest first) for display
            result_df = result_df.sort_values('Sharpe', ascending=False)
            
            # Display performance summary
            print(f"\n{period} Performance (sorted by Sharpe):")
            print(result_df.to_string(index=False, float_format=lambda x: f"{x:9.2f}" if isinstance(x, (int, float)) else f"{x:>9}"))
            
            # Calculate and display risk summary
            total_value = result_df['$Value'].sum()
            total_risk = result_df['$1moRisk'].sum()
            risk_pct = (total_risk / total_value * 100) if total_value > 0 else 0
            
            # Remove symbols with low Sharpe ratio (only from result_df, not main df)
            low_sharpe = result_df[result_df['Sharpe'] < 0.5]
            if not low_sharpe.empty:
                print(f"\nRemoving symbols with low Sharpe:")
                print(low_sharpe['Symbol'].tolist())
                # Note: We don't modify the main df here as it's used for subsequent periods
            
            # Save results
            timestamp = datetime.now().strftime('%Y-%m-%d')
            sharpe_file = Path(output_dir) / f'Portfolio_Positions_{timestamp}_{period}_by_Sharpe.tsv'
            diff_file = Path(output_dir) / f'Portfolio_Positions_{timestamp}_{period}_by_Diff$.tsv'
            
            result_df.to_csv(sharpe_file, index=False, float_format='%.2f')
            print(f"Results written to {sharpe_file}")
            
            # Calculate and save target positions
            target_positions = calculate_target_positions(result_df, period)
            target_positions.to_csv(diff_file, index=False, float_format='%.2f')
            print(f"Diff$ report written to {diff_file}")
            
            # Display consolidated summary
            print(f"\n============================================================")
            print(f"CONSOLIDATED SUMMARY FOR {period.upper()}:    does NOT account for uncorrelation") 
            print(f"============================================================")
            print(f"Total Value: ${total_value:,.2f}")
            print(f"Total $1moRisk: ${total_risk:,.2f}")
            print(f"Total 1moRisk%: {risk_pct:.2f}% of Current Value")
            print(f"Position Qty: {len(target_positions)}")
            print(f"Total new positions $: ${target_positions['$Value'].sum():,.2f}")
            print(f"Total new risk $: ${target_positions['$1moRisk'].sum():,.2f}")
            pct_of_value = target_positions['$1moRisk'].sum() / total_value * 100 if total_value != 0 else float('nan')
            print(f"% of Current Value: {pct_of_value:.2f}%")
            print(f"============================================================")
            print(f"=== End {period} analysis ===")
            
            # Pause if not last iteration
            if period != periods[-1]:
                print(f"\nPress Enter to continue...")
                # Only wait for input if running interactively
                if sys.stdin.isatty():
                    input()

def calculate_target_positions(df: pd.DataFrame, period: str = '1mo') -> pd.DataFrame:
    """Calculate optimal target positions based on risk metrics."""
    # Create a copy to avoid modifying the original
    target_df = df.copy()
    
    # Calculate target position metrics
    total_value = df['$Value'].sum()
    
    # Calculate optimal leverage using Kelly criterion
    target_df['optimalLever'] = target_df['Sharpe'] / target_df['yrStdev%']
    
    # Calculate prudent risk (1/3 of Sharpe ratio)
    target_df['prudentRisk'] = target_df['Sharpe'] / 3
    
    # Calculate nominal target notion
    target_df['nomTargNotion'] = total_value * target_df['prudentRisk'] / target_df['yrStdev%']
    
    # Calculate target notion percentage
    target_df['targNotion%'] = target_df['nomTargNotion'] / target_df['nomTargNotion'].sum() * 100
    
    # Calculate target position dollars
    target_df['targPosition$'] = target_df['targNotion%'] * total_value / 100
    
    # Calculate target 1-month risk
    target_df['targ1moRisk'] = target_df['targPosition$'] * target_df['1moRisk%'] / 100
    
    # Calculate GainSharpe (using the specified period)
    target_df['GainSharpe'] = target_df[period+'Gain%'] * target_df['Sharpe']
    
    # Store original $Value before it gets modified
    target_df['Original$Value'] = target_df['$Value'].copy()
    
    # Filter out symbols with invalid Sharpe ratios or zero/negative values
    valid_mask = (df['Sharpe'] > 0) & (df['$Value'] > 0) & (df['$1moRisk'] > 0)
    valid_df = df[valid_mask].copy()
    
    if valid_df.empty:
        print("Warning: No valid positions for target calculation")
        return target_df
    
    # Calculate risk-adjusted scores (Sharpe ratio * inverse volatility)
    # Using $1moRisk% as a proxy for volatility
    valid_df['RiskScore'] = valid_df['Sharpe'] / (valid_df['1moRisk%'] / 100)
    valid_df['RiskScore'] = valid_df['RiskScore'].fillna(0)
    
    # Calculate weights based on risk scores
    total_risk_score = valid_df['RiskScore'].sum()
    if total_risk_score > 0:
        valid_df['TargetWeight'] = valid_df['RiskScore'] / total_risk_score
    else:
        # Fallback to equal weight if risk scores are zero
        valid_df['TargetWeight'] = 1.0 / len(valid_df)
    
    # Calculate target position values
    total_portfolio_value = df['$Value'].sum()
    valid_df['TargetValue'] = valid_df['TargetWeight'] * total_portfolio_value
    
    # Update the target dataframe with calculated values using vectorized operations
    # Merge target values back to target_df (but don't merge $Value to preserve original)
    target_df = target_df.merge(
        valid_df[['Symbol', 'TargetValue', '$1moRisk']], 
        on='Symbol', 
        how='left',
        suffixes=('', '_new')
    )
    
    # Update $Value and $1moRisk for valid positions
    valid_mask_merged = target_df['TargetValue'].notna()
    target_df.loc[valid_mask_merged, '$Value'] = target_df.loc[valid_mask_merged, 'TargetValue']
    
    # Adjust risk proportionally using vectorized operation
    risk_ratio = target_df.loc[valid_mask_merged, 'TargetValue'] / target_df.loc[valid_mask_merged, 'Original$Value']
    target_df.loc[valid_mask_merged, '$1moRisk'] = target_df.loc[valid_mask_merged, '$1moRisk_new'] * risk_ratio
    
    # Drop the temporary columns
    target_df = target_df.drop(columns=['TargetValue', '$1moRisk_new'], errors='ignore')
    
    # For positions with invalid metrics, keep original values but reduce them
    invalid_mask = ~valid_mask
    if invalid_mask.any():
        # Reduce invalid positions to 10% of original
        target_df.loc[invalid_mask, '$Value'] *= 0.1
        target_df.loc[invalid_mask, '$1moRisk'] *= 0.1
        print(f"Reduced positions with low Sharpe: {df[invalid_mask]['Symbol'].tolist()}")
    
    # Restore original $Value values for Diff$ calculation
    target_df['$Value'] = target_df['Original$Value']
    
    # Calculate difference between target and current position (using original values)
    target_df['Diff$'] = target_df['targPosition$'] - target_df['$Value']
    
    # Drop the temporary Original$Value column
    target_df = target_df.drop(columns=['Original$Value'])
    
    # Reorder columns to match expected format
    cols = target_df.columns.tolist()
    
    # Put key columns in specific order (use period-specific gain column)
    if period == '1mo':
        desired_order = ['Symbol', 'targPosition$', '$Value', 'Diff$', '1moGain%', 'yrGain%', 
                         'dayStdev%', 'yrStdev%', 'Sharpe', '1moRisk%', '$1moRisk', 
                         'optimalLever', 'prudentRisk', 'nomTargNotion', 'targNotion%', 
                         'targ1moRisk', 'GainSharpe']
    elif period == '2mo':
        desired_order = ['Symbol', 'targPosition$', '$Value', 'Diff$', '2moGain%', 'yrGain%', 
                         'dayStdev%', 'yrStdev%', 'Sharpe', '1moRisk%', '$1moRisk', 
                         'optimalLever', 'prudentRisk', 'nomTargNotion', 'targNotion%', 
                         'targ1moRisk', 'GainSharpe']
    else:
        # Use the period-specific gain column dynamically
        gain_col = period + 'Gain%'
        desired_order = ['Symbol', 'targPosition$', '$Value', 'Diff$', gain_col, 'yrGain%', 
                         'dayStdev%', 'yrStdev%', 'Sharpe', '1moRisk%', '$1moRisk', 
                         'optimalLever', 'prudentRisk', 'nomTargNotion', 'targNotion%', 
                         'targ1moRisk', 'GainSharpe']
    
    # Create new column order with ONLY desired columns
    final_cols = []
    for col in desired_order:
        if col in cols:
            final_cols.append(col)
    
    # Reorder the dataframe with only these columns
    target_df = target_df[final_cols]
    
    # Sort by Diff$ (descending) as in the original
    target_df = target_df.sort_values('Diff$', ascending=False)
    
    return target_df


def main() -> None:
    """Main function to run portfolio performance analysis."""
    print(f"Analysis timestamp: {datetime.now()}")
    
    # Check if positions data is fresh before proceeding
    check_positions_freshness()
    
    # Read portfolio positions
    df, _, _ = read_portfolio_positions()
    
    if df is not None and not df.empty:
        print(f"\nLoaded {len(df)} positions from portfolio")
        # Save ticker list to myTickers.tsv
        save_myTickers_tsv(df)
    else:
        print("No portfolio data found")
    
if __name__ == "__main__":
    main()
