"""Portfolio Value and Daily Gains Plotter with Persistent Caching.
This script creates visualizations for your top 10 portfolio positions:
1. Portfolio value over 6 months
2. Daily dollar gains/losses over 6 months
Uses daily_data with persistent feather caching for historical price data and integrates with your existing portfolio modules.
Cache Location:
Data is cached to cache/1d_data/ using feather files for fast reload
"""

from datetime import datetime, timedelta
import os
import sys
import time
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import REPORT_DIR, ACCOUNT_ACTIVITY_DIR, PLOT_DPI

from Acquire.DailyData import daily_data
from Monitor.read_portfolio_positions import read_portfolio_positions

# Set up plotting style
plt.style.use('default')
sns.set_theme(style="whitegrid")

def get_top_positions_data(n: int = 10) -> Tuple[pd.DataFrame, float]:
    """Get top N positions by current value from your portfolio data."""
    df, _, _ = read_portfolio_positions()  # Unpack all 3 return values
    top_df = df.nlargest(n, 'Current Value').copy()
    
    # Calculate portfolio weights
    total_value = top_df['Current Value'].sum()
    top_df['Weight'] = top_df['Current Value'] / total_value
    
    print(f"\nTop {n} Positions by Current Value:")
    print(top_df[['Symbol', 'Current Value', 'Weight', 'Quantity']].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print(f"\nTotal Value: ${total_value:,.2f}")
    
    return top_df, total_value

def fetch_historical_data_for_positions(symbols: List[str], period: str = '6mo') -> Tuple[pd.DataFrame, List[str]]:
    """Fetch historical price data using daily_data with persistent caching for given symbols.
    Args:
        symbols: List of ticker symbols
        period: '6mo' for 6 months of data
    Returns:
        DataFrame with closing prices for all symbols
        List of failed symbols
    """
    print(f"\nFetching historical data for {len(symbols)} symbols ({period})...")
    
    # Calculate start date for 6 months
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=180)
    
    price_data = {}
    failed_symbols = []
    
    for symbol in symbols:
        success = False
        for attempt in range(3):  # 3 retry attempts
            try:
                # Fetch historical data using daily_data (with built-in caching)
                historical_data = daily_data(
                    ticker=symbol,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d')
                )
                
                if historical_data is not None and not historical_data.empty and 'Close' in historical_data.columns:
                    price_data[symbol] = historical_data['Close']
                    print(f"  OK {symbol}: {len(historical_data)} days of data")
                    success = True
                    break
                else:
                    print(f"  X {symbol}: No close price data (attempt {attempt + 1})")
                    
            except Exception as e:
                print(f"  X {symbol}: Error - {str(e)} (attempt {attempt + 1})")
                if attempt < 2:  # Don't sleep on last attempt
                    time.sleep(2 ** attempt)  # Exponential backoff: 2, 4 seconds
        
        if not success:
            failed_symbols.append(symbol)
    
    if not price_data:
        raise ValueError("No historical data could be fetched")
    
    # Combine all price data into a single DataFrame
    prices_df = pd.DataFrame(price_data)
    
    print(f"\nSuccessfully fetched data for {len(price_data)} symbols")
    if failed_symbols:
        print(f"Failed to fetch data for: {failed_symbols}")
    
    return prices_df, failed_symbols

def calculate_portfolio_value_and_gains(prices_df: pd.DataFrame, positions_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily portfolio value and dollar gains/losses.
    Args:
        prices_df: DataFrame of historical prices
        positions_df: DataFrame of positions with quantities
    Returns:
        DataFrame with daily portfolio values and gains
    """
    print("\nCalculating portfolio value and daily gains...")
    
    # Get quantities for each symbol
    quantities = positions_df.set_index('Symbol')['Quantity'].to_dict()
    
    # Calculate daily value for each position
    position_values = pd.DataFrame()
    
    for symbol in prices_df.columns:
        if symbol in quantities:
            position_values[symbol] = prices_df[symbol] * quantities[symbol]
    
    # Calculate total portfolio value
    portfolio_values = position_values.sum(axis=1)
    
    # Calculate daily gains/losses
    daily_gains = portfolio_values.diff()
    
    # Combine into results DataFrame
    results_df = pd.DataFrame({
        'Portfolio_Value': portfolio_values,
        'Daily_Gain_Loss': daily_gains,
        'Daily_Gain_Loss_Pct': portfolio_values.pct_change() * 100
    })
    
    # Add individual position values
    for symbol in position_values.columns:
        results_df[f'{symbol}_Value'] = position_values[symbol]
        results_df[f'{symbol}_Daily_Gain'] = position_values[symbol].diff()
    
    print(f"Portfolio value range: ${portfolio_values.min():,.2f} to ${portfolio_values.max():,.2f}")
    print(f"Best daily gain: ${daily_gains.max():,.2f}")
    print(f"Worst daily loss: ${daily_gains.min():,.2f}")
    
    return results_df

def create_portfolio_value_plot(results_df: pd.DataFrame, period: str = '6mo') -> str:
    """Create portfolio value over time plot.
    Side Effects:
        - Saves plot file to disk
    """
    print(f"\nCreating portfolio value plot for {period}...")
    
    plt.figure(figsize=(14, 8))
    
    # Plot portfolio value
    plt.plot(results_df.index, results_df['Portfolio_Value'], 
            label='Portfolio Value', color='black', linewidth=3, alpha=0.9)
    
    plt.title(f'Portfolio Value Over Time - {period.upper()} Analysis', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Format y-axis to show dollar values
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Add horizontal line for current value
    current_value = results_df['Portfolio_Value'].iloc[-1]
    plt.axhline(y=current_value, color='red', linestyle='--', alpha=0.7, label=f'Current: ${current_value:,.0f}')
    
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plot_file = f"{REPORT_DIR}/portfolio_value_{period}_{datetime.now().strftime('%Y%m%d')}.png"
    plt.savefig(plot_file, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  OK Saved portfolio value plot: {plot_file}")
    return plot_file

def create_daily_gains_plot(results_df: pd.DataFrame, period: str = '6mo') -> str:
    """Create daily gains/losses plot with bars and colors.
    Side Effects:
        - Saves plot file to disk
    """
    print(f"\nCreating daily gains/losses plot for {period}...")
    
    plt.figure(figsize=(14, 8))
    
    # Create color array for gains (green) and losses (red)
    colors = ['green' if x >= 0 else 'red' for x in results_df['Daily_Gain_Loss']]
    
    # Plot daily gains/losses as bars
    plt.bar(results_df.index, results_df['Daily_Gain_Loss'], 
            color=colors, alpha=0.7, width=1)
    
    plt.title(f'Daily Gains/Losses - {period.upper()} Analysis', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Daily Gain/Loss ($)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Format y-axis to show dollar values
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Add horizontal line at zero
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    # Add statistics text
    total_gain = results_df['Daily_Gain_Loss'].sum()
    avg_daily_gain = results_df['Daily_Gain_Loss'].mean()
    positive_days = (results_df['Daily_Gain_Loss'] > 0).sum()
    total_days = len(results_df['Daily_Gain_Loss'])
    
    stats_text = f'Total Gain: ${total_gain:,.0f}\nAvg Daily: ${avg_daily_gain:,.0f}\nWin Rate: {positive_days/total_days*100:.1f}%'
    plt.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plot_file = f"{REPORT_DIR}/daily_gains_{period}_{datetime.now().strftime('%Y%m%d')}.png"
    plt.savefig(plot_file, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  OK Saved daily gains plot: {plot_file}")
    return plot_file

def create_position_contribution_plot(results_df: pd.DataFrame, positions_df: pd.DataFrame, period: str = '6mo') -> str:
    """Create stacked area plot showing position contributions over time.
    Side Effects:
        - Saves plot file to disk
    """
    print(f"\nCreating position contribution plot for {period}...")
    
    plt.figure(figsize=(14, 8))
    
    # Get position value columns
    position_cols = [col for col in results_df.columns if col.endswith('_Value')]
    
    # Create stacked area plot
    plt.stackplot(results_df.index, 
                  [results_df[col] for col in position_cols],
                  labels=[col.replace('_Value', '') for col in position_cols],
                  alpha=0.7)
    
    plt.title(f'Position Contribution Over Time - {period.upper()} Analysis', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Position Value ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Format y-axis to show dollar values
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save plot
    plot_file = f"{REPORT_DIR}/position_contribution_{period}_{datetime.now().strftime('%Y%m%d')}.png"
    plt.savefig(plot_file, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  OK Saved position contribution plot: {plot_file}")
    return plot_file

def generate_value_analysis_report(results_df: pd.DataFrame, positions_df: pd.DataFrame, period: str = '6mo') -> str:
    """Generate a comprehensive analysis report.
    Side Effects:
        - Saves plot files to disk
        - Writes CSV report file to disk
    """
    print(f"\nGenerating value analysis report for {period}...")
    
    # Calculate statistics
    portfolio_values = results_df['Portfolio_Value']
    daily_gains = results_df['Daily_Gain_Loss']
    
    stats = {
        'Period': period.upper(),
        'Start_Date': results_df.index[0].strftime('%Y-%m-%d'),
        'End_Date': results_df.index[-1].strftime('%Y-%m-%d'),
        'Start_Value': f"${portfolio_values.iloc[0]:,.2f}",
        'End_Value': f"${portfolio_values.iloc[-1]:,.2f}",
        'Total_Gain_Loss': f"${portfolio_values.iloc[-1] - portfolio_values.iloc[0]:,.2f}",
        'Total_Percent_Change': f"{((portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1) * 100:.2f}%",
        'Best_Day': f"${daily_gains.max():,.2f}",
        'Worst_Day': f"${daily_gains.min():,.2f}",
        'Average_Daily_Gain': f"${daily_gains.mean():,.2f}",
        'Volatility_Std': f"${daily_gains.std():,.2f}",
        'Positive_Days': (daily_gains > 0).sum(),
        'Negative_Days': (daily_gains < 0).sum(),
        'Win_Rate': f"{(daily_gains > 0).sum() / len(daily_gains) * 100:.1f}%"
    }
    
    # Create position breakdown
    position_stats = []
    for _, position in positions_df.iterrows():
        symbol = position['Symbol']
        if f'{symbol}_Value' in results_df.columns:
            start_val = results_df[f'{symbol}_Value'].iloc[0]
            end_val = results_df[f'{symbol}_Value'].iloc[-1]
            gain_loss = end_val - start_val
            pct_change = (gain_loss / start_val) * 100 if start_val != 0 else 0
            
            position_stats.append({
                'Symbol': symbol,
                'Quantity': position['Quantity'],
                'Start_Value': f"${start_val:,.2f}",
                'End_Value': f"${end_val:,.2f}",
                'Gain_Loss': f"${gain_loss:,.2f}",
                'Percent_Change': f"{pct_change:.2f}%"
            })
    
    # Save report
    timestamp = datetime.now().strftime('%Y%m%d')
    report_file = f"{REPORT_DIR}/portfolio_value_report_{period}_{timestamp}.csv"
    
    # Create summary DataFrame
    summary_df = pd.DataFrame([stats])
    position_df = pd.DataFrame(position_stats)
    
    # Save both sections
    with open(report_file, 'w') as f:
        f.write("PORTFOLIO VALUE ANALYSIS SUMMARY\n")
        f.write("="*50 + "\n")
        summary_df.to_csv(f, index=False)
        f.write("\n\nPOSITION BREAKDOWN\n")
        f.write("="*50 + "\n")
        position_df.to_csv(f, index=False)
    
    print(f"  OK Saved value analysis report: {report_file}")
    
    # Print summary
    print(f"\n=== PORTFOLIO VALUE SUMMARY ({period.upper()}) ===")
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    return report_file

def main() -> None:
    """Main execution function."""
    print("=" * 60)
    print("PORTFOLIO VALUE AND GAINS ANALYSIS")
    print("=" * 60)
    
    # Get top 10 positions
    top_positions, total_value = get_top_positions_data(10)
    symbols = top_positions['Symbol'].tolist()
    
    try:
        # Fetch historical data
        prices_df, failed_symbols = fetch_historical_data_for_positions(symbols, '6mo')
        
        # Update positions to remove failed symbols
        successful_symbols = [s for s in symbols if s not in failed_symbols]
        top_positions_filtered = top_positions[top_positions['Symbol'].isin(successful_symbols)].copy()
        
        if len(successful_symbols) < len(symbols):
            print(f"Removed symbols: {failed_symbols}")
        
        # Calculate portfolio values and gains
        results_df = calculate_portfolio_value_and_gains(prices_df, top_positions_filtered)
        
        # Create visualizations
        value_plot = create_portfolio_value_plot(results_df, '6mo')
        gains_plot = create_daily_gains_plot(results_df, '6mo')
        contribution_plot = create_position_contribution_plot(results_df, top_positions_filtered, '6mo')
        
        # Generate report
        report_file = generate_value_analysis_report(results_df, top_positions_filtered, '6mo')
        
        print(f"\nOK 6-month value analysis completed successfully!")
        print(f"Generated 3 visualizations")
        print(f"Report saved to: {report_file}")
        print(f"\nGenerated files:")
        print(f"  - {value_plot}")
        print(f"  - {gains_plot}")
        print(f"  - {contribution_plot}")
        print(f"  - {report_file}")
        
    except Exception as e:
        print(f"\nX Error in analysis: {str(e)}")
        return
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
