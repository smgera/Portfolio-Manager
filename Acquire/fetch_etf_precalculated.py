from datetime import datetime
import json
import time
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import requests
import yfinance as yf

from config import TICKER_LISTS_DIR, ETF_PRECALCULATED_CSV
from utils.io_utils import read_tsv

def fetch_precalculated_metrics(symbols: List[str]) -> pd.DataFrame:
    """Fetch pre-calculated metrics directly from financial data providers without processing historical time series.
    Side Effects:
        - Makes network requests via yfinance for each symbol
        - Prints progress messages to stdout
    """
    results = []
    
    print(f"Fetching pre-calculated metrics for {len(symbols)} symbols...")
    
    for i, symbol in enumerate(symbols, 1):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get pre-calculated statistics directly from Yahoo Finance
            # These are already calculated by Yahoo, not derived from raw data
            
            # Performance metrics (pre-calculated by Yahoo)
            ytd_return = info.get('ytdReturn', 0) * 100 if info.get('ytdReturn') else 0
            three_year_return = info.get('threeYearAverageReturn', 0) * 100 if info.get('threeYearAverageReturn') else 0
            five_year_return = info.get('fiveYearAverageReturn', 0) * 100 if info.get('fiveYearAverageReturn') else 0
            ten_year_return = info.get('tenYearAverageReturn', 0) * 100 if info.get('tenYearAverageReturn') else 0
            
            # Risk metrics (pre-calculated)
            beta = info.get('beta', 0)
            standard_deviation = info.get('standardDeviation', 0) * 100 if info.get('standardDeviation') else 0
            sharpe_ratio = info.get('threeYearSharpeRatio', 0) if info.get('threeYearSharpeRatio') else 0
            treynor_ratio = info.get('threeYearTreynorRatio', 0) if info.get('threeYearTreynorRatio') else 0
            
            # Fund metrics (pre-calculated)
            dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
            expense_ratio = info.get('expenseRatio', 0) * 100 if info.get('expenseRatio') else 0
            morningstar_rating = info.get('morningStarOverallRating', 0)
            
            # Price metrics (pre-calculated)
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            fifty_two_week_high = info.get('fiftyTwoWeekHigh', 0)
            fifty_two_week_low = info.get('fiftyTwoWeekLow', 0)
            
            # Calculate 52-week range percentage (simple calculation, not time series)
            if fifty_two_week_high > 0 and fifty_two_week_low > 0:
                range_percent = ((current_price - fifty_two_week_low) / (fifty_two_week_high - fifty_two_week_low)) * 100
            else:
                range_percent = 0
            
            # Fund information
            fund_category = info.get('category', 'N/A')
            fund_family = info.get('fundFamily', 'N/A')
            investment_style = info.get('investmentStyle', 'N/A')
            
            # Size and valuation metrics
            market_cap = info.get('marketCap', 0)
            average_volume = info.get('averageVolume', 0)
            assets_under_management = info.get('totalAssets', 0)
            
            # Sector/asset allocation (pre-calculated)
            try:
                sector_weighting = info.get('sectorWeightings', {})
                top_sector = max(sector_weighting.items(), key=lambda x: x[1])[0] if sector_weighting else 'N/A'
                top_sector_weight = sector_weighting.get(top_sector, 0) * 100 if sector_weighting else 0
            except (ValueError, TypeError, KeyError):
                top_sector = 'N/A'
                top_sector_weight = 0
            
            # Bond metrics if applicable
            effective_duration = info.get('effectiveDuration', 0)
            average_maturity = info.get('averageMaturity', 0)
            
            results.append({
                'Symbol': symbol,
                'Name': info.get('shortName', symbol)[:60],
                'Category': fund_category,
                'Fund Family': fund_family,
                'Investment_Style': investment_style,
                'Current_Price': round(current_price, 2) if current_price else 0,
                'YTD_Return (%)': round(ytd_return, 2),
                '3Y_Annual_Return (%)': round(three_year_return, 2),
                '5Y_Annual_Return (%)': round(five_year_return, 2),
                '10Y_Annual_Return (%)': round(ten_year_return, 2),
                'Sharpe_Ratio_3Y': round(sharpe_ratio, 3),
                'Beta': round(beta, 2),
                'Std_Dev (%)': round(standard_deviation, 2),
                'Treynor_Ratio': round(treynor_ratio, 3),
                'Dividend_Yield (%)': round(dividend_yield, 2),
                'Expense_Ratio (%)': round(expense_ratio, 2),
                'Morningstar_Rating': morningstar_rating,
                '52W_Range_Pos (%)': round(range_percent, 1),
                'Top_Sector': top_sector,
                'Top_Sector_Weight (%)': round(top_sector_weight, 1),
                'AUM': f"${assets_under_management/1e9:.1f}B" if assets_under_management else 'N/A',
                'Effective_Duration': round(effective_duration, 1) if effective_duration else 'N/A',
                'Avg_Maturity': round(average_maturity, 1) if average_maturity else 'N/A'
            })
            
            if i % 10 == 0:
                print(f"Processed {i}/{len(symbols)} symbols...")
            
            # Rate limiting to avoid being blocked
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error fetching {symbol}: {str(e)}")
            continue
    
    return pd.DataFrame(results)

def fetch_additional_precalculated(symbols: List[str]) -> None:
    """
    Fetch additional pre-calculated metrics from other sources
    """
    # Could add calls to:
    # - ETF.com API
    # - Morningstar API
    # - FactSet
    # - Bloomberg API
    # These would provide pre-calculated Hurst, alpha, information ratio, etc.
    pass

def main() -> None:
    """Fetch pre-calculated ETF metrics and save results to CSV.
    Side Effects:
        - Makes network API calls via yfinance
        - Writes results CSV file to disk
    """
    file_path = str(TICKER_LISTS_DIR / "bigETFs.tsv")
    
    # Read symbols from TSV file
    try:
        df = read_tsv(file_path)
        symbols = df['Symbol'].dropna().tolist()[1:]  # Skip header
        print(f"Loaded {len(symbols)} symbols from {file_path}")
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Fetch pre-calculated metrics
    results_df = fetch_precalculated_metrics(symbols[:100])  # Process first 100 for demo
    
    if len(results_df) > 0:
        print("\n" + "="*160)
        print("PRE-CALCULATED ETF METRICS (No Historical Time Series Processing)")
        print("="*160)
        
        # Configure display
        pd.set_option('display.max_rows', 30)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', lambda x: '%.2f' if isinstance(x, float) and abs(x) < 1000 else '%.0f' if isinstance(x, float) else '%s' % x)
        
        # Sort by different pre-calculated metrics
        print("\nTop ETFs by 3-Year Sharpe Ratio:")
        print("-"*160)
        df_sorted_sharpe = results_df.sort_values('Sharpe_Ratio_3Y', ascending=False)
        print(df_sorted_sharpe[['Symbol', 'Name', 'Sharpe_Ratio_3Y', '3Y_Annual_Return (%)', 'Std_Dev (%)', 'Beta']].head(20).to_string(index=False))
        
        print("\n\nTop ETFs by YTD Return:")
        print("-"*160)
        df_sorted_ytd = results_df.sort_values('YTD_Return (%)', ascending=False)
        print(df_sorted_ytd[['Symbol', 'Name', 'YTD_Return (%)', '3Y_Annual_Return (%)', 'Category']].head(20).to_string(index=False))
        
        print("\n\nTop ETFs by Dividend Yield:")
        print("-"*160)
        df_sorted_div = results_df.sort_values('Dividend_Yield (%)', ascending=False)
        print(df_sorted_div[['Symbol', 'Name', 'Dividend_Yield (%)', 'Expense_Ratio (%)', 'Category']].head(20).to_string(index=False))
        
        print("\n\nLowest Beta ETFs:")
        print("-"*160)
        df_sorted_beta = results_df.sort_values('Beta', ascending=True)
        print(df_sorted_beta[['Symbol', 'Name', 'Beta', 'Std_Dev (%)', 'Category']].head(20).to_string(index=False))
        
        # Save complete results
        output_file = ETF_PRECALCULATED_CSV
        results_df.to_csv(output_file, index=False)
        print(f"\n\nComplete results saved to: {output_file}")
        
        # Summary statistics
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        print(f"Total ETFs analyzed: {len(results_df)}")
        print(f"Average YTD return: {results_df['YTD_Return (%)'].mean():.2f}%")
        print(f"Average 3Y annual return: {results_df['3Y_Annual_Return (%)'].mean():.2f}%")
        print(f"Average Sharpe ratio: {results_df['Sharpe_Ratio_3Y'].mean():.3f}")
        print(f"Average beta: {results_df['Beta'].mean():.2f}")
        print(f"Average expense ratio: {results_df['Expense_Ratio (%)'].mean():.2f}%")
        
        # Note about Hurst exponent
        print("\n" + "="*50)
        print("NOTE ON HURST EXPONENT")
        print("="*50)
        print("Hurst exponent is not typically provided as a pre-calculated metric by standard data providers.")
        print("To get Hurst values, you would need:")
        print("1. A specialized analytics provider (e.g., QuantConnect, AlphaSense)")
        print("2. Calculate it yourself using historical price data")
        print("3. Use a service like Portfolio Visualizer or Wolfram Alpha")
        
    else:
        print("No results to display.")

if __name__ == "__main__":
    main()
