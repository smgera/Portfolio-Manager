from datetime import datetime
import os
import sys
from typing import Union
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf

from config import REPORT_DIR, MIN_SHARPE_FILTER
from common.financial_utils import calculate_sharpe_ratio, annualize_return, annualize_volatility

from Acquire.DailyData import daily_data
from Monitor.read_portfolio_positions import read_portfolio_positions

pd.set_option('display.max_rows', 200)


def plot_seaborn_grid(result_df: pd.DataFrame, filename: Union[str, Path], by: str = 'Sharpe') -> None:
    """Plot a seaborn heatmap of the result_df, color scaling only the Sharpe column, and saves to file.
    Side Effects:
        - Saves plot file to disk
    """
    cols_to_show = ['Gain%', 'dStdev%', 'Sharpe', '$Value', 'Value%', '$1moRisk']
    data_for_heatmap = result_df.set_index('Symbol')[cols_to_show].copy()
    # Set all columns except Sharpe to a constant (e.g., the min Sharpe value) for color mapping
    sharpe_min = data_for_heatmap['Sharpe'].min()
    for col in data_for_heatmap.columns:
        if col != 'Sharpe':
            data_for_heatmap[col] = 0.3
    # Make rows tighter and move x-axis label above chart
    row_height = 0.28  # smaller for tighter rows
    fig_height = max(3, len(result_df) * row_height + 1)
    plt.figure(figsize=(10, fig_height))
    sns.set(font_scale=0.8)
    ax = sns.heatmap(
        data_for_heatmap,
        annot=result_df.set_index('Symbol')[cols_to_show].round(2),
        fmt=".2f",
        cmap="RdYlGn", # red is low, green is high
        cbar_kws={'label': 'Sharpe Ratio'},
        linewidths=.5,
        center=0.3,
        vmin=0.1, #sharpe_min,
        vmax=1 #data_for_heatmap['Sharpe'].max()
    )
    plt.title(f'Portfolio Performance by {by} (Sharpe color scaled)')
    plt.yticks(rotation=0)  # Make row labels horizontal
    # Move column labels (x-axis tick labels) to top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Metric', labelpad=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    imagefile = filename.replace('.tsv', f'.png')
    plt.savefig(imagefile)
    print(f"Heatmap saved to {imagefile}\n")


def main() -> None:
    """Run portfolio performance analysis and generate report.
    Side Effects: Saves CSV and plot files to disk."""
    date_str = datetime.now().strftime('%Y-%m-%d')

    df, newest_positions_file, timestamp = read_portfolio_positions()

    # Add column using yfinance
    # print(dir(yf.Ticker('SPY'))) # methods and properties

    tPE = []
    fPE = []
    beta = []
    for symbol in df['Symbol']:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            tPE.append(info.get('trailingPE', np.nan))
            fPE.append(info.get('forwardPE', np.nan))
            beta.append(info.get('beta', np.nan))
        except Exception:
            tPE.append(np.nan)
            fPE.append(np.nan)
            beta.append(np.nan)
    df['tPE'] = tPE
    df['fPE'] = fPE
    df['Beta'] = beta

    # save df to csv, tab separated
    ticker_list_path = Path(__file__).parent.parent / 'tickerLists' / 'myTickers.tsv'
    df.to_csv(ticker_list_path, index=False, sep='\t')
    print('saved myTickers.tsv')

    print(df[['Symbol', 'Current Value', 'Beta', 'tPE', 'fPE']].to_string(index=False, float_format=lambda x: f"{x:7.2f}"))

    # for each period
    for period in ['2mo', '1mo']:
        results = []
        for symbol, value in zip(df['Symbol'], df['Current Value']):
            # print(f"Processing {symbol}...")
            data = daily_data(symbol, period=period) # get daily data for symbol and period
            if data is not None and not data.empty and len(data)>1 and 'Close' in data.columns:
                prices = data['Close']
                if len(prices) > 1:
                    gain_pct = (prices.iloc[-1] / prices.iloc[0] - 1) * 100 # period gain
                    yrGain_pct = annualize_return(gain_pct / 100, len(data)) * 100
                    dStdev_pct = prices.pct_change().std() * 100 # daily stdev
                    yrStdev_pct = annualize_volatility(dStdev_pct)
                    moStdev_pct = (dStdev_pct * np.sqrt(21))
                    sharpe = calculate_sharpe_ratio(yrGain_pct / 100, yrStdev_pct / 100)
                    results.append({'Symbol': symbol, period+'Gain%': gain_pct, 'yrGain%': yrGain_pct,
                                    'dayStdev%': dStdev_pct, 'yrStdev%': yrStdev_pct, 'Sharpe': sharpe,
                                    '$Value': value, '1moRisk%': moStdev_pct, '$1moRisk': value * moStdev_pct/100,
                                    'Price': prices.iloc[-1]})

        # make dataframe
        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values(by='Sharpe', ascending=False)
        print(result_df)

        total_value = result_df['$Value'].sum() # investing capital, pg 93
        total_1moRisk = result_df['$1moRisk'].sum()
        print(f"Total Value: ${total_value:,.2f}")
        print(f"Total $1moRisk: ${total_1moRisk:,.2f}")
        print(f"Total 1moRisk%: {total_1moRisk/total_value*100:.2f}% of Current Value")

        # print symbols of and remove rows with negative Sharpe
        print("Removing symbols with low Sharpe:")
        print(result_df[result_df['Sharpe'] <= 0]['Symbol'].to_string(index=False))
        result_df = result_df[result_df['Sharpe'] > MIN_SHARPE_FILTER]
        # pause
        input("Press Enter to continue...")

        result_df['optimalLever'] = result_df['Sharpe'] / result_df['yrStdev%'] # formula 5, Kelly criterion
        result_df['prudentRisk'] = result_df['Sharpe'] / 3 # formula 18,  if Sharpe 3, risk 100% of capital /yr. To risk 100% without leverage, you need 100% stdev
        result_df['nomTargNotion'] = total_value * result_df['prudentRisk'] / result_df['yrStdev%'] # formula 1 and 14

        result_df['targNotion%'] = result_df['nomTargNotion'] / result_df['nomTargNotion'].sum() * 100
        result_df['targPosition$'] = result_df['targNotion%'] * total_value / 100
        result_df['targ1moRisk'] = result_df['targPosition$'] * result_df['1moRisk%'] / 100
        result_df['GainSharpe'] = result_df[period+'Gain%'] * result_df['Sharpe']

        #  sort, display, and write to csv
        # add Diff$ column
        result_df['Diff$'] = result_df['targPosition$'] - result_df['$Value']
        by = 'Diff$'
        print(f"\nSorting by {by}")
        result_df = result_df.sort_values(by=by, ascending=False)
        # reorder columns to put targPosition$ as 2nd column, $Value as 3rd, and Diff$ as 4th
        cols = result_df.columns.tolist()
        if 'targPosition$' in cols:
            cols.remove('targPosition$')
            cols.insert(1, 'targPosition$')
        if '$Value' in cols:
            cols.remove('$Value')
            cols.insert(2, '$Value')
        if 'Diff$' in cols:
            cols.remove('Diff$')
            cols.insert(3, 'Diff$')
        result_df = result_df[cols]
        print(result_df.to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
        filename = os.path.basename(newest_positions_file)
        filename = REPORT_DIR / filename
        filename = str(filename).replace('.csv', f'_{period}_by_{by}.tsv')
        result_df = result_df.round(2)
        result_df.to_csv(filename, sep='\t', index=False)
        print(f"Results written to {filename}\n")

        print(f"Position Qty: {len(result_df):,.0f}")
        print(f"Total new positions $: ${result_df['targPosition$'].sum():,.2f}")
        print(f"Total new risk $: ${result_df['targ1moRisk'].sum():,.2f}")
        print(f'% of Current Value: {result_df["targ1moRisk"].sum()/total_value*100:.2f}%')
        print()


if __name__ == '__main__':
    main()
