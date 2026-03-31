import pandas as pd
from DailyData import daily_data

# Read the tickers from the TSV file
tickers_df = pd.read_csv('tickerLists/myTickers.tsv', sep='\t')

# Loop through each ticker and print the latest close price
for _, row in tickers_df.iterrows():
    ticker = row['Symbol']
    data = daily_data(ticker)
    latest_close = data['Close'].iloc[-1]
    print(f"Ticker: {ticker}, Latest Close: {latest_close}, {data.tail(1).index[0].strftime('%Y-%m-%d')}")