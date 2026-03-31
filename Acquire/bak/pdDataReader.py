from pandas_datareader import data as pdr
import pandas as pd

# Initializing Parameters
start = "2007-01-01"
end = "2025-02-22"
symbols = ["AAPL"]

# Getting the data
data = pdr.get_data_yahoo(symbols, start, end)

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Save the data to a CSV file
df.to_csv('AAPL_history.tsv', sep='\t', index=False)