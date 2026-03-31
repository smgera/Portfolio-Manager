"""Portfolio utility functions shared across subsystems."""
import re

import numpy as np
import pandas as pd

from config import MONEY_MARKET_FUNDS
from common.symbols_from_description import extract_symbol_from_description

_OPTION_RE = re.compile(r'^[\s\-]*([A-Za-z]+)(\d{6})([CP])(\d+(?:\.\d+)?)$')


def _normalize_option_symbol(sym):
    """Normalize an options ticker to standard OCC-like format."""
    if pd.isna(sym):
        return sym
    if not isinstance(sym, str):
        raise ValueError(f"Invalid symbol: {sym}")
    m = _OPTION_RE.match(sym)
    if not m:
        return sym
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
    Side Effects:
        - Prints money market fund holdings summary to stdout
    """
    if df.empty:
        return df

    df = df.copy()

    df["Symbol"] = df["Symbol"].apply(lambda x: extract_symbol_from_description(x) if pd.notna(x) else x)
    df["Symbol"] = df["Symbol"].apply(_normalize_option_symbol)

    symbol_map = {"BRKB": "BRK-B"}
    df["Symbol"] = df["Symbol"].replace(symbol_map)

    if data_type == 'positions' and 'Description' in df.columns:
        df.loc[df['Description'] == 'S&P 500 EQUITY INDEX', 'Symbol'] = 'SPY'

    df = df[df['Symbol'].notna() & (df['Symbol'] != '')]

    if 'Current Value' in df.columns:
        df['Current Value'] = df['Current Value'].astype(str).replace(r'[\$,]', '', regex=True)
        df['Current Value'] = pd.to_numeric(df['Current Value'], errors='coerce')
        if data_type == 'positions':
            df = df[df['Current Value'].notna()]

    print('Money Market Funds may not be available to trade, may be needed for settlements:')
    print(df[df['Symbol'].isin(MONEY_MARKET_FUNDS)][['Symbol', 'Current Value']].groupby('Symbol', as_index=False).agg({'Current Value': 'sum'}), '\n')
    df = df[~df['Symbol'].isin(MONEY_MARKET_FUNDS)]

    if 'Last Price' in df.columns:
        df['Last Price'] = df['Last Price'].astype(str).replace(r'[\$,]', '', regex=True)
        df['Last Price'] = pd.to_numeric(df['Last Price'], errors='coerce')
        if data_type == 'positions':
            df.loc[(df['Last Price'].isna()) | (df['Last Price'] == 1.00), 'Symbol'] = '**'

    if 'Quantity' not in df.columns:
        if 'Current Value' in df.columns and 'Last Price' in df.columns:
            df['Quantity'] = df['Current Value'] / df['Last Price']
        else:
            df['Quantity'] = np.nan
    else:
        df['Quantity'] = df['Quantity'].astype(str).replace(r'[\$,]', '', regex=True)
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')

    df = df[~df['Symbol'].isin(MONEY_MARKET_FUNDS)]

    return df
