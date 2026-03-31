import os

import pandas as pd

from config import REPORT_DIR
from utils.io_utils import read_tsv

def compare_reports() -> None:
    """Compare two portfolio reports and display differences."""
    old_file = REPORT_DIR / "withoutReinvest_qty_comparison_20251129.csv"
    new_file = REPORT_DIR / "qty_comparison_20251129.csv"
    
    try:
        df_old = read_tsv(old_file, sep=',')
        df_new = read_tsv(new_file, sep=',')
    except FileNotFoundError as e:
        print(f"Error reading files: {e}")
        return

    old_mismatches = df_old[df_old['Status'] != 'Match']
    new_mismatches = df_new[df_new['Status'] != 'Match']
    
    print(f"Old Mismatches (without Reinvest): {len(old_mismatches)}")
    print(f"New Mismatches (with Reinvest):    {len(new_mismatches)}")
    
    # Analyze improvement
    improved = len(old_mismatches) - len(new_mismatches)
    print(f"Improvement: {improved} fewer mismatches.")
    
    # Find symbols that were fixed
    # Key: (Snapshot Date, Symbol)
    old_keys = set(zip(old_mismatches['Snapshot Date'], old_mismatches['Symbol']))
    new_keys = set(zip(new_mismatches['Snapshot Date'], new_mismatches['Symbol']))
    
    fixed_keys = old_keys - new_keys
    print(f"Unique Position Comparisons fixed: {len(fixed_keys)}")
    
    if fixed_keys:
        print("\nSample of fixed discrepancies:")
        for i, key in enumerate(list(fixed_keys)[:5]):
            date, sym = key
            old_row = old_mismatches[(old_mismatches['Snapshot Date'] == date) & (old_mismatches['Symbol'] == sym)].iloc[0]
            print(f"  {sym} on {date}: Old Diff {old_row['Difference']:.2f}")

if __name__ == "__main__":
    compare_reports()
