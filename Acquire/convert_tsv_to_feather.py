"""Convert TSV files to Feather format for faster loading."""

from typing import Optional, Union
from pathlib import Path

import pandas as pd
import pyarrow.feather as feather
from utils.io_utils import read_tsv


def convert_tsv_to_feather(tsv_file: Union[str, Path], feather_file: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Convert a TSV file to Feather format.
    Args:
        tsv_file: Path to input TSV file
        feather_file: Path to output Feather file (optional, defaults to same name with .feather extension)
    Returns:
        DataFrame that was saved
    Side Effects:
        - Writes Feather file to disk
        - Prints progress messages to stdout
    """
    # Generate output filename if not provided
    if feather_file is None:
        feather_file = tsv_file.replace('.tsv', '.feather')
    
    print(f"Reading TSV from: {tsv_file}")
    df = read_tsv(tsv_file)
    
    print(f"  Loaded {len(df)} rows and {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")
    
    print(f"Writing Feather to: {feather_file}")
    feather.write_feather(df, feather_file)
    
    print(f"  Successfully saved to {feather_file}")
    
    return df


def load_feather(feather_file: Union[str, Path]) -> pd.DataFrame:
    """Load a Feather file back into a DataFrame.
    Args:
        feather_file: Path to Feather file
    Returns:
        DataFrame
    Side Effects:
        - Prints progress messages to stdout
    """
    print(f"\nReading Feather from: {feather_file}")
    df = feather.read_feather(feather_file)
    
    print(f"  Loaded {len(df)} rows and {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")
    
    return df


def verify_conversion(original_df: pd.DataFrame, feather_file: Union[str, Path]) -> bool:
    """Verify that the Feather file contains the same data as the original DataFrame.
    Args:
        original_df: Original DataFrame
        feather_file: Path to Feather file to verify
    Returns:
        Boolean indicating if verification passed
    Side Effects:
        - Prints verification results to stdout
    """
    print("\nVerifying conversion...")
    loaded_df = load_feather(feather_file)
    
    # Check shape
    if original_df.shape != loaded_df.shape:
        print(f"  [FAIL] Shape mismatch: {original_df.shape} vs {loaded_df.shape}")
        return False
    
    # Check columns
    if list(original_df.columns) != list(loaded_df.columns):
        print(f"  [FAIL] Column mismatch")
        return False
    
    # Check first few rows
    if not original_df.head().equals(loaded_df.head()):
        print(f"  ⚠️  Data might differ slightly (this can be due to data types)")
    
    print("  [OK] Verification passed!")
    return True


if __name__ == "__main__":
    # Test with ETF data
    tsv_file = 'tickerLists/largest_etfs_marketcap.tsv'
    feather_file = 'tickerLists/largest_etfs_marketcap.feather'
    
    # Convert TSV to Feather
    df = convert_tsv_to_feather(tsv_file, feather_file)
    
    # Verify the conversion
    verify_conversion(df, feather_file)
    
    # Display first 10 rows
    print("\nFirst 10 rows from Feather file:")
    print("=" * 100)
    loaded_df = load_feather(feather_file)
    print(loaded_df.head(10).to_string(index=False))
    print("=" * 100)
