from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


def read_with_pandas(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".feather":
        return pd.read_feather(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".tsv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported file type: {path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Read all data files in a directory and print DataFrames")
    parser.add_argument(
        "directory",
        nargs="?",
        default="C:/Users/dad/Documents/GitHub/gerads/Trading/cache/1d_data",
        help="Directory containing data files (default: cache/1d_data path)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Print full DataFrame content (default prints shape, columns, and head)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of rows to preview when not printing all (default: 10)",
    )
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.exists() or not directory.is_dir():
        print(f"Directory not found or not a directory: {directory}", file=sys.stderr)
        return 1

    # Supported file extensions in priority we expect to see
    exts = {".feather", ".parquet", ".tsv", ".tsv"}

    files = sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if not files:
        print(f"No supported files found in {directory}")
        return 0

    for fp in files:
        print("\n" + "=" * 80)
        print(f"File: {fp.name}")
        try:
            df = read_with_pandas(fp)
        except Exception as e:
            print(f"Error reading {fp.name}: {e}", file=sys.stderr)
            continue

        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(df.index)
        if args.all:
            # Print full DataFrame; beware of very large outputs
            with pd.option_context(
                "display.max_rows", None,
                "display.max_columns", None,
                "display.width", 0,
                "display.max_colwidth", None,
            ):
                print(df)
        else:
            print(df.head(args.limit))
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
