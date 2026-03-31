"""I/O utility functions for reading data files."""
from typing import Any

import pandas as pd


def read_tsv(path: Any, **kwargs: Any) -> pd.DataFrame:
    """Read a TSV/CSV file and strip whitespace from all string columns.
    Defaults to tab separator for TSV files; pass sep=',' for CSV files.
    Args:
        path: File path or file-like object passed to pd.read_csv.
        **kwargs: Additional arguments forwarded to pd.read_csv.
            sep defaults to '\\t' if not provided.
    Returns:
        DataFrame with all string columns stripped of leading/trailing whitespace.
    Side Effects:
        - Reads file from disk (or file-like object)
    """
    kwargs.setdefault('sep', '\t')
    df = pd.read_csv(path, **kwargs)
    return df.apply(lambda col: col.str.strip() if col.dtype == object else col)
