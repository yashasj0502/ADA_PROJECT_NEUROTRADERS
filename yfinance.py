from __future__ import annotations
import pandas as pd
from pathlib import Path

_DATA_FILE = Path(__file__).resolve().parent / 'upload_DJIA_table.csv'


def _load_base_frame():
    if not _DATA_FILE.exists():
        raise FileNotFoundError(f"Cannot locate {_DATA_FILE}")
    df = pd.read_csv(_DATA_FILE, parse_dates=['Date'])
    df = df.sort_values('Date')
    df = df.set_index('Date')
    return df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()


def download(ticker: str, start: str | None = None, end: str | None = None, progress: bool = True, **kwargs):
    """Return historical DJIA data so the notebook can run offline."""
    df = _load_base_frame()
    trimmed = df
    if start is not None:
        start_ts = pd.to_datetime(start)
        trimmed = trimmed[trimmed.index >= start_ts]
    if end is not None:
        end_ts = pd.to_datetime(end)
        trimmed = trimmed[trimmed.index <= end_ts]
    if trimmed.empty:
        trimmed = df.copy()
    return trimmed.copy()
