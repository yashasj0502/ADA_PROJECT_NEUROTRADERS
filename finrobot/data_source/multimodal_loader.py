# finrobot/data_source/multimodal_loader.py
"""
Return synchronized datasets aligned by timestamp.
- numeric_df: market timeseries (timestamp, open, high, low, close, volume)
- text_df: articles or Reddit posts (timestamp, text)
"""

import pandas as pd
import numpy as np
from typing import Tuple

def ensure_ts(df: pd.DataFrame, ts_col="timestamp") -> pd.DataFrame:
    df = df.copy()
    if ts_col not in df.columns:
        raise ValueError(f"{ts_col} column missing in dataframe")
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    return df

def aggregate_text_features(text_df: pd.DataFrame,
                            window: str = "1h") -> pd.DataFrame:
    """
    Group text rows into hourly (or chosen window) bins.
    """
    df = ensure_ts(text_df, "timestamp")
    df = df.set_index("timestamp")
    # combine all texts in the same time bin
    out = df.resample(window).agg({
        "text": lambda x: " ".join(x.astype(str))
    })
    out = out.rename_axis("timestamp").reset_index()
    return out

def align_numeric_and_text(numeric_df: pd.DataFrame,
                           text_df: pd.DataFrame,
                           window: str = "1h",
                           on_ts: str = "timestamp") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Resample numeric data and align with aggregated text data.
    Returns (numeric_df_resampled, text_df_aggregated)
    """
    num = numeric_df.copy()
    num[on_ts] = pd.to_datetime(num[on_ts], utc=True)
    num = num.set_index(on_ts).sort_index()

    # Resample numeric to same window
    num_resampled = num.resample(window).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna().ffill()

    # Aggregate text
    txt_agg = aggregate_text_features(text_df, window=window)
    txt_agg = txt_agg.set_index("timestamp")

    # Reindex text to match numeric timestamps
    txt_agg = txt_agg.reindex(num_resampled.index).reset_index()

    return num_resampled.reset_index().rename(columns={"index": "timestamp"}), txt_agg
