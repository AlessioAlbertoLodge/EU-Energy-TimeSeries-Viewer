# src/filters.py
import pandas as pd

def infer_time_bounds(df: pd.DataFrame):
    """
    Returns (min_t, max_t) from 'utc_timestamp', robust to NaNs and order.
    """
    if "utc_timestamp" not in df.columns:
        raise KeyError("Column 'utc_timestamp' not found for bounds inference.")
    ts = pd.to_datetime(df["utc_timestamp"], errors="coerce", utc=True)
    ts = ts.dropna()
    if ts.empty:
        raise ValueError("No valid UTC timestamps in dataset.")
    return ts.min().to_pydatetime(), ts.max().to_pydatetime()

def filter_by_time(df: pd.DataFrame, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.DataFrame:
    if start_utc.tz is None:  # enforce UTC
        start_utc = start_utc.tz_localize("UTC")
    if end_utc.tz is None:
        end_utc = end_utc.tz_localize("UTC")
    mask = (df["utc_timestamp"] >= start_utc) & (df["utc_timestamp"] <= end_utc + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    return df.loc[mask].copy()
