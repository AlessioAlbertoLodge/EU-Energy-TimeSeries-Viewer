# src/heatmap_helpers.py
import numpy as np
import pandas as pd
from typing import Iterable, List, Tuple, Dict

from .derived import compute_residual_load

def compute_residual_ratio_long(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Returns a long-format DataFrame with columns:
      - utc_timestamp (datetime64[ns, UTC])
      - year  (int)
      - doy   (1..365/366)
      - hour  (0..23)
      - ratio (float in [0,1], residual/load, clipped; NaN when load<=0 or missing)

    ratio = residual_load / load_actual_entsoe_transparency
    """
    ts_col = "utc_timestamp"
    if ts_col not in df.columns:
        raise KeyError("Expected 'utc_timestamp' in DataFrame.")

    load_col = f"{prefix}_load_actual_entsoe_transparency"
    if load_col not in df.columns:
        raise KeyError(f"Missing required column: '{load_col}'")

    # Ensure residual is present (compute on the fly, don't mutate original df)
    local = df.copy()
    residual = compute_residual_load(local, prefix)
    load = local[load_col]

    # Safe ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = residual / load
    # Clip to [0,1] and blank impossible/ill-defined cases
    ratio = ratio.clip(lower=0, upper=1)
    ratio = ratio.where(load > 0)

    out = pd.DataFrame({
        "utc_timestamp": pd.to_datetime(local[ts_col], utc=True),
        "ratio": ratio
    }).dropna(subset=["utc_timestamp"])

    out["year"] = out["utc_timestamp"].dt.year.astype(int)
    out["doy"]  = out["utc_timestamp"].dt.dayofyear.astype(int)
    out["hour"] = out["utc_timestamp"].dt.hour.astype(int)
    return out[["utc_timestamp", "year", "doy", "hour", "ratio"]]

def years_available(long_df: pd.DataFrame) -> List[int]:
    """List years (sorted) present in the computed long-format residual ratio frame."""
    return sorted(long_df["year"].dropna().astype(int).unique().tolist())

def pivot_for_year(long_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Pivot a single year's data to a 24 x N_days matrix:
      rows   = hours 0..23  (we will show 00 at top in the plot by reversing y)
      cols   = day-of-year 1..365/366
      values = mean ratio (if multiple entries per hour/doy)
    """
    df_y = long_df.loc[long_df["year"] == year, ["doy", "hour", "ratio"]]
    if df_y.empty:
        # Create a blank 24 x 365 grid (fallback; leap ignored if not present)
        hours = np.arange(24)
        doys  = np.arange(1, 366)
        return pd.DataFrame(np.nan, index=hours, columns=doys)

    # Use mean if multiple points per (hour, doy)
    mat = df_y.groupby(["hour", "doy"], as_index=False)["ratio"].mean()
    # Full grid (respect leap years up to max found)
    max_doy = int(max(366, mat["doy"].max()))
    hours = np.arange(24)
    doys  = np.arange(1, max_doy + 1)

    wide = (
        mat.pivot(index="hour", columns="doy", values="ratio")
           .reindex(index=hours, columns=doys)
    )
    return wide

def make_green_red_colorscale() -> list:
    """
    Red (0) -> Green (1) colorscale for Plotly.
    We keep it simple and bright as requested.
    """
    return [
        [0.00, "#ff0000"],   # bright red
        [1.00, "#00b050"],   # Excel-like green (pleasant, readable)
    ]
