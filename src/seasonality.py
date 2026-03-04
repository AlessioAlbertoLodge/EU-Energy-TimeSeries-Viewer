# src/seasonality.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

SOLAR_SUFFIX = "solar_generation_actual"
WIND_TOTAL_SUFFIX = "wind_generation_actual"
WIND_ON_SUFFIX = "wind_onshore_generation_actual"
WIND_OFF_SUFFIX = "wind_offshore_generation_actual"

SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]
MONTH_TO_SEASON = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}

def _resolve_solar_col(df: pd.DataFrame, prefix: str) -> str | None:
    col = f"{prefix}_{SOLAR_SUFFIX}"
    return col if col in df.columns else None

def _resolve_wind_series(df: pd.DataFrame, prefix: str) -> pd.Series | None:
    """
    Returns a Series for total wind:
      - If total wind exists: use it.
      - Else: sum available onshore/offshore.
      - If nothing available: None.
    """
    p = f"{prefix}_"
    tot = p + WIND_TOTAL_SUFFIX
    on  = p + WIND_ON_SUFFIX
    off = p + WIND_OFF_SUFFIX

    if tot in df.columns:
        return df[tot]
    parts = []
    if on in df.columns:  parts.append(df[on])
    if off in df.columns: parts.append(df[off])
    if parts:
        return sum([s.fillna(0) for s in parts])
    return None

def _season_from_month(m: int) -> str:
    return MONTH_TO_SEASON.get(int(m), "Unknown")

def years_available(df: pd.DataFrame) -> List[int]:
    ts = pd.to_datetime(df["utc_timestamp"], errors="coerce", utc=True)
    return sorted(ts.dt.year.dropna().astype(int).unique().tolist())

def build_long_by_source(
    df: pd.DataFrame,
    prefix: str,
    source: str,  # "solar" or "wind"
) -> pd.DataFrame:
    """
    Returns long-format records with:
      date (UTC date), year, season, hour (0..23), value (power)
    Aggregates duplicates within hour as mean (dataset is 60-min already).
    """
    if "utc_timestamp" not in df.columns:
        raise KeyError("Expected 'utc_timestamp' in DataFrame.")
    ts = pd.to_datetime(df["utc_timestamp"], errors="coerce", utc=True)

    if source == "solar":
        scol = _resolve_solar_col(df, prefix)
        if scol is None:
            raise KeyError(f"Missing solar column for {prefix}: '{prefix}_{SOLAR_SUFFIX}'")
        values = df[scol]
    elif source == "wind":
        w = _resolve_wind_series(df, prefix)
        if w is None:
            raise KeyError(
                f"Missing wind columns for {prefix}: try '{prefix}_{WIND_TOTAL_SUFFIX}' "
                f"or sum of '{prefix}_{WIND_ON_SUFFIX}' and '{prefix}_{WIND_OFF_SUFFIX}'."
            )
        values = w
    else:
        raise ValueError("source must be 'solar' or 'wind'.")

    base = pd.DataFrame({"utc_timestamp": ts, "value": pd.to_numeric(values, errors="coerce")}).dropna()
    if base.empty:
        return base.assign(date=pd.NaT, year=np.nan, season="", hour=np.nan)

    base["date"]  = base["utc_timestamp"].dt.date
    base["year"]  = base["utc_timestamp"].dt.year.astype(int)
    base["month"] = base["utc_timestamp"].dt.month.astype(int)
    base["hour"]  = base["utc_timestamp"].dt.hour.astype(int)
    base["season"] = base["month"].map(_season_from_month)

    # One value per (date, hour)
    agg = (
        base.groupby(["date", "year", "season", "hour"], as_index=False)["value"]
            .mean()
            .sort_values(["year", "season", "date", "hour"])
    )
    return agg

def split_by_year_and_season(long_df: pd.DataFrame) -> Dict[int, Dict[str, pd.DataFrame]]:
    """
    Returns: { year: { season: df_for_that_season } }
    where each df has rows per day and columns: 'date','hour','value'
    """
    out: Dict[int, Dict[str, pd.DataFrame]] = {}
    for y, dfy in long_df.groupby("year"):
        out[y] = {}
        for s in SEASON_ORDER:
            dfs = dfy[dfy["season"] == s].copy()
            out[y][s] = dfs
    return out
