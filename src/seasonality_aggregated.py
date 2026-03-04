# src/seasonality_aggregated.py
import pandas as pd
from typing import Dict

from .derived import compute_residual_load
from .seasonality import _resolve_solar_col, _resolve_wind_series


def _ensure_ts(df: pd.DataFrame) -> pd.Series:
    """Return parsed utc_timestamp as UTC datetime."""
    if "utc_timestamp" not in df.columns:
        raise KeyError("Expected 'utc_timestamp' in DataFrame.")
    return pd.to_datetime(df["utc_timestamp"], errors="coerce", utc=True)


def _get_series(df: pd.DataFrame, prefix: str, source: str) -> pd.Series:
    """Return a numeric Series for solar, wind, or residual load."""
    if source == "solar":
        col = _resolve_solar_col(df, prefix)
        if col is None:
            raise KeyError(f"Missing solar column for {prefix}.")
        return pd.to_numeric(df[col], errors="coerce")
    elif source == "wind":
        s = _resolve_wind_series(df, prefix)
        if s is None:
            raise KeyError(f"Missing wind columns for {prefix}.")
        return pd.to_numeric(s, errors="coerce")
    elif source == "residual":
        residual = compute_residual_load(df, prefix)
        return pd.to_numeric(residual, errors="coerce")
    else:
        raise ValueError("source must be 'solar', 'wind', or 'residual'.")


def aggregate_energy(df: pd.DataFrame, prefix: str, source: str, freq: str) -> pd.DataFrame:
    """
    Aggregate energy by time frequency (1H assumed input resolution).
    freq = 'W' (weekly) or 'D' (daily)

    Returns: DataFrame with columns ['year', 'period', 'energy_MWh'] where
      - for 'W', 'period' = ISO week number (1..53)
      - for 'D', 'period' = day-of-year (1..365/366)
      - energy_MWh = sum(MW) over the period (1-hour timestep => MWh)
    """
    ts = _ensure_ts(df)
    vals = _get_series(df, prefix, source)
    base = pd.DataFrame({"utc_timestamp": ts, "value": vals}).dropna()

    if base.empty:
        return pd.DataFrame(columns=["year", "period", "energy_MWh"])

    base["year"] = base["utc_timestamp"].dt.year.astype(int)
    if freq == "W":
        base["period"] = base["utc_timestamp"].dt.isocalendar().week.astype(int)
    elif freq == "D":
        base["period"] = base["utc_timestamp"].dt.dayofyear.astype(int)
    else:
        raise ValueError("freq must be 'W' or 'D'.")

    grouped = base.groupby(["year", "period"], as_index=False)["value"].sum(min_count=1)
    grouped.rename(columns={"value": "energy_MWh"}, inplace=True)
    return grouped.sort_values(["year", "period"])


def normalize_by_year_mean(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'energy_norm' = energy_MWh / (yearly average of energy_MWh over periods).
    The average is computed separately per year.
    """
    if agg_df.empty:
        return agg_df.assign(energy_norm=pd.Series(dtype=float))
    means = agg_df.groupby("year")["energy_MWh"].mean()
    out = agg_df.copy()
    out["energy_norm"] = out.apply(lambda r: (r["energy_MWh"] / means.loc[r["year"]]) if means.loc[r["year"]] and pd.notna(means.loc[r["year"]]) else pd.NA, axis=1)
    return out


def pivot_energy(agg_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Return {year: DataFrame(period, energy_MWh)} for plotting."""
    result = {}
    for y, dfy in agg_df.groupby("year"):
        result[int(y)] = dfy[["period", "energy_MWh"]]
    return result


def pivot_energy_norm(agg_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Return {year: DataFrame(period, energy_norm)} for plotting."""
    result = {}
    for y, dfy in agg_df.groupby("year"):
        result[int(y)] = dfy[["period", "energy_norm"]]
    return result


def make_month_ticks(freq: str) -> Dict[int, str]:
    """Approximate mapping from period number to month labels for x-axis ticks."""
    if freq == "W":
        return {
            2: "Jan", 6: "Feb", 10: "Mar", 15: "Apr", 19: "May", 24: "Jun",
            28: "Jul", 33: "Aug", 37: "Sep", 42: "Oct", 46: "Nov", 51: "Dec",
        }
    elif freq == "D":
        return {
            15: "Jan", 46: "Feb", 75: "Mar", 105: "Apr", 135: "May", 166: "Jun",
            196: "Jul", 227: "Aug", 258: "Sep", 288: "Oct", 319: "Nov", 350: "Dec",
        }
    else:
        return {}
