# src/dunkelflaute.py
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Column suffixes
LOAD_SUFFIX = "load_actual_entsoe_transparency"
PRICE_SUFFIX = "price_day_ahead"
SOLAR_SUFFIX = "solar_generation_actual"
WIND_ON_SUFFIX = "wind_onshore_generation_actual"
WIND_OFF_SUFFIX = "wind_offshore_generation_actual"
WIND_TOTAL_SUFFIX = "wind_generation_actual"

@dataclass
class DunkelflauteEvent:
    start: pd.Timestamp
    end: pd.Timestamp
    duration_hours: int
    start_idx: int
    end_idx: int

# ---------- column helpers ----------

def _ensure_ts(df: pd.DataFrame) -> pd.Series:
    if "utc_timestamp" not in df.columns:
        raise KeyError("Expected 'utc_timestamp' in DataFrame.")
    return pd.to_datetime(df["utc_timestamp"], errors="coerce", utc=True)

def _col(df: pd.DataFrame, prefix: str, suffix: str) -> Optional[str]:
    name = f"{prefix}_{suffix}"
    return name if name in df.columns else None

def _load_column(df: pd.DataFrame, prefix: str) -> str:
    c = _col(df, prefix, LOAD_SUFFIX)
    if c is None:
        raise KeyError(f"Missing required load column '{prefix}_{LOAD_SUFFIX}'")
    return c

def _price_column(df: pd.DataFrame, prefix: str) -> Optional[str]:
    return _col(df, prefix, PRICE_SUFFIX)

def _solar_series(df: pd.DataFrame, prefix: str) -> pd.Series:
    c = _col(df, prefix, SOLAR_SUFFIX)
    if c is None:
        return pd.Series(0.0, index=df.index, dtype="float64")
    return pd.to_numeric(df[c], errors="coerce").fillna(0.0)

def _wind_series(df: pd.DataFrame, prefix: str, use_total_wind: bool) -> pd.Series:
    if use_total_wind:
        c = _col(df, prefix, WIND_TOTAL_SUFFIX)
        if c is None:
            return pd.Series(0.0, index=df.index, dtype="float64")
        return pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    c_on = _col(df, prefix, WIND_ON_SUFFIX)
    c_off = _col(df, prefix, WIND_OFF_SUFFIX)
    on = pd.to_numeric(df[c_on], errors="coerce").fillna(0.0) if c_on else 0.0
    off = pd.to_numeric(df[c_off], errors="coerce").fillna(0.0) if c_off else 0.0
    if isinstance(on, pd.Series) and isinstance(off, pd.Series):
        return (on + off).fillna(0.0)
    if isinstance(on, pd.Series):
        return on
    if isinstance(off, pd.Series):
        return off
    return pd.Series(0.0, index=df.index, dtype="float64")

# ---------- residual logic with toggle ----------

def compute_residual(df: pd.DataFrame, prefix: str, use_total_wind: bool = False) -> pd.Series:
    """
    residual = load - (solar + wind)
    wind = wind_onshore + wind_offshore (default) OR total wind when use_total_wind=True
    Missing components are treated as 0. Residual is clipped at >= 0.
    """
    load = pd.to_numeric(df[_load_column(df, prefix)], errors="coerce")
    solar = _solar_series(df, prefix)
    wind = _wind_series(df, prefix, use_total_wind=use_total_wind)
    residual = (load - (solar + wind)).clip(lower=0)
    return residual

def compute_residual_and_ratio(
    df: pd.DataFrame,
    prefix: str,
    use_total_wind: bool = False
) -> Tuple[pd.Series, pd.Series]:
    """
    Returns (residual_MW, ratio = residual/load). ratio NaN if load <= 0 or NaN.
    """
    load_col = _load_column(df, prefix)
    load = pd.to_numeric(df[load_col], errors="coerce")
    residual = compute_residual(df, prefix, use_total_wind=use_total_wind)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (residual / load).where(load > 0)
    # Do not cap to 1 aggressively; still clip lower to 0 for neatness.
    ratio = ratio.clip(lower=0)
    return residual, ratio

# ---------- streak detection ----------

def _consecutive_true_runs(mask: pd.Series) -> List[Tuple[int, int]]:
    """
    Return (start_idx, end_idx) inclusive for consecutive True stretches.
    """
    runs: List[Tuple[int, int]] = []
    in_run = False
    start = 0
    arr = mask.to_numpy()
    for i, v in enumerate(arr):
        if v and not in_run:
            in_run, start = True, i
        elif not v and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(arr) - 1))
    return runs

# ---------- event finder & window extraction ----------

def find_dunkelflaute_events(
    df: pd.DataFrame,
    prefix: str,
    ratio_threshold: float = 0.85,  # INTERPRETED AS residual/load >= threshold
    min_duration_hours: int = 72,
    use_total_wind: bool = False
) -> List[DunkelflauteEvent]:
    """
    Detect events where residual/load >= ratio_threshold for >= min_duration_hours consecutive hours.
    Returns events sorted oldest -> most recent.
    """
    ts = _ensure_ts(df)
    _, ratio = compute_residual_and_ratio(df, prefix, use_total_wind=use_total_wind)
    mask = (ratio >= ratio_threshold) & ratio.notna()
    runs = _consecutive_true_runs(mask)

    events: List[DunkelflauteEvent] = []
    for s_idx, e_idx in runs:
        dur = (e_idx - s_idx + 1)  # assuming hourly samples
        if dur >= min_duration_hours:
            events.append(DunkelflauteEvent(
                start=ts.iloc[s_idx],
                end=ts.iloc[e_idx],
                duration_hours=dur,
                start_idx=s_idx,
                end_idx=e_idx
            ))

    events.sort(key=lambda ev: ev.start)
    return events

def extract_event_window(
    df: pd.DataFrame,
    prefix: str,
    event: DunkelflauteEvent,
    hours_before: int = 24,
    hours_after: int = 24,
    use_total_wind: bool = False
) -> pd.DataFrame:
    """
    Return DataFrame around the event with columns:
      utc_timestamp, load_MW, residual_MW, price (if available)
    Uses actual timestamps for x-axis; attaches event_start/event_end for shading.
    """
    ts = _ensure_ts(df)
    load_col = _load_column(df, prefix)
    price_col = _price_column(df, prefix)

    residual, _ = compute_residual_and_ratio(df, prefix, use_total_wind=use_total_wind)

    start_time = event.start - pd.Timedelta(hours=hours_before)
    end_time = event.end + pd.Timedelta(hours=hours_after)
    sel = (ts >= start_time) & (ts <= end_time)

    w = pd.DataFrame({
        "utc_timestamp": ts[sel],
        "load_MW": pd.to_numeric(df.loc[sel, load_col], errors="coerce"),
        "residual_MW": pd.to_numeric(residual[sel], errors="coerce"),
    }).reset_index(drop=True)

    if price_col is not None:
        w["price"] = pd.to_numeric(df.loc[sel, price_col], errors="coerce")
    else:
        w["price"] = np.nan

    # keep event start/end (for shading)
    w.attrs["event_start"] = event.start
    w.attrs["event_end"] = event.end
    return w
