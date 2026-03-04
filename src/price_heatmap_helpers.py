# src/price_heatmap_helpers.py
import pandas as pd
import numpy as np
import re
from typing import List, Tuple, Optional

def _ensure_ts(df: pd.DataFrame) -> pd.Series:
    if "utc_timestamp" not in df.columns:
        raise KeyError("Expected 'utc_timestamp' in DataFrame.")
    return pd.to_datetime(df["utc_timestamp"], errors="coerce", utc=True)

# ---------- Flexible price column discovery ----------

def list_price_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    """
    Return a list of candidate price columns for the given prefix.
    Heuristics:
      - column starts with f"{prefix}_"
      - and contains 'price' (case-insensitive)
    Common examples: 'DE_price_day_ahead', 'DE_LU_price_day_ahead', 'NL_price_day_ahead', etc.
    """
    pfx = f"{prefix}_"
    pat = re.compile(r"price", re.IGNORECASE)
    candidates: List[str] = []
    for c in df.columns:
        if isinstance(c, str) and c.startswith(pfx) and pat.search(c):
            candidates.append(c)
    # Prefer *_price_day_ahead first, then others
    candidates.sort(key=lambda c: (0 if c.lower().endswith("price_day_ahead") else 1, c))
    return candidates

def _default_price_col(df: pd.DataFrame, prefix: str) -> Optional[str]:
    """
    Keep the original 'strict' default for backwards compatibility:
    returns f'{prefix}_price_day_ahead' if it exists; else None.
    """
    cand = f"{prefix}_price_day_ahead"
    return cand if cand in df.columns else None

# ---------- Long-format price builder ----------

def compute_price_long(df: pd.DataFrame, prefix: str, price_col: Optional[str] = None) -> pd.DataFrame:
    """
    Return long-format dataframe with columns:
      year, doy (1..365/366), hour (0..23), price
    Aggregated by mean for duplicate (year, doy, hour) bins.

    If price_col is None:
      - try the strict default '{prefix}_price_day_ahead'
      - else raise a clear KeyError listing discovered candidates (if any)
    """
    ts = _ensure_ts(df)
    use_col = price_col

    if use_col is None:
        strict = _default_price_col(df, prefix)
        if strict is not None:
            use_col = strict
        else:
            cands = list_price_columns(df, prefix)
            if not cands:
                raise KeyError(
                    f"Missing price column for '{prefix}'. Tried '{prefix}_price_day_ahead' and found no alternatives. "
                    f"Tip: rename or select a price column that starts with '{prefix}_' and contains 'price'."
                )
            # If we’re here, caller didn’t pass price_col but alternatives exist.
            # Be explicit to the caller to select one.
            raise KeyError(
                f"Multiple/alternative price columns exist for '{prefix}', but none explicitly selected. "
                f"Candidates: {', '.join(cands)}"
            )

    if use_col not in df.columns:
        raise KeyError(f"Selected price column '{use_col}' not found in DataFrame.")

    tmp = pd.DataFrame({
        "ts": ts,
        "price": pd.to_numeric(df[use_col], errors="coerce")
    }).dropna(subset=["ts"])

    tmp["year"] = tmp["ts"].dt.year
    tmp["doy"] = tmp["ts"].dt.dayofyear
    tmp["hour"] = tmp["ts"].dt.hour

    long_df = (
        tmp.groupby(["year", "doy", "hour"], as_index=False)["price"]
           .mean()
           .sort_values(["year", "doy", "hour"])
    )
    return long_df

def years_available(long_df: pd.DataFrame) -> List[int]:
    years = sorted(long_df["year"].dropna().unique().tolist())
    return [int(y) for y in years]

def pivot_for_year(long_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Return a 24 x N matrix (rows=hour 0..23; cols=DOY 1..365/366) of mean prices for that year.
    Missing bins -> NaN.
    """
    sub = long_df[long_df["year"] == year]
    if sub.empty:
        return pd.DataFrame(index=list(range(24)))
    mat = sub.pivot(index="hour", columns="doy", values="price").sort_index()
    mat = mat.reindex(index=range(24))
    return mat

def make_blue_red_colorscale() -> list:
    return [
        [0.0, "rgb(33, 113, 181)"],  # blue
        [1.0, "rgb(220, 50, 47)"],   # red
    ]

def global_min_max(long_df: pd.DataFrame, selected_years: List[int]) -> Tuple[float, float]:
    sub = long_df[long_df["year"].isin(selected_years)]
    if sub.empty or sub["price"].dropna().empty:
        return (0.0, 1.0)
    vmin = float(np.nanmin(sub["price"].values))
    vmax = float(np.nanmax(sub["price"].values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return (vmin if np.isfinite(vmin) else 0.0, (vmax if np.isfinite(vmax) else 1.0) + 1e-6)
    return vmin, vmax
