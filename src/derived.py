# src/derived.py
import pandas as pd

LOAD_SUFFIX = "load_actual_entsoe_transparency"
SOLAR_SUFFIX = "solar_generation_actual"
WIND_TOTAL_SUFFIX = "wind_generation_actual"
WIND_ON_SUFFIX = "wind_onshore_generation_actual"
WIND_OFF_SUFFIX = "wind_offshore_generation_actual"

def compute_residual_load(df: pd.DataFrame, prefix: str) -> pd.Series:
    """
    residual_load = total_load_actual - (solar_generation_actual
                                         + wind_generation_actual
                                         + wind_onshore_generation_actual
                                         + wind_offshore_generation_actual)
    Missing components are treated as 0.
    Returns a Series aligned to df.index.
    """
    p = f"{prefix}_"
    # Required: total load actual
    load_col = p + LOAD_SUFFIX
    if load_col not in df.columns:
        raise KeyError(f"Cannot compute residual load: missing '{load_col}'")

    # Optional renewable components
    renew_cols = []
    for suf in [SOLAR_SUFFIX, WIND_TOTAL_SUFFIX, WIND_ON_SUFFIX, WIND_OFF_SUFFIX]:
        col = p + suf
        if col in df.columns:
            renew_cols.append(col)

    if not renew_cols:
        # No renewable components found; residual == load
        return df[load_col].copy()

    renew_sum = df[renew_cols].fillna(0).sum(axis=1)
    return (df[load_col] - renew_sum)
