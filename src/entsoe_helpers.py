# src/entsoe_helpers.py
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Final, Iterable, Literal, List
import datetime as dt
import re

import numpy as np
import pandas as pd
from entsoe import EntsoePandasClient  # type: ignore

# --------------------------------------------------------------------------------------
# 1) Country/zone mapping
# --------------------------------------------------------------------------------------

_BIDDING_ZONE: Final[dict[str, str]] = {
    "DE": "DE_LU", "GERMANY": "DE_LU", "LU": "DE_LU", "LUXEMBOURG": "DE_LU",
    "FR": "FR", "FRANCE": "FR",
    "BE": "BE", "BELGIUM": "BE",
    "NL": "NL", "NETHERLANDS": "NL",
    "ES": "ES", "SPAIN": "ES",
    "PT": "PT", "PORTUGAL": "PT",
    "GB": "GB", "UK": "UK", "UNITED KINGDOM": "UK",
    "DK": "DK_1", "DK1": "DK_1", "DK2": "DK_2",
    "SE4": "SE_4", "SE_4": "SE_4",
    "NO2": "NO_2", "NO_2": "NO_2",
    "CH": "CH", "SWITZERLAND": "CH",
    "AT": "AT", "AUSTRIA": "AT",
    "IT": "IT_NORD", "ITALY": "IT_NORD",
}

def resolve_zone(country_or_zone: str) -> str:
    s = country_or_zone.upper()
    if s in _BIDDING_ZONE:
        return _BIDDING_ZONE[s]
    if "_" in s:
        return s
    raise ValueError(
        f"Unknown country/zone '{country_or_zone}'. Extend _BIDDING_ZONE or pass a zone like 'DE_LU'."
    )

# --------------------------------------------------------------------------------------
# 2) Client + tz helpers
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class EntsoeConfig:
    api_key: str
    tz_out: Literal["UTC", "Europe/Brussels"] = "UTC"

@lru_cache(maxsize=16)
def _client(api_key: str) -> EntsoePandasClient:
    return EntsoePandasClient(api_key=api_key)

def _to_tz(ts: pd.DatetimeIndex | pd.Series, tz_out: str) -> pd.Series:
    s = pd.to_datetime(ts, errors="coerce")
    if isinstance(s, pd.DatetimeIndex):
        s = pd.Series(s)
    try:
        return s.dt.tz_convert(tz_out)
    except Exception:
        s = s.dt.tz_localize("Europe/Brussels", nonexistent="shift_forward", ambiguous="NaT")
        return s.dt.tz_convert(tz_out)

# --------------------------------------------------------------------------------------
# 3) Frame normalizers (avoid MultiIndex, ensure clean merge)
# --------------------------------------------------------------------------------------

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join([str(p) for p in tup if p is not None and str(p) != "None"]).strip("_")
            for tup in df.columns.tolist()
        ]
    return df

def _to_named_frame(obj: pd.Series | pd.DataFrame, colname: str) -> pd.DataFrame:
    if isinstance(obj, pd.Series):
        df = obj.to_frame(name=colname)
    else:
        df = _flatten_columns(obj).copy()
        if df.shape[1] == 1:
            df.columns = [colname]
        else:
            df = pd.to_numeric(df, errors="coerce").sum(axis=1).to_frame(name=colname)
    return df

def _finalize_time_frame(df: pd.DataFrame, tz_out: str, time_col_first: bool = True) -> pd.DataFrame:
    df = _flatten_columns(df)
    if "time" not in df.columns:
        idx = df.index
        time = _to_tz(idx, tz_out)
        df = df.copy()
        df["time"] = time.values
    cols = list(df.columns)
    if time_col_first:
        cols = ["time"] + [c for c in cols if c != "time"]
    df = df[cols].reset_index(drop=True).sort_values("time", ignore_index=True)
    return df

# --------------------------------------------------------------------------------------
# 4) Low-level fetchers (prices, load, generation)
# --------------------------------------------------------------------------------------

def get_day_ahead_prices_range(
    country_or_zone: str,
    start: dt.date,
    end: dt.date,
    *,
    cfg: EntsoeConfig,
) -> pd.DataFrame:
    zone = resolve_zone(country_or_zone)
    start_br = pd.Timestamp(start, tz="Europe/Brussels")
    end_br = pd.Timestamp(end + dt.timedelta(days=1), tz="Europe/Brussels")
    obj = _client(cfg.api_key).query_day_ahead_prices(zone, start=start_br, end=end_br)
    df = _to_named_frame(obj, "price")
    df = _finalize_time_frame(df, cfg.tz_out, time_col_first=True)
    return df[["time", "price"]]

def get_actual_total_load_range(
    country_or_zone: str,
    start: dt.date,
    end: dt.date,
    *,
    cfg: EntsoeConfig,
) -> pd.DataFrame:
    zone = resolve_zone(country_or_zone)
    start_br = pd.Timestamp(start, tz="Europe/Brussels")
    end_br = pd.Timestamp(end + dt.timedelta(days=1), tz="Europe/Brussels")
    obj = _client(cfg.api_key).query_load(zone, start=start_br, end=end_br)
    df = _to_named_frame(obj, "load_MW")
    df = _finalize_time_frame(df, cfg.tz_out, time_col_first=True)
    return df[["time", "load_MW"]]

# --------------------------------------------------------------------------------------
# 5) Generation (robust PSR detection)
# --------------------------------------------------------------------------------------

# Accept either PSR codes or human labels, any case/spacing
_PSR_MATCHES = {
    "solar": {"b16", "solar"},
    "wind_onshore": {"b19", "windonshore", "wind_onshore", "onshorewind", "onshore"},
    "wind_offshore": {"b18", "windoffshore", "wind_offshore", "offshorewind", "offshore"},
}

def _norm(s: str) -> str:
    # lower + keep only letters/numbers
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def _pick_sum_cols(df: pd.DataFrame, keys: set[str]) -> pd.Series:
    """Sum all columns whose normalized name contains ANY of the keys."""
    if df.empty or df.shape[1] == 0:
        return pd.Series(index=df.index, dtype="float64")
    sel = []
    for c in df.columns:
        cn = _norm(c)
        if any(k in cn for k in keys):
            sel.append(c)
    if not sel:
        return pd.Series(index=df.index, dtype="float64")
    return pd.to_numeric(df[sel], errors="coerce").sum(axis=1)

def get_actual_generation_range(
    country_or_zone: str,
    start: dt.date,
    end: dt.date,
    *,
    cfg: EntsoeConfig,
) -> pd.DataFrame:
    zone = resolve_zone(country_or_zone)
    start_br = pd.Timestamp(start, tz="Europe/Brussels")
    end_br = pd.Timestamp(end + dt.timedelta(days=1), tz="Europe/Brussels")

    gen = _client(cfg.api_key).query_generation(zone, start=start_br, end=end_br, psr_type=None)
    if gen is None or gen.empty:
        idx = pd.date_range(start_br, end_br, freq="H", inclusive="left", tz="Europe/Brussels")
        out = pd.DataFrame(index=idx)
        out = _finalize_time_frame(out, cfg.tz_out)
        for c in ("solar_MW", "wind_onshore_MW", "wind_offshore_MW", "wind_total_MW"):
            out[c] = np.nan
        return out[["time", "solar_MW", "wind_onshore_MW", "wind_offshore_MW", "wind_total_MW"]]

    gen = _flatten_columns(gen).copy()
    # Build time in tz_out
    gen["time"] = _to_tz(gen.index, cfg.tz_out).values

    # Everything except 'time' is candidate numeric
    cand = gen.drop(columns=[c for c in gen.columns if c == "time"], errors="ignore")

    # Robust matching by code or label
    solar = _pick_sum_cols(cand, _PSR_MATCHES["solar"])
    won   = _pick_sum_cols(cand, _PSR_MATCHES["wind_onshore"])
    woff  = _pick_sum_cols(cand, _PSR_MATCHES["wind_offshore"])

    out = pd.DataFrame({
        "time": gen["time"].values,
        "solar_MW": solar.reindex(cand.index).values if len(solar) else np.nan,
        "wind_onshore_MW": won.reindex(cand.index).values if len(won) else np.nan,
        "wind_offshore_MW": woff.reindex(cand.index).values if len(woff) else np.nan,
    })

    out = _finalize_time_frame(out, cfg.tz_out, time_col_first=True)

    # wind_total
    if "wind_onshore_MW" in out and "wind_offshore_MW" in out:
        out["wind_total_MW"] = out["wind_onshore_MW"].fillna(0) + out["wind_offshore_MW"].fillna(0)
        out.loc[
            out[["wind_onshore_MW", "wind_offshore_MW"]].isna().all(axis=1),
            "wind_total_MW"
        ] = np.nan
    elif "wind_onshore_MW" in out:
        out["wind_total_MW"] = out["wind_onshore_MW"]
    elif "wind_offshore_MW" in out:
        out["wind_total_MW"] = out["wind_offshore_MW"]
    else:
        out["wind_total_MW"] = np.nan

    return out[["time", "solar_MW", "wind_onshore_MW", "wind_offshore_MW", "wind_total_MW"]]

# --------------------------------------------------------------------------------------
# 6) High-level merge
# --------------------------------------------------------------------------------------

Signal = Literal["price", "load", "solar", "wind_onshore", "wind_offshore", "wind_total"]

def fetch_signals(
    country_or_zone: str,
    start: dt.date,
    end: dt.date,
    *,
    cfg: EntsoeConfig,
    signals: Iterable[Signal] = ("price","load","solar","wind_onshore","wind_offshore","wind_total"),
) -> pd.DataFrame:
    want = set(signals)
    dfs: List[pd.DataFrame] = []

    if "price" in want:
        dfs.append(get_day_ahead_prices_range(country_or_zone, start, end, cfg=cfg))
    if "load" in want:
        dfs.append(get_actual_total_load_range(country_or_zone, start, end, cfg=cfg))
    if any(x in want for x in ("solar","wind_onshore","wind_offshore","wind_total")):
        dfs.append(get_actual_generation_range(country_or_zone, start, end, cfg=cfg))

    if not dfs:
        return pd.DataFrame(columns=["time"]).astype({"time": "datetime64[ns, UTC]"})

    out = dfs[0]
    for d in dfs[1:]:
        d = _finalize_time_frame(d, cfg.tz_out)
        out = _finalize_time_frame(out, cfg.tz_out)
        out = out.merge(d, on="time", how="outer")

    out = out.loc[:, ~out.columns.duplicated()].sort_values("time", ignore_index=True)

    # Ensure requested columns exist
    need = []
    if "price" in want: need.append("price")
    if "load" in want: need.append("load_MW")
    if "solar" in want: need.append("solar_MW")
    if "wind_onshore" in want: need.append("wind_onshore_MW")
    if "wind_offshore" in want: need.append("wind_offshore_MW")
    if "wind_total" in want: need.append("wind_total_MW")
    for c in need:
        if c not in out.columns:
            out[c] = np.nan

    for c in out.columns:
        if c != "time":
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out

# --------------------------------------------------------------------------------------
# 7) Residual utilities
# --------------------------------------------------------------------------------------

def compute_residual_strict(df: pd.DataFrame) -> pd.Series:
    load = pd.to_numeric(df.get("load_MW"), errors="coerce")
    solar = pd.to_numeric(df.get("solar_MW"), errors="coerce").fillna(0)
    won = pd.to_numeric(df.get("wind_onshore_MW"), errors="coerce").fillna(0)
    woff = pd.to_numeric(df.get("wind_offshore_MW"), errors="coerce").fillna(0)
    return (load - (solar + won + woff)).clip(lower=0)

def add_residual_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["residual_MW"] = compute_residual_strict(out)
    if "wind_total_MW" in out and out["wind_total_MW"].notna().any():
        out["vre_MW"] = pd.to_numeric(out.get("solar_MW"), errors="coerce").fillna(0) + \
                        pd.to_numeric(out.get("wind_total_MW"), errors="coerce").fillna(0)
        if "wind_onshore_MW" in out and "wind_offshore_MW" in out:
            comp = out["wind_onshore_MW"].fillna(0) + out["wind_offshore_MW"].fillna(0)
            out["vre_MW"] = np.where(out["wind_total_MW"].notna(), out["vre_MW"], comp + out.get("solar_MW", 0).fillna(0))
    else:
        out["vre_MW"] = pd.to_numeric(out.get("solar_MW"), errors="coerce").fillna(0) + \
                        pd.to_numeric(out.get("wind_onshore_MW"), errors="coerce").fillna(0) + \
                        pd.to_numeric(out.get("wind_offshore_MW"), errors="coerce").fillna(0)
    load = pd.to_numeric(out.get("load_MW"), errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        out["residual_ratio"] = (out["residual_MW"] / load).where(load > 0).clip(0, 1)
    return out

def years_present(df: pd.DataFrame, time_col: str = "time") -> List[int]:
    if time_col not in df.columns:
        return []
    yrs = pd.to_datetime(df[time_col], errors="coerce").dt.year.dropna().unique()
    return sorted(int(y) for y in yrs)
