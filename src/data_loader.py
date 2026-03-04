# src/data_loader.py
import os
import re
import pandas as pd
from functools import lru_cache
from typing import List, Tuple

_PREFIX_RE = re.compile(r"^([A-Z0-9_]+?)_[a-z]")

@lru_cache(maxsize=2)
def load_dataset(path: str) -> pd.DataFrame:
    """Load CSV and return a sorted DataFrame with a parsed UTC timestamp."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path, low_memory=False)

    if "utc_timestamp" not in df.columns:
        raise KeyError("Column 'utc_timestamp' not found in CSV.")

    # Robust UTC parsing
    df["utc_timestamp"] = pd.to_datetime(
        df["utc_timestamp"], errors="coerce", utc=True, infer_datetime_format=True
    )

    df = df.dropna(subset=["utc_timestamp"]).sort_values("utc_timestamp").reset_index(drop=True)
    return df

def list_prefixes(df: pd.DataFrame) -> list:
    """Return sorted list of country/zone prefixes (e.g., IT, DE_LU, SE_3)."""
    prefixes = set()
    for c in df.columns:
        if c in ("utc_timestamp", "cet_cest_timestamp"):
            continue
        m = _PREFIX_RE.match(c)
        if m:
            prefixes.add(m.group(1))
    return sorted(prefixes)

def list_fields_for_prefix(df: pd.DataFrame, prefix: str) -> list:
    """Return field names (suffixes) available under a given prefix."""
    plen = len(prefix) + 1
    fields = []
    for c in df.columns:
        if c.startswith(prefix + "_"):
            fields.append(c[plen:])
    return sorted(fields)

def split_fields_for_prefix(df: pd.DataFrame, prefix: str) -> Tuple[List[str], List[str]]:
    """
    Split fields into (power_fields, price_fields) for a given prefix.
    Heuristic:
      - price fields: suffix endswith 'price_day_ahead'
      - power fields: everything else numeric under the prefix
    """
    fields = list_fields_for_prefix(df, prefix)
    price_fields = [f for f in fields if f.endswith("price_day_ahead")]
    # Power = any remaining columns under prefix that are numeric
    pref = prefix + "_"
    numeric_under_prefix = []
    for f in fields:
        col = pref + f
        if col in df.columns:
            s = df[col]
            if pd.api.types.is_numeric_dtype(s.dtype):
                numeric_under_prefix.append(f)
    power_fields = [f for f in numeric_under_prefix if f not in price_fields]
    return sorted(power_fields), sorted(price_fields)
