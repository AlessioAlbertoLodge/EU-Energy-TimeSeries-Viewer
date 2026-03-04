# try_calling_api.py
from datetime import date
import os

# Streamlit is optional here; we only use it to read secrets if available
try:
    import streamlit as st
except Exception:  # running without Streamlit
    st = None

from src.entsoe_helpers import EntsoeConfig, fetch_signals, add_residual_columns


def get_entsoe_api_key() -> str:
    """
    Robust getter:
      1) .streamlit/secrets.toml under [api_keys].entsoe
      2) .streamlit/secrets.toml top-level ENTSOE_API_KEY (if you ever switch)
      3) ENV vars: ENTSOE_API_KEY or ENTSOE
    Raises RuntimeError if not found.
    """
    # 1) Streamlit secrets (works only when streamlit runs; else st is None/empty)
    if st is not None and hasattr(st, "secrets"):
        if "api_keys" in st.secrets and "entsoe" in st.secrets["api_keys"]:
            val = st.secrets["api_keys"]["entsoe"]
            if isinstance(val, str) and val.strip():
                return val.strip()
        if "ENTSOE_API_KEY" in st.secrets:
            val = st.secrets["ENTSOE_API_KEY"]
            if isinstance(val, str) and val.strip():
                return val.strip()

    # 2) Environment variables
    for env_name in ("ENTSOE_API_KEY", "ENTSOE"):
        val = os.environ.get(env_name, "").strip()
        if val:
            return val

    raise RuntimeError(
        "ENTSO-E API key not found.\n"
        "Put it in .streamlit/secrets.toml as [api_keys].entsoe = \"...\"\n"
        "or set ENV var ENTSOE_API_KEY / ENTSOE."
    )


def main():
    api_key = get_entsoe_api_key()
    cfg = EntsoeConfig(api_key=api_key, tz_out="UTC")

    df = fetch_signals(
        "DE",
        start=date(2020, 1, 1),
        end=date(2020, 1, 31),
        cfg=cfg,
        signals=("price", "load", "solar", "wind_onshore", "wind_offshore", "wind_total"),
    )
    df = add_residual_columns(df)

    # Print a small sample to console
    print(df.head(12))
    print("\nColumns:", list(df.columns))


if __name__ == "__main__":
    main()
