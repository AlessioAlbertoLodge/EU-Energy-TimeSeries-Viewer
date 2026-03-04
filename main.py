# main.py
import pandas as pd
import streamlit as st

from src.data_loader import (
    load_dataset,
    list_prefixes,
    split_fields_for_prefix,   # power vs price splitter
)
from src.filters import infer_time_bounds, filter_by_time
from src.viz import make_dualaxis_figure
from src.derived import compute_residual_load  # residual load helper

st.set_page_config(page_title="EU Energy – 60-min Viewer", layout="wide")

st.title("EU Energy – 60-min Time Series (Power vs Price)")
st.caption(
    "Left y-axis: power (MW). Right y-axis: price (EUR/GBP). "
    "Click legend items to toggle series."
)

# --- Path to data (editable) ---
default_csv = r"C:\Users\aless\Desktop\Streamlit_Apps_Backup\Dataset1_streamlit_app\data\time_series_60min_singleindex.csv"
data_path = st.text_input("CSV path", value=default_csv)

# --- Load dataset ---
with st.spinner("Loading dataset..."):
    try:
        df = load_dataset(data_path)
    except Exception as e:
        st.error(f"Could not load CSV. Check path & file. Error: {e}")
        st.stop()

# --- Time bounds & prefixes ---
min_t, max_t = infer_time_bounds(df)
prefixes = list_prefixes(df)
if not prefixes:
    st.error("No country/zone prefixes discovered in this dataset.")
    st.stop()

sorted_prefixes = sorted(prefixes)
default_prefix = "NL" if "NL" in prefixes else sorted_prefixes[0]
with st.sidebar:
    st.header("Filters")

    # NL as default selection (falls back to first if NL not present)
    prefix = st.selectbox(
        "Country / Zone prefix",
        options=sorted_prefixes,
        index=sorted_prefixes.index(default_prefix),
    )

    # Split available fields
    power_fields, price_fields = split_fields_for_prefix(df, prefix)

    # Residual toggle: keep True by default
    add_residual = st.checkbox("Add residual load (load − RES)", value=True)
    if add_residual and "residual_load" not in power_fields:
        power_fields = ["residual_load"] + power_fields

    # --- DEFAULTS: if NL, select ALL power fields except forecast loads ---
    if prefix == "NL":
        default_power = [
            f for f in power_fields
            if not f.endswith("load_forecast_entsoe_transparency")
        ]
        default_price = price_fields[:]  # keep all prices
    else:
        default_power = [
            f for f in power_fields
            if any(x in f for x in [
                "load_actual",
                "residual_load",
                "solar_generation_actual",
                "wind_generation_actual",
                "wind_onshore_generation_actual",
                "wind_offshore_generation_actual",
            ])
        ][:3] or power_fields[:3]
        default_price = [f for f in price_fields if "price_day_ahead" in f] or price_fields[:1]

    power_sel = st.multiselect(
        "Power fields (MW, left axis)",
        options=power_fields,
        default=default_power,
        help="All non-price numeric series under the chosen prefix.",
    )

    price_sel = st.multiselect(
        "Price fields (right axis)",
        options=price_fields,
        default=default_price,
        help="Typically *_price_day_ahead (EUR, or GBP for GB_GBN).",
    )

    st.caption(f"Available window (UTC): **{min_t} → {max_t}**")
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input(
            "Start date", value=min_t.date(), min_value=min_t.date(), max_value=max_t.date()
        )
    with c2:
        end_date = st.date_input(
            "End date", value=max_t.date(), min_value=min_t.date(), max_value=max_t.date()
        )

# Must pick at least one series
if not power_sel and not price_sel:
    st.info("Select at least one power or price series to plot.")
    st.stop()

# --- Filter time range (inclusive of whole days) ---
df_slice = filter_by_time(
    df,
    pd.to_datetime(start_date).tz_localize("UTC"),
    pd.to_datetime(end_date).tz_localize("UTC"),
)

# --- Compute residual if requested & selected ---
if "residual_load" in power_sel:
    try:
        df_slice[f"{prefix}_residual_load"] = compute_residual_load(df_slice, prefix)
    except KeyError as e:
        st.warning(str(e))

# --- Build column lists to plot ---
def col_for(prefix: str, field: str) -> str:
    """Return the concrete column name for a prefix+field (handles residual)."""
    return f"{prefix}_residual_load" if field == "residual_load" else f"{prefix}_{field}"

power_cols = [col_for(prefix, f) for f in power_sel if col_for(prefix, f) in df_slice.columns]
price_cols = [col_for(prefix, f) for f in price_sel if col_for(prefix, f) in df_slice.columns]

if not power_cols and not price_cols:
    st.warning("No matching columns for this selection in the chosen date window.")
    st.stop()

# --- Title & plot ---
title_bits = []
if power_sel: title_bits.append(f"Power: {', '.join(power_sel)}")
if price_sel: title_bits.append(f"Price: {', '.join(price_sel)}")
title = f"{prefix} – " + " | ".join(title_bits)

# Optional currency note for GB_GBN
right_label = "Price"
if prefix == "GB_GBN":
    right_label = "Price (GBP)"
elif any(f.endswith("price_day_ahead") for f in price_sel):
    right_label = "Price (EUR)"

fig = make_dualaxis_figure(
    df_slice,
    x_col="utc_timestamp",
    power_cols=power_cols,
    price_cols=price_cols,
    title=title,
    left_label="Power (MW)",
    right_label=right_label,
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Preview first rows (filtered)"):
    preview_cols = ["utc_timestamp"] + power_cols + price_cols
    st.dataframe(df_slice[preview_cols].head(50))
