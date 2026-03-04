# pages/07_💸_Price_Heatmaps.py
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_loader import load_dataset, list_prefixes
from src.filters import infer_time_bounds
from src.price_heatmap_helpers import (
    compute_price_long,
    years_available,
    pivot_for_year,
    make_blue_red_colorscale,
    global_min_max,
    list_price_columns,
)

st.set_page_config(page_title="Electricity Price Heatmaps", layout="wide")
st.title("Electricity Prices – Yearly Hour/Day Heatmaps")
st.caption(
    "One heatmap per year. **X** = day of year (1..365/366). **Y** = hour (00 at top → 23 at bottom). "
    "Color: **blue = low price**, **red = high price**. Single colorbar at the top."
)

# --- Data path (same default as main) ---
default_csv = r"C:\Users\lodgeaa\Desktop\Private Projects\Dataset1_streamlit_app\data\time_series_60min_singleindex.csv"
data_path = st.text_input("CSV path", value=default_csv)

with st.spinner("Loading dataset..."):
    try:
        df = load_dataset(data_path)
    except Exception as e:
        st.error(f"Could not load CSV. Error: {e}")
        st.stop()

min_t, max_t = infer_time_bounds(df)
prefixes = list_prefixes(df)
if not prefixes:
    st.error("Could not discover any prefixes in this CSV.")
    st.stop()

col1, col2 = st.columns([1, 2], vertical_alignment="center")
with col1:
    prefix = st.selectbox("Country / Zone prefix", options=sorted(prefixes), index=0)
with col2:
    st.write(f"Available UTC window: **{min_t} → {max_t}**")

# --- Let the user pick the price column (flexible) ---
cands = list_price_columns(df, prefix)
price_col = None
if cands:
    price_col = st.selectbox(
        "Select price column",
        options=cands,
        index=0,
        help="Price columns are discovered automatically for this prefix."
    )
else:
    # Still allow the strict default if present under a non-matching discovery pattern
    strict = f"{prefix}_price_day_ahead"
    if strict in df.columns:
        price_col = strict
    else:
        st.error(
            f"No price-like columns found for '{prefix}'. "
            f"Expected columns that start with '{prefix}_' and contain 'price', e.g. '{prefix}_price_day_ahead' "
            f"or zone-specific variants like '{prefix}_LU_price_day_ahead'."
        )
        st.stop()

# --- Build long-format prices using the chosen price column ---
with st.spinner("Preparing price matrices..."):
    try:
        long_df = compute_price_long(df, prefix, price_col=price_col)
    except KeyError as e:
        st.error(str(e))
        st.stop()

all_years = years_available(long_df)
if not all_years:
    st.warning("No yearly price data available for the selected column.")
    st.stop()

years_select = st.multiselect(
    "Years to display (vertical stack)",
    options=all_years,
    default=all_years,
)
if not years_select:
    st.info("Select at least one year.")
    st.stop()

# Global color scale bounds across the selected years
zmin, zmax = global_min_max(long_df, years_select)
colorscale = make_blue_red_colorscale()

# --- LAYOUT CONTROLS ---
lc1, lc2, lc3 = st.columns([1, 1, 1])
with lc1:
    vspace = st.slider("Subplot spacing", 0.00, 0.10, 0.03, 0.005)
with lc2:
    row_height = st.slider("Row height (px / year)", 180, 600, 340, 10)
with lc3:
    fig_width = st.slider("Figure width (px)", 1400, 3600, 2400, 50, help="Increase for much wider heatmaps.")

global_font = 10
title_font_size = 12
n_rows = len(years_select)

# Plotly spacing guard
if n_rows <= 1:
    vspace_eff = 0.0
else:
    max_allowed = 1.0 / (n_rows - 1) - 1e-6
    vspace_eff = min(vspace, max_allowed)
    if vspace_eff < vspace:
        st.caption(
            f"ℹ️ Subplot spacing auto-clamped from **{vspace:.4f}** to **{vspace_eff:.4f}** "
            f"for **{n_rows}** rows."
        )

fig = make_subplots(
    rows=n_rows,
    cols=1,
    shared_xaxes=False,
    shared_yaxes=False,
    vertical_spacing=vspace_eff,
    subplot_titles=[str(y) for y in years_select],
)

# Shared coloraxis for a single colorbar at the top
fig.update_layout(
    coloraxis=dict(
        colorscale=colorscale,
        cmin=zmin,
        cmax=zmax,
        colorbar=dict(
            title=f"Price (€/MWh) – {price_col}",
            orientation="h",
            y=1.06,
            x=0.5,
            xanchor="center",
            len=0.7,
            thickness=18,
            tickformat=",d",  # integers, no scientific notation
        )
    )
)

for i, yr in enumerate(years_select, start=1):
    mat = pivot_for_year(long_df, yr)  # 24 x N_doys
    mat = mat.reindex(sorted(mat.columns.tolist()), axis=1)  # sort DOY ascending
    z = mat.values
    x_vals = mat.columns.tolist()
    y_vals = mat.index.tolist()

    heat = go.Heatmap(
        z=z,
        x=x_vals,
        y=y_vals,
        coloraxis="coloraxis",
        hovertemplate=(
            "Year: %{customdata[0]}<br>"
            "DOY: %{x}<br>"
            "Hour: %{y}:00<br>"
            "Avg Price: %{z:.0f} €/MWh<extra></extra>"
        ),
        customdata=[[yr] * len(y_vals)] * len(x_vals),
        zsmooth=False,
        showscale=False,
    )
    fig.add_trace(heat, row=i, col=1)

    show_xticks = (i == n_rows)
    fig.update_xaxes(
        title_text="Day of Year" if show_xticks else None,
        showticklabels=show_xticks,
        ticks="outside" if show_xticks else "",
        row=i, col=1,
        tickmode="array",
        tickfont=dict(size=global_font),
        tickvals=[d for d in x_vals if (d % 30 == 1 or d in (1, 182, 366))] if show_xticks else None,
    )
    fig.update_yaxes(
        title_text="Hour (UTC)",
        row=i, col=1,
        autorange="reversed",
        tickmode="array",
        tickvals=list(range(0, 24, 2)),
        ticks="outside",
        tickfont=dict(size=global_font),
        title_font=dict(size=global_font),
    )

fig.update_layout(
    title=f"{prefix} – Electricity Price Heatmaps",
    height=max(300, int(row_height * n_rows)),
    width=int(fig_width),
    margin=dict(l=10, r=60, t=100, b=10),
    font=dict(size=global_font),
)

# Shrink subplot title fonts a bit
if "annotations" in fig["layout"]:
    for ann in fig["layout"]["annotations"]:
        if "text" in ann and ann["text"] in [str(y) for y in years_select]:
            ann["font"] = {"size": title_font_size}

st.plotly_chart(fig, use_container_width=False)

with st.expander("Notes"):
    st.markdown(f"""
- Price column used: **`{price_col}`** (change it above if multiple are available for this country).
- Values are **means** per (hour, day-of-year) bin; colorbar scale shared across all selected years.
- Integers in hover; no per-cell numbers for a cleaner view.
- Adjust **Figure width** and **Row height** to make plots much larger.
""")
