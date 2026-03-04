# pages/02_🗺️_Residual_Load_Heatmaps.py
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_loader import load_dataset, list_prefixes
from src.filters import infer_time_bounds
from src.heatmap_helpers import (
    compute_residual_ratio_long,
    years_available,
    pivot_for_year,
    make_green_red_colorscale,
)

st.set_page_config(page_title="Residual Load Heatmaps", layout="wide")
st.title("Residual Load – Yearly Heatmaps")
st.caption(
    "Each heatmap shows a year. **X** = day of year (1..365/366). "
    "**Y** = hour of day (00 at top → 23 at bottom). "
    "Color: **red = 0% residual**, **green = 100% residual** (residual/load)."
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

# --- Build long-format residual ratio data ---
with st.spinner("Preparing residual ratios..."):
    try:
        long_df = compute_residual_ratio_long(df, prefix)
    except KeyError as e:
        st.error(str(e))
        st.stop()

all_years = years_available(long_df)
if not all_years:
    st.warning("No yearly data available for the selected prefix.")
    st.stop()

years_select = st.multiselect(
    "Years to display (vertical stack)",
    options=all_years,
    default=all_years,  # show all by default
)

if not years_select:
    st.info("Select at least one year.")
    st.stop()

# --- Create one heatmap per year, stacked vertically ---
# --- Create one heatmap per year, stacked vertically ---
colorscale = make_green_red_colorscale()
n_rows = len(years_select)

# tighter vertical spacing (was 0.08)
fig = make_subplots(
    rows=n_rows,
    cols=1,
    shared_xaxes=False,
    shared_yaxes=False,
    vertical_spacing=0.03,   # closer heatmaps
    subplot_titles=[str(y) for y in years_select],
)

for i, yr in enumerate(years_select, start=1):
    mat = pivot_for_year(long_df, yr)  # 24 x N_doys
    z = mat.values
    x_vals = mat.columns.tolist()
    y_vals = mat.index.tolist()

    heat = go.Heatmap(
        z=z,
        x=x_vals,
        y=y_vals,
        colorscale=colorscale,
        zmin=0.0,
        zmax=1.0,
        colorbar=dict(title="Residual / Load", x=1.02),
        showscale=(i == n_rows),
        hovertemplate=(
            "Year: %{customdata[0]}<br>"
            "DOY: %{x}<br>"
            "Hour: %{y}:00<br>"
            "Residual/Load: %{z:.2f}<extra></extra>"
        ),
        customdata=[[yr]*len(y_vals)] * len(x_vals),
    )
    fig.add_trace(heat, row=i, col=1)

    # Only show x-axis labels/ticks on bottom subplot
    show_xticks = (i == n_rows)
    fig.update_xaxes(
        title_text="Day of Year" if show_xticks else None,
        showticklabels=show_xticks,
        ticks="outside" if show_xticks else "",
        row=i, col=1,
    )
    fig.update_yaxes(
        title_text="Hour (UTC)",
        row=i, col=1,
        autorange="reversed",
        tickmode="array",
        tickvals=list(range(0, 24, 2)),
        ticks="outside",
    )

# Reduce total height since subplots are closer together
fig.update_layout(
    title=f"{prefix} – Residual Load Heatmaps",
    height=max(300, 200 * n_rows),   # was 260 * n_rows
    margin=dict(l=10, r=10, t=60, b=10),
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Notes"):
    st.markdown("""
- **Residual / Load** is clipped to [0, 1] and blank when actual load ≤ 0 or missing.
- Leap years will show an extra column (**DOY=366**) when present.
- The heatmap aggregates multiple points per (hour, DOY) using the **mean**.
""")
