# pages/06_🌫️_Dunkelflaute_Evaluate.py
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from src.data_loader import load_dataset, list_prefixes
from src.filters import infer_time_bounds
from src.dunkelflaute import find_dunkelflaute_events, extract_event_window

st.set_page_config(page_title="Dunkelflaute – Evaluate", layout="wide")
st.title("Dunkelflaute – Frequencies & Event Plots")

st.caption(
    "Detect extended low-renewable periods (Dunkelflauten). Each event shows −24h / +24h around the event "
    "with load, residual, and prices. Residual is computed as either:\n"
    "- **Mode A (default):** load − (solar + wind_onshore + wind_offshore)\n"
    "- **Mode B:** load − (solar + total wind)"
)

# --- Data path ---
default_csv = r"C:\Users\lodgeaa\Desktop\Private Projects\Dataset1_streamlit_app\data\time_series_60min_singleindex.csv"
data_path = st.text_input("CSV path", value=default_csv)

with st.spinner("Loading dataset..."):
    try:
        df = load_dataset(data_path)
    except Exception as e:
        st.error(f"Could not load CSV. Error: {e}")
        st.stop()

# Parse time bounds & available years for filtering
min_t, max_t = infer_time_bounds(df)
if "utc_timestamp" not in df.columns:
    st.error("Dataset missing 'utc_timestamp' column.")
    st.stop()
ts_all = pd.to_datetime(df["utc_timestamp"], errors="coerce", utc=True)
years_available = sorted({t.year for t in ts_all.dropna()})
default_start_year = 2017 if 2017 in years_available else (years_available[0] if years_available else 2017)

prefixes = list_prefixes(df)
if not prefixes:
    st.error("No prefixes found in this dataset.")
    st.stop()
sorted_prefixes = sorted(prefixes)
default_prefix = "NL" if "NL" in prefixes else sorted_prefixes[0]

c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
with c1:
    prefix = st.selectbox("Country / Zone", options=sorted_prefixes, index=sorted_prefixes.index(default_prefix))
with c2:
    # Dunkelflaute threshold: residual / load >= X%
    thr_pct = st.slider("Residual / Load threshold (%)", min_value=80, max_value=100, value=85, step=1)
    threshold = thr_pct / 100.0
with c3:
    min_hours = st.number_input("Minimum duration (hours)", min_value=24, max_value=240, value=72, step=6)
with c4:
    use_total_wind = st.toggle("Use total wind instead of wind_on+off", value=False)

# Year filter + layout controls
c5, c6, c7 = st.columns([1, 1, 2])
with c5:
    start_year = st.selectbox("Analyze from year (inclusive)", options=years_available, index=years_available.index(default_start_year))
with c6:
    vspace_user = st.slider("Subplot spacing", min_value=0.00, max_value=0.10, value=0.02, step=0.005,
                            help="Gap between stacked subplots (Plotly max shrinks as rows increase).")
with c7:
    row_height = st.slider("Row height (px per subplot)", min_value=120, max_value=400, value=260, step=10)

st.write(
    f"Detecting where residual/load ≥ **{thr_pct}%** for ≥ **{int(min_hours)} h** "
    f"(Mode: {'Total Wind' if use_total_wind else 'Separated Wind Components'}), "
    f"**from {start_year} onward**."
)

# --- Filter DF by year for detection (plots still use full DF for ±24h padding) ---
start_dt = pd.Timestamp(f"{start_year}-01-01 00:00:00", tz="UTC")
df_detect = df[ts_all >= start_dt].reset_index(drop=True)

# --- Find events ---
with st.spinner("Scanning for Dunkelflauten..."):
    events = find_dunkelflaute_events(
        df_detect,
        prefix,
        ratio_threshold=threshold,   # interpreted as residual/load >= threshold
        min_duration_hours=int(min_hours),
        use_total_wind=use_total_wind
    )

n_events = len(events)
st.subheader(f"Detected events from {start_year}: {n_events}")

# --- Number of events to show ---
if n_events == 0:
    st.info("No events found for the chosen parameters and time window.")
    st.stop()
elif n_events == 1:
    st.write("Only one event detected.")
    max_to_show = 1
else:
    max_to_show = st.slider("Number of events to plot", min_value=1, max_value=n_events, value=min(8, n_events))

sel_events = events[:max_to_show]
rows = len(sel_events)

# Compute safe vertical spacing for Plotly
if rows <= 1:
    vspace_effective = 0.0
else:
    max_allowed = 1.0 / (rows - 1) - 1e-6  # tiny buffer for float precision
    vspace_effective = min(vspace_user, max_allowed)
    if vspace_effective < vspace_user:
        st.caption(
            f"ℹ️ Subplot spacing auto-clamped from **{vspace_user:.4f}** to **{vspace_effective:.4f}** "
            f"due to Plotly’s limit for **{rows}** rows."
        )

# --- Build stacked plot ---
fig = make_subplots(
    rows=rows,
    cols=1,
    shared_xaxes=False,
    shared_yaxes=False,
    vertical_spacing=vspace_effective,
    specs=[[{"secondary_y": True}] for _ in sel_events],
    subplot_titles=[
        f"{i+1}. {ev.start.strftime('%Y-%m-%d %H:%M')} → {ev.end.strftime('%Y-%m-%d %H:%M')}  ({ev.duration_hours} h)"
        for i, ev in enumerate(sel_events)
    ],
)

for i, ev in enumerate(sel_events, start=1):
    w = extract_event_window(
        df, prefix, ev,
        hours_before=24, hours_after=24,
        use_total_wind=use_total_wind
    )

    # Load (red)
    fig.add_trace(
        go.Scatter(
            x=w["utc_timestamp"], y=w["load_MW"],
            mode="lines", line=dict(width=2, color="red"),
            name="Actual Load (MW)",
            hovertemplate="%{x|%Y-%m-%d %H:%M} UTC<br>Load: %{y:.0f} MW<extra></extra>",
            showlegend=False,
        ),
        row=i, col=1, secondary_y=False
    )

    # Residual (blue)
    fig.add_trace(
        go.Scatter(
            x=w["utc_timestamp"], y=w["residual_MW"],
            mode="lines", line=dict(width=2, color="blue"),
            name="Residual Load (MW)",
            hovertemplate="%{x|%Y-%m-%d %H:%M} UTC<br>Residual: %{y:.0f} MW<extra></extra>",
            showlegend=False,
        ),
        row=i, col=1, secondary_y=False
    )

    # Price (right axis)
    if "price" in w.columns and w["price"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=w["utc_timestamp"], y=w["price"],
                mode="lines", line=dict(width=1.5, color="orange", dash="dot"),
                name="Price (€/MWh)",
                hovertemplate="%{x|%Y-%m-%d %H:%M} UTC<br>Price: %{y:.2f}<extra></extra>",
                showlegend=False,
            ),
            row=i, col=1, secondary_y=True
        )

    # Shaded event duration
    fig.add_vrect(
        x0=w.attrs.get("event_start"), x1=w.attrs.get("event_end"),
        fillcolor="gray", opacity=0.15, line_width=0,
        row=i, col=1
    )

    # X-axis labels
    fig.update_xaxes(
        showticklabels=True,
        tickformat="%b %d\n%H:%M",
        row=i, col=1
    )
    fig.update_yaxes(title_text="Power (MW)", row=i, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Price (€/MWh)", row=i, col=1, secondary_y=True)

fig.update_layout(
    height=max(400, int(row_height * rows)),
    margin=dict(l=10, r=10, t=60, b=10),
)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------------
# TABLES
# ------------------------------------------------------------------------------------
st.header("Event Tables")

# Helper to build metrics per event using only the event core (no ±24h padding)
def _event_metrics_df(events_list):
    rows = []
    load_col = f"{prefix}_load_actual_entsoe_transparency"
    price_col = f"{prefix}_price_day_ahead" if f"{prefix}_price_day_ahead" in df.columns else None

    for idx, ev in enumerate(events_list, start=1):
        # limit to the core event interval in absolute time
        core_mask = (ts_all >= ev.start) & (ts_all <= ev.end)

        # Load (MW) and residual (MW) for the full dataframe, but select core
        # We reuse extract_event_window to honor the 'use_total_wind' toggle correctly,
        # then crop to the core timestamps to avoid recomputing wind/solar here.
        w_full = extract_event_window(df, prefix, ev, hours_before=0, hours_after=0, use_total_wind=use_total_wind)

        # If for any reason w_full is empty (shouldn't be), skip safely
        if w_full.empty:
            continue

        # Use w_full directly; it already corresponds exactly to event start..end
        load_series = pd.to_numeric(w_full["load_MW"], errors="coerce").fillna(0.0)
        residual_series = pd.to_numeric(w_full["residual_MW"], errors="coerce").fillna(0.0)
        vre_series = (load_series - residual_series).clip(lower=0.0)

        # Energy (MWh) since data are hourly
        total_load_MWh = float(load_series.sum())
        total_vre_MWh = float(vre_series.sum())

        # VRE share over the event (requested as 'residual load in % (so VRE/total)')
        # To avoid confusion, we explicitly compute VRE share (%)
        vre_share_pct = float((total_vre_MWh / total_load_MWh) * 100.0) if total_load_MWh > 0 else np.nan

        # Average price during event core
        if price_col is not None:
            price_series = pd.to_numeric(w_full["price"], errors="coerce")
            avg_price = float(price_series.mean(skipna=True)) if price_series.notna().any() else np.nan
        else:
            avg_price = np.nan

        rows.append({
            "Event #": idx,
            "Start (UTC)": ev.start,
            "End (UTC)": ev.end,
            "Duration (h)": ev.duration_hours,
            "Total Load (MWh)": round(total_load_MWh, 2),
            "Total VRE (MWh)": round(total_vre_MWh, 2),
            "VRE share (%)": round(vre_share_pct, 2),
            "Avg Price": round(avg_price, 2) if not np.isnan(avg_price) else np.nan,
        })
    return pd.DataFrame(rows)

# ---- Table 1: detailed list of events (sortable) ----
st.subheader("Table 1 — Detailed Event List")
sort_choice = st.selectbox(
    "Order table by",
    options=["Chronological (start time)", "Duration (desc)", "VRE share (asc)", "VRE share (desc)"],
    index=0
)

events_df = _event_metrics_df(events)

if not events_df.empty:
    if sort_choice == "Duration (desc)":
        events_df = events_df.sort_values(["Duration (h)", "Start (UTC)"], ascending=[False, True]).reset_index(drop=True)
    elif sort_choice == "VRE share (asc)":
        events_df = events_df.sort_values(["VRE share (%)", "Start (UTC)"], ascending=[True, True]).reset_index(drop=True)
    elif sort_choice == "VRE share (desc)":
        events_df = events_df.sort_values(["VRE share (%)", "Start (UTC)"], ascending=[False, True]).reset_index(drop=True)
    else:
        events_df = events_df.sort_values(["Start (UTC)"]).reset_index(drop=True)
else:
    events_df = pd.DataFrame(columns=[
        "Event #","Start (UTC)","End (UTC)","Duration (h)",
        "Total Load (MWh)","Total VRE (MWh)","VRE share (%)","Avg Price"
    ])

st.dataframe(
    events_df,
    use_container_width=True,
    hide_index=True
)

# ---- Table 2: counts by year (≥72h at the selected threshold) ----
st.subheader("Table 2 — Count of Dunkelflauten by Year (≥72h)")

counts_rows = []
for yr in range(start_year, (max_t.year if isinstance(max_t, pd.Timestamp) else years_available[-1]) + 1):
    n = sum((ev.start.year == yr) and (ev.duration_hours >= 72) for ev in events)  # fixed 72h per spec
    counts_rows.append({"Year": yr, "Events (≥72h)": n})

counts_df = pd.DataFrame(counts_rows)
st.dataframe(counts_df, use_container_width=True, hide_index=True)

with st.expander("Notes & Tips"):
    st.markdown(f"""
- **Event rule:** residual / load ≥ **{thr_pct}%** (i.e., wind+solar share ≤ {100 - thr_pct}%).
- **Residual definition**
  - **Mode A (default):** Load − (Solar + Wind Onshore + Wind Offshore)
  - **Mode B (toggle):** Load − (Solar + Total Wind)
- **Table 1** uses the core event interval only (no ±24h padding) to compute:
  - Total Load (MWh), Total VRE (MWh), VRE share (%), and Avg Price.
- **Table 2** reports counts **by event start year** for events with **duration ≥ 72h** at the selected threshold.
- Use **Subplot spacing** and **Row height** to fit many plots; **From year** restricts detection window.
""")
