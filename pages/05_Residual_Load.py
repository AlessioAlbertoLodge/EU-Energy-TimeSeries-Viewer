# pages/05_⚡_Residual_Frequencies_and_Metrics.py
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from src.data_loader import load_dataset, list_prefixes
from src.filters import infer_time_bounds
from src.seasonality_aggregated import make_month_ticks, pivot_energy_norm
from src.residual_metrics import (
    get_normalized_daily_residual,
    compute_residual_metrics,
    compute_frequency_analysis,
)

st.set_page_config(page_title="Residual Load: Frequencies & Metrics", layout="wide")
st.title("Residual Load – Frequencies and Metrics")
st.caption("""
Explore **residual load** dynamics normalized to each year's average daily energy.
You can adjust the low-load threshold and examine both temporal and frequency patterns.
""")

# --- Load dataset ---
default_csv = r"C:\Users\lodgeaa\Desktop\Private Projects\Dataset1_streamlit_app\data\time_series_60min_singleindex.csv"
data_path = st.text_input("CSV path", value=default_csv)

with st.spinner("Loading dataset..."):
    try:
        df = load_dataset(data_path)
    except Exception as e:
        st.error(f"Could not load dataset. Error: {e}")
        st.stop()

min_t, max_t = infer_time_bounds(df)
prefixes = list_prefixes(df)
sorted_prefixes = sorted(prefixes)
default_prefix = "NL" if "NL" in prefixes else sorted_prefixes[0]
prefix = st.selectbox("Select country prefix", options=sorted_prefixes, index=sorted_prefixes.index(default_prefix))

# --- Threshold selector ---
col1, col2 = st.columns([1, 4])
with col1:
    threshold_pct = st.slider("Low-load threshold (%)", 5, 95, 30, step=1)
threshold = threshold_pct / 100.0
with col2:
    st.write(f"**Days considered 'low residual load'** are those below **{threshold_pct}%** of that year's average residual energy.")

# --- Prepare normalized daily residual data ---
with st.spinner("Computing normalized residual load..."):
    normed = get_normalized_daily_residual(df, prefix)
if normed.empty:
    st.error("No residual data found.")
    st.stop()

# --- Top: daily normalized residual load (1 subplot per year) ---
st.header("Normalized Daily Residual Load (by Year)")
month_ticks = make_month_ticks("D")
by_year = pivot_energy_norm(normed)
years = sorted(by_year.keys())

fig = make_subplots(
    rows=len(years), cols=1,
    shared_xaxes=False, shared_yaxes=False,
    vertical_spacing=0.03,
    subplot_titles=[f"{y}" for y in years],
)
for i, y in enumerate(years, start=1):
    d = by_year[y]
    fig.add_trace(
        go.Scatter(
            x=d["period"],
            y=d["energy_norm"],
            mode="lines",
            line=dict(width=3),
            showlegend=False,
            hovertemplate="DOY %{x}<br>Normalized Residual: %{y:.2f}<extra></extra>",
        ),
        row=i, col=1
    )
    fig.add_hline(
        y=threshold, line_dash="dot", line_color="red", row=i, col=1,
        annotation_text=f"{threshold_pct}%", annotation_position="top right"
    )
    show_x = (i == len(years))
    fig.update_xaxes(
        title_text="Day of Year" if show_x else None,
        showticklabels=show_x,
        tickmode="array",
        tickvals=list(month_ticks.keys()),
        ticktext=list(month_ticks.values()),
        ticks="outside" if show_x else "",
        row=i, col=1
    )
    fig.update_yaxes(title_text="Residual / Mean", row=i, col=1)

fig.update_layout(
    height=max(400, 200 * len(years)),
    margin=dict(l=10, r=10, t=40, b=10),
)
st.plotly_chart(fig, use_container_width=True)

# --- Metrics table ---
st.header(f"Yearly Low-Residual Metrics (< {threshold_pct}% of yearly mean)")
metrics = compute_residual_metrics(normed, threshold=threshold)
st.dataframe(metrics.style.format(precision=0), use_container_width=True)

# --- Frequency analysis ---
st.header("Frequency & Seasonality Analysis")

spec = compute_frequency_analysis(normed)
fig_fft = go.Figure()
fig_fft.add_trace(
    go.Scatter(
        x=spec["frequency_1_per_day"],
        y=spec["amplitude"],
        mode="lines",
        line=dict(width=2),
    )
)
fig_fft.update_layout(
    title="FFT Spectrum (Residual Load – Normalized)",
    xaxis_title="Frequency (1/day)",
    yaxis_title="Amplitude",
    xaxis=dict(type="log", showgrid=True),
    yaxis=dict(showgrid=True),
    height=400,
)
st.plotly_chart(fig_fft, use_container_width=True)

# --- Autocorrelation plot ---
st.subheader("Autocorrelation of Normalized Residual Load")
vals = normed["energy_norm"].dropna().values
lags = np.arange(1, 181)
autocorr = [np.corrcoef(vals[:-lag], vals[lag:])[0, 1] for lag in lags]

fig_ac = go.Figure()
fig_ac.add_trace(go.Scatter(x=lags, y=autocorr, mode="lines", line=dict(width=2)))
fig_ac.update_layout(
    title="Autocorrelation (lag in days)",
    xaxis_title="Lag (days)",
    yaxis_title="Correlation",
    height=400,
)
st.plotly_chart(fig_ac, use_container_width=True)

with st.expander("Analysis Notes"):
    st.markdown(f"""
- **Threshold control:** adjustable from **5 %** to **95 %** of each year's average residual energy.  
- **Metrics:**  
  - *Days below {threshold_pct}%*: count of days with very low residual load.  
  - *3+ day streaks*: independent low-load runs (≥ 3 consecutive days).  
  - *Longest streak*: maximum duration of any low-load run.  
- **Red dotted line** marks the chosen threshold in each year's plot.  
- **FFT spectrum** indicates recurring periodicities (peaks near 1/365 ≈ annual or 1/7 ≈ weekly).  
- **Autocorrelation** quantifies temporal persistence and cyclical patterns.  
""")
