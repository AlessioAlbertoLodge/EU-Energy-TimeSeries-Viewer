# pages/04_📆_Seasonality_Aggregated.py
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_loader import load_dataset, list_prefixes
from src.filters import infer_time_bounds
from src.seasonality_aggregated import (
    aggregate_energy,
    normalize_by_year_mean,
    pivot_energy,
    pivot_energy_norm,
    make_month_ticks,
)

st.set_page_config(page_title="Seasonality (Aggregated Energy)", layout="wide")
st.title("Seasonality (Aggregated Energy by Week & Day)")
st.caption(
    "Shows **weekly** and **daily** aggregated energy (MWh) per year for **solar**, **wind**, and **residual load**. "
    "Each year is a **separate subplot** stacked vertically (thick lines, compact). "
    "Below each, the **normalized** versions (÷ yearly average) are also shown."
)

# --- Load dataset ---
default_csv = r"C:\Users\aless\Desktop\Streamlit_Apps_Backup\Dataset1_streamlit_app\data\time_series_60min_singleindex.csv"
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
    st.error("No prefixes found.")
    st.stop()

sorted_prefixes = sorted(prefixes)
default_prefix = "NL" if "NL" in prefixes else sorted_prefixes[0]

col1, col2 = st.columns([1, 2], vertical_alignment="center")
with col1:
    prefix = st.selectbox("Country / Zone prefix", options=sorted_prefixes, index=sorted_prefixes.index(default_prefix))
with col2:
    st.write(f"Available UTC window: **{min_t} → {max_t}**")

sources = ["solar", "wind", "residual"]
titles = {
    "solar": "Solar – Aggregated Energy",
    "wind": "Wind – Aggregated Energy",
    "residual": "Residual Load – Aggregated Energy",
}


def _plot_stacked_per_year(by_year_dict, month_ticks, x_label, y_label, section_title):
    """Render one subplot per year (tight vertical spacing)."""
    years = sorted(by_year_dict.keys())
    if not years:
        st.info("No data to plot.")
        return

    st.subheader(section_title)
    n_rows = len(years)
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=False,
        shared_yaxes=False,
        vertical_spacing=0.03,
        subplot_titles=[f"{y}" for y in years],
    )

    for i, y in enumerate(years, start=1):
        d = by_year_dict[y]
        fig.add_trace(
            go.Scatter(
                x=d["period"],
                y=d.iloc[:, 1],
                mode="lines",
                line=dict(width=3),
                name=str(y),
                showlegend=False,
                # ✅ FIXED hovertemplate (escaped braces)
                hovertemplate="Year: %{customdata}<br>Period: %{x}<br>Value: %{y:.3f}<extra></extra>",
                customdata=[y] * len(d),
            ),
            row=i, col=1
        )

        show_x = (i == n_rows)
        fig.update_xaxes(
            title_text=x_label if show_x else None,
            showticklabels=show_x,
            tickmode="array",
            tickvals=list(month_ticks.keys()),
            ticktext=list(month_ticks.values()),
            ticks="outside" if show_x else "",
            row=i, col=1
        )
        fig.update_yaxes(title_text=y_label, row=i, col=1, ticks="outside")

    fig.update_layout(
        height=max(420, 200 * n_rows),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def _section(prefix, source, freq, normalized=False):
    """Render one section (weekly or daily, normalized or not)."""
    label = "Weekly" if freq == "W" else "Daily"
    month_ticks = make_month_ticks(freq)
    base_title = titles[source]
    section_title = f"{base_title} ({label}{' – Normalized' if normalized else ''})"

    with st.spinner(f"Aggregating {label.lower()} {source} energy..."):
        try:
            agg = aggregate_energy(df, prefix, source, freq=freq)
        except KeyError as e:
            st.warning(str(e))
            return

    if agg.empty:
        st.info(f"No {source} data for {prefix}.")
        return

    if normalized:
        agg = normalize_by_year_mean(agg)
        by_year = pivot_energy_norm(agg)
        y_label = "Normalized (÷ yearly avg)"
    else:
        by_year = pivot_energy(agg)
        y_label = "Energy (MWh)"

    x_label = "Week of Year" if freq == "W" else "Day of Year"
    _plot_stacked_per_year(by_year, month_ticks, x_label, y_label, section_title)


# --- Weekly (raw) ---
st.header("Weekly Aggregated Energy")
for src in sources:
    _section(prefix, src, freq="W", normalized=False)
    st.divider()

# --- Weekly (normalized) ---
st.header("Weekly Aggregated Energy – Normalized (÷ Yearly Average)")
for src in sources:
    _section(prefix, src, freq="W", normalized=True)
    st.divider()

# --- Daily (raw) ---
st.header("Daily Aggregated Energy")
for src in sources:
    _section(prefix, src, freq="D", normalized=False)
    st.divider()

# --- Daily (normalized) ---
st.header("Daily Aggregated Energy – Normalized (÷ Yearly Average)")
for src in sources:
    _section(prefix, src, freq="D", normalized=True)
    st.divider()

with st.expander("Notes"):
    st.markdown("""
- Each **year** is displayed as its **own subplot** (no overlapping lines).
- **Weekly** uses ISO week numbers; **Daily** uses day-of-year.
- **Normalized** = divided by that year's **average** of the chosen period totals (weekly or daily).
- X-axis month markers appear only on the **bottom** subplot of each stacked figure.
- Lines are **thick** and spacing is **compact** for visual comparability across years.
""")
