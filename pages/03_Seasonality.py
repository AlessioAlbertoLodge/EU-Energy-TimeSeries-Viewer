# pages/03_🍂_Seasonality.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_loader import load_dataset, list_prefixes
from src.filters import infer_time_bounds
from src.seasonality import (
    years_available,
    build_long_by_source,
    split_by_year_and_season,
    SEASON_ORDER,
)

st.set_page_config(page_title="Seasonality (Daily Profiles by Season & Year)", layout="wide")
st.title("Seasonality – Daily Profiles by Season & Year")
st.caption(
    "Per **year** and **season**, show all days' 24-hour profiles overlaid.\n"
    "Section 1: **Solar**, then Section 2: **Wind (aggregated)**.\n"
    "X = time of day (UTC hours), Y = generation (MW)."
)

# --- Data path (same default) ---
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
    st.error("No prefixes discovered in this CSV.")
    st.stop()

# Default to NL if present
sorted_prefixes = sorted(prefixes)
default_prefix = "NL" if "NL" in prefixes else sorted_prefixes[0]

col1, col2 = st.columns([1, 2], vertical_alignment="center")
with col1:
    prefix = st.selectbox(
        "Country / Zone prefix",
        options=sorted_prefixes,
        index=sorted_prefixes.index(default_prefix),
    )
with col2:
    st.write(f"Available UTC window: **{min_t} → {max_t}**")

# Optional year filter
all_years = years_available(df)
years_sel = st.multiselect(
    "Years to include",
    options=all_years,
    default=all_years
)
if not years_sel:
    st.info("Select at least one year.")
    st.stop()

def _draw_section(source: str, section_title: str):
    """Draw a grid: rows = years, cols = 4 (seasons)."""
    st.subheader(section_title)

    with st.spinner(f"Preparing {source} daily curves..."):
        # Build long-format daily curves for the selected source
        long_df = build_long_by_source(df, prefix, source=source)
        if long_df.empty:
            st.warning(f"No {source} data for {prefix}.")
            return
        # keep only selected years
        long_df = long_df[long_df["year"].isin(years_sel)]
        grid = split_by_year_and_season(long_df)  # {year: {season: df}}

    n_rows = len(years_sel)
    n_cols = 4  # Winter, Spring, Summer, Autumn

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        shared_xaxes=False, shared_yaxes=False,
        horizontal_spacing=0.04, vertical_spacing=0.06,
        subplot_titles=[f"{y} – {s}" for y in years_sel for s in SEASON_ORDER],
    )

    # For consistent x range (hours 0..23)
    x_hours = list(range(24))

    # Add daily traces (thin, transparent)
    for r, year in enumerate(years_sel, start=1):
        for c, season in enumerate(SEASON_ORDER, start=1):
            dfs = grid.get(year, {}).get(season, pd.DataFrame())
            if dfs.empty:
                # add a tiny invisible trace to keep axes stable
                fig.add_trace(go.Scatter(x=[0, 23], y=[0, 0], mode="lines", line=dict(width=0.5), opacity=0.0, showlegend=False), row=r, col=c)
            else:
                for date_val, dfd in dfs.groupby("date"):
                    # ensure full 24-hour coverage per day
                    day = dfd.set_index("hour").reindex(x_hours)["value"]
                    fig.add_trace(
                        go.Scatter(
                            x=x_hours,
                            y=day.values,
                            mode="lines",
                            line=dict(width=1),
                            opacity=0.25,           # slightly transparent
                            name=str(date_val),
                            hovertemplate=(
                                f"Year: {year} • {season}<br>"
                                f"Date: {date_val}<br>"
                                "Hour: %{x}:00<br>"
                                "Value: %{y:.2f} MW<extra></extra>"
                            ),
                            showlegend=False,
                        ),
                        row=r, col=c
                    )

            # Axes cosmetics
            # Only show x labels on bottom row
            show_xticks = (r == n_rows)
            fig.update_xaxes(
                title_text="Hour of Day (UTC)" if show_xticks else None,
                showticklabels=show_xticks,
                tickmode="array",
                tickvals=[0, 6, 12, 18, 23],
                ticks="outside" if show_xticks else "",
                row=r, col=c
            )
            # Only show y labels on first column
            show_y = (c == 1)
            fig.update_yaxes(
                title_text="Power (MW)" if show_y else None,
                ticks="outside",
                row=r, col=c
            )

    # Size: compact but readable
    fig.update_layout(
        height=max(400, 180 * n_rows),   # 4 seasons per row; many years ⇒ tall figure
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Render sections ---
_draw_section("solar", "Solar – Daily Profiles by Season & Year")
st.divider()
_draw_section("wind", "Wind (Aggregated) – Daily Profiles by Season & Year")

with st.expander("Notes"):
    st.markdown("""
- **Seasons** use meteorological grouping: **Winter (Dec–Feb)**, **Spring (Mar–May)**, **Summer (Jun–Aug)**, **Autumn (Sep–Nov)**.
- **Wind (Aggregated)** uses `*_wind_generation_actual` if available; otherwise sums `*_wind_onshore_generation_actual` + `*_wind_offshore_generation_actual` (using whichever are present).
- Each line is a **single day** (24 points for 00:00–23:00). Missing hours are shown as gaps.
- X-axis labels appear **only on the bottom row**; Y-axis labels appear **only on the left column** to save space.
""")
