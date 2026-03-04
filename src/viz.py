# src/viz.py
import pandas as pd
import plotly.graph_objects as go

def make_dualaxis_figure(
    df: pd.DataFrame,
    x_col: str,
    power_cols: list,
    price_cols: list,
    title: str,
    left_label: str = "Power (MW)",
    right_label: str = "Price"
):
    fig = go.Figure()

    # Left axis: power series
    for col in power_cols:
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[col],
                mode="lines",
                name=col,  # full column name for clarity
                hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>%{y:.2f}<extra>%{fullData.name}</extra>",
                connectgaps=False,
                yaxis="y1",
            )
        )

    # Right axis: price series
    for col in price_cols:
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[col],
                mode="lines",
                name=col,
                hovertemplate="%{x|%Y-%m-%d %H:%M UTC}<br>%{y:.2f}<extra>%{fullData.name}</extra>",
                connectgaps=False,
                yaxis="y2",
            )
        )

    fig.update_layout(
        title=title,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
        legend_title="Series (click to toggle)",
        xaxis=dict(title="Time (UTC, 60-min)"),
        yaxis=dict(title=left_label, rangemode="tozero"),
        yaxis2=dict(title=right_label, overlaying="y", side="right", rangemode="tozero"),
    )
    return fig
