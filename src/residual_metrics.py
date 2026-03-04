# src/residual_metrics.py
import numpy as np
import pandas as pd
from scipy.signal import detrend

from .seasonality_aggregated import aggregate_energy, normalize_by_year_mean


def get_normalized_daily_residual(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Return normalized daily aggregated residual load dataframe."""
    agg = aggregate_energy(df, prefix, "residual", freq="D")
    normed = normalize_by_year_mean(agg)
    return normed


def _find_consecutive_periods(mask: pd.Series) -> list:
    """Identify consecutive True segments in a boolean series."""
    streaks = []
    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            streaks.append((start, i - 1, i - start))
            start = None
    if start is not None:
        streaks.append((start, len(mask) - 1, len(mask) - start))
    return streaks


def compute_residual_metrics(normed_df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
    """
    Compute metrics for residual load (normalized daily data):
    - days_below_<threshold>
    - num_3day_streaks
    - longest_streak
    """
    out_rows = []
    for year, dfy in normed_df.groupby("year"):
        vals = dfy["energy_norm"].values
        below = vals < threshold
        n_below = np.sum(below)

        streaks = _find_consecutive_periods(pd.Series(below))
        streak_lengths = [l for _, _, l in streaks]
        long_streaks = [l for l in streak_lengths if l >= 3]

        out_rows.append({
            "year": int(year),
            f"days_below_{int(threshold*100)}pct": int(n_below),
            "num_streaks_3days": len(long_streaks),
            "longest_streak": int(max(streak_lengths)) if streak_lengths else 0
        })

    return pd.DataFrame(out_rows)


def compute_frequency_analysis(normed_df: pd.DataFrame) -> pd.DataFrame:
    """Compute FFT amplitude spectrum for seasonality analysis."""
    vals = normed_df["energy_norm"].dropna().values
    detr = detrend(vals)
    n = len(detr)
    freqs = np.fft.rfftfreq(n, d=1)  # 1 sample per day
    amps = np.abs(np.fft.rfft(detr))
    return pd.DataFrame({"frequency_1_per_day": freqs, "amplitude": amps})
