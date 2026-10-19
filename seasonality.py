"""Pure, testable seasonality calculations used by the Streamlit app."""
from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def monthly_returns(prices: pd.Series, asof: _dt.date | None = None) -> pd.Series:
    """Simple monthly returns from month-end closes, complete months only.

    * Return of month m = close(last day of m) / close(last day of m-1) - 1.
      The first month of the data has no prior month-end close and is dropped,
      so a series that starts mid-month never yields a partial first month.
    * The month containing ``asof`` (default: today) is not finished and is
      dropped, as is any later month.
    """
    prices = prices.dropna()
    if prices.empty:
        return pd.Series(dtype=float)
    month_end_close = prices.groupby(prices.index.to_period("M")).last()
    rets = month_end_close.pct_change().dropna()
    asof = asof or _dt.date.today()
    current = pd.Period(asof, freq="M")
    rets = rets[rets.index < current]
    rets.index = rets.index.to_timestamp(how="end").normalize()
    return rets


def heatmap_table(rets: pd.Series) -> pd.DataFrame:
    """Year x month table of monthly returns in percent (12 columns always)."""
    df = pd.DataFrame({"Year": rets.index.year, "Month": rets.index.month, "r": rets.values * 100})
    table = df.pivot_table(index="Year", columns="Month", values="r").reindex(columns=range(1, 13))
    table.columns = MONTHS
    table.index.name = "Year"
    return table


def month_stats(rets: pd.Series) -> pd.DataFrame:
    """Per-calendar-month statistics in percent: mean, median, hit rate, n, t-stat."""
    df = pd.DataFrame({"Month": rets.index.month, "r": rets.values * 100})
    g = df.groupby("Month")["r"]
    out = pd.DataFrame({
        "mean_pct": g.mean(),
        "median_pct": g.median(),
        "hit_rate_pct": g.apply(lambda s: (s > 0).mean() * 100),
        "n": g.count(),
        "std_pct": g.std(ddof=1),
    }).reindex(range(1, 13))
    out["t_stat"] = out["mean_pct"] / (out["std_pct"] / np.sqrt(out["n"]))
    out.index = MONTHS
    out.index.name = "Month"
    return out


def growth_of_one(prices: pd.Series) -> pd.Series:
    """Growth of $1 invested at the first close (starts at exactly 1.0, no NaN)."""
    prices = prices.dropna()
    return prices / prices.iloc[0]
