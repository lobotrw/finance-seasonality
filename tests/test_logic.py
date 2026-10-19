"""Offline unit tests for seasonality.py (no network)."""
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import seasonality as s  # noqa: E402


def daily(start, end, values=None, freq="D"):
    idx = pd.date_range(start, end, freq=freq)
    if values is None:
        values = np.linspace(100, 200, len(idx))
    return pd.Series(values, index=idx)


def test_month_return_is_month_end_to_month_end():
    p = daily("2020-01-01", "2020-03-31")
    r = s.monthly_returns(p, asof=dt.date(2021, 1, 1))
    jan_end, feb_end, mar_end = p["2020-01-31"], p["2020-02-29"], p["2020-03-31"]
    assert list(r.index.month) == [2, 3]          # January has no prior month-end -> dropped
    assert r.iloc[0] == pytest.approx(feb_end / jan_end - 1)
    assert r.iloc[1] == pytest.approx(mar_end / feb_end - 1)


def test_partial_first_month_is_excluded():
    p = daily("2014-09-17", "2014-12-31")           # BTC-style mid-month start
    r = s.monthly_returns(p, asof=dt.date(2015, 6, 1))
    assert 9 not in r.index.month


def test_current_unfinished_month_is_excluded():
    p = daily("2026-01-01", "2026-10-19")
    r = s.monthly_returns(p, asof=dt.date(2026, 10, 19))
    assert r.index[-1].month == 9 and r.index[-1].year == 2026


def test_finished_last_month_is_kept():
    p = daily("2026-01-01", "2026-09-30")
    r = s.monthly_returns(p, asof=dt.date(2026, 10, 19))
    assert r.index[-1].month == 9


def test_month_returns_do_not_depend_on_future_data():
    """No lookahead: adding later data never changes an already-complete month."""
    full = daily("2020-01-01", "2021-12-31", values=np.random.default_rng(0).lognormal(0, 0.02, 731).cumprod())
    early = s.monthly_returns(full[:"2020-12-31"], asof=dt.date(2030, 1, 1))
    late = s.monthly_returns(full, asof=dt.date(2030, 1, 1))
    pd.testing.assert_series_equal(early, late.loc[early.index])


def test_stats_are_percent_scaled_once():
    idx = pd.to_datetime(["2020-10-31", "2021-10-31", "2022-10-31"])
    r = pd.Series([0.10, 0.20, -0.30], index=idx)
    st = s.month_stats(r)
    assert st.loc["Oct", "mean_pct"] == pytest.approx(0.0, abs=1e-9)
    assert st.loc["Oct", "median_pct"] == pytest.approx(10.0)
    assert st.loc["Oct", "hit_rate_pct"] == pytest.approx(200 / 3)
    assert st.loc["Oct", "n"] == 3
    assert st.shape[0] == 12 and np.isnan(st.loc["Jan", "mean_pct"])
    assert s.heatmap_table(r).loc[2021, "Oct"] == pytest.approx(20.0)


def test_growth_of_one_starts_at_one():
    g = s.growth_of_one(daily("2020-01-01", "2020-01-10"))
    assert g.iloc[0] == 1.0 and not g.isna().any()
