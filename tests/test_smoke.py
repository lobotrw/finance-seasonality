"""Smoke test: the app runs without exceptions for every ticker and edge-case start years.
Needs network access (Yahoo Finance)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "1_📈_Seasonality.py")


@pytest.mark.parametrize("ticker", ["^GSPC", "ETH-USD", "BTC-USD", "^IXIC"])
def test_app_runs_for_each_ticker(ticker, monkeypatch):
    monkeypatch.chdir(ROOT)
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.sidebar.selectbox[0].set_value(ticker).run()
    assert not at.exception, [e.message for e in at.exception]


@pytest.mark.parametrize("year", [1960, 2014, None])   # None = current year (partial year of data)
def test_app_runs_for_edge_start_years(year, monkeypatch):
    monkeypatch.chdir(ROOT)
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    slider = at.slider[0]
    slider.set_value(year if year is not None else slider.max).run()
    assert not at.exception, [e.message for e in at.exception]


def test_about_page_runs(monkeypatch):
    monkeypatch.chdir(ROOT)
    at = AppTest.from_file(str(ROOT / "pages" / "2_@_About.py"), default_timeout=60)
    at.run()
    assert not at.exception
