import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import datetime

import seasonality


#-------------------------------------------------------------------------------------------------------
# Page Configurations
#-------------------------------------------------------------------------------------------------------

st.set_page_config(page_title = "Seasonality", layout="wide", initial_sidebar_state="expanded", 
                   page_icon= "📈")

# Titles
st.header("Seasonality Performance", divider="gray")

st.sidebar.header("Seasonality")


#-------------------------------------------------------------------------------------------------------
# INPUTS
#-------------------------------------------------------------------------------------------------------

ticker = st.sidebar.selectbox(
    'Ticker:', ('^GSPC', 'ETH-USD', 'BTC-USD', "^IXIC"))

max_value = datetime.date.today().year
year = st.slider("Start Year", min_value=1960, max_value=max_value, value=2000, step=1)


#-------------------------------------------------------------------------------------------------------
# GET DATA
#-------------------------------------------------------------------------------------------------------

# Define a function to download data
@st.cache_data
def download_data(ticker, start_date, end_date):
    # auto_adjust=False keeps the "Adj Close" column; multi_level_index=False keeps flat columns (yfinance >= 0.2.48)
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False,
                       multi_level_index=False, progress=False)
    return data

# Start ~2 weeks early so the selected first January has a prior month-end close (it is dropped otherwise).
data = download_data(ticker, datetime.date(year, 1, 1) - datetime.timedelta(days=14), datetime.date.today() + datetime.timedelta(days=1))

if data.empty or len(data) < 2:
    st.warning(f"No price data for {ticker} from {year}. Pick an earlier start year.")
    st.stop()

# Set the app title and sidebar description
if ticker == "^GSPC":
    ticker = "S&P500"
if ticker == "^IXIC":
    ticker = "NASDAQ"


#-------------------------------------------------------------------------------------------------------
# RETURNS (complete months only; see seasonality.py)
#-------------------------------------------------------------------------------------------------------

prices = data["Adj Close"]
first_date = prices.dropna().index[0].date()
if first_date > datetime.date(year, 1, 1):
    st.info(f"{ticker} data starts on {first_date:%Y-%m-%d}, later than the selected start year {year}.")

monthly_returns = seasonality.monthly_returns(prices)
if monthly_returns.empty:
    st.warning("Not enough data for one complete month. Pick an earlier start year.")
    st.stop()
growth = seasonality.growth_of_one(prices[prices.index >= pd.Timestamp(year, 1, 1)])
heatmap_data = seasonality.heatmap_table(monthly_returns)
stats = seasonality.month_stats(monthly_returns)
st.caption(f"Complete months only: {monthly_returns.index[0]:%b %Y} – {monthly_returns.index[-1]:%b %Y} "
           f"({len(monthly_returns)} months). The current, unfinished month and a partial first month are excluded.")

#-------------------------------------------------------------------------------------------------------
# GROWTH OF $1 CHART
ret = go.Figure()
ret.add_trace(go.Scatter(x=growth.index, y=growth, mode='lines', name='Growth of $1',
                         line=dict(color='gray', width=2), fill='tozeroy',
                         fillcolor='rgba(220, 220, 220, 0.5)'))
ret.update_layout(title=f"{ticker} growth of $1 (adjusted close)", yaxis_title="Growth of $1")

#-------------------------------------------------------------------------------------------------------
# MONTHLY RETURNS HEAT MAP (values in %)
heatmap_fig = px.imshow(heatmap_data.round(2),
                        labels=dict(x="Month", y="Year", color="Monthly return (%)"),
                        title=f"Heatmap of monthly returns (%) for {ticker}",
                        color_continuous_scale=["red", "white", "green"],
                        color_continuous_midpoint=0, text_auto=True, height=1000)
heatmap_fig.update_yaxes(title_text="Year")

#-------------------------------------------------------------------------------------------------------
# AVERAGE / MEDIAN MONTHLY RETURNS (values in %, already scaled once)
percentage_changes_fig = go.Figure()
percentage_changes_fig.add_trace(go.Bar(x=stats.index, y=stats["mean_pct"].round(2), name="Mean",
                                        marker=dict(color=stats["mean_pct"], colorscale="rdbu", cmid=0,
                                                    line=dict(color="rgb(0, 0, 0)", width=1)),
                                        customdata=np.stack([stats["n"], stats["hit_rate_pct"], stats["t_stat"]], axis=-1),
                                        hovertemplate="%{x}: mean %{y:.2f}%<br>n=%{customdata[0]}, "
                                                      "positive %{customdata[1]:.0f}%, t=%{customdata[2]:.2f}<extra></extra>"))
percentage_changes_fig.add_trace(go.Bar(x=stats.index, y=stats["median_pct"].round(2), name="Median",
                                        marker=dict(color="rgba(120,120,120,0.6)", line=dict(color="rgb(0,0,0)", width=1))))
percentage_changes_fig.update_layout(title_text='Mean and median monthly return (%)', height=500, barmode="group")
percentage_changes_fig.update_yaxes(title_text="Monthly return (%)")
percentage_changes_fig.add_hline(y=0)
percentage_changes_fig.add_annotation(text="@VanHelsing", showarrow=False, x=1, y=0.1, xref='paper', yref='paper',
                                      xanchor='right', yanchor='bottom', font=dict(size=12, color="grey"))

# Streamlit integration
st.plotly_chart(ret, width='stretch')
st.plotly_chart(heatmap_fig, width='stretch')
st.plotly_chart(percentage_changes_fig, width='content')
st.markdown("**Per-month statistics** (|t| < 2 means the mean is not distinguishable from zero; "
            "with ~10–30 years per month most seasonal 'edges' are noise)")
st.dataframe(stats.drop(columns="std_pct").round(2), width='content')

st.write(
    "About\n",
    "\nSeasonality Performance is not financial advice.\n",
)
