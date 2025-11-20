import streamlit as st
import pandas as pd
from datetime import date, time as dt_time

from sma_strategy import compute_runs_dashboard  # your functions live here

# ───────────────────────────────────────────────
# Streamlit page config
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="SMA Runs Dashboard",
    layout="wide",
)

st.title("📈 SMA200 / SMA50 Runs Dashboard")

st.markdown(
    """
This dashboard:
1. Fetches intraday minute data from Polygon  
2. Computes indicators (SMA, EMA, RSI, TSI, z-scores, etc.)  
3. Detects **UP / DOWN runs** based on SMA200 velocity z-score  
4. Shows:
   - An interactive Plotly chart  
   - Summary tables for UP and DOWN runs  
"""
)

# ───────────────────────────────────────────────
# Sidebar controls
# ───────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

default_ticker = "PLTR"
ticker_str = st.sidebar.text_input("Ticker(s) (comma-separated)", value=default_ticker)
tickers = [t.strip().upper() for t in ticker_str.split(",") if t.strip()]

today = date.today()

data_date = st.sidebar.date_input(
    "Data date (for fetching Polygon minute bars)",
    value=today,
)

start_time_data = st.sidebar.time_input("Start time (data window, ET)", value=dt_time(4, 0))
end_time_data   = st.sidebar.time_input("End time (data window, ET)",   value=dt_time(20, 0))

st.sidebar.markdown("---")
st.sidebar.markdown("**Trading / analysis window (subset of data window)**")

start_time_trade = st.sidebar.time_input("Start time (trade window, ET)", value=dt_time(9, 30))
end_time_trade   = st.sidebar.time_input("End time (trade window, ET)",   value=dt_time(16, 0))

multiplier = st.sidebar.number_input("Bar size multiplier", min_value=1, value=1, step=1)
timespan   = st.sidebar.selectbox("Timespan", options=["minute", "hour"], index=0)
limit      = st.sidebar.number_input("Max bars to fetch", min_value=1000, value=50000, step=1000)

run_button = st.sidebar.button("🔄 Run / Refresh")

# ───────────────────────────────────────────────
# Main logic
# ───────────────────────────────────────────────
if run_button:
    try:
        with st.spinner("Fetching data from Polygon and computing runs..."):
            runs_up, runs_down, fig = compute_runs_dashboard(
                tickers=tickers,
                start_date_data=data_date,
                end_date_data=data_date,
                start_time_data=start_time_data,
                end_time_data=end_time_data,
                start_date_trade=data_date,
                end_date_trade=data_date,
                start_time_trade=start_time_trade,
                end_time_trade=end_time_trade,
                multiplier=multiplier,
                timespan=timespan,
                limit=limit,
            )

        # Plot
        st.subheader("📊 SMA Runs Chart")
        st.plotly_chart(fig, use_container_width=True)

        # Tables
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🟩 UP Runs")
            if isinstance(runs_up, pd.DataFrame) and not runs_up.empty:
                st.dataframe(runs_up)
            else:
                st.info("No UP runs detected in the selected window.")

        with col2:
            st.subheader("🟥 DOWN Runs")
            if isinstance(runs_down, pd.DataFrame) and not runs_down.empty:
                st.dataframe(runs_down)
            else:
                st.info("No DOWN runs detected in the selected window.")

    except Exception as e:
        st.error(f"Error while computing runs: {e}")
else:
    st.info("Set your parameters in the sidebar and click **Run / Refresh**.")
