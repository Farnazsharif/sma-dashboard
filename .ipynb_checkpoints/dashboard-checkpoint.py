#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from datetime import date, time as dt_time

from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output

# 🌟 Import your function from sma_strategy.py
from sma_strategy import compute_runs_dashboard

# ===== CONFIG =====
DEFAULT_TICKERS = ["PLTR"]
REFRESH_MS = 55_000   # 55 seconds
ROWS_PER_PAGE = 15

# ===== DASH APP SETUP =====
app = Dash(
    __name__,
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1.0, maximum-scale=1.0",
        }
    ],
)
server = app.server  # in case you later want deployment

# ---- one initial computation (so app starts with something) ----
today = date.today()

runs_up_init, runs_down_init, fig_init = compute_runs_dashboard(
    tickers=DEFAULT_TICKERS,
    start_date_data=today,
    end_date_data=today,
    start_time_data=dt_time(0, 0),
    end_time_data=dt_time(20, 0),
    start_date_trade=today,
    end_date_trade=today,
    start_time_trade=dt_time(0, 0),
    end_time_trade=dt_time(23, 59),
)

# Make sure we always have DataFrames
if runs_up_init is None:
    runs_up_init = pd.DataFrame()
if runs_down_init is None:
    runs_down_init = pd.DataFrame()

# ===== LAYOUT =====
app.layout = html.Div(
    style={
        "fontFamily": "Arial, sans-serif",
        "padding": "10px",
        "maxWidth": "1200px",
        "margin": "0 auto",
    },
    children=[
        html.H2(
            "SMA Runs Dashboard",
            style={"textAlign": "center", "marginBottom": "10px"},
        ),

        # Last update time
        html.Div(
            id="last-update",
            style={
                "textAlign": "right",
                "fontSize": "12px",
                "marginBottom": "8px",
                "color": "#555",
            },
        ),

        # Ticker selector
        html.Div(
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "8px",
                "marginBottom": "10px",
            },
            children=[
                html.Label("Ticker:", style={"fontWeight": "bold"}),
                dcc.Input(
                    id="ticker-input",
                    type="text",
                    value="PLTR",
                    style={"width": "80px"},
                ),
            ],
        ),

        # FIGURE
        dcc.Graph(
            id="runs-fig",
            figure=fig_init,
            style={"height": "420px"},
        ),

        html.H4("Up Runs", style={"marginTop": "15px"}),

        dash_table.DataTable(
            id="table-up",
            columns=[{"name": c, "id": c} for c in runs_up_init.columns],
            data=runs_up_init.to_dict("records"),
            page_size=ROWS_PER_PAGE,
            style_table={"overflowX": "auto", "border": "1px solid #ddd"},
            style_header={
                "backgroundColor": "#f7f7f7",
                "fontWeight": "bold",
                "borderBottom": "1px solid #ccc",
            },
            style_cell={
                "fontSize": 12,
                "padding": "4px",
                "textAlign": "center",
                "whiteSpace": "normal",
            },
        ),

        html.H4("Down Runs", style={"marginTop": "15px"}),

        dash_table.DataTable(
            id="table-down",
            columns=[{"name": c, "id": c} for c in runs_down_init.columns],
            data=runs_down_init.to_dict("records"),
            page_size=ROWS_PER_PAGE,
            style_table={"overflowX": "auto", "border": "1px solid #ddd"},
            style_header={
                "backgroundColor": "#f7f7f7",
                "fontWeight": "bold",
                "borderBottom": "1px solid #ccc",
            },
            style_cell={
                "fontSize": 12,
                "padding": "4px",
                "textAlign": "center",
                "whiteSpace": "normal",
            },
        ),

        # Interval for auto-refresh, like your while True loop
        dcc.Interval(
            id="interval-component",
            interval=REFRESH_MS,  # in milliseconds
            n_intervals=0,
        ),
    ],
)

# ===== CALLBACK: auto-refresh data and fig =====
@app.callback(
    [
        Output("table-up", "data"),
        Output("table-up", "columns"),
        Output("table-down", "data"),
        Output("table-down", "columns"),
        Output("runs-fig", "figure"),
        Output("last-update", "children"),
    ],
    [
        Input("interval-component", "n_intervals"),
        Input("ticker-input", "value"),
    ],
)
def update_dashboard(n, ticker_value):
    # use ticker input; fall back to PLTR if empty
    ticker_value = (ticker_value or "PLTR").strip().upper()
    tickers = [ticker_value]

    today = date.today()

    runs_up, runs_down, fig = compute_runs_dashboard(
        tickers=tickers,
        start_date_data=today,
        end_date_data=today,
        start_time_data=dt_time(0, 0),
        end_time_data=dt_time(20, 0),
        start_date_trade=today,
        end_date_trade=today,
        start_time_trade=dt_time(0, 0),
        end_time_trade=dt_time(23, 59),
    )

    if runs_up is None:
        runs_up = pd.DataFrame()
    if runs_down is None:
        runs_down = pd.DataFrame()

    up_cols = [{"name": c, "id": c} for c in runs_up.columns]
    down_cols = [{"name": c, "id": c} for c in runs_down.columns]

    last_update_str = (
        "Last update: "
        + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    return (
        runs_up.to_dict("records"),
        up_cols,
        runs_down.to_dict("records"),
        down_cols,
        fig,
        last_update_str,
    )


if __name__ == "__main__":
    # host='0.0.0.0' so your iPhone can see it on the network
    app.run(host="0.0.0.0", port=8050, debug=False)


