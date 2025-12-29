# Project: ETF Dashboard (Dash)
# File: app.py

from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from data import get_top_holdings, get_daily_returns, get_daily_prices, get_insider_trades, transactions_to_cumm_ts, validate_and_classify_ticker
from multiprocessing import Pool
from components.tables import get_top_holdings_with_performance
from components.charts import combined_chart, corr_heatmap, rolling_insider_chart
from dash.exceptions import PreventUpdate
import json
import numpy as np
import os

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

app.layout = dbc.Container([
    html.H2("ETF Dashboard"),

dbc.Row([
    dbc.Col([
        # dbc.InputGroup([
        html.Label("Enter ETF Ticker:"),
        dbc.Input(id="etf-input", placeholder="Enter ETF (e.g., VOO)", type="text", value="")
            # dbc.Button("Load Data", id="load-button", n_clicks=0,
            #            style={"backgroundColor": "#111", "color": "#8cb4ff"})
    ],
        width=3
    ),
    dbc.Col([
        html.Label("Stock List (comma separated):"),
        dbc.Input(id="stock-list-input", placeholder="AAPL, MSFT, GOOGL", type="text", value="")
    ], width=6),
    dbc.Col([
        dbc.Button("Load Data", id="load-button", n_clicks=0,
                   style={"backgroundColor": "#111", "color": "#8cb4ff", "marginTop": "24px"})
    ], width=3)
], className="mt-4"),
    dbc.Row([
        dbc.Col([
            html.Label("Select Time Period:"),
            dcc.Dropdown(
                id='time-period',
                options=[
                    {"label": "1Y", "value": "1Y"},
                    {"label": "3Y", "value": "3Y"},
                    {"label": "5Y", "value": "5Y"},
                    {"label": "7Y", "value": "7Y"},
                    {"label": "10Y", "value": "10Y"},
                ],
                value='1Y',
                clearable=False,
                disabled=True
            )
        ], width=3),
    dbc.Col([
        html.Label("Return Aggregation Method:"),
        dcc.Dropdown(
            id="cum-method",
            options=[
                {"label": "Cum Sum", "value": "cumsum"},
                {"label": "Cum Product", "value": "cumprod"},
                ],
                value="cumprod",
                clearable=False,
                disabled=True
            )
        ], width=3),
        dbc.Col([
            html.Label("Correlation Frequency:"),
            dcc.Dropdown(
                id="corr-frequency-dropdown",
                options=[
                    {"label": "Daily", "value": "Daily"},
                    {"label": "Weekly", "value": "Weekly"},
                    {"label": "Monthly", "value": "Monthly"},
                ],
                value="Daily",
                clearable=False,
                style={"marginBottom": "10px"},
                disabled=True
            )
        ], width=6)
    ], className="mt-4"),

    html.Hr(),
    dcc.Store(id="ui-state"),
    dcc.Loading(
        id="loading", type="circle",
        children=[html.Div(id='combined-chart-output')]
    ),
], fluid=True)


@app.callback(
    Output('combined-chart-output', 'children'),
    Output('ui-state', 'data'),
    # Output("time-period", "disabled"), 
    Input('load-button', 'n_clicks'),
    State('etf-input', 'value'),
    State("stock-list-input", "value"),
    Input('time-period', 'value'),
    Input('cum-method', 'value'),
    Input('corr-frequency-dropdown','value'),
    prevent_initial_call=True
)

def update_chart(n_clicks, etf, stock_list, period, cum_method, corr_freq):

    if n_clicks is None or n_clicks == 0:
        raise PreventUpdate
    
    etf = (etf or "").strip().upper()
    stock_list = (stock_list or "").strip()

    # ❌ neither provided
    if not etf and not stock_list:
        return dbc.Alert(
            "Please enter either an ETF ticker OR a list of stocks.",
            color="danger"
        ), {
            "inputs_disabled": True
        }

    # ❌ both provided
    if etf and stock_list:
        return dbc.Alert(
            "Please enter only ONE input: ETF OR Stock List (not both).",
            color="danger"
        ), {
            "inputs_disabled": True
        }

    # ✅ ETF or stocks are provided
    if etf:
        if not validate_and_classify_ticker(etf)[1]:
            return dbc.Alert(
                f"The ticker '{etf}' is not a valid ETF. Please check the ticker and try again.",
                color="danger"
            ), {
            "inputs_disabled": True
        }
        holdings = get_top_holdings(etf)
        tickers = [etf] + holdings        
    else:
        holdings = []
        for t in stock_list.split(","):
            stock = t.strip().upper()
            if not stock:
                continue
            if not validate_and_classify_ticker(stock)[0]:
                return dbc.Alert(
                    f"The ticker '{stock}' is not a valid stock ticker. Please check the ticker and try again.",
                    color="danger"
                ), True
            holdings.append(stock)
        tickers = holdings

    if tickers is None:
        return dbc.Alert(
            f"Error loading holdings. Please check the etf/tickers and try again.",
            color="danger"
        ), {
            "inputs_disabled": True
        }

    df_returns = get_daily_returns(tickers, period)

    fig_line = combined_chart(df_returns, tickers, agg_method=cum_method)

    df_prices = get_daily_prices(tickers, period, corr_freq)    

    fig_corr = corr_heatmap(df_prices, corr_freq)

    df_display, style_rules = get_top_holdings_with_performance(etf,tickers)
    outstanding_shares = { row['Ticker']: row['Total Shares'] for _,row in df_display.iterrows()}
    df_display.drop(columns=["Total Shares"], inplace=True)
    stocks = {}
    # Set up multiprocessing to get the insider trades for each company
    pool = Pool(processes=5)
    results = pool.map(func=get_insider_trades, iterable= [tick for tick in holdings]) # type: ignore
    # Wait for all processes to finish
    pool.close()
    pool.join()
    
    warnings = []
    for tick, trades in results:
        if trades and "ERROR" in trades[0].keys():
            warnings.append(dbc.Alert(
                f"Error loading insider trades for {tick}: {trades[0]['ERROR']}",
                color="warning"))
            stocks[tick] = []
        else:
            stocks[tick] = trades

    bar_fig = rolling_insider_chart(stocks, outstanding_shares, 30)

    return html.Div([ dbc.Row([
        dbc.Col([dcc.Graph(figure=fig_line)], width=7),
        dbc.Col([dcc.Graph(figure=fig_corr)], width=5),
    ], className="mt-4"), 
    
    html.Hr(),

    # === Bottom full-width dataframe ===
    dbc.Row([
        dbc.Col([
            dash_table.DataTable(
                id='holdings-table',
                columns=[{"name": c, "id": c} for c in df_display.columns],
                data=df_display.to_dict("records"), # type: ignore
                page_size=25,
                style_data_conditional=style_rules, # type: ignore
                # Enable horizontal scrolling
                style_table={
                    "overflowX": "auto",
                    "width": "100%",
                    "backgroundColor": "rgb(30, 30, 30)"
                },
                # Cell styles
                style_cell={
                    "backgroundColor": "rgb(30, 30, 30)",   # dark background
                    "color": "white",                       # white text
                    "textAlign": "left",
                    "padding": "6px",
                    "whiteSpace": "normal",
                    "height": "auto",
                    "border": "1px solid rgb(50, 50, 50)"   # subtle borders
                },

                # Header styles
                style_header={
                    "backgroundColor": "rgb(45, 45, 45)",    # slightly lighter header
                    "color": "white",
                    "fontWeight": "bold",
                    "borderBottom": "2px solid rgb(80, 80, 80)"
                }
            )
        ], width=12)
    ], className="mt-4"), 
    html.Div(warnings, className="mt-4"),
    html.Hr(),
    dbc.Row([dcc.Graph(figure=bar_fig, id="insider-bar-chart")], className="mt-4"),
    html.Hr(),
    dcc.Loading(
        children=[
            dcc.Graph(id="insider-line-chart",
                      style={"display": "none"}
                      )
        ],
        type="circle"
    )
    ]), {
            "inputs_disabled": False
        }

@app.callback(
    Output("insider-line-chart", "figure"),
    Output("insider-line-chart", "style"),
    Input("insider-bar-chart", "clickData"),
    prevent_initial_call=True
)

def update_line_chart(clickData):
    ticker = clickData["points"][0]["customdata"][0]
    net_shares = clickData["points"][0]["x"]
    print(net_shares)
    file_path = f"./data/insider_trades/{ticker}.json"
    print(f"Loading insider trades for {ticker} from {file_path}")
    if not os.path.exists(file_path):
        raise Exception(f"No insider trades data found for {ticker}")
    
    with open(f'./data/insider_trades/{ticker}.json') as f:
        insider_trades = json.load(f)

    ts, dots = transactions_to_cumm_ts(
        insider_trades, net_shares
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=ts['date'],
            y=ts["cum_shares"],
            mode="lines",
            name=ticker
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dots["date"],
            y=dots["shares"].cumsum(),  # aligns vertically
            mode="markers",
            name="Transactions",
            marker=dict(
                size=9,
                color=dots["shares"],
                colorscale="RdYlGn",
                showscale=True,
            ),
            customdata=np.stack(
                [
                    dots['accession_number'],
                    dots['date'],
                    dots["insider"],
                    dots["role"],
                    dots["shares"]
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Date: %{customdata[1]}<br>"
                "Insider: %{customdata[2]}<br>"
                "Role: %{customdata[3]}<br>"
                "Shares: %{customdata[4]:,.0f}<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"Cumulative Insider Flow — {ticker}",
        xaxis_title="Date",
        yaxis_title="Cumulative Net Flow",
        template="plotly_dark",
        showlegend=False
    )

    return fig, {"display": "block"}

@app.callback(
    Output("time-period", "disabled"),
    Output("cum-method", "disabled"),
    Output("corr-frequency-dropdown", "disabled"),
    Input("ui-state", "data"),
    prevent_initial_call=True
)
def toggle_inputs(ui_state):
    if not ui_state:
        return True, True, True

    disabled = ui_state.get("inputs_disabled", False)
    return disabled, disabled, disabled

if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=8050,
        debug=False
    )
