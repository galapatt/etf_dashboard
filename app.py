# Project: ETF Dashboard (Dash)
# File: app.py

from dash import Dash, dash, dcc, html, Input, Output, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from multiprocessing import Pool
import os
import plotly.graph_objs as go
import json

from components.ui_layout import factor_model_section, holdings_chart, insider_section, returns_chart
from components.tables import build_table_columns_by_ticker, fetch_yoy_financials, get_top_holdings_with_performance
from components.charts import combined_chart, corr_heatmap, create_line_chart, rolling_insider_chart, yoy_compare_bar, generate_quarter_options
from data.analytics import get_top_holdings, get_daily_returns, get_daily_prices, run_factor_model, transactions_to_cumm_ts, validate_and_classify_ticker
from data.insider_data import get_insider_trades, refresh_company_tickers
from src.config import INSIDER_DATA_PATH


app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

DAYS = 180

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
    dcc.Store(id="holdings-store"),
    dcc.Loading(
        id="loading", type="circle",
        children=[html.Div(id='combined-chart-output'),
                  html.Div(id="financial-compare-section"),
                  html.Div(id="factor-model-output"),
                  html.Div(id='insider-output')
                  ]
    ) 
], fluid=True)


@app.callback(
    Output('combined-chart-output', 'children'),
    Output('insider-output', 'children'),
    Output('ui-state', 'data'),
    Output('holdings-store', 'data'),
    Output('factor-model-output', 'children'),
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
        }, None, None, None

    # ❌ both provided
    if etf and stock_list:
        return dbc.Alert(
            "Please enter only ONE input: ETF OR Stock List (not both).",
            color="danger"
        ), {
            "inputs_disabled": True
        }, None, None, None

    # ✅ ETF or stocks are provided
    if etf:
        if not validate_and_classify_ticker(etf)[1]:
            return dbc.Alert(
                f"The ticker '{etf}' is not a valid ETF. Please check the ticker and try again.",
                color="danger"
            ), {
            "inputs_disabled": True
        }, None, None, None
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
                ), True, None, None, None
            holdings.append(stock)
        tickers = holdings

    if tickers is None:
        return dbc.Alert(
            f"Error loading holdings. Please check the etf/tickers and try again.",
            color="danger"
        ), {
            "inputs_disabled": True
        }, None, None, None

    df_returns, ret_warnings = get_daily_returns(tickers, period)
    ret_warnings = [dbc.Alert(warn, color="primary") for warn in ret_warnings] # type: ignore

    fig_line = combined_chart(df_returns, tickers, agg_method=cum_method) # type: ignore

    df_prices = get_daily_prices(tickers, period, corr_freq)    

    fig_corr = corr_heatmap(df_prices, corr_freq)

    df_display, style_rules = get_top_holdings_with_performance(etf,tickers)
    outstanding_shares = { row['Ticker']: row['Total Shares'] for _,row in df_display.iterrows()}

    factor_display = run_factor_model(tickers)
    df_display.drop(columns=["Total Shares"], inplace=True)
    if not refresh_company_tickers(): # ensure we have the latest tickers before loading insider trades
        return dbc.Alert(
            "Failed to refresh company tickers.",
            color="danger"
        ), {
            "inputs_disabled": True
        }, None, None, None
    stocks = {}
    # Set up multiprocessing to get the insider trades for each company
    pool = Pool(processes=5)
    results = pool.map(func=get_insider_trades, iterable= [tick for tick in holdings]) # type: ignore
    # Wait for all processes to finish
    pool.close()
    pool.join()
    
    warnings = []
    for tick, trades in results:
        if not trades:
            warnings.append(dbc.Alert(
                f"No insider trades for {tick}", color="primary"))
            stocks[tick] = []
        elif "ERROR" in trades[0].keys():
            warnings.append(dbc.Alert(
                f"Error loading insider trades for {tick}: {trades[0]['ERROR']}",
                color="primary"))
            stocks[tick] = []
        else:
            stocks[tick] = trades

    bar_fig = rolling_insider_chart(stocks, outstanding_shares, DAYS)

    return html.Div([
        returns_chart(fig_line, fig_corr),
    html.Div(ret_warnings, className="mt-4"),
    html.Hr(),
    holdings_chart(df_display, style_rules), 
    ]), html.Div([
        html.Div(warnings, className="mt-4"),
        html.Hr(),
        insider_section(bar_fig),
    ]), {"inputs_disabled": False}, holdings, factor_model_section(factor_display)

@app.callback(
    Output("insider-line-chart", "figure"),
    Output("insider-line-chart", "style"),
    Output("net-shares-output", "children"),
    Output("quarter-dropdown-container", "style"),
    Output("quarter-dropdown", "options"),
    Output("quarter-dropdown", "value"),
    Output("time-series-header", "children"),
    Input("insider-bar-chart", "clickData"),
    prevent_initial_call=True,
)

def on_bar_click(clickData):

    if clickData is None:
        raise PreventUpdate
    
    ticker = clickData["points"][0]["x"]
    both_shares = (clickData["points"][0]["y"],clickData["points"][1]["y"])
    net_shares = both_shares[0] + both_shares[1]

    net_shares_stmt = dbc.Alert([f"Clicked ticker: {ticker}; Total Shares Traded over Past {DAYS} Days: {both_shares[0]} + {both_shares[1]} = ",
                                 html.B(f"{net_shares:,.0f}")], color="dark", style={"backgroundColor": "#222", "color": "white"}, # still useful for text color and override bg
                                 dismissable=True) # 
    
    file_path = os.path.join(INSIDER_DATA_PATH, f"{ticker}.json")
    print(f"Loading insider trades for {ticker} from {file_path}")
    if not os.path.exists(file_path):
        raise Exception(f"No insider trades data found for {ticker}")
    
    with open(file_path) as f:
        insider_trades = json.load(f)

    earliest_date = insider_trades[-1]['date']
    options       = generate_quarter_options(earliest_date)
    default_val   = options[0]["value"]

    ts, dots, q_ends = transactions_to_cumm_ts( insider_trades, net_shares )#start_date)
    line_fig = create_line_chart(ticker, ts, dots, q_ends)

    return line_fig, {"display": "block"}, net_shares_stmt, {"display": "block"}, options, default_val, f"{ticker} Insider Time Series"

@app.callback(
    Output("insider-line-chart", "figure", allow_duplicate=True),
    Input("quarter-dropdown", "value"),
    State("insider-bar-chart", "clickData"),
    prevent_initial_call=True,
)
def on_quarter_change(quarter_start, clickData):
    if clickData is None or quarter_start is None:
        raise PreventUpdate

    ticker      = clickData["points"][0]["x"]
    both_shares = (clickData["points"][0]["y"], clickData["points"][1]["y"])
    net_shares  = both_shares[0] + both_shares[1]

    file_path = os.path.join(INSIDER_DATA_PATH, f"{ticker}.json")
    with open(file_path) as f:
        insider_trades = json.load(f)

    ts, dots, q_ends = transactions_to_cumm_ts(insider_trades, net_shares, start_date=quarter_start)

    return create_line_chart(ticker, ts, dots, q_ends)

@app.callback(
    Output("sec-link", "href"),
    Output("sec-link", "style"),
    Input("insider-line-chart", "clickData"),
    prevent_initial_call=True,
)
def open_sec_filing(clickData):
    if clickData is None:
        raise PreventUpdate
    
    url = clickData["points"][0]["customdata"][5]  # whatever index you put the url at
    
    return url, {"display": "block"}

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

@app.callback(
    Output("financial-compare-section", "children"),
    Input("holdings-store", "data"),
    prevent_initial_call=True
)
def render_compare_dropdowns(holdings):

    if not holdings:
        raise PreventUpdate

    options = [{"label": t, "value": t} for t in holdings] 

    return dbc.Card(
        dbc.CardBody([

            html.H5("Compare Financial Growth", className="mb-3"),

            dbc.Row([
                dbc.Col([
                    html.Label("Stock A"),
                    dcc.Dropdown(
                        id="stock-a",
                        options=options, # type: ignore
                        value=holdings[0],
                        clearable=False
                    )
                ], width=3),

                dbc.Col([
                    html.Label("Stock B"),
                    dcc.Dropdown(
                        id="stock-b",
                        options=options, # type: ignore
                        value=holdings[1] if len(holdings) > 1 else holdings[0],
                        clearable=False
                    )
                ], width=3),
            ], className="mt-4"),

            dcc.Graph(id="financial-compare-bar"),
            dash_table.DataTable(id="financial-compare-table",
            merge_duplicate_headers=True, # type: ignore
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#000", "textAlign": "center"},
            style_cell={"backgroundColor": "#000", "textAlign": "center"},
        )],
        style={
            "backgroundColor": "#000000"
        }),
        className="mt-4",
        style={
            "backgroundColor": "#000000",
            "border": "1px solid #222"
        }
    )

@app.callback(
    Output("financial-compare-bar", "figure"),
    Output("financial-compare-table", "data"),
    Output("financial-compare-table", "columns"),
    Output("financial-compare-table", "style_data_conditional"),
    Output("financial-compare-table", "style_header_conditional"),
    Input("stock-a", "value"),
    Input("stock-b", "value"),
)
def update_financial_compare(stock_a, stock_b):

    if not stock_a or not stock_b:
        return go.Figure() # empty figure

    df_a = fetch_yoy_financials(stock_a)
    df_b = fetch_yoy_financials(stock_b)

    if df_a.empty or df_b.empty:
        raise PreventUpdate
    
    table_df = build_table_columns_by_ticker(stock_a, stock_b, df_a, df_b)
    fig = yoy_compare_bar(stock_a, stock_b, df_a, df_b)
    
    # Flatten columns for DataTable IDs
    flat_columns = ["_".join(col) for col in table_df.columns] # type: ignore
    table_df_flat = table_df.copy()
    table_df_flat.columns = flat_columns

    # Reset index so metric becomes a column
    table_df_flat = table_df_flat.reset_index()
    table_df_flat = table_df_flat.rename(columns={"index": "Metric"})

    # Build column definitions
    columns = [
        {"name": ["", "Metric"], "id": "Metric"}
    ]

    columns += [
        {"name": list(col), "id": "_".join(col)}
        for col in table_df.columns # type: ignore
    ]

    data = table_df_flat.to_dict("records")

    # --- Dynamic Color Matching From Bar Chart ---
    # Extract ticker -> color from Plotly traces
    color_map = {
        trace.name: trace.marker.color # type: ignore
        for trace in fig.data
    }
    # Build mapping from ticker to column IDs
    ticker_to_cols = {ticker: [] for ticker in color_map.keys()}

    for col in table_df_flat.columns: # type: ignore
        for ticker in color_map.keys():
            if col.startswith(ticker + "_"):
                ticker_to_cols[ticker].append(col)

    style_data_conditional = []
    style_header_conditional = []

    for ticker, cols in ticker_to_cols.items():
        color = color_map[ticker]
        # Data cells
        for c in cols:
        # Bold Total columns
            if c.endswith("_Total"):
                style_data_conditional.append({
                    "if": {"column_id": c},
                    "color": color,
                    "fontWeight": "bold"   # combine fontWeight and color in one rule
                })
            else:
                style_data_conditional.append({
                    "if": {"column_id": c},
                    "color": color
                })

        # Column headers
        for c in cols:
            style_header_conditional.append({
                "if": {"column_id": c},
                "color": color,
                "fontWeight": "700"
            })

    style_header_conditional.append({
        "if": {"header_index": 0},
        "fontWeight": "bold",
        "fontSize": "16px",
    })
    style_data_conditional.insert(0, {
    "if": {"column_id": "Metric"},
        "color": "#ffffff"
    })
    style_header_conditional.insert(0, {
        "if": {"column_id": "Metric"},
        "color": "#ffffff",
        "fontWeight": "bold"
    })

    return fig, data, columns, style_data_conditional, style_header_conditional

# cloudflared tunnel --url http://localhost:8050   
if __name__ == '__main__':
    app.run(
        host="127.0.0.1",
        port=8050,
        debug=True
    )
