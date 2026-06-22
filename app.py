# Project: ETF Dashboard (Dash)
# File: app.py

from dash import Dash, dash, dcc, html, Input, Output, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from multiprocessing import Pool
import base64
import os
import plotly.graph_objs as go
import json
from io import BytesIO, StringIO
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile
import numpy as np
import pandas as pd

from components.ui_layout import etf_activity_section, factor_model_section, holdings_chart, insider_section, returns_chart
from components.tables import build_table_columns_by_ticker, fetch_yoy_financials, get_top_holdings_with_performance
from components.charts import combined_chart, corr_heatmap, create_line_chart, rolling_insider_chart, yoy_compare_bar, generate_quarter_options
from data.analytics import get_etf_trading_activity, get_top_holdings, get_daily_returns, get_daily_prices, run_factor_model, transactions_to_cumm_ts, validate_and_classify_ticker
from data.insider_data import get_insider_trades, refresh_company_tickers
from src.config import INSIDER_DATA_PATH


app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

DAYS = 180


def build_export_dataframe(chart_data: dict, corr_freq: str) -> pd.DataFrame:
    df_returns = pd.read_json(StringIO(chart_data["prices_json"]), orient="split")
    df_holdings = pd.read_json(StringIO(chart_data["holdings_json"]), orient="records")
    df_factors = pd.read_json(StringIO(chart_data["factor_json"]), orient="records")

    tickers = chart_data["tickers"]
    benchmark = chart_data.get("benchmark") or tickers[0]
    source_label = chart_data.get("source_label", "Export")

    corr = df_returns.corr()
    corr_values = {}
    for ticker in tickers:
        if benchmark in corr.columns and ticker in corr.index and ticker != benchmark:
            corr_values[ticker] = round(corr.loc[ticker, benchmark], 3)
        elif ticker in corr.columns:
            peers = corr.loc[ticker].drop(labels=[ticker], errors="ignore")
            corr_values[ticker] = round(peers.mean(), 3) if not peers.empty else None
        else:
            corr_values[ticker] = None

    holdings_by_ticker = df_holdings.set_index("Ticker").to_dict("index") if "Ticker" in df_holdings.columns else {}
    factors_by_ticker = df_factors.set_index("Ticker").to_dict("index") if "Ticker" in df_factors.columns else {}

    rows = []
    for ticker in tickers:
        holding = holdings_by_ticker.get(ticker, {})
        factor = factors_by_ticker.get(ticker, {})
        rows.append({
            "ETF / Theme": source_label,
            "Stock": ticker,
            "Weight (%)": holding.get("Weight", ""),
            "Correlation": corr_values.get(ticker),
            "Sector": factor.get("Sector", ""),
            "Factor Score": factor.get("Factor Score", ""),
            "FCF Yield": factor.get("FCF Yield", ""),
            "Shareholder Yield": factor.get("Shareholder Yield", ""),
            "Momentum (12-1)": factor.get("Momentum (12-1)", ""),
            "3Y Share Change": factor.get("3Y Share Change", ""),
            "Leverage (D/E)": factor.get("Leverage (D/E)", ""),
            "1Y Return": holding.get("1Y Return", ""),
            "3Y Return": holding.get("3Y Return", ""),
            "Recent Revenue": holding.get("Recent Revenue", ""),
            "Current Price": holding.get("Current Price", ""),
            "Target Price": holding.get("Target Price", ""),
            "Recommendation": holding.get("Recommendation", ""),
            "Trailing PE": holding.get("Trailing PE", ""),
            "Forward PE": holding.get("Forward PE", ""),
        })

    return pd.DataFrame(rows)


def excel_column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def xlsx_cell(cell_ref: str, value) -> str:
    if value is None or pd.isna(value):
        return f'<c r="{cell_ref}"/>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{cell_ref}"><v>{value}</v></c>'
    return f'<c r="{cell_ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def write_simple_xlsx(buffer: BytesIO, df: pd.DataFrame, sheet_name: str = "Stocks") -> None:
    rows = [list(df.columns)] + df.astype(object).where(pd.notna(df), None).values.tolist()
    sheet_rows = []

    for row_idx, row in enumerate(rows, start=1):
        cells = [
            xlsx_cell(f"{excel_column_name(col_idx)}{row_idx}", value)
            for col_idx, value in enumerate(row, start=1)
        ]
        sheet_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    widths = []
    for col_idx, column in enumerate(df.columns, start=1):
        values = [column] + df[column].astype(str).replace("nan", "").tolist()
        width = min(max(len(value) for value in values) + 2, 42)
        widths.append(
            f'<col min="{col_idx}" max="{col_idx}" width="{width}" customWidth="1"/>'
        )

    worksheet_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<cols>{"".join(widths)}</cols>
<sheetData>{"".join(sheet_rows)}</sheetData>
</worksheet>'''

    workbook_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="{escape(sheet_name)}" sheetId="1" r:id="rId1"/></sheets>
</workbook>'''

    rels_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>'''

    workbook_rels_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>'''

    content_types_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>'''

    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml)
        archive.writestr("_rels/.rels", rels_xml)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet_xml)


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
                   style={"backgroundColor": "#111", "color": "#8cb4ff", "marginTop": "24px"}),
        dbc.Button(
            "Export Excel",
            id="export-excel-button",
            n_clicks=0,
            color="success",
            disabled=True,
            style={"marginTop": "24px", "marginLeft": "8px"}
        ),
        dcc.Download(id="excel-download")
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
    dcc.Store(id="chart-data-store"),
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
    Output('chart-data-store', 'data'),
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
        }, None, None, None, None

    # ❌ both provided
    if etf and stock_list:
        return dbc.Alert(
            "Please enter only ONE input: ETF OR Stock List (not both).",
            color="danger"
        ), {
            "inputs_disabled": True
        }, None, None, None, None

    # ✅ ETF or stocks are provided
    if etf:
        if not validate_and_classify_ticker(etf)[1]:
            return dbc.Alert(
                f"The ticker '{etf}' is not a valid ETF. Please check the ticker and try again.",
                color="danger"
            ), {
            "inputs_disabled": True
        }, None, None, None, None
        holdings = get_top_holdings(etf)
        etf_activity_display = get_etf_trading_activity(etf)
        etf_activity_output = (
            dbc.Alert(f"Could not load ETF trading activity for {etf}.", color="warning")
            if etf_activity_display.empty
            else etf_activity_section(etf_activity_display, etf)
        )
        tickers = [etf] + holdings        
        source_label = etf
    else:
        etf_activity_output = None
        holdings = []
        for t in stock_list.split(","):
            stock = t.strip().upper()
            if not stock:
                continue
            if not validate_and_classify_ticker(stock)[0]:
                return dbc.Alert(
                    f"The ticker '{stock}' is not a valid stock ticker. Please check the ticker and try again.",
                    color="danger"
                ), {"inputs_disabled": True}, None, None, None, None
            holdings.append(stock)
        tickers = holdings
        source_label = "Custom Stock List"

    if tickers is None:
        return dbc.Alert(
            f"Error loading holdings. Please check the etf/tickers and try again.",
            color="danger"
        ), {
            "inputs_disabled": True
        }, None, None, None, None

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
        }, None, None, None, None
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
    chart_data = {
        "tickers": tickers,
        "benchmark": etf or tickers[0],
        "source_label": source_label,
        "prices_json": df_prices.to_json(date_format="iso", orient="split"),
        "holdings_json": df_display.to_json(orient="records"),
        "factor_json": factor_display.to_json(orient="records"),
    }

    return html.Div([
        etf_activity_output,
        html.Hr() if etf_activity_output else None,
        returns_chart(fig_line, fig_corr),
    html.Div(ret_warnings, className="mt-4"),
    html.Hr(),
    holdings_chart(df_display, style_rules), 
    ]), html.Div([
        html.Div(warnings, className="mt-4"),
        html.Hr(),
        insider_section(bar_fig),
    ]), {"inputs_disabled": False}, holdings, factor_model_section(factor_display), chart_data


def as_plotly_list(values) -> list:
    if values is None:
        return []
    if isinstance(values, dict):
        if "bdata" in values and "dtype" in values:
            try:
                decoded = base64.b64decode(values["bdata"])
                return np.frombuffer(decoded, dtype=np.dtype(values["dtype"])).tolist()
            except Exception:
                return []
        try:
            return [values[key] for key in sorted(values, key=lambda k: int(k))]
        except (TypeError, ValueError):
            return list(values.values())
    return list(values)


def as_float(value) -> float:
    if isinstance(value, dict):
        decoded = as_plotly_list(value)
        value = decoded[0] if decoded else 0
    return float(value or 0)


def get_clicked_insider_shares(clickData, bar_figure) -> tuple[str, tuple[float, float]]:
    if not clickData or not clickData.get("points"):
        raise PreventUpdate

    ticker = clickData["points"][0]["x"]
    shares = [0.0, 0.0]

    for trace_idx, trace in enumerate((bar_figure or {}).get("data", [])[:2]):
        x_values = as_plotly_list(trace.get("x", []))
        y_values = as_plotly_list(trace.get("y", []))
        try:
            point_idx = x_values.index(ticker)
        except ValueError:
            continue
        if point_idx < len(y_values):
            shares[trace_idx] = as_float(y_values[point_idx])

    if not any(shares):
        for idx, point in enumerate(clickData.get("points", [])[:2]):
            shares[idx] = as_float(point.get("y"))

    return ticker, (shares[0], shares[1])


@app.callback(
    Output("insider-line-chart", "figure"),
    Output("insider-line-chart", "style"),
    Output("net-shares-output", "children"),
    Output("quarter-dropdown-container", "style"),
    Output("quarter-dropdown", "options"),
    Output("quarter-dropdown", "value"),
    Output("time-series-header", "children"),
    Input("insider-bar-chart", "clickData"),
    State("insider-bar-chart", "figure"),
    prevent_initial_call=True,
)

def on_bar_click(clickData, bar_figure):

    if clickData is None:
        raise PreventUpdate
    
    ticker, both_shares = get_clicked_insider_shares(clickData, bar_figure)
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
    State("insider-bar-chart", "figure"),
    prevent_initial_call=True,
)
def on_quarter_change(quarter_start, clickData, bar_figure):
    if clickData is None or quarter_start is None:
        raise PreventUpdate

    ticker, both_shares = get_clicked_insider_shares(clickData, bar_figure)
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
    Output("excel-download", "data"),
    Input("export-excel-button", "n_clicks"),
    State("chart-data-store", "data"),
    State("corr-frequency-dropdown", "value"),
    prevent_initial_call=True
)
def export_excel(n_clicks, chart_data, corr_freq):
    if not n_clicks or not chart_data:
        raise PreventUpdate

    export_df = build_export_dataframe(chart_data, corr_freq)
    source_label = chart_data.get("source_label", "export").replace(" ", "_")
    filename = f"{source_label}_stock_export.xlsx"

    def write_excel(buffer):
        write_simple_xlsx(buffer, export_df)

    return dcc.send_bytes(write_excel, filename)


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
    Output("export-excel-button", "disabled"),
    Input("chart-data-store", "data"),
    Input("ui-state", "data"),
    prevent_initial_call=True
)
def toggle_export_button(chart_data, ui_state):
    if not chart_data or (ui_state and ui_state.get("inputs_disabled", False)):
        return True
    return False

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
