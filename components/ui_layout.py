import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dash_table.Format import Format, Scheme, Symbol

def returns_chart(fig_line, fig_corr):
    return dbc.Row([
        dbc.Col([dcc.Graph(figure=fig_line)], width=6),
        dbc.Col([dcc.Graph(figure=fig_corr)], width=6),
    ], className="mt-4")

def holdings_chart(df_display, style_rules):
    # === Bottom full-width dataframe ===
    return dbc.Row([
        dbc.Col(
            dash_table.DataTable(
                id='holdings-table',
                columns=[{"name": c, "id": c} for c in df_display.columns],
                data=df_display.to_dict("records"),
                page_size=25,
                style_data_conditional=style_rules,
                style_table={
                    "overflowX": "auto",
                    "width": "100%",
                    "backgroundColor": "rgb(30, 30, 30)"
                },
                style_cell={
                    "backgroundColor": "rgb(30, 30, 30)",
                    "color": "white",
                    "textAlign": "left",
                    "padding": "6px",
                    "whiteSpace": "normal",
                    "height": "auto",
                    "border": "1px solid rgb(50, 50, 50)"
                },
                style_header={
                    "backgroundColor": "rgb(45, 45, 45)",
                    "color": "white",
                    "fontWeight": "bold",
                    "borderBottom": "2px solid rgb(80, 80, 80)"
                }
            ),
            width=12
        )
    ], className="mt-4")


def build_columns(df):
    pct_cols = {
        "Shareholder Yield", "FCF Yield", "Momentum (12-1)", "3Y Share Change",
        "Dividend Yield", "Buyback Yield", "Dilution Yield"
    }
    ratio_cols = {"Leverage (D/E)"}
    num_cols   = {"Factor Score"}

    columns = []
    for c in df.columns:
        if c.startswith("s_"):        # drop component score columns
            continue

        col = {"name": c, "id": c}

        if c in pct_cols:
            col["type"]   = "numeric"
            col["format"] = Format(scheme=Scheme.percentage, precision=2)

        elif c in ratio_cols:
            col["type"]   = "numeric"
            col["format"] = Format(precision=2, scheme=Scheme.fixed).symbol(Symbol.yes).symbol_suffix("x")

        elif c in num_cols:
            col["type"]   = "numeric"
            col["format"] = Format(precision=2, scheme=Scheme.fixed)

        columns.append(col)
    return columns


def factor_model_section(factor_display):
    return html.Div([
        html.H5("Factor Model Exposure (from Quarterly Cash Flows)", className="mb-3"),
        dash_table.DataTable(
            id='factor-model-table',
            columns=build_columns(factor_display),
            data=factor_display.to_dict("records"),
            page_size=25,
            style_table={
                "overflowX": "auto",
                "width": "100%",
                "backgroundColor": "rgb(30, 30, 30)"
            },
            style_cell={
                "backgroundColor": "rgb(30, 30, 30)",
                "color": "white",
                "textAlign": "left",
                "padding": "6px",
                "whiteSpace": "normal",
                "height": "auto",
                "border": "1px solid rgb(50, 50, 50)"
            },
            style_header={
                "backgroundColor": "rgb(45, 45, 45)",
                "color": "white",
                "fontWeight": "bold",
                "borderBottom": "2px solid rgb(80, 80, 80)"
            },
            style_data_conditional=[
                {
                    "if": {"filter_query": '{Signal} = "BUY"', "column_id": "Signal"},
                    "color": "#00c853",
                    "fontWeight": "bold"
                },
                {
                    "if": {"filter_query": '{Signal} = "SELL"', "column_id": "Signal"},
                    "color": "#ff1744",
                    "fontWeight": "bold"
                },
                {
                    "if": {"filter_query": '{Signal} = "HOLD"', "column_id": "Signal"},
                    "color": "#ffd600",
                    "fontWeight": "bold"
                },
                {
                    "if": {"filter_query": "{Factor Score} > 0.15", "column_id": "Factor Score"},
                    "color": "#00c853"
                },
                {
                    "if": {"filter_query": "{Factor Score} < -0.20", "column_id": "Factor Score"},
                    "color": "#ff1744"
                },
            ] # type: ignore
        )
    ], className="mt-4")

def insider_section(bar_fig):
    return html.Div([
        html.H5("Insider/10% Owner Flows (scraped from SEC)", className="mb-3"),
        dbc.Row(dcc.Graph(figure=bar_fig, id="insider-bar-chart"), className="mt-4"),
        html.Div(id="net-shares-output"),
        html.Hr(),
        html.H6(id="time-series-header", className="mb-3"),
        html.A("View SEC Filing",
            id="sec-link",
            target="_blank",   # opens in new tab
            style={"display": "none"},
        ),
        dcc.Loading(
        [html.Div(id="quarter-dropdown-container",
                  children=[
                      dcc.Dropdown(id="quarter-dropdown", clearable=False,
                        style={"width": "180px", "color": "#000"},
        )],
        style={"display": "none"}), # hidden until first click  
        dcc.Graph(id="insider-line-chart", style={"display": "none"})], 
        type="circle"
        )
    ])