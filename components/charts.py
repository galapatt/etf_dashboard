# -----------------------------
# File: components/charts.py
# -----------------------------

import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime


def combined_chart(df: pd.DataFrame, tickers: list[str], agg_method: str) -> go.Figure:
    
    if agg_method == "cumsum":
        df = df.cumsum()*100
    elif agg_method == "cumprod":
        df = ((1 + df).cumprod() - 1)*100
    
    fig = go.Figure()

    for t in tickers:
        if t in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[t],
                mode='lines',
                name=t
            ))

    fig.update_layout(
        title="ETF vs Top 10 Holdings",
        xaxis_title="Date",
        yaxis_title="Percent Return (%)",
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font=dict(color="#8cb4ff")
    )

    return fig


def corr_heatmap(returns_df: pd.DataFrame, freq: str) -> go.Figure:
    """
    Generate a correlation heatmap of daily returns.

    Args:
        df (pd.DataFrame): DataFrame with daily returns for each symbol.
        freq (str): Frequency for resampling ('D', 'W', 'M', etc.)

    Returns:
        go.Figure: Plotly Figure object representing the heatmap.
    """

    corr = returns_df.corr()

    heatmap = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="RdBu",
        text=np.round(corr.values, 2),  # round to 2 decimals
        texttemplate="%{text}", 
        zmin=-1,
        zmax=1
    ))

    heatmap.update_layout(
        title=f'Correlation Heatmap ({freq} Returns)',
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font=dict(color="#8cb4ff")
    )

    return heatmap

def rolling_insider_chart(insider_trades: dict, outstanding_shares: dict, days: int) -> go.Figure: # type: ignore
    """
    Generate a rolling insider trading chart.

    Args:
        list (insider_trades): list containing tuples with ticker and trades pulled from API call.
        dict (outstanding_shares): Dictionary containing total number of shares for normalization.
        int (days): number of days to look back for trades.
    Returns:
        go.Figure: Plotly Figure object representing the rolling insider trading chart.
    """
    flows ={}

    for ticker,trades in insider_trades.items(): # type: ignore
        if not trades:
            continue
        amount = 0
        for trade in trades:
            if (datetime.strptime(trade['date'],"%m/%d/%Y") - datetime.now()).days > days:
                break
            nd_trades = trade['non_deriv_trades']
            for nd_trade in nd_trades:
                if nd_trade['change'] == "A":
                    amount += nd_trade['amount']
                elif nd_trade['change'] == "D":
                    amount -= nd_trade['amount']
        flows[ticker] = amount

    df = (
            pd.DataFrame.from_dict(flows, orient="index", columns=["net_shares"])
            .reset_index()
            .rename(columns={"index": "ticker"})
        )    
        
    df['shares_outstanding'] = df['ticker'].map(outstanding_shares)
    df["pct_outstanding"] = (df["net_shares"] / df["shares_outstanding"]) * 100
    df["color"] = df["net_shares"].apply(lambda x: "green" if x > 0 else "red")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["net_shares"],
        y=df["ticker"],
        orientation="h",
        marker_color=df["color"],
        hovertemplate="%{y}: %{x:,.0f}<extra></extra>"
    ))

    fig.update_layout(
        title=f"Net Insider / Large-Holder Flow (Trailing {days} Days)",
        xaxis=dict(
            title="Net Buy / Sell Value",
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="white",
        ),
        yaxis=dict(
            autorange="reversed"  # top-to-bottom order
        ),
        template="plotly_dark",
        height=400,
        margin=dict(l=80, r=40, t=50, b=40),
    )    

    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Net Shares: %{x:,.0f}<br>"
            "% of Shares Outstanding: %{customdata[1]:.3f}%"
            "<extra></extra>"
        ),
        customdata=df[["ticker", "pct_outstanding"]].values
    )

    return fig