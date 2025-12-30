# -----------------------------
# File: components/charts.py
# -----------------------------

import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


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

def sign_colors(values, alpha=1.0):
    colors = []
    for v in values:
        if v > 0:
            colors.append(f"rgba(0, 160, 0, {alpha})")  # green
        elif v < 0:
            colors.append(f"rgba(200, 0, 0, {alpha})")  # red
        else:
            colors.append(f"rgba(150, 150, 150, {alpha})")  # neutral
    return colors

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

    today = datetime.now()
    agg_monthly_flows = []
    for ticker,trades in insider_trades.items(): # type: ignore
        flow_dict = {"ticker": ticker}
        if not trades:
            continue
        amount = 0
        for trade in trades:
            num_days = (today - datetime.strptime(trade['date'],"%m/%d/%Y")).days
            if num_days >= days//2 and f"{days//2}_day_flow" not in flow_dict:
                flow_dict[f"{days//2}_day_flow"] = amount
                amount = 0
            elif num_days >= days:
                flow_dict[f"{days}_day_flow"] = amount 
                break
            nd_trades = trade['non_deriv_trades']
            for nd_trade in nd_trades:
                if nd_trade['change'] == "A":
                    amount += nd_trade['amount']
                elif nd_trade['change'] == "D":
                    amount -= nd_trade['amount']
        agg_monthly_flows.append(flow_dict)


    df = pd.DataFrame.from_records(agg_monthly_flows)
    df['shares_outstanding'] = df['ticker'].map(outstanding_shares)
    df["delta"] = df[f"{days//2}_day_flow"] - df[f"{days}_day_flow"]
    df[f"{days//2} pct_outstanding"] = (df[f"{days//2}_day_flow"] / df["shares_outstanding"]) * 100
    df[f"{days} pct_outstanding"] = (df[f"{days}_day_flow"] / df["shares_outstanding"]) * 100

    fig = go.Figure()

    fig.add_bar(
        x=df["ticker"],
        y=df[f"{days}_day_flow"],
        name=f"Previous {days//2}-{days} Days ({(today - timedelta(days=days)).strftime('%m/%d/%Y')}-{(today - timedelta(days=days//2)).strftime('%m/%d/%Y')})",
        marker_color=sign_colors(df[f"{days}_day_flow"], alpha=.35),
        customdata=df[[f"{days} pct_outstanding"]].values,
        hovertemplate=(
            f"Net Shares ({days//2}-{days}d): %{{y:,.0f}}<br>"
            "% of Shares Outstanding: %{customdata[0]:.3f}%"
            "<extra></extra>"))

    fig.add_bar(
        x=df["ticker"],
        y=df[f"{days//2}_day_flow"],
        name=f"Last {days//2} Days ({(today - timedelta(days=days//2)).strftime('%m/%d/%Y')}-{today.strftime('%m/%d/%Y')})",
        marker_color=sign_colors(df[f"{days//2}_day_flow"], alpha=1),
        customdata=df[[f"{days//2} pct_outstanding","delta"]].values,
        hovertemplate=(
            f"Net Shares (0-{days//2}d): %{{y:,.0f}}<br>"
            "% of Shares Outstanding: %{customdata[0]:.3f}%<br>"
            "Δ vs prior: %{customdata[1]:,.0f}"
            "<extra></extra>"))
    
    fig.update_layout(
        barmode='group',
        title=f"Net Insider / Large-Holder Flow (Last vs. Prior {days//2} Day Windows)",
        xaxis=dict(
            title="Net Buy / Sell Value",
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="white",
        ),
        yaxis=dict(
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="black" # top-to-bottom order
        ),
        template="plotly_dark",
        height=400,
        margin=dict(l=80, r=40, t=50, b=40),
        hovermode = 'x unified',
        legend=dict(orientation="h", y=1.05)
    )    

    return fig


def create_line_chart(ticker: str, ts: pd.DataFrame, dots: pd.DataFrame, quarter_ranges: list) -> go.Figure:
    """
    Generate line chart of cumulative insider trades with markers.

    Args:
        ticker (str): stock ticker symbol.
        pd.DataFrame (ts_df): dataframe with cumulative time series data.
        pd.DataFrame (dots): dataframe with significant trade markers.
        quarter_ranges (list): list of (start,end) tuples for each quarter in the data.
        
    Returns:
        go.Figure: Plotly Figure object representing the rolling insider trading chart.
    """
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
            y=dots["cum_shares"],  # aligns vertically
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
    # Add quarter background shading
    colors = ['rgba(200,200,255,0.2)', 'rgba(200,255,200,0.2)']  # alternating colors

    for i, (start, end) in enumerate(quarter_ranges):
        fig.add_shape(
            type="rect",
            xref="x",
            yref="paper",  # full height of plot
            x0=start,
            x1=end,
            y0=0,
            y1=1,
            fillcolor=colors[i % len(colors)],
            line=dict(width=0),
            layer="below"  # behind the line chart
        )
    fig.update_layout(
        title=f"Cumulative Insider Flow — {ticker}",
        xaxis_title="Date",
        yaxis_title="Cumulative Net Flow",
        template="plotly_dark",
        showlegend=False
    )

    # Add quarter background shading
    colors = ['rgba(200,200,255,0.2)', 'rgba(200,255,200,0.2)']  # alternating colors

    return fig