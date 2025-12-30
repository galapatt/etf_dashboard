import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import math


def human_format(num):
    num = float(num)
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num/1_000:.2f}K"
    else:
        return str(num)
    
def color_scale(value, alpha=0.4) -> str:
    """
    Apply color scaling for specific columns 
    
    Args:
        value (float): Value between -1 and +1 (should be percentage change for trailing vs forward PE, or price vs target price).
        alpha (float): Opacity level for the color.
    
    Returns: 
        str: RGBA color string.
    """
    # value ∈ [-1, +1] for -100% to +100%
    value = max(-1, min(1, value))  # clip
    if value >= 0:
        # white → green
        green = int(255 - (255 * (1 - value)))
        return f"rgba(0, {green}, 0,  {alpha})"
    else:
        # red → white
        red = int(255 - (255 * (1 + value)))
        return f"rgba({red}, 0, 0,  {alpha})"

def get_top_holdings_with_performance(etf_ticker, stock_list, n=10):
    """
    Get top N holdings of an ETF with 1-year and 5-year performance.

    Args:
        etf_ticker (str): ETF ticker symbol.
        stock_list (list): List of stock tickers to use instead of ETF holdings.
        n (int): Number of top holdings to retrieve.

    Returns:
        pd.DataFrame: DataFrame containing holdings, weights, and performance.
        dict: conditional formatting rules for Dash DataTable. 
    """
    if etf_ticker in stock_list:
        etf = yf.Ticker(etf_ticker)
    
        try:
            # Get holdings
            holdings = etf.get_funds_data().top_holdings # type: ignore
            df = pd.DataFrame(holdings)
        except Exception:
            raise ValueError(f"Could not retrieve holdings for {etf_ticker}")
        
        df = df.head(10).reset_index(drop=False)
        df.rename(columns={'Symbol': 'Ticker', 'Holding Percent': 'Weight'}, inplace=True)
        df['Weight'] = df['Weight'] * 100  # convert to percentage
    else:
        df = pd.DataFrame({"Ticker": stock_list})
        df = df.head(n).reset_index(drop=True)
        df['Weight'] = None
    # Calculate performance for each holding
    one_year_ago = datetime.now() - timedelta(days=365)
    
    performances = []
    for ticker in df["Ticker"]:
        perf_dict = {}
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5y")

        if hist.empty:
            one_year_return = None
            five_year_return = None
        else:
            # Remove timezone to avoid comparison errors
            hist.index = hist.index.tz_localize(None) # type: ignore

            # Compute returns
            latest_price = hist["Close"].iloc[-1]
            one_year_price = hist.loc[hist.index >= one_year_ago]["Close"].iloc[0] if any(hist.index >= one_year_ago) else None
            five_year_price = hist["Close"].iloc[0] if len(hist) > 1 else None
            
            one_year_return = (latest_price / one_year_price - 1) * 100 if one_year_price else None
            five_year_return = (latest_price / five_year_price - 1) * 100 if five_year_price else None

        # --- Get most recent revenue ---
        try:
            income_stmt = stock.quarterly_income_stmt
            if income_stmt is not None and not income_stmt.empty and "Total Revenue" in income_stmt.index:
                recent_revenue = income_stmt.loc["Total Revenue"].iloc[0] if not math.isnan(income_stmt.loc["Total Revenue"].iloc[0]) else income_stmt.loc["Total Revenue"].iloc[1] # type: ignore
            else:
                recent_revenue = None
        except Exception:
            recent_revenue = None

        # --- Get target price and rating ---
        info = stock.info
        target_price = info.get('targetMeanPrice',None)
        current_price = info.get('currentPrice',None)
        pe_ratio = info.get('trailingPE', None)
        forward_pe = info.get('forwardPE', None)
        rec = info.get('recommendationKey', None)
        total_shares = info.get('sharesOutstanding', None)
        performances.append({
            "1Y Return": one_year_return,
            "5Y Return": five_year_return,
            "Recent Revenue": recent_revenue,
            "Current Price": current_price,
            "Target Price": target_price,
            "Recommendation": rec,
            "Trailing PE": pe_ratio,
            "Forward PE": forward_pe,
            "Total Shares": total_shares
        })
    
    perf_df = pd.DataFrame(performances)
    df = pd.concat([df, perf_df], axis=1)

    # Formatting
    df["Recent Revenue"] = df["Recent Revenue"].apply(human_format) # type: ignore
    style_rules = get_conditional_formatting(df)

    return df.round(2), style_rules


def get_conditional_formatting(df: pd.DataFrame):
    """
    Generate conditional formatting rules for Dash DataTable based on performance.

    Args:
        df (pd.DataFrame): DataFrame containing performance columns.
    Returns:
        list: list of style dictionaries for Dash DataTable.
    """
    px_pct_diff = (df["Target Price"] - df["Current Price"]) / df["Current Price"]
    inv_pe_pct_diff = -(df["Forward PE"] - df["Trailing PE"]) / df["Trailing PE"] 

    style_rules = [{
        "if": {"row_index": "odd"},
        "backgroundColor": "rgb(38, 38, 38)", # type: ignore
    }]
    for i, _ in df.iterrows():
        value = px_pct_diff[i]
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        color = color_scale(px_pct_diff[i]) # type: ignore
        style_rules.append({
            "if": {"row_index": i, "column_id": "Current Price"},
            "backgroundColor": color,
            'color': 'white'
        })
        style_rules.append({
            "if": {"row_index": i, "column_id": "Target Price"},
            "backgroundColor": color,
            'color': 'white'
        })

        value = inv_pe_pct_diff[i]
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        color = color_scale(inv_pe_pct_diff[i]) # type: ignore
        style_rules.append({
            "if": {"row_index": i, "column_id": "Trailing PE"},
            "backgroundColor": color,
            'color': 'white'
        })
        style_rules.append({
            "if": {"row_index": i, "column_id": "Forward PE"},
            "backgroundColor": color,
            'color': 'white'
        })

    return style_rules