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
        df['Weight'] = df['Weight'].apply(
            lambda v: f"{v*100:.2f}%" if pd.notna(v) else "N/A"
        )
    else:
        df = pd.DataFrame({"Ticker": stock_list})
        df = df.head(n).reset_index(drop=True)
        df['Weight'] = None
    # Calculate performance for each holding
    one_year_ago = datetime.now() - timedelta(days=365)
    three_year_ago = datetime.now() - timedelta(days=365*3)  
    
    performances = []
    for ticker in df["Ticker"]:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5y")

        if hist.empty:
            one_year_return = None
            five_year_return = None
            three_year_return = None
        else:
            # Remove timezone to avoid comparison errors
            hist.index = hist.index.tz_localize(None) # type: ignore

            # Compute returns
            latest_price = hist["Close"].iloc[-1]
            one_year_price = hist.loc[hist.index >= one_year_ago]["Close"].iloc[0] if any(hist.index >= one_year_ago) else None
            three_year_price = hist.loc[hist.index >= three_year_ago]["Close"].iloc[0] if any(hist.index >= three_year_ago) else None
            five_year_price = hist["Close"].iloc[0] if len(hist) > 1 else None
            
            one_year_return = f"{(latest_price / one_year_price - 1) * 100:.2f}%" if one_year_price else None
            three_year_return = f"{(latest_price / three_year_price - 1) * 100:.2f}%" if three_year_price else None
            five_year_return = f"{(latest_price / five_year_price - 1) * 100:.2f}%" if five_year_price else None

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
            "3Y Return": three_year_return,
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
        value = px_pct_diff[i] # type: ignore
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        color = color_scale(value)
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

        value = inv_pe_pct_diff[i] # type: ignore
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        color = color_scale(value)
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


def fetch_yoy_financials(ticker: str) -> pd.DataFrame:
    """
    Pull annual income statement and compute YoY % growth
    for Revenue and Net Income.
    """

    yf_ticker = yf.Ticker(ticker)
    inc_stmt = yf_ticker.financials

    if inc_stmt.empty:
        return pd.DataFrame()

    required_rows = {
        "Revenue": "Total Revenue",
        "Net Income": "Net Income"
    }

    data = {}

    for output_label, row_label in required_rows.items():
        if row_label not in inc_stmt.index:
            continue

        series = inc_stmt.loc[row_label].dropna()

        # Ensure datetime index and sort chronologically
        series = series.sort_index()

        # Custom YoY calculation (handles negative base better)
        prior = series.shift(1)
        yoy = (series - prior) / prior.abs() * 100

        # Convert index to fiscal year string AFTER calc
        yoy.index = yoy.index.year.astype(str) # type: ignore

        data[output_label] = yoy

    df = pd.DataFrame(data)

    # Remove first row (NaN from shift)
    df = df.dropna(how="all")

    return df.round(2)

def build_comparison_table(ticker_a, ticker_b, df_a, df_b):

    def geometric_total(yoy_series: pd.Series) -> float:
        growth_factors = 1 + (yoy_series / 100)
        return (growth_factors.prod() - 1) * 100 # type: ignore

    # Align years (important)
    common_years = df_a.index.intersection(df_b.index)
    df_a = df_a.loc[common_years]
    df_b = df_b.loc[common_years]

    rows = []

    for ticker, df in [(ticker_a, df_a), (ticker_b, df_b)]:
        for metric in df.columns:

            row = df[metric].copy()

            total = geometric_total(row)

            row["Total"] = round(total, 2)

            rows.append(
                pd.DataFrame(
                    [row.values],
                    columns=row.index,
                    index=pd.MultiIndex.from_tuples(
                        [(ticker, metric)],
                        names=["Ticker", "Metric"]
                    )
                )
            )

    final_df = pd.concat(rows)

    return final_df.round(2)

def build_table_columns_by_ticker(ticker_a, ticker_b, df_a, df_b):

    def geometric_total(yoy_series: pd.Series) -> float:
        growth_factors = 1 + (yoy_series / 100)
        return (growth_factors.prod() - 1) * 100 # type: ignore

    common_years = df_a.index.intersection(df_b.index)
    df_a = df_a.loc[common_years]
    df_b = df_b.loc[common_years]

    frames = []

    for ticker, df in [(ticker_a, df_a), (ticker_b, df_b)]:

        df = df.T  # Metrics as rows

        # Add Total column
        df["Total"] = df.apply(geometric_total, axis=1)

        # Create multi-level columns
        df.columns = pd.MultiIndex.from_product(
            [[ticker], df.columns]
        )

        frames.append(df)

    final_df = pd.concat(frames, axis=1)

    return final_df.round(2)