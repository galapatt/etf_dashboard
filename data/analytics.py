# -----------------------------
# File: data.py
# -----------------------------

import re

import numpy as np
from yahooquery import Ticker
import yfinance as yf
import pandas as pd
import pandas as pd
from datetime import datetime

from src.config import TRADE_WEIGHTS

USD_EQUIVALENT_MAP = {
    "CCO.TO": "CCJ",
    "NXE.TO": "NXE",
    "SU.TO": "SU",
    "CNQ.TO": "CNQ",
    "SHOP.TO": "SHOP",
}

PLAN_KEYWORDS = ['savings', '401k', '401(k)', 'retirement', 'deferred compensation', 'pension', 'plan', 'ira']
TRUST_KEYWORDS = ['trust', 'llc', 'lp', 'holding', 'family']
SPOUSE_KEYWORDS = ['spouse', 'husband', 'wife', 'child', 'son', 'daughter']

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
# Weights sourced from:
#   FCF Yield:         Novy-Marx (2013) "The Other Side of Value"
#   Shareholder Yield: Faber (2013) "Shareholder Yield"
#   Momentum (12-1):   Jegadeesh & Titman (1993) "Returns to Buying Winners"
#   Share Issuance:    Pontiff & Woodgate (2008) "Share Issuance and Cross-sectional Returns"
#   Leverage:          Fama & French (2015) five-factor model
# Absolute weights sum to 1.0; sign encodes economic direction.
FACTOR_WEIGHTS = {
    "FCF Yield":          0.35, # strongest single value factor (Novy-Marx)
    "Shareholder Yield":  0.30, # dividends + buybacks - dilution (Faber)
    "Momentum (12-1)":    0.20, # 12-1 month price momentum (Jegadeesh & Titman)
    "3Y Share Change":   -0.10, # penalize share dilution (Pontiff & Woodgate)
    "Leverage (D/E)":    -0.05, # penalize high debt (Fama & French)
}

FACTOR_THRESHOLDS = {
    #                    (bad,   neutral, good)
    "FCF Yield":         (0.005, 0.015,  0.035),  # lowered: ~2% is now market avg
    "Shareholder Yield": (0.005, 0.015,  0.040),  # lowered accordingly
    "Momentum (12-1)":   (-0.10, 0.05,   0.20),   # unchanged — still valid
    "3Y Share Change":   (0.05,  0.00,  -0.03),   # unchanged
    "Leverage (D/E)":    (2.00,  1.00,   0.40),   # unchanged
}

BUY_THRESHOLD  =  0.15   # loosened slightly
SELL_THRESHOLD = -0.20

def get_usd_fx_rate(currency: str) -> float:
    """Return 1.0 for USD, otherwise fetch live rate from yfinance."""
    if currency == "USD":
        return 1.0
    try:
        fx_ticker = f"{currency}USD=X"
        rate = yf.Ticker(fx_ticker).fast_info["lastPrice"]
        if rate and rate > 0:
            return float(rate)
    except Exception:
        pass
    print(f"[WARN] Could not fetch FX rate for {currency}, skipping ticker")
    return None   # type: ignore # caller should return None and skip the ticker

# ─────────────────────────────────────────────
# SINGLE TICKER EXTRACTION
# ─────────────────────────────────────────────
def extract_factors(ticker_symbol: str) -> dict | None:
    try:
        ticker = yf.Ticker(ticker_symbol)
        info   = ticker.info

        financial_currency = info.get("financialCurrency", "USD")
        fx_rate = get_usd_fx_rate(financial_currency)
        if fx_rate is None:
            return None

        market_cap = info.get("marketCap", 0) or 0
        if market_cap == 0:
            print(f"[SKIP] {ticker_symbol}: no market cap")
            return None

        sector = info.get("sector", "Unknown")

        # ── Dividend Yield ──────────────────────────────────────────────
        dividend_yield = (info.get("dividendYield", 0) or 0) / 100

# ── Cashflow: prefer quarterly (TTM), fall back to annual (1Y) ──
        cashflow_q = ticker.quarterly_cashflow
        cashflow_a = ticker.cashflow   # annual, fetched lazily only if needed

        # Determine which cashflow source to use
        use_quarterly = cashflow_q is not None and not cashflow_q.empty
        use_annual    = not use_quarterly and cashflow_a is not None and not cashflow_a.empty

        if use_quarterly:
            cf        = cashflow_q
            n_periods = 4       # sum 4 quarters = TTM
        elif use_annual:
            cf        = cashflow_a
            n_periods = 1       # take most recent annual year
        else:
            cf        = None
            n_periods = 0
 
        cf_source = "quarterly (TTM)" if use_quarterly else "annual (1Y fallback)" if use_annual else "unavailable"
        print(f"[INFO] {ticker_symbol}: cashflow source = {cf_source}")

        buyback_yield  = None
        dilution_yield = None
        fcf_yield      = None

        if cf is not None:
 
            def ttm_sum(df, rows, n):
                """
                Sum first n periods for any matched row labels.
                Returns None  if NO rows exist at all (truly missing data).
                Returns None  if rows exist but every cell is NaN (entirely empty).
                Returns float if at least one real value exists — remaining NaNs treated as 0.
                """
                matched = [r for r in rows if r in df.index]
                if not matched:
                    return None                      # no rows at all → NaN
                vals = df.loc[matched].iloc[:, :n]
                if vals.isna().all().all():
                    return None                      # rows present but all empty → NaN
                return float(vals.fillna(0).values.flatten().sum())
 
            # ── Buyback / Dilution ─────────────────────────────────────
            if "Net Common Stock Issuance" in cf.index:
                raw = cf.loc["Net Common Stock Issuance"].iloc[:n_periods] # type: ignore
                if raw.isna().all(): # type: ignore
                    buyback_sum  = None
                    dilution_sum = None
                else:
                    raw          = raw.fillna(0)
                    buyback_sum  = float(abs(raw[raw < 0].sum())) # type: ignore
                    dilution_sum = float(raw[raw > 0].sum()) # type: ignore
            else:
                buyback_rows  = ["Repurchase Of Capital Stock", "Repurchase Of Stock",
                                  "Purchase Of Stock"]
                issuance_rows = ["Issuance Of Capital Stock", "Common Stock Issuance"]
                buyback_sum   = ttm_sum(cf, buyback_rows, n_periods) # type: ignore
                if buyback_sum is None:
                        print(f"[MISSING] {ticker_symbol}: none of {buyback_rows} found in cashflow")
                else:
                    buyback_sum = abs(buyback_sum)
                dilution_sum  = ttm_sum(cf, issuance_rows, n_periods) # type: ignore
                if dilution_sum is None:
                    print(f"[MISSING] {ticker_symbol}: none of {issuance_rows} found in cashflow")
                else:
                    dilution_sum = abs(dilution_sum)
 
            if buyback_sum is not None:
                buyback_yield = (buyback_sum * fx_rate) / market_cap
            if dilution_sum is not None:
                dilution_yield = (dilution_sum * fx_rate) / market_cap
 
            # ── FCF Yield ──────────────────────────────────────────────
            if "Free Cash Flow" in cf.index:
                raw_fcf = cf.loc["Free Cash Flow"].iloc[:n_periods] # type: ignore
                if raw_fcf.isna().all(): #type: ignore
                    print(f"[MISSING] {ticker_symbol}: 'Free Cash Flow' row exists but all NaN")
                    fcf_yield = None
                else:
                    fcf_ttm   = float(raw_fcf.fillna(0).sum()) * fx_rate # type: ignore
                    fcf_yield = fcf_ttm / market_cap
            elif "Operating Cash Flow" in cf.index:
                raw_ocf = cf.loc["Operating Cash Flow"].iloc[:n_periods] # type: ignore
                if raw_ocf.isna().all(): # type: ignore
                    print(f"[MISSING] {ticker_symbol}: no FCF or Operating Cash Flow rows found")
                    fcf_yield = None
                else:
                    ocf   = float(raw_ocf.fillna(0).sum()) * fx_rate # type: ignore
                    capex = ttm_sum(cf, ["Capital Expenditure", "Purchase Of PPE"], n_periods) # type: ignore
                    fcf_yield = (ocf - abs((capex or 0) * fx_rate)) / market_cap

        # ── Shareholder Yield ──────────────────────────────────────────
        # Use 0 for missing buyback/dilution so a valid dividend still produces
        # a result. Only return None if ALL three components are absent.
        _bk = buyback_yield  or 0.0
        _dl = dilution_yield or 0.0
        if dividend_yield == 0 and buyback_yield is None and dilution_yield is None:
            shareholder_yield = None
        else:
            shareholder_yield = dividend_yield + _bk - _dl

        # ── Share Count Change (true 3-year) ───────────────────────────
        shares       = ticker.get_shares_full()
        share_factor = None
        if shares is not None and len(shares) >= 36:
            share_factor = float(shares.iloc[-1] / shares.iloc[-36] - 1)
        elif shares is not None and len(shares) >= 12:
            share_factor = float(shares.iloc[-1] / shares.iloc[-12] - 1)
        # print(f"{ticker_symbol} share count change: {share_factor:.2%} over {len(shares)} months")

        # ── Price Momentum (12-1 month) ────────────────────────────────
        hist     = ticker.history(period="13mo")
        momentum = None
        if hist is not None and len(hist) >= 252:
            price_12m_ago = hist["Close"].iloc[0]
            price_1m_ago  = hist["Close"].iloc[-21]
            if price_12m_ago > 0:
                momentum = float(price_1m_ago / price_12m_ago - 1)

        # ── Leverage (Debt / Equity) ───────────────────────────────────
        total_debt   = info.get("totalDebt", None)
        total_equity = info.get("bookValue", None)
        shares_out   = info.get("sharesOutstanding", None)
        leverage     = None
        if total_debt and total_equity and shares_out and total_equity > 0:
            equity_total = total_equity * shares_out
            leverage     = total_debt * fx_rate / equity_total

        return {
            "Ticker":            ticker_symbol,
            "Sector":            sector,
            "Dividend Yield":    dividend_yield,
            "Buyback Yield":     buyback_yield,     # None = no data, not 0
            "Dilution Yield":    dilution_yield,    # None = no data, not 0
            "Shareholder Yield": shareholder_yield, # None = all components absent
            "FCF Yield":         fcf_yield,         # None = no data, not 0
            "3Y Share Change":   share_factor,
            "Momentum (12-1)":   momentum,
            "Leverage (D/E)":    leverage,
        }

    except Exception as e:
        print(f"[ERROR] {ticker_symbol}: {e}")
        return None

# ─────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────
def score_factor(value: float, bad: float, neutral: float, good: float) -> float:
    """
    Map a raw factor value to [-1, +1] using fixed anchors.
    Clamps outside the bad/good range.
    """
    if value is None or np.isnan(value):
        return 0.0   # neutral for missing data

    if good > bad:   # higher is better (FCF yield, momentum)
        if value >= good:    return  1.0
        if value <= bad:     return -1.0
        if value >= neutral: return  (value - neutral) / (good - neutral)
        return                      -(neutral - value) / (neutral - bad)
    else:            # lower is better (leverage, share dilution)
        if value <= good:    return  1.0
        if value >= bad:     return -1.0
        if value <= neutral: return  (neutral - value) / (neutral - good)
        return                      -(value - neutral) / (bad - neutral)


def compute_factor_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for factor, (bad, neutral, good) in FACTOR_THRESHOLDS.items():
        if factor == "Shareholder Yield":
            df[f"s_{factor}"] = df.apply(
                lambda row: score_factor(
                    0.0 if (row["Shareholder Yield"] is None and row["Dividend Yield"] == 0)
                    else row["Shareholder Yield"],
                    bad, neutral, good
                ), axis=1
            )
        else:
            df[f"s_{factor}"] = df[factor].apply(
                lambda v: score_factor(v, bad, neutral, good)
            )

    df["Factor Score"] = sum(
        df[f"s_{factor}"] * weight
        for factor, weight in FACTOR_WEIGHTS.items()
    )

    return df

def get_signal(score: float) -> str:
    if score >  BUY_THRESHOLD:  return "BUY"
    if score < SELL_THRESHOLD:  return "SELL"
    return "HOLD"

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run_factor_model(tickers: list[str]) -> pd.DataFrame:
    records = [extract_factors(t) for t in tickers]
    df = pd.DataFrame([r for r in records if r is not None])

    if df.empty:
        raise ValueError("No valid tickers returned data.")

    df = compute_factor_scores(df)
    df["Signal"] = df["Factor Score"].apply(get_signal)
    df = df.sort_values("Factor Score", ascending=False).reset_index(drop=True)

    return df


def get_top_holdings(etf):
    """
    Get the top holdings for a given ETF.
    
    Args:
        etf (str): ETF ticker symbol
    Returns:
        holdings (list[str]): list of top ten holdings' ticker symbols
    """
    info = yf.Ticker(etf)
    holdings = info.get_funds_data().top_holdings # type: ignore
    holdings.sort_values(by='Holding Percent', ascending=False, inplace=True)
    return [USD_EQUIVALENT_MAP.get(holding, holding) for holding in holdings.index.tolist()]

def get_daily_prices(tickers: list[str], period: str, freq: str) -> pd.DataFrame:
    """
    Get the top holdings for a given ETF.
    
    Args:
        etf (str): ETF ticker symbol
    Returns:
        holdings (list[str]): list of top ten holdings' ticker symbols
    """
    # Download adjusted close prices
    data = yf.download(tickers, period = period, auto_adjust=True)['Close'] # type: ignore
    data.dropna(axis=1, how='all', inplace=True) # type: ignore

    # interval = "Daily" or "Weekly" from dropdown
    if freq == "Weekly":
        # Resample weekly using last price and compute pct_change
        data = data.resample('W').last()
    elif freq == "Monthly":
        # Resample monthly using last price and compute pct_change
        data = data.resample('M').last()

    returns = data.pct_change().dropna()

    return returns # type: ignore


def get_daily_returns(holdings: list[str], period: str) -> tuple[pd.DataFrame,list]:  
    """
    Fetches daily returns for each ticker between start_date and end_date,
    adjusting for expense ratios if available.

    Args:
        tickers (Ticker): yahooquery.Ticker object containing tickers to fetch.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame with daily returns for each symbol.
        list: Warnings encountered during data retrieval.
    """

    tickers = Ticker(holdings)
    warnings = []
    hist = tickers.history(period = period, interval="1d")

    hist = hist.reset_index()[["symbol", "date", "adjclose"]]
    fund_profiles = tickers.fund_profile

    result = pd.DataFrame()

    for symbol, _ in fund_profiles.items():
        df = hist[hist["symbol"] == symbol].copy()
        if df.empty:
            warnings.append(f"No price data for {symbol}, skipped.")
            continue
        
        # if securities are in different timezones, normalize to date only
        df['date']=pd.to_datetime(df['date'],utc=True).dt.tz_convert(None).dt.date
        df = df.set_index("date").sort_index()

        # Price relative to first date
        df["ret"] = df["adjclose"].pct_change()

        # Expense ratio adjustment (annual, so prorate daily)
        exp_ratio = None
        if fund_profiles.get(symbol) and type(fund_profiles.get(symbol))==dict and "feesExpensesInvestment" in fund_profiles[symbol]:
            exp_ratio = fund_profiles[symbol]["feesExpensesInvestment"].get("annualReportExpenseRatio", 0) # type: ignore
            
        if exp_ratio and exp_ratio > 0:
            daily_drag = (1 - exp_ratio) ** (1 / 252)
            df["ret"] = (1 + df["ret"]) * daily_drag - 1

        result[symbol] = df["ret"].dropna(axis=0)

    return result, warnings

def validate_and_classify_ticker(ticker: str, min_days: int = 250) -> tuple[bool, bool]:
    """
    Validate a ticker and classify whether it is an ETF or a stock.

    Args:
        ticker (str): Ticker symbol (e.g. 'AAPL', 'SPY')
        min_days (int): Minimum price history rows required

    Returns:
        valid (bool): True if ticker is valid, False otherwise
        etf (bool): True if ticker is an ETF, False if stock
    """
    valid, etf = False, False
    try:
        t = yf.Ticker(ticker)

        # 1️⃣ Hard validity check (price history)
        hist = t.history(period="1y", interval="1d")

        if hist.empty or len(hist) < min_days:
            return valid, etf

        valid = True

        # 2️⃣ Asset classification
        info = t.info or {}
        quote_type = info.get("quoteType")
        if quote_type == "ETF":
            etf = True

        return valid, etf

    except Exception as e:
        return False, False

def get_min_transaction_size(dots: pd.DataFrame) -> int:
    """Dynamically determine minimum transaction size so there are at most 50 dots."""
    thresholds = [500, 1000, 2500, 5000, 10000, 25000, 50000, 100000]
    for threshold in thresholds:
        if len(dots[dots['weighted_shares'].abs() >= threshold]) <= 50:
            return threshold
    return thresholds[-1]


def get_indirect_weight(indirect_nature: str) -> float:
    # Strip footnote references like (3) before matching
    text = re.sub(r'\(\d+\)', '', indirect_nature).lower().strip()
    
    if any(k in text for k in PLAN_KEYWORDS):
        return 0.0    # automatic, no signal
    if any(k in text for k in SPOUSE_KEYWORDS):
        return 0.4    # household decision but independent, moderate discount
    if any(k in text for k in TRUST_KEYWORDS):
        return 0.75   # controlled entity, strong but slight discount
    return 0.25       # unknown indirect, cautious default

def get_weighted_shares(nd_trade: dict) -> tuple[int, float]:
    """
    Calculate the weighted shares for a non-derivative trade based on its amount, transaction code, and ownership type.
    For direct ownership, apply the weight from TRADE_WEIGHTS based on the transaction code.
    For indirect ownership, apply a weight based on the nature of the indirect ownership (e.g., plan, trust, spouse).
    """
    amount = int(nd_trade['amount']) if nd_trade['amount'] else 0
    signed = amount if nd_trade['change'] == "A" else -amount

    if nd_trade.get('ownership_type') == 'I':
        weight = get_indirect_weight(nd_trade.get('indirect_nature', ''))
        # print(f"Indirect nature: '{nd_trade.get('indirect_nature', '')}', weight: {weight}")
        return signed, weight
    
    return signed, TRADE_WEIGHTS.get(nd_trade.get('transaction_code', 'P'), 1.0)


def transactions_to_cumm_ts(insider_trades: list[dict], net_shares: float, start_date: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Convert transaction dictionaries into a cumulative buy/sell time series.

    Args:
        insider_trades (list[dict]): list of transaction records (dictionaries)
        net_shares (float): total net shares traded
        start_date (str | None): ISO date string (YYYY-MM-DD) to filter trades from.
                                 Defaults to earliest trade date if None.

    Returns:
        pd.DataFrame: indexed by date with cumulative shares (daily time series)
        pd.DataFrame: dots data for individual transactions
        list: quarter_ranges: list of (start,end) tuples for each quarter in the data
    """

    INSIDER_ROLES = {
        "ceo", "cfo", "coo", "cto", "president", "director", "general counsel"
    }
    
    cleaned_trades = []
    for trd in insider_trades:
        if "ERROR" in trd:
            continue
        trd_info = {
            "accession_number": trd['accession_number'],
            "insider": trd['reporter_name'],
            "role": ", ".join(trd['role']) if type(trd['role']) is list else trd['role'],
            "date": trd['date']
        }
        trd_amount, trd_weighted = 0, 0
        for nd_trade in trd['non_deriv_trades']:
            signed, weight = get_weighted_shares(nd_trade)
            if weight == 0:
                continue
            trd_amount += signed
            trd_weighted += signed * weight

        if trd_amount == 0 and trd_weighted == 0:
            continue  # skip trades with no meaningful activity

        trd_info['shares'] = trd_amount           # raw shares for hover
        trd_info['weighted_shares'] = trd_weighted # weighted for cumulative chart
        trd_info['url'] = trd['url']
        cleaned_trades.append(trd_info)

    records = pd.DataFrame(cleaned_trades)
    records["date"] = pd.to_datetime(records["date"], utc=True).dt.date

    parsed_start = datetime.strptime(start_date, "%m/%d/%Y").date() if start_date else records["date"].min()
    end_date = datetime.today().date()

    records = records[records["date"] >= parsed_start]

    daily = (
        records.groupby("date")["weighted_shares"]
        .sum()
        .rename("net_shares")
        .to_frame()
    )

    full_idx = pd.date_range(start=parsed_start, end=end_date, freq="D").date
    daily = daily.reindex(full_idx, fill_value=0)
    daily["cum_shares"] = daily["net_shares"].cumsum()

    dots = (
        records
        .groupby(["date", "accession_number", "url", "insider", "role"], as_index=False)
        .agg({"shares": "sum", "weighted_shares": "sum"})
    )

    dots['cum_shares'] = dots.weighted_shares.cumsum()
    min_size = get_min_transaction_size(dots)
    dots_filtered = dots[(dots['weighted_shares'].abs() >= min_size) | dots['role'].str.lower().str.strip().isin(INSIDER_ROLES)]

    quarters = pd.date_range(start=parsed_start, end=end_date, freq='QS')
    quarter_ranges = [(q, q + pd.offsets.QuarterEnd()) for q in quarters]

    return daily.reset_index().rename(columns={"index": "date"}), dots_filtered, quarter_ranges