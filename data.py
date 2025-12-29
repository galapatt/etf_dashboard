# -----------------------------
# File: data.py
# -----------------------------

from yahooquery import Ticker
import yfinance as yf
import pandas as pd
import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import re
import time
from multiprocessing.dummy import Pool
from datetime import datetime
import os

INSIDER_DATA_PATH = "./data/insider_trades/"

USD_EQUIVALENT_MAP = {
    "CCO.TO": "CCJ",
    "NXE.TO": "NXE",
    "SU.TO": "SU",
    "CNQ.TO": "CNQ",
    "SHOP.TO": "SHOP",
}

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
    data = data.dropna()

    # interval = "Daily" or "Weekly" from dropdown
    if freq == "Weekly":
        # Resample weekly using last price and compute pct_change
        data = data.resample('W').last()
    elif freq == "Monthly":
        # Resample monthly using last price and compute pct_change
        data = data.resample('M').last()

    returns = data.pct_change().dropna()

    return returns # type: ignore


def get_daily_returns(holdings: list[str], period: str) -> pd.DataFrame:  
    """
    Fetches daily returns for each ticker between start_date and end_date,
    adjusting for expense ratios if available.

    Args:
        tickers (Ticker): yahooquery.Ticker object containing tickers to fetch.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame with daily returns for each symbol.
    """

    tickers = Ticker(holdings)
    hist = tickers.history(period = period, interval="1d")

    hist = hist.reset_index()[["symbol", "date", "adjclose"]]
    fund_profiles = tickers.fund_profile

    result = pd.DataFrame()

    for symbol, _ in fund_profiles.items():
        df = hist[hist["symbol"] == symbol].copy()
        if df.empty:
            print(f"No data for {symbol}, skipping.")
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

    return result

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
    
# SHIFT+ALT+F: Format JSON file
def get_insider_trades(ticker: str) -> (str, list[dict]):  # type: ignore
    """
    Given CIK key for a company, get insider trades from the SEC website
    params: 
    cik_key: unique identifier for a company in SEC site
    returns:
    ticker: ticker symbol of the company
    a4_list: list of dictionaries with insider trading information collected from each form
    """
    a4_list = []
    emp_pat = r'\(.*?\)'
    mon_pat = r'[^\d.]'
    header = {'User-Agent': "address@email.com"}

    # Load ticket to CIK mappings from JSON file
    with open('./assets/company_tickers.json') as f:
        cik_dict = json.load(f)    
    
    cik_key = cik_dict.get(ticker, "")
    if not cik_key:
        print(f"Ticker {ticker} not found in company_tickers.json")
        return ticker, [{"ERROR": f"{ticker} not found in company_tickers.json"}]
    
    file_name = os.path.join(INSIDER_DATA_PATH, f'{ticker}.json')
    if os.path.exists(file_name):
        with open(file_name) as f:
            insider_trades = json.load(f)
        print(f"Loading insider trades for {ticker} from {file_name}: {len(insider_trades)} trades")
        saved_trades = insider_trades if insider_trades else []
    else:
        print(f"Insider trades file for {ticker} not found, loading data from SEC")
        saved_trades = []
    
    latest_trade = saved_trades[0] if saved_trades else None
    filingMetadata = requests.get(
        f'https://data.sec.gov/submissions/CIK{cik_key}.json',
        headers = header
        ).json()

    ticker = filingMetadata['tickers'][0]
    allForms = pd.concat([pd.DataFrame(), pd.DataFrame.from_dict(filingMetadata['filings']['recent']).set_index('accessionNumber', drop=True)])
    
    insider_trades = allForms[allForms['form'].isin(['4','4/A'])].sort_values(by='filingDate', ascending=False) # type: ignore

    print(f"Downloading insider trades for {ticker}...")

    for acc_num,row in insider_trades.iterrows():
        
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_key}/{acc_num.replace('-','')}/{row.primaryDocument}" # type: ignore
        file = requests.get(url, headers = header)
        time.sleep(1)
        
        if file.status_code == 429:
            print(file.text)
            raise Exception(f"Too many requests: Try again later {url}")

        soup = BeautifulSoup(file.text, 'xml')
        tbls = soup.find_all('table')
        a = True if len(tbls)==16 else False

        try:
            spans,date = tbls[5-a].find_all('span'),""
            for i,span in enumerate(spans):
                if "Date of Earliest Transaction" in span.text:
                    date = spans[i+1].text
                    break
            
            dt_date = datetime.strptime(date, '%m/%d/%Y')

            if (datetime.now() - dt_date).days > 700 or (latest_trade and latest_trade['date'] == date 
                                                        and latest_trade['accession_number'] == acc_num):
                break

            name_html = tbls[6-a].find('a') if tbls[6-a].find('a') else ""
            reporter_name = name_html.text if name_html else ""
            reporter_cik = name_html['href'].split('=')[-1] if name_html and name_html.has_attr('href') else "" # type: ignore
            xml_name = row.primaryDocument

            role_lines = tbls[11-a].find_all('td')
            roles = []
            for i in range(len(role_lines)-1):
                if role_lines[i].text == "X":
                    roles.append(role_lines[i+1].text.split(" (")[0])

            non_deriv_changes = tbls[13-a].find_all('tr')
            deriv_changes = tbls[14-a].find_all('tr')
            non_deriv_trades, deriv_trades = [],[]

            if len(non_deriv_changes) >= 3:
                non_deriv_changes = non_deriv_changes[3:]
                for row in non_deriv_changes:
                    row_values = [re.sub(emp_pat,'',obj.text).strip() for obj in row.find_all('td')]
                    trade = {
                        'sec_title': row_values[0],
                        'date': row_values[1],
                        'amount': float(re.sub(mon_pat,'',row_values[5])) if re.sub(mon_pat,'',row_values[5]) else "",
                        'change': row_values[6],
                        'price': float(re.sub(mon_pat,'',row_values[7])) if re.sub(mon_pat,'',row_values[7]) else "",
                        'total_owned': float(row_values[8].replace(',','')) if row_values[8].replace(',','') else "",
                        'ownership_type': row_values[9],
                    }
                    non_deriv_trades.append(trade)

            if len(deriv_changes) >= 3:
                deriv_changes = deriv_changes[3:]
                for row in deriv_changes:
                    row_values = [re.sub(emp_pat,'',obj.text).strip() for obj in row.find_all('td')]
                    if len(row_values) == 1: break
                    amount = row_values[6] if row_values[6] else row_values[7]
                    change = "A" if row_values[6] else "D"
                    trade = {
                        'sec_title': row_values[0],
                        'date': row_values[2],
                        'amount': float(re.sub(mon_pat,'',amount)) if re.sub(mon_pat,'',amount) else "",
                        'change': change,
                        'date_exercisable': row_values[8],
                        'exp_date': row_values[9],
                        'title_of_underlying': row_values[10],
                        'amount_of_underlying': float(re.sub(mon_pat,'',row_values[11])) if re.sub(mon_pat,'',row_values[11]) else "",
                        'price': float(re.sub(mon_pat,'',row_values[12])) if re.sub(mon_pat,'',row_values[12]) else "",
                        'ownership_type': row_values[14]
                    }
                    deriv_trades.append(trade)
        except Exception as e:
            print(f"Error parsing {url}: {e}")
            return ticker, [{"ERROR": f"Error parsing {url}: {e}"}]

        resulting_dict = {
            'accession_number': acc_num,
            'date': date,
            'reporter_name': reporter_name,
            'reporter_cik': reporter_cik,
            'role': roles,
            'non_deriv_trades': non_deriv_trades,
            'deriv_trades': deriv_trades, 
            'xml': xml_name
        }

        a4_list.append(resulting_dict)

    a4_list = a4_list + saved_trades
    print(f"Finished downloading {len(a4_list)} insider trades for {ticker}")

    print(f"Updating insider trades to {file_name};\nTotal {len(a4_list)} trades for {ticker} ")
    with open(file_name,'w') as f:
        json.dump(a4_list, f) 

    return ticker, a4_list

def transactions_to_cumm_ts(insider_trades: list[dict], net_shares: float) -> tuple[pd.DataFrame, pd.DataFrame]:  # type: ignore
    """
    Convert transaction dictionaries into a cumulative buy/sell time series.

    Args:
        insider_trades (list[dict]): list of transaction records (dictionaries)
    
    Returns:
        pd.DataFrame: indexed by date with cumulative shares (daily time series)
        pd.DataFrame: dots data for individual transactions
    """
    cleaned_trades = []
    for trd in insider_trades:
        trd_info = {
            "accession_number": trd['accession_number'],
            "insider": trd['reporter_name'],
            "role": ", ".join(trd['role']) if type(trd['role']) is list else trd['role'],
            "date": trd['date']
        }
        trd_amount = 0
        for nd_trade in trd['non_deriv_trades']:
            amount = int(nd_trade['amount']) if nd_trade['amount'] else 0 # type: ignore
            trd_amount += amount if nd_trade['change'] == "A" else -amount    
        
        trd_info['shares'] = trd_amount
        cleaned_trades.append(trd_info)

    records = pd.DataFrame(cleaned_trades)

    records["date"] = pd.to_datetime(records["date"], utc=True).dt.date # type: ignore
    start_date = pd.to_datetime("1/1/2024").date()
    records = records[records["date"] >= start_date]

    # Aggregate daily net shares
    daily = (
        records.groupby("date")["shares"]
        .sum()
        .rename("net_shares")
        .to_frame()
    )

    # Create full date index
    full_idx = pd.date_range(
        start=start_date,
        end=datetime.now().date(),
        freq="D"
    ).date

    # Reindex and fill missing with 0
    daily = daily.reindex(full_idx, fill_value=0)

    # Recompute cumulative sum
    daily["cum_shares"] = daily["net_shares"].cumsum()

    dots = (
        records
        .groupby(["date","accession_number","insider","role"],
                  as_index = False)
        .agg({"shares":"sum"})
    )
    dots['impact'] = dots['shares'].abs() / abs(net_shares)
    dots_filtered = dots[dots['impact'] >= 0.01]  # filter small transactions
    return daily.reset_index().rename(columns={"index": "date"}), dots_filtered # type: ignore