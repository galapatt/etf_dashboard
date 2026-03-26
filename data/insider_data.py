from datetime import datetime
import json
import os
import re
import time
from bs4 import BeautifulSoup
import requests
import pandas as pd

from src.config import INSIDER_DATA_PATH, CIK_PATH, HEADER, EMP_PAT, MON_PAT, MAX_TRADE_AGE_DAYS



def refresh_company_tickers(max_age_days: int = 30) -> bool:
    """
    Load company tickers from local JSON file. If the file is older than
    max_age_days, fetch a fresh copy from SEC, reformat it, and save with a new timestamp.
    """
    if os.path.exists(CIK_PATH):
        with open(CIK_PATH) as f:
            data = json.load(f)
        
        last_updated = data.get("last_updated")
        if last_updated:
            age = (datetime.today() - datetime.fromisoformat(last_updated)).days
            if age <= max_age_days:
                print(f"Company tickers up to date (last updated {age} days ago)")
                return data

    print("Refreshing company tickers from SEC...")
    response = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=HEADER
    )

    if response.status_code != 200:
        raise Exception(f"Failed to fetch company tickers: {response.status_code}")

    raw = response.json()

    # Reformat from SEC structure to ticker -> CIK mapping
    tickers = {
        v["ticker"]: str(v["cik_str"]).zfill(10)
        for v in raw.values()
        if "ticker" in v and "cik_str" in v
    }
    tickers["last_updated"] = datetime.today().isoformat()

    with open(CIK_PATH, 'w') as f:
        json.dump(tickers, f, indent=4)

    print(f"Company tickers refreshed and saved to {CIK_PATH}")
    return True

def load_cik_key(ticker: str) -> str:
    """Load the CIK key for a given ticker from the local JSON file."""
    with open(CIK_PATH) as f:
        cik_dict = json.load(f)
    
    cik_key = cik_dict.get(ticker, "")
    if not cik_key:
        raise ValueError(f"Ticker {ticker} not found in company_tickers.json")
    return cik_key

def load_saved_trades(ticker: str) -> list[dict]:
    """Load previously saved insider trades from a local JSON file if it exists."""
    file_name = os.path.join(INSIDER_DATA_PATH, f'{ticker}.json')
    if os.path.exists(file_name):
        with open(file_name) as f:
            insider_trades = json.load(f)
        print(f"Loading insider trades for {ticker} from {file_name}: {len(insider_trades)} trades")
        return insider_trades or []
    
    print(f"Insider trades file for {ticker} not found, loading data from SEC")
    return []

def fetch_sec_filings(cik_key: str) -> tuple[str, pd.DataFrame]:
    """Fetch recent SEC filings metadata for a company and return the ticker and Form 4 filings."""
    filing_metadata = requests.get(
        f'https://data.sec.gov/submissions/CIK{cik_key}.json',
        headers=HEADER
    ).json()

    ticker = filing_metadata['tickers'][0]
    all_forms = pd.DataFrame.from_dict(filing_metadata['filings']['recent']).set_index('accessionNumber', drop=True)
    form4s = all_forms[all_forms['form'].isin(['4', '4/A'])].sort_values(by='filingDate', ascending=False)
    
    return ticker, form4s

def parse_footnotes(tbls, a: bool) -> dict:
    """Parse footnotes table from Form 4, returns a dict of footnote number -> text."""
    footnotes = {}
    try:
        # Footnotes table is typically the last table in the form
        footnote_table = tbls[15 - a]
        rows = footnote_table.find_all('tr')
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 2:
                # Footnote number is in first cell, text in second
                num = re.sub(r'[^\d]', '', cells[0].text.strip())
                text = cells[1].text.strip()
                if num:
                    footnotes[num] = text
    except (IndexError, AttributeError):
        pass
    return footnotes

def parse_non_deriv_trades(table) -> list[dict]:
    """Parse non-derivative trades from a Form 4 table."""
    emp_pat = EMP_PAT
    mon_pat = MON_PAT

    trades = []
    rows = table.find_all('tr')
    if len(rows) < 3:
        return trades
    
    for row in rows[3:]:
        row_values = [re.sub(emp_pat, '', obj.text).strip() for obj in row.find_all('td')]
        trade = {
            'sec_title': row_values[0],
            'date': row_values[1],
            'transaction_code': row_values[3],
            'amount': float(re.sub(mon_pat, '', row_values[5])) if re.sub(mon_pat, '', row_values[5]) else "",
            'change': row_values[6],
            'price': float(re.sub(mon_pat, '', row_values[7])) if re.sub(mon_pat, '', row_values[7]) else "",
            'total_owned': float(row_values[8].replace(',', '')) if row_values[8].replace(',', '') else "",
            'ownership_type': row_values[9],
            'indirect_nature': row_values[10] if len(row_values) > 10 else ""
        }
        trades.append(trade)
    return trades

def parse_deriv_trades(table) -> list[dict]:
    """Parse derivative trades from a Form 4 table."""
    emp_pat = EMP_PAT
    mon_pat = MON_PAT   

    trades = []
    rows = table.find_all('tr')
    if len(rows) < 3:
        return trades
    
    for row in rows[3:]:
        row_values = [re.sub(emp_pat, '', obj.text).strip() for obj in row.find_all('td')]
        if len(row_values) == 1:
            break
        amount = row_values[6] if row_values[6] else row_values[7]
        change = "A" if row_values[6] else "D"
        trade = {
            'sec_title': row_values[0],
            'date': row_values[2],
            'amount': float(re.sub(mon_pat, '', amount)) if re.sub(mon_pat, '', amount) else "",
            'change': change,
            'date_exercisable': row_values[8],
            'exp_date': row_values[9],
            'title_of_underlying': row_values[10],
            'amount_of_underlying': float(re.sub(mon_pat, '', row_values[11])) if re.sub(mon_pat, '', row_values[11]) else "",
            'price': float(re.sub(mon_pat, '', row_values[12])) if re.sub(mon_pat, '', row_values[12]) else "",
            'ownership_type': row_values[14]
        }
        trades.append(trade)
    return trades

def parse_form4(url: str, acc_num: str, primary_doc: str) -> dict | None:
    """Fetch and parse a single Form 4 filing. Returns a dict of trade info or None on failure."""
    file = requests.get(url, headers=HEADER)
    time.sleep(1)

    if file.status_code == 429:
        raise Exception(f"Too many requests: Try again later {url}")
    
    if file.status_code == 503:
        raise Exception(f"SEC.gov is under maintenance, try again later: {url}")
    
    if file.status_code != 200:
        print(f"Error fetching {url}: status code {file.status_code}")
        return {
            "ERROR": f"SEC returned status {file.status_code}",
            "accession_number": acc_num,
            "url": url
        }

    soup = BeautifulSoup(file.text, 'xml')
    tbls = soup.find_all('table')
   
    if len(tbls) == 0:
        print(f"Skipping {url}: no tables found, filing may be corrupted or unavailable")
        return {
            "ERROR": f"No tables found, filing may be corrupted or unavailable now",
            "accession_number": acc_num,
            "url": url
        }  # Signal to skip this filing

    a = True if len(tbls) == 16 else False

    spans, date = tbls[5 - a].find_all('span'), ""
    for i, span in enumerate(spans):
        if "Date of Earliest Transaction" in span.text:
            date = spans[i + 1].text
            break

    dt_date = datetime.strptime(date, '%m/%d/%Y')
    if (datetime.now() - dt_date).days > MAX_TRADE_AGE_DAYS:
        return None  # Signal to stop fetching older filings

    name_html = tbls[6 - a].find('a') or ""
    reporter_name = name_html.text if name_html else ""
    reporter_cik = name_html['href'].split('=')[-1] if name_html and name_html.has_attr('href') else "" # type: ignore

    if len(tbls) == 18: a -= 1
    
    role_lines = tbls[11 - a].find_all('td')
    roles = [
        role_lines[i + 1].text.split(" (")[0]
        for i in range(len(role_lines) - 1)
        if role_lines[i].text == "X"
    ]

    non_deriv_trades = parse_non_deriv_trades(tbls[13 - a])
    deriv_trades = parse_deriv_trades(tbls[14 - a])

    return {
        'accession_number': acc_num,
        'date': date,
        'reporter_name': reporter_name,
        'reporter_cik': reporter_cik,
        'role': roles,
        'non_deriv_trades': non_deriv_trades,
        'deriv_trades': deriv_trades,
        'xml': primary_doc,
        'url': url
    }

def save_trades(ticker: str, trades: list[dict]) -> None:
    """Save insider trades to a local JSON file."""
    file_name = os.path.join(INSIDER_DATA_PATH, f'{ticker}.json')
    print(f"Updating insider trades to {file_name};\nTotal {len(trades)} trades for {ticker}")
    with open(file_name, 'w') as f:
        json.dump(trades, f)

def retry_failed_trades(saved_trades: list[dict]) -> list[dict]:
    retried = []
    for trade in saved_trades:
        if "ERROR" not in trade:
            retried.append(trade)
            continue

        acc_num = trade.get("accession_number")
        url = trade.get("url")

        if not acc_num or not url:
            print(f"Skipping failed trade, missing accession number or url")
            retried.append(trade)
            continue

        primary_doc = url.split('/')[-1]
        print(f"Retrying failed trade: {url}")
        try:
            result = parse_form4(url, acc_num, primary_doc)
            if result and "ERROR" not in result:
                print(f"Successfully retried: {url}")
                retried.append(result)
            else:
                print(f"Retry still failed for {url}: {result.get('ERROR')}") # type: ignore
                retried.append(result)  # keep updated error dict in place
        except Exception as e:
            print(f"Retry raised exception for {url}: {e}")
            retried.append({"ERROR": str(e), "accession_number": acc_num, "url": url})

    return retried

def get_insider_trades(ticker: str) -> tuple[str, list[dict]]:
    """
    Main entry point. Loads saved trades, fetches any new Form 4 filings from SEC,
    and returns the combined result.
    """

    try:
        cik_key = load_cik_key(ticker)
    except ValueError as e:
        return ticker, [{"ERROR": str(e)}]

    saved_trades = load_saved_trades(ticker)
    latest_trade = saved_trades[0] if saved_trades else None

    ticker, form4s = fetch_sec_filings(cik_key)

    # Retry any failed trades from previous runs before fetching new ones
    saved_trades = retry_failed_trades(saved_trades)

    print(f"Downloading insider trades for {ticker}...")

    new_trades = []
    for acc_num, row in form4s.iterrows():
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_key}/{acc_num.replace('-', '')}/{row.primaryDocument}" # type: ignore
        try:
            trade = parse_form4(url, acc_num, row.primaryDocument) # type: ignore
        except Exception as e:
            print(f"Error parsing {url}: {e}")
            return ticker, [{"ERROR": f"Error parsing {url}: {e}"}]

        if trade == {}:  # Skip this filing but keep checking older ones
            continue
        if trade is None:  # Older than 700 days, stop
            break
        if latest_trade and latest_trade['date'] == trade['date'] and latest_trade['accession_number'] == acc_num:
            break  # Already have this trade saved

        new_trades.append(trade)

    all_trades = new_trades + saved_trades
    print(f"Finished downloading {len(all_trades)} insider trades for {ticker}")
    save_trades(ticker, all_trades)

    return ticker, all_trades