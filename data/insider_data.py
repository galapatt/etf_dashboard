from datetime import datetime
import json
import os
import re
import socket
import time
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import requests
import pandas as pd

from src.config import INSIDER_DATA_PATH, CIK_PATH, HEADER, EMP_PAT, MON_PAT, MAX_TRADE_AGE_DAYS


def clear_dead_loopback_proxies() -> None:
    """Remove dead localhost proxy env vars that make SEC requests fail."""
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        value = os.environ.get(key)
        if not value:
            continue

        parsed = urlparse(value)
        if parsed.hostname not in {"127.0.0.1", "localhost"} or not parsed.port:
            continue

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.2)
        try:
            if sock.connect_ex((parsed.hostname, parsed.port)) != 0:
                os.environ.pop(key, None)
        finally:
            sock.close()


clear_dead_loopback_proxies()


def sec_get(url: str) -> requests.Response:
    clear_dead_loopback_proxies()
    return requests.get(url, headers=HEADER)


def safe_get_json(url: str) -> dict:
    """Fetch JSON from SEC and raise a readable error when the response is empty or invalid."""
    response = sec_get(url)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch SEC JSON ({response.status_code}) from {url}")

    try:
        return response.json()
    except requests.exceptions.JSONDecodeError as exc:
        body = response.text[:200].strip()
        preview = body if body else "<empty response>"
        raise Exception(f"SEC returned invalid JSON for {url}: {preview}") from exc


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
    raw = safe_get_json("https://www.sec.gov/files/company_tickers.json")

    # Reformat from SEC structure to ticker -> CIK mapping
    tickers = {
        v["ticker"]: str(v["cik_str"]).zfill(10)
        for v in raw.values()
        if "ticker" in v and "cik_str" in v
    }
    tickers["last_updated"] = datetime.today().isoformat()

    try:
        with open(CIK_PATH, 'w') as f:
            json.dump(tickers, f, indent=4)
        print(f"Company tickers refreshed and saved to {CIK_PATH}")
    except PermissionError as exc:
        print(f"[WARN] Could not save refreshed company tickers to {CIK_PATH}: {exc}")
    return True

def load_company_ticker_map() -> dict:
    with open(CIK_PATH) as f:
        return json.load(f)


def load_cik_key(ticker: str) -> str:
    """Load the CIK key for a given ticker from the local JSON file."""
    cik_dict = load_company_ticker_map()
    
    cik_key = cik_dict.get(ticker, "")
    if not cik_key:
        raise ValueError(f"Ticker {ticker} not found in company_tickers.json")
    return cik_key

def sibling_tickers_for_cik(ticker: str, cik_key: str) -> list[str]:
    cik_dict = load_company_ticker_map()
    siblings = [
        symbol
        for symbol, cik in cik_dict.items()
        if symbol != "last_updated" and cik == cik_key and symbol != ticker
    ]
    return sorted(siblings)


def load_saved_trades(ticker: str, cik_key: str | None = None) -> list[dict]:
    """Load previously saved insider trades from a local JSON file if it exists."""
    file_name = os.path.join(INSIDER_DATA_PATH, f'{ticker}.json')
    if os.path.exists(file_name):
        with open(file_name) as f:
            insider_trades = json.load(f)
        print(f"Loading insider trades for {ticker} from {file_name}: {len(insider_trades)} trades")
        return insider_trades or []

    if cik_key:
        for sibling in sibling_tickers_for_cik(ticker, cik_key):
            sibling_file = os.path.join(INSIDER_DATA_PATH, f'{sibling}.json')
            if os.path.exists(sibling_file):
                with open(sibling_file) as f:
                    insider_trades = json.load(f)
                print(
                    f"Loading insider trades for {ticker} from shared-CIK cache "
                    f"{sibling_file}: {len(insider_trades)} trades"
                )
                return insider_trades or []
    
    print(f"Insider trades file for {ticker} not found, loading data from SEC")
    return []

def fetch_sec_filings(cik_key: str) -> pd.DataFrame:
    """Fetch recent SEC filings metadata for a company and return Form 4 filings."""
    filing_metadata = safe_get_json(f'https://data.sec.gov/submissions/CIK{cik_key}.json')

    all_forms = pd.DataFrame.from_dict(filing_metadata['filings']['recent']).set_index('accessionNumber', drop=True)
    form4s = all_forms[all_forms['form'].isin(['4', '4/A'])].sort_values(by='filingDate', ascending=False)
    
    return form4s

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


def parse_numeric_cell(value: str):
    cleaned = re.sub(MON_PAT, '', value or '')
    if not cleaned or cleaned == ".":
        return ""
    return float(cleaned)


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
        if len(row_values) < 10 or (not row_values[1] and not row_values[3]):
            continue
        trade = {
            'sec_title': row_values[0],
            'date': row_values[1],
            'transaction_code': row_values[3],
            'amount': parse_numeric_cell(row_values[5]),
            'change': row_values[6],
            'price': parse_numeric_cell(row_values[7]),
            'total_owned': parse_numeric_cell(row_values[8]),
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
        if len(row_values) < 15 or (not row_values[2] and not row_values[4]):
            continue
        amount = row_values[6] if row_values[6] else row_values[7]
        change = "A" if row_values[6] else "D"
        trade = {
            'sec_title': row_values[0],
            'date': row_values[2],
            'amount': parse_numeric_cell(amount),
            'change': change,
            'date_exercisable': row_values[8],
            'exp_date': row_values[9],
            'title_of_underlying': row_values[10],
            'amount_of_underlying': parse_numeric_cell(row_values[11]),
            'price': parse_numeric_cell(row_values[12]),
            'ownership_type': row_values[14]
        }
        trades.append(trade)
    return trades


def tag_text(parent, name: str, default: str = "") -> str:
    tag = parent.find(name) if parent else None
    return tag.text.strip() if tag and tag.text else default


def tag_float(parent, name: str):
    value = tag_text(parent, name)
    cleaned = re.sub(MON_PAT, "", value)
    return float(cleaned) if cleaned else ""


def parse_xml_non_deriv_trades(soup: BeautifulSoup) -> list[dict]:
    trades = []
    for tx in soup.find_all("nonDerivativeTransaction"):
        trade = {
            "sec_title": tag_text(tx.find("securityTitle"), "value"),
            "date": normalize_form4_date(tag_text(tx.find("transactionDate"), "value")),
            "transaction_code": tag_text(tx.find("transactionCoding"), "transactionCode"),
            "amount": tag_float(tx.find("transactionAmounts"), "transactionShares"),
            "change": tag_text(tx.find("transactionAmounts"), "transactionAcquiredDisposedCode"),
            "price": tag_float(tx.find("transactionAmounts"), "transactionPricePerShare"),
            "total_owned": tag_float(tx.find("postTransactionAmounts"), "sharesOwnedFollowingTransaction"),
            "ownership_type": tag_text(tx.find("ownershipNature"), "directOrIndirectOwnership"),
            "indirect_nature": tag_text(tx.find("ownershipNature"), "natureOfOwnership"),
        }
        trades.append(trade)
    return trades


def parse_xml_deriv_trades(soup: BeautifulSoup) -> list[dict]:
    trades = []
    for tx in soup.find_all("derivativeTransaction"):
        amounts = tx.find("transactionAmounts")
        acquired_disposed = tag_text(amounts, "transactionAcquiredDisposedCode")
        trade = {
            "sec_title": tag_text(tx.find("securityTitle"), "value"),
            "date": normalize_form4_date(tag_text(tx.find("transactionDate"), "value")),
            "amount": tag_float(amounts, "transactionShares"),
            "change": acquired_disposed,
            "date_exercisable": tag_text(tx.find("exerciseDate"), "value"),
            "exp_date": tag_text(tx.find("expirationDate"), "value"),
            "title_of_underlying": tag_text(tx.find("underlyingSecurity"), "underlyingSecurityTitle"),
            "amount_of_underlying": tag_float(tx.find("underlyingSecurity"), "underlyingSecurityShares"),
            "price": tag_float(amounts, "transactionPricePerShare"),
            "ownership_type": tag_text(tx.find("ownershipNature"), "directOrIndirectOwnership"),
        }
        trades.append(trade)
    return trades


def normalize_form4_date(raw_date: str) -> str:
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw_date, fmt).strftime("%m/%d/%Y")
        except ValueError:
            continue
    return raw_date


def extract_form4_date(soup: BeautifulSoup, tbls: list, a: bool, url: str) -> str:
    """
    Extract the effective filing date from a Form 4.

    Prefer raw XML tags because SEC filings are not always rendered into the
    same table/span layout. Fall back to the legacy table parser for older logic.
    """
    for tag_name in ("periodOfReport", "dateOfEarliestTransaction"):
        tag = soup.find(tag_name)
        if tag and tag.text.strip():
            date = normalize_form4_date(tag.text.strip())
            if date:
                return date

    try:
        spans = tbls[5 - a].find_all('span')
        for i, span in enumerate(spans):
            if "Date of Earliest Transaction" in span.text and i + 1 < len(spans):
                raw_date = spans[i + 1].text.strip()
                if raw_date:
                    return normalize_form4_date(raw_date)
    except (IndexError, AttributeError):
        pass

    full_text = soup.get_text("\n", strip=True)
    label_match = re.search(
        r"Date of Earliest Transaction.*?(\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})",
        full_text,
        re.DOTALL,
    )
    if label_match:
        return normalize_form4_date(label_match.group(1))

    raise ValueError(f"Could not find transaction date in filing: {url}")


def find_form4_table(tbls: list, heading: str):
    candidates = []
    for table in tbls:
        text = table.get_text(" ", strip=True)
        if not text:
            continue
        if text.startswith(f"Table I - {heading}") or text.startswith(f"Table II - {heading}"):
            return table
        if heading in text and "UNITED STATES SECURITIES AND EXCHANGE COMMISSION" not in text:
            candidates.append((len(text), table))
    return sorted(candidates, key=lambda item: item[0])[0][1] if candidates else None


def parse_form4(url: str, acc_num: str, primary_doc: str) -> dict | None:
    """Fetch and parse a single Form 4 filing. Returns a dict of trade info or None on failure."""
    file = sec_get(url)
    time.sleep(0.2)

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

    date = extract_form4_date(soup, tbls, a, url)
    dt_date = datetime.strptime(date, '%m/%d/%Y')
    if (datetime.now() - dt_date).days > MAX_TRADE_AGE_DAYS:
        return None  # Signal to stop fetching older filings

    name_html = tbls[6 - a].find('a') or ""
    reporter_name = name_html.text if name_html else ""
    reporter_cik = name_html['href'].split('=')[-1] if name_html and name_html.has_attr('href') else "" # type: ignore

    if len(tbls) == 18: a -= 1
    if any("(Country)" in tbl.get_text(" ", strip=True) for tbl in tbls[:5]):
        a -= 1
    
    role_lines = tbls[11 - a].find_all('td') if len(tbls) > 11 - a else []
    roles = [
        role_lines[i + 1].text.split(" (")[0]
        for i in range(len(role_lines) - 1)
        if role_lines[i].text == "X"
    ]

    non_deriv_trades = parse_xml_non_deriv_trades(soup)
    deriv_trades = parse_xml_deriv_trades(soup)

    non_deriv_table = find_form4_table(
        tbls,
        "Non-Derivative Securities Acquired, Disposed of, or Beneficially Owned",
    )
    if not non_deriv_trades and non_deriv_table:
        non_deriv_trades = parse_non_deriv_trades(non_deriv_table)

    deriv_table = find_form4_table(
        tbls,
        "Derivative Securities Acquired, Disposed of, or Beneficially Owned",
    )
    if not deriv_trades and deriv_table:
        rows = deriv_table.find_all("tr")[3:]
        no_derivative_trades = next(
            (
                "Name and Address of Reporting Person" in row.get_text(" ", strip=True)
                for row in rows
                if row.get_text(" ", strip=True)
            ),
            False
        )
        deriv_trades = parse_deriv_trades(deriv_table) if not no_derivative_trades else []

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
    try:
        with open(file_name, 'w') as f:
            json.dump(trades, f)
    except PermissionError as exc:
        print(f"[WARN] Could not save insider trades for {ticker} to {file_name}: {exc}")

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

    requested_ticker = ticker
    saved_trades = load_saved_trades(requested_ticker, cik_key)

    form4s = fetch_sec_filings(cik_key)

    # Retry any failed trades from previous runs before fetching new ones
    saved_trades = retry_failed_trades(saved_trades)
    saved_accessions = {
        trade.get("accession_number")
        for trade in saved_trades
        if (
            "ERROR" not in trade
            and trade.get("accession_number")
        )
    }

    print(f"Checking insider trades for {requested_ticker}...")

    new_trades = []
    cutoff_date = pd.Timestamp(datetime.now().date() - pd.Timedelta(days=MAX_TRADE_AGE_DAYS))
    for acc_num, row in form4s.iterrows():
        if acc_num in saved_accessions:
            break

        report_date = pd.to_datetime(getattr(row, "reportDate", None), errors="coerce")
        filing_date = pd.to_datetime(getattr(row, "filingDate", None), errors="coerce")
        if pd.notna(report_date) and report_date < cutoff_date:
            if pd.notna(filing_date) and filing_date < cutoff_date:
                break
            continue

        url = f"https://www.sec.gov/Archives/edgar/data/{cik_key}/{acc_num.replace('-', '')}/{row.primaryDocument}" # type: ignore
        try:
            trade = parse_form4(url, acc_num, row.primaryDocument) # type: ignore
        except Exception as e:
            print(f"Error parsing {url}: {e}")
            return requested_ticker, [{"ERROR": f"Error parsing {url}: {e}"}]

        if trade == {}:  # Skip this filing but keep checking older ones
            continue
        if trade is None:  # Older than the trade-age window, skip but keep scanning by filing date
            continue

        new_trades.append(trade)

    all_trades_by_accession = {}
    for trade in new_trades + saved_trades:
        accession = trade.get("accession_number")
        if accession and accession not in all_trades_by_accession:
            all_trades_by_accession[accession] = trade

    all_trades = sorted(
        all_trades_by_accession.values(),
        key=lambda trade: datetime.strptime(trade["date"], "%m/%d/%Y") if trade.get("date") else datetime.min,
        reverse=True,
    )
    if new_trades:
        print(f"Finished downloading {len(all_trades)} insider trades for {requested_ticker}")
        save_trades(requested_ticker, all_trades)
    else:
        print(f"No new insider trades for {requested_ticker}; using {len(all_trades)} cached trades")

    return requested_ticker, all_trades
