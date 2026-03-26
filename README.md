# ETF Dashboard

## Overview
A dashboard application for tracking and analyzing Exchange Traded Funds (ETFs).

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation
```bash
git clone https://github.com/galapatt/etf_dashboard.git
cd etf_dashboard
pip install -r requirements.txt
```

### Usage
```bash
python main.py
```

## Features
- Real-time ETF price tracking
- Portfolio analysis
- Performance metrics
- Data visualization

## Project Structure
```
etf_dashboard/
├── src/
├── tests/
├── requirements.txt
└── README.md
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first.

## License
MIT License

# Cash Flow Factor Model — Reference Guide

## Overview

This model scores stocks across five factors derived from cash flow statements, price history, and balance sheet data. Each factor is mapped to a score between –1 (bad) and +1 (good) using fixed thresholds, then combined into a weighted composite Factor Score that drives the BUY / HOLD / SELL signal.

---

## Factors

### 1. FCF Yield — Weight: +0.35
**What it measures:** How much free cash flow the company generates relative to its market value. The highest-weighted factor, grounded in Novy-Marx (2013).

**Formula:** `FCF TTM / Market Cap`

**Data source:** `ticker.quarterly_cashflow` — row `"Free Cash Flow"`. Falls back to Operating Cash Flow minus Capital Expenditure if Free Cash Flow is not available.

**Time period:** TTM (trailing twelve months) — sum of the 4 most recent quarterly periods.

**Thresholds:**

| Rating | Value |
|--------|-------|
| Bad | < 0.5% |
| Neutral | 1.5% |
| Good | ≥ 3.5% |

**Missing data:** If the row is entirely absent or all NaN → `N/A` in the table, treated as neutral (0.0) in scoring.

---

### 2. Shareholder Yield — Weight: +0.30
**What it measures:** Total cash returned to shareholders via dividends and buybacks, minus dilution from new share issuance. Based on Faber (2013).

**Formula:** `Dividend Yield + Buyback Yield − Dilution Yield`

**Data sources:**
- **Dividend Yield** — `ticker.info["dividendYield"]`, divided by 100 (yfinance returns e.g. 0.92 meaning 0.92%)
- **Buyback Yield** — quarterly cashflow row `"Net Common Stock Issuance"` (negative values = buybacks). Falls back to `"Repurchase Of Capital Stock"` / `"Repurchase Of Stock"` / `"Purchase Of Stock"`
- **Dilution Yield** — positive values from the same issuance rows, or `"Issuance Of Capital Stock"` / `"Common Stock Issuance"`

All cashflow values converted to USD using a live FX rate before dividing by market cap.

**Time period:** TTM — sum of the 4 most recent quarterly periods.

**Thresholds:**

| Rating | Value |
|--------|-------|
| Bad | < 0.5% |
| Neutral | 1.5% |
| Good | ≥ 4.0% |

**Missing data:** If buyback/dilution rows are absent but a dividend exists, Shareholder Yield = Dividend Yield alone. If all three components are absent → `N/A`.

---

### 3. Momentum (12-1) — Weight: +0.20
**What it measures:** Price return over the past 12 months, skipping the most recent month. The skip avoids short-term reversal noise. Based on Jegadeesh & Titman (1993).

**Formula:** `Price 1 month ago / Price 12 months ago − 1`

**Data source:** `ticker.history(period="13mo")` — uses `Close` prices at `iloc[0]` (12 months ago) and `iloc[-21]` (approximately 1 month ago, ~21 trading days).

**Time period:** 12-month window, ending 1 month before today.

**Thresholds:**

| Rating | Value |
|--------|-------|
| Bad | < –10% |
| Neutral | +5% |
| Good | ≥ +20% |

> **Note:** The good threshold caps at +20%, so strong momentum years (e.g. semis in 2024) hit the ceiling early. Consider switching to rank-based normalization within your universe for this factor, as momentum is regime-dependent.

**Missing data:** Requires at least 252 trading days of history. If not available → `N/A`, treated as neutral.

---

### 4. 3Y Share Change — Weight: −0.10
**What it measures:** Change in share count over 3 years. Increasing shares = dilution = penalized. Based on Pontiff & Woodgate (2008).

**Formula:** `Shares today / Shares 36 months ago − 1`

**Data source:** `ticker.get_shares_full()` — returns a monthly time series of shares outstanding.

**Time period:** 36 months (true 3-year). Falls back to 12 months if fewer than 36 observations are available.

**Thresholds:**

| Rating | Value |
|--------|-------|
| Bad (dilution) | > +5% |
| Neutral | 0% |
| Good (buyback) | ≤ −3% |

**Missing data:** If share history unavailable → `N/A`, treated as neutral.

---

### 5. Leverage (D/E) — Weight: −0.05
**What it measures:** Total debt relative to book equity. High leverage is penalized, especially when combined with large buyback programs financed by debt. Based on Fama & French (2015).

**Formula:** `Total Debt × FX Rate / (Book Value Per Share × Shares Outstanding)`

**Data sources:** All from `ticker.info`:
- `"totalDebt"` — reported in the company's local currency, converted to USD
- `"bookValue"` — per share, in USD (yfinance normalizes this)
- `"sharesOutstanding"`

**Time period:** Point-in-time (most recent reported balance sheet).

**Thresholds:**

| Rating | Value |
|--------|-------|
| Bad | > 2.0× |
| Neutral | 1.0× |
| Good | ≤ 0.4× |

**Missing data:** If any of the three inputs are absent or equity is zero → `N/A`, treated as neutral.

> **Known issue:** Foreign ADRs (e.g. TSM) can show distorted leverage if `totalDebt` is in local currency and not properly converted. The FX adjustment mitigates this, but verify manually for non-USD reporters.

---

## FX Handling

For tickers whose financials are reported in a non-USD currency (detected via `ticker.info["financialCurrency"]`), a live exchange rate is fetched using `yfinance` (e.g. `TWDUSD=X`). All cashflow values are multiplied by this rate before dividing by market cap. If the FX rate cannot be fetched, the ticker is skipped entirely.

---

## Scoring

Each raw factor value is passed through `score_factor()`, which maps it linearly to [–1, +1] using the three anchors (bad, neutral, good). Values outside the range are clamped to ±1. Missing values return 0.0 (neutral — no reward, no penalty).

The weighted composite is:

```
Factor Score = (s_FCF Yield × 0.35)
             + (s_Shareholder Yield × 0.30)
             + (s_Momentum × 0.20)
             − (s_3Y Share Change × 0.10)
             − (s_Leverage × 0.05)
```

Absolute weights sum to 1.0. Negative weights on Share Change and Leverage mean higher raw values produce lower scores.

---

## Signal

| Condition | Signal |
|-----------|--------|
| Factor Score > 0.15 | BUY |
| Factor Score < –0.20 | SELL |
| Otherwise | HOLD |

---

## Key Limitations

- **Small universe:** Z-score and rank normalization are unreliable below ~30 stocks. Fixed thresholds (used here) are more robust at small universe sizes.
- **Sector mixing:** Comparing raw FCF yield across sectors (e.g. tech vs. mining) is imperfect. Sector-neutral scoring is the proper fix for larger universes.
- **Momentum ceiling:** The +20% good threshold caps strong performers. Consider rank-based normalization for this factor specifically.
- **Point-in-time leverage:** Balance sheet data is as of the last reported quarter, not TTM-averaged.
- **yfinance reliability:** Data can occasionally be stale, mis-labelled, or missing rows. Always sanity-check outliers with the debug print statements in the code.