import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INSIDER_DATA_PATH = os.path.join(ROOT_DIR, 'forms', 'a4_forms')
CIK_PATH = os.path.join(ROOT_DIR, 'assets', 'company_tickers.json')
HEADER = {'User-Agent': "address@email.com"}
EMP_PAT = r'\(.*?\)'
MON_PAT = r'[^\d.]'
MAX_TRADE_AGE_DAYS = 700

TRADE_WEIGHTS = {
    # Strong signal (sums to 3.0)
    # Discretionary cash decisions that directly reflect insider conviction
    'P': 2.0,   # Open market purchase — insider spends their own cash at market price, strongest and rarest conviction signal. Research shows ~2x more informative than sells
    'S': 1.0,   # Open market sale — meaningful but discounted, insiders sell for many non-bearish reasons such as diversification, taxes, and lifestyle spending

    # Moderate signal (sums to 2.0)
    # Insider takes on some risk or shows discretion but no fresh cash involved
    'J': 0.9,   # Other acquisition/disposition — research shows highly informed transactions with ~8% abnormal returns. Kept just below S so J + S nets a slight negative (-0.1)
    'M': 0.7,   # Option exercise — positive if held showing price risk acceptance, nets moderate negative (-0.3) with immediate sale reflecting exercise-and-dump behaviour
    'K': 0.4,   # Equity swap — synthetic exposure change, more likely hedging than conviction so heavily discounted

    # Dampened signal (sums to 1.0)
    # Not discretionary but holding behaviour after still carries weak signal
    'A': 0.3,   # Grant/award — routine compensation, neutral to receive but large grant with no subsequent sale is mildly positive
    'F': 0.2,   # Tax withholding sale — automatic disposal to cover vesting tax, not discretionary but still a share reduction
    'W': 0.2,   # Inheritance — involuntary acquisition, no conviction but holding after is weakly positive
    'G': 0.1,   # Gift — wealth transfer with no market view, disposing via gift is less bearish than an open market sale
    'D': 0.1,   # Disposition to company — typically clawback or return of shares, involuntary and weakly negative
    'U': 0.1,   # Tender offer — responding to external acquisition offer, largely involuntary once deal is announced
}