# ===============================
# Configuration
# ===============================
MARKET_TICKER = "^GSPC"
TARGET_TICKER = "AMD"
START_DATE = "1992-01-01"
END_DATE = "2019-12-31"

RAW_DATA_DIR = "FinD_Generator/data/raw"
os.makedirs(RAW_DATA_DIR, exist_ok=True)

PROCESSED_DATA_DIR = "FinD_Generator/data/processed"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

GRAPH_DIR = 'FinD_Generator/image/graph/'
os.makedirs(GRAPH_DIR, exist_ok=True)

FRED_DAILY_IDS = {
        "T10Y2Y": "yield_curve",  # Yield Curve (daily)
}

FRED_MONTHLY_IDS = {
    "CPIAUCSL": "cpi",            # Consumer Price Index (monthly)
    "UNRATE": "unemployment",     # Unemployment Rate (monthly)
    "FEDFUNDS": "interest_rate",  # Interest Rates (monthly)
    "BOPGSTB": "trade_balance",   # US Trade Balance (monthly)
}

FRED_QUARTERLY_IDS = {
    "GDPC1": "gdp",                           # Real GDP (quarterly)
    "GFDEBTN": "gov_debt",                    # Debt Level (quarterly)
    "M318501Q027NBEA": "gov_fiscal_balance",  # US Government Fiscal Balance (quarterly)
    "W006RC1Q027SBEA": "tax_receipts",        # US Government Tax Receipts (quarterly)
    "FGEXPND": "gov_spending"                 # US Government Spending (quarterly)
}

WAVELET = "db4"
WAVELET_LEVEL = 3
PCA_VARIANCE = 0.95  # explained variance ratio for PCA
DEFAULT_SEQ_LEN = 60
DEFAULT_HORIZON = 5
DEFAULT_BATCH = 32
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15