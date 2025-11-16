"""
config.py - Configuration for FinD_Generator

IMPORTANT: This project uses TWO different model architectures:
1. Custom ConditionalTimeGrad (in src/models/)
2. Standard PTS TimeGrad (NOT USED - kept for reference)

Choose ONE and remove the other's config.
"""

import os

# ===============================
# Data Configuration
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

# FRED Data Sources
FRED_DAILY_IDS = {
    "T10Y2Y": "yield_curve",
}

FRED_MONTHLY_IDS = {
    "CPIAUCSL": "cpi",
    "UNRATE": "unemployment",
    "FEDFUNDS": "interest_rate",
    "BOPGSTB": "trade_balance",
}

FRED_QUARTERLY_IDS = {
    "GDPC1": "gdp",
    "GFDEBTN": "gov_debt",
    "M318501Q027NBEA": "gov_fiscal_balance",
    "W006RC1Q027SBEA": "tax_receipts",
    "FGEXPND": "gov_spending"
}

# ===============================
# Preprocessing Configuration
# ===============================
WAVELET = "db4"
WAVELET_LEVEL = 3
PCA_VARIANCE = 0.95

DEFAULT_SEQ_LEN = 60
DEFAULT_HORIZON = 5
DEFAULT_BATCH = 32

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

# ===============================
# Model Configuration
# ===============================
MODEL_DIR = "FinD_Generator/models"
os.makedirs(MODEL_DIR, exist_ok=True)

CHECKPOINT_DIR = "FinD_Generator/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ===============================
# ⚠️ WARNING: CONFLICTING CONFIGURATIONS BELOW
# ===============================
# The following configs are for STANDARD PTS TimeGrad (GluonTS-based)
# If you're using ConditionalTimeGrad (your custom model), DELETE THESE:

# --- TimeGrad/GluonTS Parameters (ONLY if using standard TimeGrad) ---
# FREQ = "B"  # Business day frequency
# CONTEXT_LENGTH = DEFAULT_SEQ_LEN
# PREDICTION_LENGTH = DEFAULT_HORIZON
# TARGET_DIM = 4  # ⚠️ WRONG - TimeGrad expects univariate (1)
# LAGS_SEQ = [1, 2, 3, 4, 5, 6, 7, 30, 60, 90]
# from gluonts.time_feature import time_features_from_frequency_str
# TIME_FEATURES = time_features_from_frequency_str(FREQ)

# ===============================
# Recommended: Use ConditionalTimeGrad Config
# ===============================
# Your custom model doesn't need these GluonTS-specific parameters
# It uses the dataset's dynamic shapes instead

CONDITIONAL_TIMEGRAD_CONFIG = {
    'target_dim': 1,  # ✅ Univariate (required)
    'diff_steps': 100,
    'beta_end': 0.1,
    'beta_schedule': 'linear',
    'residual_layers': 8,
    'residual_channels': 8,
    'dilation_cycle_length': 2,
}